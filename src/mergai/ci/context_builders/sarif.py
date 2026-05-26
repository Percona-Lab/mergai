"""Context builder for workflows that upload a SARIF file artifact.

Used by the ``clang-tidy`` workflow: clang-tidy.yml uploads
``clang-tidy-results.sarif`` as the ``clang-tidy-results`` artifact
(alongside the Code Scanning upload).

SARIF (Static Analysis Results Interchange Format) is a JSON schema for
reporting static-analysis findings. The parser here extracts enough to
build an AI prompt: affected files and a per-finding summary of
rule-id + location + message. It is deliberately tolerant of missing or
odd fields — real-world SARIF files vary.

The builder picks its source from what the caller provides:

* ``artifacts_dir`` set: read the SARIF from the downloaded artifact.
  If the workflow failed before producing one (e.g. a Bazel error
  during ``compile_commands.json`` generation), fall back to the
  failing job's log so the agent sees the build error rather than a
  silent crash. This is the path used when the watched workflow_run
  fails.
* ``head_sha`` set (no artifacts): fetch the SARIF for that commit via
  Code Scanning's analyses API. This is the path used when the
  workflow_run *passed* but ``code_scanning_check`` is enabled — the
  build was clean, but Code Scanning has findings to fix.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ...config import WorkflowContextConfig
from ._job_log import LOG_TAIL_BYTES as _LOG_TAIL_BYTES
from ._job_log import fetch_failing_job_log
from .base import WorkflowContext, WorkflowContextBuilder

log = logging.getLogger(__name__)


class SARIFContextBuilder(WorkflowContextBuilder):
    """Parses a SARIF JSON file from a workflow artifact.

    Falls back to the failing job's log via the GitHub API when the
    SARIF artifact is missing — see module docstring.
    """

    def build_context(
        self,
        config: WorkflowContextConfig,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        artifacts_dir: str | None,
        head_sha: str | None = None,
    ) -> WorkflowContext:
        if artifacts_dir is not None:
            return self._build_from_artifact(
                config, workflow_name, run_id, pr_number, artifacts_dir
            )
        if head_sha is not None:
            return self._build_from_code_scanning(
                workflow_name, run_id, pr_number, head_sha
            )
        raise FileNotFoundError(
            f"SARIF context for '{workflow_name}' needs either an "
            f"artifacts directory (workflow_run trigger) or a head_sha "
            f"(check_run trigger from Code Scanning); neither was provided."
        )

    def _build_from_artifact(
        self,
        config: WorkflowContextConfig,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        artifacts_dir: str,
    ) -> WorkflowContext:
        """Read SARIF from the downloaded workflow artifact.

        Falls back to the failing job's log when the artifact is missing
        (workflow failed before producing the SARIF report).
        """
        if config.source != "artifact":
            raise NotImplementedError(
                f"SARIF source '{config.source}' is not supported yet "
                f"(only 'artifact'). See PSMDB-1972 follow-ups."
            )
        if len(config.artifact_name) != 1:
            raise ValueError(
                f"Workflow '{workflow_name}' sarif context requires exactly one "
                f"'context.artifact_name'; got {len(config.artifact_name)}"
            )

        sarif_path = self._try_find_sarif(
            artifacts_dir, config.artifact_name[0], workflow_name
        )
        if sarif_path is None:
            return self._build_log_fallback_context(
                workflow_name=workflow_name,
                run_id=run_id,
                pr_number=pr_number,
                artifacts_dir=artifacts_dir,
            )

        sarif_data = json.loads(sarif_path.read_text())
        return self._context_from_sarif(
            sarif_data,
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            extra_raw={"source": "artifact"},
            artifacts_dir=artifacts_dir,
        )

    def _build_from_code_scanning(
        self,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        head_sha: str,
    ) -> WorkflowContext:
        """Fetch SARIF from Code Scanning for the given commit + tool.

        Used when the watched workflow_run *passed* but the per-workflow
        config opts in via ``code_scanning_check: true`` — Code Scanning
        flagged findings even though the build itself was clean.
        """
        if self.app.gh is None:
            raise FileNotFoundError(
                f"Cannot fetch Code Scanning SARIF for '{workflow_name}': "
                f"no GitHub token available."
            )

        analysis = self.find_code_scanning_analysis(
            tool_name=workflow_name, head_sha=head_sha, pr_number=pr_number
        )
        if analysis is None:
            raise FileNotFoundError(
                f"No Code Scanning analyses found for tool '{workflow_name}' "
                f"on refs/pull/{pr_number}/merge."
            )
        sarif_data = self._download_sarif_for_analysis(analysis["id"])
        return self._context_from_sarif(
            sarif_data,
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            extra_raw={
                "source": "code-scanning",
                "analysis_id": analysis["id"],
                "analysis_commit_sha": analysis["commit_sha"],
                "analysis_ref": analysis["ref"],
            },
        )

    def _context_from_sarif(
        self,
        sarif_data: dict[str, Any],
        *,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        extra_raw: dict[str, Any],
        artifacts_dir: str | None = None,
    ) -> WorkflowContext:
        findings = self._flatten_findings(sarif_data)
        files_affected = sorted({f["file"] for f in findings if f["file"]})
        details = self._format_details(findings)

        count = len(findings)
        summary = (
            f"{workflow_name} reported {count} finding{'s' if count != 1 else ''} "
            f"in {len(files_affected)} file{'s' if len(files_affected) != 1 else ''}"
        )

        return WorkflowContext(
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            summary=summary,
            files_affected=files_affected,
            details=details,
            raw_data={"findings": findings, **extra_raw},
            artifacts_dir=artifacts_dir,
        )

    def find_code_scanning_analysis(
        self, *, tool_name: str, head_sha: str, pr_number: int
    ) -> dict[str, Any] | None:
        """Look up the Code Scanning analysis for a commit + tool, or None.

        Filters ``/code-scanning/analyses`` by the PR's merge ref and
        tool name, then matches by ``commit_sha``. Falls back to the
        most-recent analysis on the same ref+tool if no commit_sha match
        — useful when analysis ingestion lags slightly behind the
        workflow_run event, or the merge-commit SHA drifted.

        Returned dicts have ``id``, ``commit_sha``, ``ref``,
        ``results_count``, etc. Used by both this builder
        (``_build_from_code_scanning``) and the orchestrator (to
        pre-check ``results_count`` before running the resolve handler).
        """
        repo = self.app.gh_repo
        ref = f"refs/pull/{pr_number}/merge"

        _, analyses = repo._requester.requestJsonAndCheck(
            "GET",
            f"{repo.url}/code-scanning/analyses",
            parameters={"ref": ref, "tool_name": tool_name, "per_page": 30},
        )

        analysis = next(
            (a for a in analyses if a.get("commit_sha") == head_sha),
            None,
        )
        if analysis is None and analyses:
            log.info(
                "No analysis matches commit %s on %s; using most recent",
                head_sha[:7],
                ref,
            )
            analysis = analyses[0]
        return analysis

    def _download_sarif_for_analysis(self, analysis_id: int) -> dict[str, Any]:
        """Download SARIF for a Code Scanning analysis ID.

        Uses ``Accept: application/sarif+json`` so the API returns the
        original SARIF document GitHub stored — same shape that
        :meth:`_flatten_findings` consumes from artifact files.
        """
        repo = self.app.gh_repo
        _, sarif_data = repo._requester.requestJsonAndCheck(
            "GET",
            f"{repo.url}/code-scanning/analyses/{analysis_id}",
            headers={"Accept": "application/sarif+json"},
        )
        result: dict[str, Any] = sarif_data
        return result

    @staticmethod
    def _try_find_sarif(
        artifacts_dir: str | None, artifact_name: str, workflow_name: str
    ) -> Path | None:
        """Locate the SARIF file in the downloaded artifacts, or return None.

        Returns None when the artifact directory is absent (the workflow
        failed before producing its SARIF report) or contains no
        ``*.sarif`` file. Callers fall back to a log-based context.
        """
        if artifacts_dir is None:
            return None

        artifact_dir = Path(artifacts_dir) / artifact_name
        if not artifact_dir.is_dir():
            return None

        candidate = artifact_dir / f"{workflow_name}-results.sarif"
        if candidate.exists():
            return candidate

        for entry in sorted(artifact_dir.iterdir()):
            if entry.is_file() and entry.suffix == ".sarif":
                return entry

        return None

    def _build_log_fallback_context(
        self,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        artifacts_dir: str | None = None,
    ) -> WorkflowContext:
        """Build a context from the failing job's log when SARIF is absent.

        Pulls the run's failing jobs via PyGithub, downloads the log of
        the first one, and trims it to the tail. The agent reads the
        result as raw build/CI output and can fix the underlying issue
        (for clang-tidy this is typically a BUILD.bazel mismatch after
        an upstream merge).
        """
        try:
            job_log = fetch_failing_job_log(self.app, run_id, _LOG_TAIL_BYTES)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No SARIF file found for '{workflow_name}' and {e}"
            ) from e

        step_label = (
            f" at step '{job_log.failing_step}'" if job_log.failing_step else ""
        )
        summary = (
            f"{workflow_name} failed before producing SARIF; using log of "
            f"job '{job_log.job_name}'{step_label}"
        )

        return WorkflowContext(
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            summary=summary,
            files_affected=[],
            details=job_log.details,
            raw_data={
                "fallback": "job_log",
                "job_id": job_log.job_id,
                "job_name": job_log.job_name,
                "failing_step": job_log.failing_step,
                "truncated": job_log.truncated,
            },
            artifacts_dir=artifacts_dir,
        )

    @staticmethod
    def _flatten_findings(sarif: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ``{rule_id, level, message, file, line}`` per finding."""
        findings: list[dict[str, Any]] = []
        for run in sarif.get("runs", []):
            for result in run.get("results", []):
                rule_id = result.get("ruleId", "")
                level = result.get("level", "warning")
                message = (result.get("message") or {}).get("text", "")

                file_path = ""
                line = 0
                locations = result.get("locations") or []
                if locations:
                    phys = (locations[0] or {}).get("physicalLocation") or {}
                    file_path = (phys.get("artifactLocation") or {}).get("uri", "")
                    line = (phys.get("region") or {}).get("startLine", 0)

                findings.append(
                    {
                        "rule_id": rule_id,
                        "level": level,
                        "message": message,
                        "file": file_path,
                        "line": line,
                    }
                )
        return findings

    @staticmethod
    def _format_details(findings: list[dict[str, Any]]) -> str:
        """Render findings as a Markdown list suitable for AI prompts."""
        if not findings:
            return "(no findings reported)"

        lines = []
        for f in findings:
            loc = f"{f['file']}:{f['line']}" if f["file"] else "(unknown location)"
            header = f"- [{f['level']}] {f['rule_id']} at {loc}"
            lines.append(header)
            if f["message"]:
                lines.append(f"    {f['message']}")
        return "\n".join(lines)
