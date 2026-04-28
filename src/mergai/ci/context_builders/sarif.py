"""Context builder for workflows that upload a SARIF file artifact.

Used by the ``clang-tidy`` workflow: clang-tidy.yml uploads
``clang-tidy-results.sarif`` as the ``clang-tidy-results`` artifact
(alongside the Code Scanning upload).

SARIF (Static Analysis Results Interchange Format) is a JSON schema for
reporting static-analysis findings. The parser here extracts enough to
build an AI prompt: affected files and a per-finding summary of
rule-id + location + message. It is deliberately tolerant of missing or
odd fields — real-world SARIF files vary.

When the SARIF artifact is absent — typically because the workflow
failed before clang-tidy could run (e.g. a Bazel build error during
``compile_commands.json`` generation) — the builder falls back to
fetching the failing job's log via the GitHub API. The agent then sees
the actual build error rather than a silent crash.
"""

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ...config import WorkflowContextConfig
from .base import WorkflowContext, WorkflowContextBuilder

log = logging.getLogger(__name__)

# Cap for the log fallback. Job logs can be megabytes; the agent doesn't
# need every Bazel progress line. Tail-only because GHA build errors and
# their summary lines ("ERROR: Build did NOT complete successfully")
# always appear at the end of the failing step's output.
_LOG_TAIL_BYTES = 64 * 1024


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
    ) -> WorkflowContext:
        if config.source != "artifact":
            raise NotImplementedError(
                f"SARIF source '{config.source}' is not supported yet "
                f"(only 'artifact'). See PSMDB-1972 follow-ups."
            )
        if not config.artifact_name:
            raise ValueError(
                f"Workflow '{workflow_name}' sarif context requires "
                f"'context.artifact_name' to be set"
            )

        sarif_path = self._try_find_sarif(
            artifacts_dir, config.artifact_name, workflow_name
        )
        if sarif_path is None:
            return self._build_log_fallback_context(
                workflow_name=workflow_name,
                run_id=run_id,
                pr_number=pr_number,
            )

        sarif_data = json.loads(sarif_path.read_text())
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
            raw_data={"findings": findings},
        )

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
    ) -> WorkflowContext:
        """Build a context from the failing job's log when SARIF is absent.

        Pulls the run's failing jobs via PyGithub, downloads the log of
        the first one, and trims it to the tail. The agent reads the
        result as raw build/CI output and can fix the underlying issue
        (for clang-tidy this is typically a BUILD.bazel mismatch after
        an upstream merge).
        """
        if self.app.gh is None:
            raise FileNotFoundError(
                f"No SARIF file found for '{workflow_name}' and no GitHub "
                f"token available to fetch the job log as fallback."
            )

        run = self.app.gh_repo.get_workflow_run(int(run_id))
        failing_job = next(
            (j for j in run.jobs() if j.conclusion == "failure"),
            None,
        )
        if failing_job is None:
            raise FileNotFoundError(
                f"No SARIF file found for '{workflow_name}' and run {run_id} "
                f"has no failing job to inspect."
            )

        failing_step_name = next(
            (s.name for s in (failing_job.steps or []) if s.conclusion == "failure"),
            None,
        )

        log_text = self._download_job_log(failing_job.logs_url())
        details = self._failing_step_excerpt(log_text, _LOG_TAIL_BYTES)

        step_label = f" at step '{failing_step_name}'" if failing_step_name else ""
        summary = (
            f"{workflow_name} failed before producing SARIF; using log of "
            f"job '{failing_job.name}'{step_label}"
        )

        return WorkflowContext(
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            summary=summary,
            files_affected=[],
            details=details,
            raw_data={
                "fallback": "job_log",
                "job_id": failing_job.id,
                "job_name": failing_job.name,
                "failing_step": failing_step_name,
                "truncated": len(log_text) > len(details),
            },
        )

    @staticmethod
    def _download_job_log(url: str) -> str:
        """Fetch a job log from the presigned URL returned by PyGithub.

        ``WorkflowJob.logs_url()`` resolves the ``/jobs/{id}/logs`` 302
        and returns the Location it points at — a presigned URL that
        needs no auth headers.
        """
        try:
            with urllib.request.urlopen(
                url, timeout=30
            ) as resp:  # noqa: S310 — GitHub-issued URL
                data: bytes = resp.read()
                return data.decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            log.warning("Failed to download job log: %s", e)
            raise FileNotFoundError(f"Could not download job log: {e}") from e

    @staticmethod
    def _failing_step_excerpt(text: str, max_bytes: int) -> str:
        """Excerpt the failing step's section of a job log.

        GHA writes ``##[group]Run <command>`` at the start of every step
        and ``##[error]Process completed with exit code N.`` at the end
        of failing ones. We anchor on the *first* error marker (the
        actual step that broke the run, not downstream cleanup noise)
        and walk back to the preceding ``##[group]Run`` to delimit the
        section. If that section is too big to keep whole, return its
        head + tail joined by an omission marker — root-cause errors
        from build tools usually appear at the *start* of the output
        while the failure summary appears at the *end*, so a plain tail
        loses the original error.

        Falls back to a tail-only excerpt if no error marker is found.
        """
        error_marker = "##[error]Process completed with exit code"
        error_idx = text.find(error_marker)
        if error_idx < 0:
            return SARIFContextBuilder._tail(text, max_bytes)

        line_start = text.rfind("\n", 0, error_idx) + 1
        error_end = text.find("\n", error_idx)
        section_end = error_end + 1 if error_end >= 0 else len(text)

        group_marker = "##[group]Run "
        group_idx = text.rfind(group_marker, 0, line_start)
        if group_idx < 0:
            return SARIFContextBuilder._tail(text[:section_end], max_bytes)

        # Walk back to the start of the line that contains ##[group]Run.
        section_start = text.rfind("\n", 0, group_idx) + 1
        section_size = section_end - section_start

        if section_size <= max_bytes:
            return text[section_start:section_end]

        # Section too large — keep head + tail so the agent sees both
        # the originating error and the failure summary.
        half = max_bytes // 2
        head = text[section_start : section_start + half]
        tail = text[section_end - half : section_end]

        # Trim head to a line boundary (last full line) and tail to start
        # at a line boundary (first full line) for readability.
        last_nl = head.rfind("\n")
        if last_nl >= 0:
            head = head[: last_nl + 1]
        first_nl = tail.find("\n")
        if 0 <= first_nl < len(tail) - 1:
            tail = tail[first_nl + 1 :]

        omitted = section_size - len(head) - len(tail)
        return f"{head}... (~{omitted // 1024} KiB omitted) ...\n{tail}"

    @staticmethod
    def _tail(text: str, max_bytes: int) -> str:
        """Return the last ``max_bytes`` of ``text``, prefixed with a marker."""
        if len(text) <= max_bytes:
            return text
        truncated = text[-max_bytes:]
        # Drop the partial first line so the agent doesn't see a
        # mid-line fragment.
        newline = truncated.find("\n")
        if 0 <= newline < len(truncated) - 1:
            truncated = truncated[newline + 1 :]
        return (
            f"... (log truncated; showing last ~{max_bytes // 1024} KiB)\n{truncated}"
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
