"""Context builder for Bazel build/test workflows.

Surfaces three things to the agent and lets it decide what to read:

* A high-level summary derived from the Build Event Protocol (BEP)
  JSON stream when present (aborted / action / test failure counts +
  failing target labels). Just enough for the agent to know what kind
  of failure it is.
* Absolute paths to the failing job logs from the GitHub Actions API,
  saved to disk under ``<artifacts_dir>/_mergai_job_logs/``. These are
  the *general* "what happened in CI" view that applies to any
  workflow type, not just Bazel.
* Absolute path to the artifacts directory, which holds whatever the
  workflow uploaded (BEP, per-target test logs, JUnit XML, etc.).

The agent uses its filesystem tools to read whatever portion is
relevant. Mergai does not embed log contents in the prompt: full
test.log files routinely run to multiple MB and the assertion / failure
point sits buried in the middle, where head+tail truncation would drop
it. Pointer-based context keeps the prompt small while giving the agent
access to everything.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...config import WorkflowContextConfig
from ._job_log import _download_job_log
from .artifacts import resolve_artifact_dir
from .base import WorkflowContext, WorkflowContextBuilder

log = logging.getLogger(__name__)

# Cap the per-failure target listing in case a build aborts thousands of
# targets — only impacts the markdown listing, not the agent's access
# to the underlying files.
_MAX_FAILURE_LINES = 64

_LABEL_TO_PATH = re.compile(r"^//([^:]*):(.+)$")
_SOURCE_PATH = re.compile(r"\bsrc/[\w./-]+\.(?:cpp|cc|c|h|hpp)\b")
_SAFE_FILENAME = re.compile(r"[^\w.-]+")
_JOB_LOGS_SUBDIR = "_mergai_job_logs"


def _safe_filename(name: str) -> str:
    """Coerce an arbitrary job name into a safe filename stem."""
    return _SAFE_FILENAME.sub("_", name).strip("_") or "job"


class BazelContextBuilder(WorkflowContextBuilder):
    """Parses bazel-bep.json + bazel-testlogs/ from one of several artifacts.

    Iterates the configured artifact names and uses the first one whose
    directory exists in the downloaded artifacts. This lets one config
    entry cover multi-job Bazel workflows where each job uploads its own
    artifact on failure.
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
        if config.source != "artifact":
            raise NotImplementedError(
                f"Bazel source '{config.source}' is not supported "
                f"(only 'artifact')."
            )
        if artifacts_dir is None:
            raise FileNotFoundError(
                f"Workflow '{workflow_name}' needs artifacts_dir (bazel context)"
            )

        artifact_dir = (
            resolve_artifact_dir(artifacts_dir, config.artifact_name)
            if config.artifact_name
            else None
        )

        failures: list[dict[str, Any]] = []
        bep_paths: list[Path] = []
        if artifact_dir is not None:
            # Discover every BEP stream in the artifact. The build/unittests
            # jobs upload a single `bazel-bep.json`, but the jstests job runs
            # resmoke in several invocations and uploads one BEP per invocation
            # (`bazel-bep.json` for the reliable batch plus
            # `bazel-bep-<suite>.json` per load-sensitive suite). Parsing only
            # the fixed name would miss failures isolated to a load-sensitive
            # suite, so glob and concatenate all of them.
            bep_paths = sorted(
                p for p in artifact_dir.glob("bazel-bep*.json") if p.is_file()
            )
            if bep_paths:
                for p in bep_paths:
                    failures.extend(self._parse_bep(p))
            else:
                log.info(
                    "Artifact %s has no bazel-bep*.json; BEP summary unavailable",
                    artifact_dir.name,
                )
        else:
            log.info(
                "No configured bazel artifact present (%s); "
                "agent will work from job logs only",
                ", ".join(config.artifact_name or ()),
            )

        job_logs = self._save_failing_job_logs(
            run_id, Path(artifacts_dir) / _JOB_LOGS_SUBDIR
        )

        files_affected = self._files_affected_from_failures(failures)
        n_aborted = sum(1 for f in failures if f["kind"] == "aborted")
        n_actions = sum(1 for f in failures if f["kind"] == "action")
        n_tests = sum(1 for f in failures if f["kind"] == "test")
        if failures:
            summary = (
                f"{workflow_name} failed: "
                f"{n_aborted} aborted, "
                f"{n_actions} action error{'s' if n_actions != 1 else ''}, "
                f"{n_tests} test failure{'s' if n_tests != 1 else ''} "
                f"(from {artifact_dir.name if artifact_dir else 'BEP unavailable'})"
            )
        else:
            failing_job_names = [name for name, _ in job_logs]
            if failing_job_names:
                summary = (
                    f"{workflow_name} failed in "
                    f"{len(failing_job_names)} job(s): "
                    f"{', '.join(failing_job_names)}"
                )
            else:
                summary = f"{workflow_name} failed (no parsable failure detail)"

        details = self._render_details(
            artifacts_dir=artifacts_dir,
            artifact_dir=artifact_dir,
            bep_paths=bep_paths,
            failures=failures,
            job_logs=job_logs,
        )

        return WorkflowContext(
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            summary=summary,
            files_affected=sorted(files_affected),
            details=details,
            raw_data={
                "source": "artifact",
                "artifact_dir": artifact_dir.name if artifact_dir else None,
                "failing_jobs": [name for name, _ in job_logs],
                "failure_count": {
                    "aborted": n_aborted,
                    "action": n_actions,
                    "test": n_tests,
                },
            },
            artifacts_dir=artifacts_dir,
        )

    # ---- failing-job logs --------------------------------------------------

    def _save_failing_job_logs(
        self, run_id: str, dest_dir: Path
    ) -> list[tuple[str, Path]]:
        """Save each failing job's full log to disk; return (name, path) pairs.

        Logs are written verbatim — no truncation. The agent reads them
        on demand. Returns an empty list when GitHub auth is missing,
        the run can't be fetched, or no jobs failed.
        """
        if self.app.gh is None:
            log.info("No GitHub token available; skipping job-log download.")
            return []
        try:
            run = self.app.gh_repo.get_workflow_run(int(run_id))
        except Exception as e:  # noqa: BLE001 — best-effort enrichment
            log.warning("Could not fetch workflow run %s: %s", run_id, e)
            return []

        saved: list[tuple[str, Path]] = []
        for job in run.jobs():
            if job.conclusion != "failure":
                continue
            try:
                log_text = _download_job_log(job.logs_url())
            except FileNotFoundError as e:
                log.warning("Could not download log for job %r: %s", job.name, e)
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            filename = _safe_filename(job.name) + ".log"
            out_path = dest_dir / filename
            out_path.write_text(log_text)
            saved.append((job.name, out_path))
            log.info("Saved failing job log: %r -> %s", job.name, out_path)
        return saved

    # ---- details rendering -------------------------------------------------

    @staticmethod
    def _render_details(
        *,
        artifacts_dir: str,
        artifact_dir: Path | None,
        bep_paths: list[Path],
        failures: list[dict[str, Any]],
        job_logs: list[tuple[str, Path]],
    ) -> str:
        """Render pointer-based Markdown for the agent's prompt.

        Lists failing job logs, failing bazel targets (label + kind, no
        per-target file paths), and the artifacts directory. The agent
        reads any file it needs with its filesystem tools.
        """
        sections: list[str] = []

        if job_logs:
            lines = ["## Failing job logs (full logs saved to disk)"]
            for name, path in job_logs:
                lines.append(f"- `{name}` -> `{path}`")
            sections.append("\n".join(lines))
        else:
            sections.append(
                "## Failing job logs\n\n"
                "_None saved; mergai could not fetch them from the "
                "GitHub Actions API._"
            )

        if failures:
            lines = ["## Failing bazel targets"]
            if bep_paths:
                src = ", ".join(f"`{p}`" for p in bep_paths)
                lines.append(f"_Source: {src}_")
            lines.append("")
            for entry in failures[:_MAX_FAILURE_LINES]:
                lines.append(f"- `{entry['label']}` ({entry['kind']})")
            if len(failures) > _MAX_FAILURE_LINES:
                lines.append(
                    f"- ...{len(failures) - _MAX_FAILURE_LINES} more "
                    f"(read the Build Event Protocol stream(s) for the full list)"
                )
            sections.append("\n".join(lines))

        nav_lines = ["## Where to find more"]
        nav_lines.append(f"- Artifacts directory: `{artifacts_dir}`")
        if artifact_dir is not None:
            nav_lines.append(f"- Bazel artifact directory: `{artifact_dir}`")
        for p in bep_paths:
            nav_lines.append(f"- Build Event Protocol stream: `{p}`")
        nav_lines.append("")
        nav_lines.append(
            "Use your filesystem tools (Read, Bash, Glob, Grep) to "
            "inspect any of the files above. Per-target test outputs "
            "(test.log, test.xml, attempt logs) live under the bazel "
            "artifact directory if the workflow uploaded them."
        )
        sections.append("\n".join(nav_lines))

        return "\n\n".join(sections)

    @staticmethod
    def _files_affected_from_failures(failures: list[dict[str, Any]]) -> set[str]:
        """Derive ``files_affected`` hints from BEP failures.

        Pulls source paths out of action stderr (e.g.
        ``src/foo/bar.cpp:123:1: error: ...``) and adds the
        ``<package>/`` directory for failing test labels. Best-effort
        only — the agent doesn't depend on this list.
        """
        out: set[str] = set()
        for entry in failures:
            for f in BazelContextBuilder._extract_paths(entry.get("message", "")):
                out.add(f)
            if entry["kind"] == "test":
                src_hint = BazelContextBuilder._label_to_source_hint(entry["label"])
                if src_hint:
                    out.add(src_hint)
        return out

    # ---- artifact location -------------------------------------------------

    # ---- BEP parsing -------------------------------------------------------

    @staticmethod
    def _parse_bep(bep_path: Path) -> list[dict[str, Any]]:
        """Extract failure records from a BEP newline-JSON file.

        Returns a list of ``{kind, label, message}`` dicts in file order.
        ``kind`` is ``aborted`` (build couldn't proceed), ``action``
        (failed action — compile/link/codegen), or ``test`` (test
        result other than PASSED).
        """
        failures: list[dict[str, Any]] = []
        with bep_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                aborted = event.get("aborted")
                if aborted:
                    failures.append(
                        {
                            "kind": "aborted",
                            "label": aborted.get("reason", "unknown"),
                            "message": (aborted.get("description") or "").strip(),
                        }
                    )
                    continue

                action = event.get("action")
                if action and action.get("success") is False:
                    label = (event.get("id", {}).get("actionCompleted") or {}).get(
                        "label", "?"
                    )
                    failures.append(
                        {
                            "kind": "action",
                            "label": label,
                            "message": BazelContextBuilder._action_message(action),
                        }
                    )
                    continue

                test_result = event.get("testResult")
                if test_result:
                    status = test_result.get("status", "")
                    if status and status != "PASSED":
                        label = (event.get("id", {}).get("testResult") or {}).get(
                            "label", "?"
                        )
                        failures.append(
                            {
                                "kind": "test",
                                "label": label,
                                "message": status,
                            }
                        )
        return failures

    @staticmethod
    def _action_message(action: dict[str, Any]) -> str:
        """Best-effort error message from a failed BEP action event.

        Prefers ``stderr.contents`` (inline), then
        ``failureDetail.message``, then the raw ``stderr.name`` URI as a
        last hint. ``contents`` is only present when bazel was run with
        ``--experimental_build_event_text_pb_file_path_conversion`` or
        the inline-output flag; with the default JSON output stderr is a
        file URI we can't open from the CI artifact.
        """
        stderr = action.get("stderr") or {}
        contents = stderr.get("contents")
        if contents:
            return str(contents)
        failure_detail = action.get("failureDetail") or {}
        msg = failure_detail.get("message")
        if msg:
            return str(msg)
        uri = stderr.get("name") or stderr.get("uri")
        if uri:
            return f"(stderr at {uri})"
        return "(no stderr captured)"

    # ---- failure-derived helpers ------------------------------------------

    @staticmethod
    def _label_to_source_hint(label: str) -> str | None:
        """``//pkg:tgt`` → ``pkg/`` (best-effort source-tree hint)."""
        m = _LABEL_TO_PATH.match(label)
        return f"{m.group(1)}/" if m and m.group(1) else None

    @staticmethod
    def _extract_paths(text: str) -> Iterable[str]:
        """Pull repo-relative source paths out of compiler/link errors.

        Looks for ``src/<...>.{cpp,h,hpp,cc,c}`` tokens — enough to seed
        ``files_affected`` without depending on Bazel-specific URI
        parsing.
        """
        return _SOURCE_PATH.findall(text or "")
