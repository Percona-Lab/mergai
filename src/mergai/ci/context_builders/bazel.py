"""Context builder for Bazel build/test workflows.

Reads the Build Event Protocol (BEP) JSON stream + bazel-testlogs/
uploaded as failure artifacts and extracts:

* Aborted events (build couldn't complete — analysis errors, etc.)
* Failed actions (compile/link errors) with their stderr
* Failed test results, augmented with each test's ``test.log``

Used by PSMDB's ``build-and-test`` workflow, which uploads
``build-failure-artifacts`` (bazel-bep.json) from the build job and
``unittest-failure-artifacts`` (bazel-bep.json + bazel-testlogs/) from
the unittests job. The unittests job needs the build job, so a given
failing run carries exactly one of those artifacts — listing both in
``context.artifact_name`` lets one config entry cover the whole
workflow.

If no configured artifact is present (the job crashed before uploading
anything), falls back to the failing job's log via the GitHub API,
same as the SARIF builder.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from ...config import WorkflowContextConfig
from ._job_log import LOG_TAIL_BYTES, fetch_failing_job_log
from .base import WorkflowContext, WorkflowContextBuilder

log = logging.getLogger(__name__)

# Bounded details so prompts don't blow up. Bazel can emit MBs of BEP
# events and tens of failing-test logs.
_MAX_DETAILS_BYTES = 96 * 1024
# Cap per-section so a single noisy failure doesn't crowd out others.
_MAX_SECTION_BYTES = 16 * 1024

_LABEL_TO_PATH = re.compile(r"^//([^:]*):(.+)$")
_SOURCE_PATH = re.compile(r"\bsrc/[\w./-]+\.(?:cpp|cc|c|h|hpp)\b")


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
        if not config.artifact_name:
            raise ValueError(
                f"Workflow '{workflow_name}' bazel context requires at least "
                f"one 'context.artifact_name'"
            )
        if artifacts_dir is None:
            raise FileNotFoundError(
                f"Workflow '{workflow_name}' needs artifacts_dir (bazel context)"
            )

        artifact_dir = self._pick_artifact_dir(artifacts_dir, config.artifact_name)
        if artifact_dir is None:
            log.info(
                "No configured bazel artifact present (%s); falling back to job log",
                ", ".join(config.artifact_name),
            )
            return self._build_log_fallback(
                workflow_name=workflow_name,
                run_id=run_id,
                pr_number=pr_number,
                artifacts_dir=artifacts_dir,
            )

        bep_path = artifact_dir / "bazel-bep.json"
        testlogs_dir = artifact_dir / "bazel-testlogs"

        if not bep_path.is_file():
            log.info(
                "Artifact %s has no bazel-bep.json; falling back to job log",
                artifact_dir.name,
            )
            return self._build_log_fallback(
                workflow_name=workflow_name,
                run_id=run_id,
                pr_number=pr_number,
                artifacts_dir=artifacts_dir,
            )

        failures = self._parse_bep(bep_path)
        sections, files_affected = self._format_failures(failures, testlogs_dir)
        details = self._truncate_total(sections, _MAX_DETAILS_BYTES)

        n_aborted = sum(1 for f in failures if f["kind"] == "aborted")
        n_actions = sum(1 for f in failures if f["kind"] == "action")
        n_tests = sum(1 for f in failures if f["kind"] == "test")
        summary = (
            f"{workflow_name} failed: "
            f"{n_aborted} aborted, "
            f"{n_actions} action error{'s' if n_actions != 1 else ''}, "
            f"{n_tests} test failure{'s' if n_tests != 1 else ''} "
            f"(from {artifact_dir.name})"
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
                "artifact_dir": artifact_dir.name,
                "failure_count": {
                    "aborted": n_aborted,
                    "action": n_actions,
                    "test": n_tests,
                },
            },
            artifacts_dir=artifacts_dir,
        )

    # ---- artifact location -------------------------------------------------

    @staticmethod
    def _pick_artifact_dir(artifacts_dir: str, candidates: list[str]) -> Path | None:
        """Return the first existing artifact subdirectory, or None."""
        root = Path(artifacts_dir)
        for name in candidates:
            d = root / name
            if d.is_dir():
                return d
        return None

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

    # ---- formatting --------------------------------------------------------

    @staticmethod
    def _format_failures(
        failures: list[dict[str, Any]], testlogs_dir: Path
    ) -> tuple[list[str], set[str]]:
        """Render each failure as a Markdown section. Returns (sections, files)."""
        sections: list[str] = []
        files_affected: set[str] = set()

        for entry in failures:
            kind = entry["kind"]
            label = entry["label"]
            message = entry["message"]

            if kind == "test":
                test_log = BazelContextBuilder._read_test_log(testlogs_dir, label)
                body = test_log if test_log else f"(status: {message})"
                header = f"### Test failed: `{label}`"
            elif kind == "action":
                body = message
                header = f"### Action failed: `{label}`"
            else:
                body = message or "(no description)"
                header = f"### Build aborted: {label}"

            body = BazelContextBuilder._cap(body, _MAX_SECTION_BYTES)
            sections.append(f"{header}\n\n```\n{body}\n```")

            for f in BazelContextBuilder._extract_paths(message):
                files_affected.add(f)
            if kind == "test":
                src_hint = BazelContextBuilder._label_to_source_hint(label)
                if src_hint:
                    files_affected.add(src_hint)

        return sections, files_affected

    @staticmethod
    def _read_test_log(testlogs_dir: Path, label: str) -> str:
        """Resolve ``//pkg:tgt`` → ``bazel-testlogs/pkg/tgt/test.log``."""
        m = _LABEL_TO_PATH.match(label)
        if not m:
            return ""
        pkg, tgt = m.group(1), m.group(2)
        log_path = testlogs_dir / pkg / tgt / "test.log"
        if not log_path.is_file():
            return ""
        try:
            return log_path.read_text(errors="replace")
        except OSError as e:
            log.warning("Could not read %s: %s", log_path, e)
            return ""

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

    # ---- sizing ------------------------------------------------------------

    @staticmethod
    def _cap(text: str, max_bytes: int) -> str:
        if len(text) <= max_bytes:
            return text
        half = max_bytes // 2
        head = text[:half]
        tail = text[-half:]
        omitted = len(text) - len(head) - len(tail)
        return f"{head}\n... ({omitted // 1024} KiB omitted) ...\n{tail}"

    @staticmethod
    def _truncate_total(sections: list[str], max_bytes: int) -> str:
        if not sections:
            return "(no failures parsed from bazel-bep.json)"

        out: list[str] = []
        used = 0
        for s in sections:
            piece = s + "\n\n"
            if used + len(piece) > max_bytes:
                remaining = len(sections) - len(out)
                out.append(f"... ({remaining} more failure(s) omitted)\n")
                break
            out.append(piece)
            used += len(piece)
        return "".join(out)

    # ---- fallback ----------------------------------------------------------

    def _build_log_fallback(
        self,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        artifacts_dir: str,
    ) -> WorkflowContext:
        """Build a context from the failing job's log when no BEP is present."""
        try:
            job_log = fetch_failing_job_log(self.app, run_id, LOG_TAIL_BYTES)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No bazel-bep.json found for '{workflow_name}' and {e}"
            ) from e

        step_label = (
            f" at step '{job_log.failing_step}'" if job_log.failing_step else ""
        )
        summary = (
            f"{workflow_name} failed before producing bazel-bep.json; "
            f"using log of job '{job_log.job_name}'{step_label}"
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
