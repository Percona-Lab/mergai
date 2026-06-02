"""Shared job-log fallback for context builders.

When the expected artifact (SARIF, BEP, ...) is missing because the
workflow failed before producing it, builders fall back to the failing
job's log via the GitHub API. The log fetching + excerpting is identical
across builders; only the resulting ``WorkflowContext`` summary and
``raw_data`` differ. This module exposes the shared parts so each
builder can keep its own framing.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from ...app import AppContext

log = logging.getLogger(__name__)

# Cap for the log fallback. Job logs can be megabytes; the agent doesn't
# need every progress line. Tail-oriented because GHA build errors and
# their summary lines always appear at the end of the failing step.
LOG_TAIL_BYTES = 64 * 1024


@dataclass
class FailingJobLog:
    """Excerpt of a failing job's log plus identifying metadata.

    Attributes:
        job_id: GitHub Actions job id.
        job_name: Display name of the failing job.
        failing_step: Name of the failing step, or ``None`` if unknown.
        details: Excerpt of the job log (head+tail when truncated).
        truncated: Whether ``details`` was shortened from the full log.
    """

    job_id: int
    job_name: str
    failing_step: str | None
    details: str
    truncated: bool


def fetch_failing_job_log(
    app: AppContext, run_id: str, max_bytes: int = LOG_TAIL_BYTES
) -> FailingJobLog:
    """Fetch + excerpt the first failing job's log for a workflow run.

    Raises:
        FileNotFoundError: When no GitHub token is available, the run has
            no failing job, or the log download fails.
    """
    if app.gh is None:
        raise FileNotFoundError(
            "No GitHub token available to fetch the job log as fallback."
        )

    run = app.gh_repo.get_workflow_run(int(run_id))
    failing_job = next(
        (j for j in run.jobs() if j.conclusion == "failure"),
        None,
    )
    if failing_job is None:
        raise FileNotFoundError(f"Run {run_id} has no failing job to inspect.")

    failing_step_name = next(
        (s.name for s in (failing_job.steps or []) if s.conclusion == "failure"),
        None,
    )

    log_text = _download_job_log(failing_job.logs_url())
    details = _failing_step_excerpt(log_text, max_bytes)

    return FailingJobLog(
        job_id=failing_job.id,
        job_name=failing_job.name,
        failing_step=failing_step_name,
        details=details,
        truncated=len(log_text) > len(details),
    )


def _download_job_log(url: str) -> str:
    """Fetch a job log from the presigned URL returned by PyGithub.

    ``WorkflowJob.logs_url()`` resolves the ``/jobs/{id}/logs`` 302 and
    returns the Location it points at — a presigned URL that needs no
    auth headers.
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


def _failing_step_excerpt(text: str, max_bytes: int) -> str:
    """Excerpt the failing step's section of a job log.

    GHA writes ``##[group]Run <command>`` at the start of every step and
    ``##[error]Process completed with exit code N.`` at the end of
    failing ones. We anchor on the first error marker (the actual step
    that broke the run, not downstream cleanup noise) and walk back to
    the preceding ``##[group]Run`` to delimit the section. If that
    section is too big to keep whole, return its head + tail joined by
    an omission marker — root-cause errors from build tools usually
    appear at the start of the output while the failure summary
    appears at the end, so a plain tail loses the original error.

    Falls back to a tail-only excerpt if no error marker is found.
    """
    error_marker = "##[error]Process completed with exit code"
    error_idx = text.find(error_marker)
    if error_idx < 0:
        return _tail(text, max_bytes)

    line_start = text.rfind("\n", 0, error_idx) + 1
    error_end = text.find("\n", error_idx)
    section_end = error_end + 1 if error_end >= 0 else len(text)

    group_marker = "##[group]Run "
    group_idx = text.rfind(group_marker, 0, line_start)
    if group_idx < 0:
        return _tail(text[:section_end], max_bytes)

    section_start = text.rfind("\n", 0, group_idx) + 1
    section_size = section_end - section_start

    if section_size <= max_bytes:
        return text[section_start:section_end]

    half = max_bytes // 2
    head = text[section_start : section_start + half]
    tail = text[section_end - half : section_end]

    last_nl = head.rfind("\n")
    if last_nl >= 0:
        head = head[: last_nl + 1]
    first_nl = tail.find("\n")
    if 0 <= first_nl < len(tail) - 1:
        tail = tail[first_nl + 1 :]

    omitted = section_size - len(head) - len(tail)
    return f"{head}... (~{omitted // 1024} KiB omitted) ...\n{tail}"


def _tail(text: str, max_bytes: int) -> str:
    """Return the last ``max_bytes`` of ``text``, prefixed with a marker."""
    if len(text) <= max_bytes:
        return text
    truncated = text[-max_bytes:]
    newline = truncated.find("\n")
    if 0 <= newline < len(truncated) - 1:
        truncated = truncated[newline + 1 :]
    return f"... (log truncated; showing last ~{max_bytes // 1024} KiB)\n{truncated}"
