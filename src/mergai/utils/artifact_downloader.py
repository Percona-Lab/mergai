"""Download and extract artifacts from a GitHub Actions workflow run.

Used by ``mergai ci fix`` so the workflow YAML doesn't have to call
the artifacts API itself. The layout matches what the context builders
expect: each artifact extracted into ``<dest>/<artifact_name>/``.
"""

import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import click
from github.WorkflowRun import WorkflowRun

log = logging.getLogger(__name__)

# Per-artifact download timeout — artifacts can be tens of MB so be
# generous. The presigned S3 URL is the slow part. This is the socket
# timeout: it bounds inactivity per chunk, not the total download.
_DOWNLOAD_TIMEOUT_SECONDS = 120
_CHUNK_SIZE = 256 * 1024
_TTY_UPDATE_INTERVAL_S = 0.2


def download_workflow_run_artifacts(run: WorkflowRun, dest: Path) -> list[str]:
    """Download every artifact of ``run`` into ``dest/<artifact_name>/``.

    Returns the list of artifact names that were extracted. Skips
    expired artifacts and logs (rather than raises) on individual
    download failures so a single transient hiccup doesn't lose the
    whole run's context.
    """
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []

    # GitHub returns artifacts cumulatively across all attempts of a
    # workflow run, with no per-attempt artifacts endpoint. Artifacts
    # from prior attempts describe state that was overwritten by the
    # rerun, so honor `run_started_at` (the latest attempt's start
    # time) and drop anything older.
    attempt_started_at = getattr(run, "run_started_at", None)

    for artifact in run.get_artifacts():
        if artifact.expired:
            log.info("Skipping expired artifact %r", artifact.name)
            continue
        if _is_from_prior_attempt(artifact, attempt_started_at):
            log.info(
                "Skipping artifact %r from prior attempt "
                "(created %s, attempt started %s)",
                artifact.name,
                artifact.created_at,
                attempt_started_at,
            )
            continue
        target_dir = dest / artifact.name
        expected_size = getattr(artifact, "size_in_bytes", None)
        click.echo(
            f"Downloading artifact '{artifact.name}' "
            f"({_format_size(expected_size)})..."
        )
        try:
            zip_url = _resolve_artifact_zip_url(run, artifact)
            zip_bytes = _fetch_bytes(zip_url, expected_size=expected_size)
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(target_dir)
            extracted.append(artifact.name)
            log.info("Downloaded artifact %r to %s", artifact.name, target_dir)
        except (urllib.error.URLError, zipfile.BadZipFile, OSError) as e:
            log.warning("Failed to download artifact %r: %s", artifact.name, e)

    return extracted


def _is_from_prior_attempt(
    artifact: object, attempt_started_at: datetime | None
) -> bool:
    """True when ``artifact`` predates the run's latest attempt.

    Defensive on missing timestamps: if either value is absent, fall
    through to "keep" — better to over-include than to silently drop
    an artifact the caller would otherwise see.
    """
    if attempt_started_at is None:
        return False
    created_at = getattr(artifact, "created_at", None)
    if not isinstance(created_at, datetime):
        return False
    return created_at < attempt_started_at


def _resolve_artifact_zip_url(run: WorkflowRun, artifact: object) -> str:
    """Resolve the presigned zip URL for an artifact.

    The Actions API at ``/repos/.../artifacts/{id}/zip`` answers with a
    302 to a presigned S3 URL; the requester returns the ``Location``
    header verbatim. PyGithub doesn't expose a high-level downloader,
    so we go through ``_requester`` directly (same pattern as
    ``WorkflowJob.logs_url``).
    """
    api_url = artifact.archive_download_url  # type: ignore[attr-defined]
    headers, _ = run._requester.requestBlobAndCheck("GET", api_url)
    location: str = headers["location"]
    return location


def _fetch_bytes(url: str, *, expected_size: int | None = None) -> bytes:
    """Download a presigned URL into memory, reporting progress.

    Streamed in chunks rather than a single ``read()`` so we can emit
    progress instead of blocking silently for tens of seconds on a
    multi-MB artifact.
    """
    buf = io.BytesIO()
    with urllib.request.urlopen(  # noqa: S310 — GitHub-issued URL
        url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
    ) as resp:
        total = expected_size
        if total is None:
            cl = resp.headers.get("Content-Length")
            if cl is not None and cl.isdigit():
                total = int(cl)

        progress = _Progress(total=total)
        while True:
            chunk = resp.read(_CHUNK_SIZE)
            if not chunk:
                break
            buf.write(chunk)
            progress.update(buf.tell())
        progress.finish(buf.tell())
    return buf.getvalue()


class _Progress:
    """Download progress display.

    Carriage-return overwrite on a TTY; one discrete line per 10% bucket
    when stdout is redirected (CI logs, files) so progress is still
    visible without spamming.
    """

    def __init__(self, total: int | None):
        self.total = total
        self.stream = click.get_text_stream("stdout")
        self.tty = self.stream.isatty()
        self.last_tty_emit = 0.0
        self.last_bucket = -1

    def update(self, downloaded: int) -> None:
        if self.tty:
            now = time.monotonic()
            if now - self.last_tty_emit < _TTY_UPDATE_INTERVAL_S:
                return
            self.last_tty_emit = now
            click.echo(f"\r  {self._format(downloaded)}", nl=False)
        elif self.total:
            bucket = downloaded * 10 // self.total
            if bucket != self.last_bucket:
                self.last_bucket = bucket
                click.echo(f"  {self._format(downloaded)}")

    def finish(self, downloaded: int) -> None:
        if self.tty:
            click.echo(f"\r  {self._format(downloaded)}")
        else:
            click.echo(f"  {self._format(downloaded)}")

    def _format(self, downloaded: int) -> str:
        d = _format_size(downloaded)
        if self.total:
            t = _format_size(self.total)
            pct = downloaded * 100 // self.total
            return f"{d} / {t} ({pct}%)"
        return d


def _format_size(n: int | None) -> str:
    if n is None:
        return "unknown size"
    if n < 1024:
        return f"{n} B"
    size = float(n)
    for unit in ("KB", "MB", "GB"):
        size /= 1024
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} GB"
