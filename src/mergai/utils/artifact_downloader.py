"""Download and extract artifacts from a GitHub Actions workflow run.

Used by ``mergai ci fix`` so the workflow YAML doesn't have to call
the artifacts API itself. The layout matches what the context builders
expect: each artifact extracted into ``<dest>/<artifact_name>/``.
"""

import io
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from github.WorkflowRun import WorkflowRun

log = logging.getLogger(__name__)

# Per-artifact download timeout — artifacts can be tens of MB so be
# generous. The presigned S3 URL is the slow part.
_DOWNLOAD_TIMEOUT_SECONDS = 120


def download_workflow_run_artifacts(run: WorkflowRun, dest: Path) -> list[str]:
    """Download every artifact of ``run`` into ``dest/<artifact_name>/``.

    Returns the list of artifact names that were extracted. Skips
    expired artifacts and logs (rather than raises) on individual
    download failures so a single transient hiccup doesn't lose the
    whole run's context.
    """
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []

    for artifact in run.get_artifacts():
        if artifact.expired:
            log.info("Skipping expired artifact %r", artifact.name)
            continue
        target_dir = dest / artifact.name
        try:
            zip_url = _resolve_artifact_zip_url(run, artifact)
            zip_bytes = _fetch_bytes(zip_url)
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(target_dir)
            extracted.append(artifact.name)
            log.info("Downloaded artifact %r to %s", artifact.name, target_dir)
        except (urllib.error.URLError, zipfile.BadZipFile, OSError) as e:
            log.warning("Failed to download artifact %r: %s", artifact.name, e)

    return extracted


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


def _fetch_bytes(url: str) -> bytes:
    """Download a presigned URL into memory.

    Artifacts are small enough (tens of MB at most for our use case)
    that in-memory decoding is simpler than streaming to disk before
    unzipping.
    """
    with urllib.request.urlopen(  # noqa: S310 — GitHub-issued URL
        url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
    ) as resp:
        data: bytes = resp.read()
        return data
