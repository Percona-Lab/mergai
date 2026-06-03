"""Shared helpers for locating downloaded workflow artifacts on disk.

Each workflow artifact is extracted into a subdirectory named after it under
the run's ``artifacts_dir``. These helpers centralize the path/name resolution
the individual context builders would otherwise each reimplement.
"""

from pathlib import Path

from ...config import WorkflowContextConfig


def resolve_artifact_dir(artifacts_dir: str, names: list[str]) -> Path | None:
    """Return the first ``<artifacts_dir>/<name>`` that is a directory, or None.

    Multi-job workflows upload one of several artifacts depending on which job
    failed, so callers pass all candidate names and take whichever is present.
    """
    root = Path(artifacts_dir)
    for name in names:
        d = root / name
        if d.is_dir():
            return d
    return None


def require_single_artifact_name(
    config: WorkflowContextConfig, workflow_name: str, *, context_label: str
) -> str:
    """Return the sole configured artifact name, or raise if not exactly one.

    The ``diff`` and ``sarif`` builders each consume exactly one artifact;
    ``context_label`` ("diff" / "sarif") is woven into the error message.
    """
    if len(config.artifact_name) != 1:
        raise ValueError(
            f"Workflow '{workflow_name}' {context_label} context requires exactly "
            f"one 'context.artifact_name'; got {len(config.artifact_name)}"
        )
    return config.artifact_name[0]
