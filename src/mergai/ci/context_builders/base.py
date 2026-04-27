"""Base types for workflow context builders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...config import WorkflowContextConfig


@dataclass
class WorkflowContext:
    """Structured context extracted from a failed CI workflow run.

    Context builders populate this from workflow artifacts, the GitHub API,
    or logs. Handlers consume it: ``CommandHandler`` mostly uses it for
    reporting, while ``ResolveHandler`` feeds ``summary`` + ``details`` +
    ``files_affected`` into the AI prompt.

    Attributes:
        workflow_name: The failing workflow's name (e.g. ``"format"``).
        run_id: GitHub workflow run ID that produced this failure.
        pr_number: Pull request number the run is associated with.
        summary: One-line human-readable summary of the failure.
        files_affected: Paths (repo-relative) implicated by the failure.
        details: Full text content for the AI prompt (the diff, SARIF
            findings, log excerpt — whatever the builder extracts).
        raw_data: Original parsed data, kept for storage/debugging.
    """

    workflow_name: str
    run_id: str
    pr_number: int
    summary: str
    files_affected: list[str] = field(default_factory=list)
    details: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)


class WorkflowContextBuilder(ABC):
    """Abstract base class for context builders.

    A builder maps a ``WorkflowContextConfig`` (with a ``type`` and
    ``source``) to a concrete :class:`WorkflowContext`. Subclasses are
    registered by type via
    :func:`mergai.ci.context_builders.get_context_builder`.
    """

    @abstractmethod
    def build_context(
        self,
        config: WorkflowContextConfig,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        artifacts_dir: str | None,
    ) -> WorkflowContext:
        """Build a :class:`WorkflowContext` for a given failed run.

        Args:
            config: The per-workflow context config (``type``, ``source``,
                ``artifact_name``, ``extract_pattern``).
            workflow_name: Name of the failing workflow.
            run_id: GitHub workflow run ID.
            pr_number: PR number.
            artifacts_dir: Directory with downloaded workflow artifacts.
                Each artifact is extracted into a subdirectory named after
                the artifact.

        Returns:
            Populated WorkflowContext.

        Raises:
            FileNotFoundError: If the expected artifact is missing.
            ValueError: If the artifact content can't be parsed.
        """
