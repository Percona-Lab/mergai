"""GitHub workflow API utilities for MergAI.

This module provides functions for interacting with GitHub's workflow runs API,
including querying workflow status, getting run URLs, and checking completion.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from github import Repository as GithubRepository

log = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of a GitHub workflow run."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"
    QUEUED = "queued"
    PENDING = "pending"
    UNKNOWN = "unknown"

    @classmethod
    def from_conclusion(cls, conclusion: str | None, status: str) -> "WorkflowStatus":
        """Create WorkflowStatus from GitHub API conclusion and status.

        Args:
            conclusion: The workflow run conclusion (success, failure, etc.)
                       or None if still running.
            status: The workflow run status (completed, in_progress, queued, etc.).

        Returns:
            WorkflowStatus enum value.
        """
        if status == "queued":
            return cls.QUEUED
        if status == "in_progress":
            return cls.IN_PROGRESS
        if status == "pending":
            return cls.PENDING

        if conclusion is None:
            return cls.UNKNOWN

        conclusion_lower = conclusion.lower()
        if conclusion_lower == "success":
            return cls.SUCCESS
        if conclusion_lower == "failure":
            return cls.FAILURE
        if conclusion_lower == "cancelled":
            return cls.CANCELLED
        if conclusion_lower == "skipped":
            return cls.SKIPPED

        return cls.UNKNOWN

    @property
    def is_complete(self) -> bool:
        """Check if status indicates completion (success, failure, cancelled, skipped)."""
        return self in (
            WorkflowStatus.SUCCESS,
            WorkflowStatus.FAILURE,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.SKIPPED,
        )

    @property
    def is_failed(self) -> bool:
        """Check if status indicates failure."""
        return self == WorkflowStatus.FAILURE

    @property
    def is_success(self) -> bool:
        """Check if status indicates success."""
        return self == WorkflowStatus.SUCCESS


@dataclass
class WorkflowRunInfo:
    """Information about a GitHub workflow run.

    Attributes:
        workflow_name: Name of the workflow.
        run_id: Unique identifier for this run.
        run_url: URL to the workflow run page.
        status: Current status of the run.
        head_sha: Commit SHA that triggered the run.
        head_branch: Branch name that triggered the run.
    """

    workflow_name: str
    run_id: int
    run_url: str
    status: WorkflowStatus
    head_sha: str
    head_branch: str

    @property
    def is_complete(self) -> bool:
        """Check if this run is complete."""
        return self.status.is_complete

    @property
    def is_failed(self) -> bool:
        """Check if this run failed."""
        return self.status.is_failed

    @property
    def is_success(self) -> bool:
        """Check if this run succeeded."""
        return self.status.is_success


@dataclass
class WorkflowStatusSummary:
    """Summary of workflow statuses for a PR.

    Attributes:
        workflows: Dict mapping workflow name to WorkflowRunInfo.
        all_complete: Whether all configured workflows have completed.
        any_failed: Whether any workflow has failed.
        missing_workflows: List of configured workflow names with no runs found.
    """

    workflows: dict[str, WorkflowRunInfo]
    all_complete: bool
    any_failed: bool
    missing_workflows: list[str]

    def get_failed_workflows(self) -> list[WorkflowRunInfo]:
        """Get list of failed workflow runs."""
        return [w for w in self.workflows.values() if w.is_failed]

    def get_incomplete_workflows(self) -> list[WorkflowRunInfo]:
        """Get list of incomplete workflow runs."""
        return [w for w in self.workflows.values() if not w.is_complete]


def get_workflow_runs_for_pr(
    gh_repo: GithubRepository.Repository,
    pr_number: int,
    workflow_names: list[str],
) -> WorkflowStatusSummary:
    """Get workflow run status for a PR.

    Queries GitHub API to find the latest workflow runs for the given PR
    and workflow names.

    Args:
        gh_repo: PyGithub Repository object.
        pr_number: Pull request number.
        workflow_names: List of workflow names to check.

    Returns:
        WorkflowStatusSummary with status of all requested workflows.

    Raises:
        Exception: If GitHub API call fails.
    """
    # Get the PR to find the head SHA
    pr = gh_repo.get_pull(pr_number)
    head_sha = pr.head.sha
    head_branch = pr.head.ref

    log.debug(f"Checking workflows for PR #{pr_number}, HEAD: {head_sha[:11]}")

    workflows: dict[str, WorkflowRunInfo] = {}
    missing_workflows: list[str] = []

    for workflow_name in workflow_names:
        run_info = _get_latest_workflow_run(
            gh_repo, workflow_name, head_sha, head_branch
        )
        if run_info:
            workflows[workflow_name] = run_info
        else:
            missing_workflows.append(workflow_name)
            log.debug(f"No run found for workflow '{workflow_name}'")

    all_complete = (
        len(missing_workflows) == 0
        and len(workflows) > 0
        and all(w.is_complete for w in workflows.values())
    )

    any_failed = any(w.is_failed for w in workflows.values())

    return WorkflowStatusSummary(
        workflows=workflows,
        all_complete=all_complete,
        any_failed=any_failed,
        missing_workflows=missing_workflows,
    )


def _get_latest_workflow_run(
    gh_repo: GithubRepository.Repository,
    workflow_name: str,
    head_sha: str,
    head_branch: str,
) -> WorkflowRunInfo | None:
    """Get the latest workflow run for a specific workflow and commit.

    Args:
        gh_repo: PyGithub Repository object.
        workflow_name: Name of the workflow to find.
        head_sha: Commit SHA to filter runs.
        head_branch: Branch name to filter runs.

    Returns:
        WorkflowRunInfo if found, None otherwise.
    """
    try:
        # Get workflow runs for the branch
        runs = gh_repo.get_workflow_runs(
            branch=head_branch,
            head_sha=head_sha,
        )

        # Find runs matching the workflow name
        for run in runs:
            if run.name == workflow_name:
                status = WorkflowStatus.from_conclusion(run.conclusion, run.status)
                return WorkflowRunInfo(
                    workflow_name=workflow_name,
                    run_id=run.id,
                    run_url=run.html_url,
                    status=status,
                    head_sha=run.head_sha,
                    head_branch=run.head_branch,
                )

        return None

    except Exception as e:
        log.warning(f"Error getting workflow runs for '{workflow_name}': {e}")
        return None


def get_workflow_run_by_id(
    gh_repo: GithubRepository.Repository,
    run_id: int,
) -> WorkflowRunInfo | None:
    """Get workflow run info by run ID.

    Args:
        gh_repo: PyGithub Repository object.
        run_id: Workflow run ID.

    Returns:
        WorkflowRunInfo if found, None otherwise.
    """
    try:
        run = gh_repo.get_workflow_run(run_id)
        status = WorkflowStatus.from_conclusion(run.conclusion, run.status)
        return WorkflowRunInfo(
            workflow_name=run.name,
            run_id=run.id,
            run_url=run.html_url,
            status=status,
            head_sha=run.head_sha,
            head_branch=run.head_branch,
        )
    except Exception as e:
        log.warning(f"Error getting workflow run {run_id}: {e}")
        return None


def format_status_icon(status: WorkflowStatus, width: int = 11) -> str:
    """Get an icon/symbol for a workflow status.

    Args:
        status: WorkflowStatus to represent.
        width: Minimum width for the icon (for alignment). Default 11 for "[CANCELLED]".

    Returns:
        Text symbol representing the status, padded to width.
    """
    icons = {
        WorkflowStatus.SUCCESS: "OK",
        WorkflowStatus.FAILURE: "FAIL",
        WorkflowStatus.CANCELLED: "CANCELLED",
        WorkflowStatus.SKIPPED: "SKIPPED",
        WorkflowStatus.IN_PROGRESS: "RUNNING",
        WorkflowStatus.QUEUED: "QUEUED",
        WorkflowStatus.PENDING: "PENDING",
        WorkflowStatus.UNKNOWN: "?",
    }
    icon = icons.get(status, "?")
    # Pad the content inside brackets to achieve consistent width
    # width includes the brackets, so content width is width - 2
    content_width = width - 2
    return f"[{icon:<{content_width}}]"
