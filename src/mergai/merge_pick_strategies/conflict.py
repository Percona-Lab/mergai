"""Conflict strategy - prioritizes commits that would cause merge conflicts."""

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

from .base import MergePickStrategy, MergePickStrategyResult, MergePickStrategyContext

if TYPE_CHECKING:
    from git import Repo, Commit


@dataclass
class ConflictResult(MergePickStrategyResult):
    """Result for conflict strategy match.

    Attributes:
        conflicting_files: List of files that would conflict.
    """

    conflicting_files: List[str] = field(default_factory=list)

    def format_short(self) -> str:
        """Return a short description of the conflict match."""
        if len(self.conflicting_files) == 1:
            return f"conflicts in {self.conflicting_files[0]}"
        if len(self.conflicting_files) > 1:
            return f"conflicts in {len(self.conflicting_files)} files"
        return "would cause conflicts"


@dataclass
class ConflictStrategyConfig:
    """Configuration for conflict strategy.

    Currently no configuration options, but the class exists for
    future extensibility (e.g., specific files to check).
    """

    @classmethod
    def from_dict(cls, data) -> "ConflictStrategyConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dict or True for defaults.

        Returns:
            ConflictStrategyConfig instance.
        """
        return cls()


class ConflictStrategy(MergePickStrategy):
    """Strategy that prioritizes commits that would cause merge conflicts.

    NOTE: Not yet fully implemented. Currently returns None for all commits.

    When implemented, this will check if cherry-picking or merging the
    commit would cause conflicts with the current fork state.
    """

    def __init__(self, config: ConflictStrategyConfig):
        """Initialize the strategy with configuration.

        Args:
            config: ConflictStrategyConfig instance.
        """
        self.config = config

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "conflict"

    def check(
        self, repo: "Repo", commit: "Commit", context: MergePickStrategyContext
    ) -> Optional[ConflictResult]:
        """Check if commit would cause merge conflicts.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Strategy context with fork_ref.

        Returns:
            ConflictResult if the commit would conflict, None otherwise.

        NOTE: Not yet implemented - always returns None.
        """
        # TODO: Implement conflict detection
        # This would check if cherry-picking/merging this commit would cause conflicts
        # Possible implementation:
        # if git_utils.commit_would_conflict(repo, commit, context.fork_ref):
        #     conflicting_files = git_utils.get_conflicting_files(repo, commit, context.fork_ref)
        #     return ConflictResult(conflicting_files=conflicting_files)
        return None
