"""Conflict strategy - prioritizes commits that would cause merge conflicts."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import MergePickStrategy, MergePickStrategyContext, MergePickStrategyResult

if TYPE_CHECKING:
    from git import Commit, Repo


@dataclass
class ConflictResult(MergePickStrategyResult):
    """Result for conflict strategy match.

    Attributes:
        conflicting_files: List of files that would conflict.
    """

    conflicting_files: list[str] = field(default_factory=list)

    def format_short(self) -> str:
        """Return a short description of the conflict match.

        Shows file names for 1-3 files, otherwise shows count.
        Includes 'first conflict' prefix to indicate this strategy
        only marks the first conflicting commit.
        """
        count = len(self.conflicting_files)
        if count == 0:
            return "first conflict"
        if count == 1:
            return f"first conflict in {self.conflicting_files[0]}"
        if count <= 3:
            return f"first conflict in {', '.join(self.conflicting_files)}"
        return f"first conflict in {count} files"

    def to_dict(self) -> dict:
        """Return a dictionary representation for JSON serialization."""
        return {
            "conflicting_files": self.conflicting_files,
        }


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

    Checks if merging the commit into the fork branch would cause conflicts
    using git merge-tree (read-only operation that doesn't modify working tree).

    This strategy is stateful: it only marks the FIRST commit that would
    introduce a conflict. Once a conflict is found, subsequent calls return
    None immediately (O(1)) since later commits would also conflict with
    the same files. This optimization is critical for performance when
    checking many commits.

    Since commits are evaluated in chronological order (oldest first), this
    ensures we identify the earliest commit that introduces the conflict.
    """

    def __init__(self, config: ConflictStrategyConfig):
        """Initialize the strategy with configuration.

        Args:
            config: ConflictStrategyConfig instance.
        """
        self.config = config
        self._conflict_found = False

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "conflict"

    def check(
        self, repo: "Repo", commit: "Commit", context: MergePickStrategyContext
    ) -> ConflictResult | None:
        """Check if commit would cause merge conflicts.

        Uses git merge-tree to perform a read-only merge simulation.
        This does not modify the working tree or index.

        Once a conflict is found, subsequent calls return None immediately
        to avoid expensive merge-tree operations for every remaining commit.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Strategy context with fork_ref.

        Returns:
            ConflictResult if this is the first commit that would conflict,
            None otherwise (including if a conflict was already found).
        """
        # Skip expensive check if we already found a conflict
        if self._conflict_found:
            return None

        from ..utils import git_utils

        if not context.fork_ref:
            return None

        has_conflict, conflicting_files = git_utils.commit_would_conflict(
            repo, commit, context.fork_ref
        )

        if has_conflict:
            self._conflict_found = True
            return ConflictResult(conflicting_files=conflicting_files)

        return None
