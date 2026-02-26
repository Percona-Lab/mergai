"""Base classes for merge-pick strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from git import Commit

if TYPE_CHECKING:
    from git import Commit, Repo

    from ..utils.git_utils import CommitStats


@dataclass
class MergePickStrategyContext:
    """Context passed to strategies during evaluation.

    This avoids passing multiple parameters through the call chain.
    New context fields can be added here as strategies need them.

    Attributes:
        upstream_ref: The upstream reference (e.g., "origin/master").
        fork_ref: The fork reference (e.g., "HEAD").
        commit_stats_cache: Pre-computed commit stats (sha -> CommitStats).
            Used by huge_commit and important_files strategies.
        branching_points_cache: Pre-computed branching points (sha -> child_count).
            Only commits with >1 child are included.
    """

    upstream_ref: str | None = None
    fork_ref: str | None = None

    # Batch data caches for performance optimization
    commit_stats_cache: dict[str, "CommitStats"] = field(default_factory=dict)
    branching_points_cache: dict[str, int] = field(default_factory=dict)


class MergePickStrategyResult(ABC):
    """Base class for strategy match results.

    Each strategy returns its own result subclass with specific details
    about why the commit matched.
    """

    @abstractmethod
    def format_short(self) -> str:
        """Return a short formatted description of the match.

        This is used for display in the merge-pick command output.

        Returns:
            A concise string describing the match details.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a dictionary representation for JSON serialization.

        This is used for JSON output format in CLI commands.

        Returns:
            A dictionary with strategy-specific match details.
        """
        pass


class MergePickStrategy(ABC):
    """Abstract base class for merge-pick strategies.

    Each strategy checks if a commit matches its criterion and returns
    a result with match details if successful.

    To add a new strategy:
    1. Create a new module in the strategies package
    2. Define a MergePickStrategyConfig dataclass with from_dict() class method
    3. Define a MergePickStrategyResult dataclass with format_short() method
    4. Define a Strategy class inheriting from MergePickStrategy
    5. Register the strategy in registry.py
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name (e.g., 'huge_commit').

        This is used as an identifier in output and logs.

        Returns:
            The strategy name string.
        """
        pass

    @abstractmethod
    def check(
        self, repo: "Repo", commit: "Commit", context: MergePickStrategyContext
    ) -> MergePickStrategyResult | None:
        """Check if commit matches this strategy.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Additional context (upstream_ref, fork_ref, etc.)

        Returns:
            MergePickStrategyResult subclass if matched, None otherwise.
        """
        pass


@dataclass
class MergePickCommit:
    """A commit that matched a priority strategy.

    Attributes:
        commit: The git commit object.
        strategy_name: Name of the strategy that matched.
        result: The strategy result with match details.
    """

    commit: Commit
    strategy_name: str
    result: MergePickStrategyResult
