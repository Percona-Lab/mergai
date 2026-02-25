"""Branching point strategy - prioritizes commits with multiple children in upstream."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import MergePickStrategy, MergePickStrategyContext, MergePickStrategyResult

if TYPE_CHECKING:
    from git import Commit, Repo


@dataclass
class BranchingPointResult(MergePickStrategyResult):
    """Result for branching point strategy match.

    Attributes:
        child_count: Number of children this commit has in upstream.
    """

    child_count: int

    def format_short(self) -> str:
        """Return a short description of the branching point match."""
        return f"{self.child_count} children in upstream"


@dataclass
class BranchingPointStrategyConfig:
    """Configuration for branching point strategy.

    Currently no configuration options, but the class exists for
    future extensibility (e.g., min_children threshold).
    """

    @classmethod
    def from_dict(cls, data) -> "BranchingPointStrategyConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dict or True for defaults.

        Returns:
            BranchingPointStrategyConfig instance.
        """
        return cls()


class BranchingPointStrategy(MergePickStrategy):
    """Strategy that prioritizes commits that are branching points.

    A branching point is a commit with multiple children in the upstream
    history, indicating where branches diverged. These can be important
    merge points as they often represent significant decision points in
    the upstream development.
    """

    def __init__(self, config: BranchingPointStrategyConfig):
        """Initialize the strategy with configuration.

        Args:
            config: BranchingPointStrategyConfig instance.
        """
        self.config = config

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "branching_point"

    def check(
        self, repo: "Repo", commit: "Commit", context: MergePickStrategyContext
    ) -> BranchingPointResult | None:
        """Check if commit is a branching point.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Strategy context with upstream_ref and optional branching_points_cache.

        Returns:
            BranchingPointResult if the commit is a branching point, None otherwise.
        """
        from ..utils import git_utils

        if not context.upstream_ref:
            return None

        # Use cached branching points if available
        if context.branching_points_cache:
            # Cache only contains commits with >1 child
            if commit.hexsha in context.branching_points_cache:
                child_count = context.branching_points_cache[commit.hexsha]
                return BranchingPointResult(child_count=child_count)
            return None

        # Fallback to individual check if no cache
        is_bp, child_count = git_utils.is_branching_point(
            repo, commit, context.upstream_ref
        )
        if is_bp:
            return BranchingPointResult(child_count=child_count)
        return None
