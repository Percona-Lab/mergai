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
        branches: List of branch names where children of this commit exist.
    """

    branches: list[str]

    def format_short(self) -> str:
        """Return a short description of the branching point match."""
        return f"branches: {', '.join(self.branches)}"

    def to_dict(self) -> dict:
        """Return a dictionary representation for JSON serialization."""
        return {
            "branches": self.branches,
        }


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

        # Use cached branching points if available (None means not loaded)
        if context.branching_points_cache is not None:
            # Cache only contains commits with children on multiple branches
            if commit.hexsha in context.branching_points_cache:
                branches = context.branching_points_cache[commit.hexsha]
                return BranchingPointResult(branches=branches)
            # Cache is loaded but this commit is not a branching point
            return None

        # Fallback to individual check if cache is not loaded
        # Warning: This is expensive as it runs git rev-list --all --children
        is_bp, branches = git_utils.is_branching_point(
            repo, commit, context.upstream_ref
        )
        if is_bp:
            return BranchingPointResult(branches=branches)
        return None
