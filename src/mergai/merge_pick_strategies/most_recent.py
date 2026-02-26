"""Most recent commit fallback strategy - selects the newest unmerged commit."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import MergePickStrategy, MergePickStrategyContext, MergePickStrategyResult

if TYPE_CHECKING:
    from git import Commit, Repo


@dataclass
class MostRecentResult(MergePickStrategyResult):
    """Result for most recent fallback match.

    This result is returned when the most_recent_fallback option is enabled
    and no other strategy found a match for any commit.
    """

    def format_short(self) -> str:
        """Return a short description of the fallback match."""
        return "fallback - most recent unmerged"

    def to_dict(self) -> dict:
        """Return a dictionary representation for JSON serialization."""
        return {
            "fallback": True,
        }


@dataclass
class MostRecentStrategyConfig:
    """Configuration for most recent fallback strategy.

    This strategy has no configuration options - it's a simple fallback
    that always matches when invoked.
    """

    @classmethod
    def from_dict(cls, data) -> "MostRecentStrategyConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dict or True for defaults.

        Returns:
            MostRecentStrategyConfig instance.
        """
        return cls()


class MostRecentStrategy(MergePickStrategy):
    """Fallback strategy that matches the most recent unmerged commit.

    This strategy always returns a match when checked - it's intended to be
    used as a fallback when no other strategy found a match. The caller
    (get_prioritized_commits) is responsible for only invoking this on the
    most recent commit when appropriate.

    Usage in config:
        fork:
          merge_picks:
            most_recent_fallback: true  # Enable this fallback
            strategies:
              - huge_commit: "num_of_files >= 100"
              - branching_point: true
    """

    def __init__(self, config: MostRecentStrategyConfig):
        """Initialize the strategy with configuration.

        Args:
            config: MostRecentStrategyConfig instance.
        """
        self.config = config

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "most_recent"

    def check(
        self, repo: "Repo", commit: "Commit", context: MergePickStrategyContext
    ) -> MostRecentResult | None:
        """Check if commit matches this strategy.

        This strategy always returns a match - it's a fallback strategy
        that should only be called for the most recent unmerged commit
        when no other strategy has matched.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Strategy context (not used by this strategy).

        Returns:
            MostRecentResult always (this is a fallback strategy).
        """
        return MostRecentResult()
