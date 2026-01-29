"""Base classes for pick strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from git import Repo, Commit


@dataclass
class StrategyContext:
    """Context passed to strategies during evaluation.

    This avoids passing multiple parameters through the call chain.
    New context fields can be added here as strategies need them.

    Attributes:
        upstream_ref: The upstream reference (e.g., "origin/master").
        fork_ref: The fork reference (e.g., "HEAD").
    """

    upstream_ref: Optional[str] = None
    fork_ref: Optional[str] = None


class StrategyResult(ABC):
    """Base class for strategy match results.

    Each strategy returns its own result subclass with specific details
    about why the commit matched.
    """

    @abstractmethod
    def format_short(self) -> str:
        """Return a short formatted description of the match.

        This is used for display in the pick command output.

        Returns:
            A concise string describing the match details.
        """
        pass


class PickStrategy(ABC):
    """Abstract base class for pick strategies.

    Each strategy checks if a commit matches its criterion and returns
    a result with match details if successful.

    To add a new strategy:
    1. Create a new module in the strategies package
    2. Define a StrategyConfig dataclass with from_dict() class method
    3. Define a StrategyResult dataclass with format_short() method
    4. Define a Strategy class inheriting from PickStrategy
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
        self, repo: "Repo", commit: "Commit", context: StrategyContext
    ) -> Optional[StrategyResult]:
        """Check if commit matches this strategy.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Additional context (upstream_ref, fork_ref, etc.)

        Returns:
            StrategyResult subclass if matched, None otherwise.
        """
        pass
