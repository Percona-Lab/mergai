"""Huge commit strategy - prioritizes commits based on expression evaluation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simpleeval import InvalidExpression, simple_eval

from .base import MergePickStrategy, MergePickStrategyContext, MergePickStrategyResult

if TYPE_CHECKING:
    from git import Commit, Repo


@dataclass
class HugeCommitResult(MergePickStrategyResult):
    """Result for huge commit strategy match.

    Attributes:
        expression: The expression that was evaluated.
        evaluated_vars: Dictionary of variable names to their values.
    """

    expression: str
    evaluated_vars: dict[str, Any]

    def format_short(self) -> str:
        """Return a short description showing the evaluated variables."""
        vars_str = ", ".join(f"{k}={v}" for k, v in self.evaluated_vars.items())
        return f"({vars_str}) matched"

    def to_dict(self) -> dict:
        """Return a dictionary representation for JSON serialization."""
        return {
            "expression": self.expression,
            "evaluated_vars": self.evaluated_vars,
        }


@dataclass
class HugeCommitStrategyConfig:
    """Configuration for huge commit strategy.

    Attributes:
        expression: A simpleeval expression to evaluate against commit stats.
            Available variables:
            - num_of_files: Number of files changed
            - num_of_lines: Total lines changed (added + deleted)
            - lines_added: Lines added
            - lines_deleted: Lines deleted
            - num_of_dirs: Number of unique directories modified
    """

    expression: str = ""

    @classmethod
    def from_dict(cls, data) -> "HugeCommitStrategyConfig":
        """Create config from expression string.

        Args:
            data: Expression string for evaluation.

        Returns:
            HugeCommitStrategyConfig instance.

        Raises:
            ValueError: If data is not a valid expression string.
        """
        if isinstance(data, str) and data.strip():
            return cls(expression=data.strip())
        raise ValueError(
            "huge_commit strategy requires an expression string. "
            "Example: 'num_of_files > 100 or num_of_lines > 1000'"
        )


class HugeCommitStrategy(MergePickStrategy):
    """Strategy that prioritizes commits based on expression evaluation.

    A commit matches if the configured expression evaluates to True.
    The expression can use commit statistics as variables.

    Available variables:
        - num_of_files: Number of files changed
        - num_of_lines: Total lines changed (added + deleted)
        - lines_added: Lines added
        - lines_deleted: Lines deleted
        - num_of_dirs: Number of unique directories modified

    Example expressions:
        - "num_of_files > 100"
        - "num_of_files > 100 or num_of_lines > 1000"
        - "(num_of_files > 1000 or num_of_dirs > 20) and num_of_lines > 500"
    """

    def __init__(self, config: HugeCommitStrategyConfig):
        """Initialize the strategy with configuration.

        Args:
            config: HugeCommitStrategyConfig instance.
        """
        self.config = config

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "huge_commit"

    def check(
        self, repo: "Repo", commit: "Commit", context: MergePickStrategyContext
    ) -> HugeCommitResult | None:
        """Check if commit matches the expression.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Strategy context with optional commit_stats_cache.

        Returns:
            HugeCommitResult if the expression evaluates to True, None otherwise.
        """
        from ..utils import git_utils

        # Use cached stats if available, otherwise compute individually
        if context.commit_stats_cache and commit.hexsha in context.commit_stats_cache:
            stats = context.commit_stats_cache[commit.hexsha]
        else:
            stats = git_utils.get_commit_stats(repo, commit)

        # Build variable dictionary for expression evaluation
        variables = {
            "num_of_files": stats.files_changed,
            "num_of_lines": stats.total_lines,
            "lines_added": stats.lines_added,
            "lines_deleted": stats.lines_deleted,
            "num_of_dirs": stats.num_of_dirs,
        }

        try:
            result = simple_eval(self.config.expression, names=variables)
        except InvalidExpression as e:
            raise ValueError(
                f"Invalid huge_commit expression '{self.config.expression}': {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error evaluating huge_commit expression '{self.config.expression}': {e}"
            ) from e

        if result:
            return HugeCommitResult(
                expression=self.config.expression,
                evaluated_vars=variables,
            )
        return None
