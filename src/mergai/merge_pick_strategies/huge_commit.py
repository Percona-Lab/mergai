"""Huge commit strategy - prioritizes commits based on expression evaluation."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING

from simpleeval import simple_eval, InvalidExpression

from .base import MergePickStrategy, MergePickStrategyResult, MergePickStrategyContext

if TYPE_CHECKING:
    from git import Repo, Commit


@dataclass
class HugeCommitResult(MergePickStrategyResult):
    """Result for huge commit strategy match.

    Attributes:
        expression: The expression that was evaluated.
        evaluated_vars: Dictionary of variable names to their values.
    """

    expression: str
    evaluated_vars: Dict[str, Any]

    def format_short(self) -> str:
        """Return a short description showing the evaluated variables."""
        vars_str = ", ".join(f"{k}={v}" for k, v in self.evaluated_vars.items())
        return f"({vars_str}) matched"


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
    ) -> Optional[HugeCommitResult]:
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
            )
        except Exception as e:
            raise ValueError(
                f"Error evaluating huge_commit expression '{self.config.expression}': {e}"
            )

        if result:
            return HugeCommitResult(
                expression=self.config.expression,
                evaluated_vars=variables,
            )
        return None
