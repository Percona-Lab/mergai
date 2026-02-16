"""Huge commit strategy - prioritizes commits with many changed files/lines."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .base import MergePickStrategy, MergePickStrategyResult, MergePickStrategyContext

if TYPE_CHECKING:
    from git import Repo, Commit


@dataclass
class HugeCommitResult(MergePickStrategyResult):
    """Result for huge commit strategy match.

    Attributes:
        files_changed: Number of files changed in the commit.
        lines_changed: Total lines changed (added + deleted).
    """

    files_changed: int
    lines_changed: int

    def format_short(self) -> str:
        """Return a short description of the huge commit match."""
        return f"{self.files_changed} files, {self.lines_changed} lines"


@dataclass
class HugeCommitStrategyConfig:
    """Configuration for huge commit strategy.

    Attributes:
        min_changed_files: Minimum number of changed files to consider huge.
        min_changed_lines: Minimum number of changed lines to consider huge.
    """

    min_changed_files: int = 100
    min_changed_lines: int = 1000

    @classmethod
    def from_dict(cls, data) -> "HugeCommitStrategyConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dict or True for defaults.

        Returns:
            HugeCommitStrategyConfig instance.
        """
        if isinstance(data, dict):
            return cls(
                min_changed_files=data.get("min_changed_files", 100),
                min_changed_lines=data.get("min_changed_lines", 1000),
            )
        # Allow `huge_commit: true` to use defaults
        return cls()


class HugeCommitStrategy(MergePickStrategy):
    """Strategy that prioritizes commits with many changed files/lines.

    A commit matches if it changes at least min_changed_files files
    OR at least min_changed_lines lines.
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
        """Check if commit is a huge commit.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Strategy context (not used by this strategy).

        Returns:
            HugeCommitResult if the commit is huge, None otherwise.
        """
        from ..utils import git_utils

        stats = git_utils.get_commit_stats(repo, commit)
        if (
            stats.files_changed >= self.config.min_changed_files
            or stats.total_lines >= self.config.min_changed_lines
        ):
            return HugeCommitResult(
                files_changed=stats.files_changed,
                lines_changed=stats.total_lines,
            )
        return None
