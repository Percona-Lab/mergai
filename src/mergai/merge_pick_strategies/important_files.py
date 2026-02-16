"""Important files strategy - prioritizes commits touching specific files."""

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

from .base import MergePickStrategy, MergePickStrategyResult, MergePickStrategyContext

if TYPE_CHECKING:
    from git import Repo, Commit


@dataclass
class ImportantFilesResult(MergePickStrategyResult):
    """Result for important files strategy match.

    Attributes:
        matched_files: List of important files that were modified.
    """

    matched_files: List[str]

    def format_short(self) -> str:
        """Return a short description of the important files match."""
        if len(self.matched_files) == 1:
            return self.matched_files[0]
        return f"{len(self.matched_files)} important files"


@dataclass
class ImportantFilesStrategyConfig:
    """Configuration for important files strategy.

    Attributes:
        files: List of file paths considered important.
    """

    files: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data) -> "ImportantFilesStrategyConfig":
        """Create config from dictionary or list.

        Args:
            data: List of file paths or empty/invalid data.

        Returns:
            ImportantFilesStrategyConfig instance.
        """
        if isinstance(data, list):
            return cls(files=data)
        return cls()


class ImportantFilesStrategy(MergePickStrategy):
    """Strategy that prioritizes commits touching important files.

    A commit matches if it modifies any file in the configured list
    of important files.
    """

    def __init__(self, config: ImportantFilesStrategyConfig):
        """Initialize the strategy with configuration.

        Args:
            config: ImportantFilesStrategyConfig instance.
        """
        self.config = config

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "important_files"

    def check(
        self, repo: "Repo", commit: "Commit", context: MergePickStrategyContext
    ) -> Optional[ImportantFilesResult]:
        """Check if commit modifies any important files.

        Args:
            repo: GitPython Repo object.
            commit: The commit to check.
            context: Strategy context (not used by this strategy).

        Returns:
            ImportantFilesResult if important files are modified, None otherwise.
        """
        from ..utils import git_utils

        if not self.config.files:
            return None

        modified = git_utils.get_commit_modified_files(repo, commit)
        matches = sorted(set(modified) & set(self.config.files))
        if matches:
            return ImportantFilesResult(matched_files=matches)
        return None
