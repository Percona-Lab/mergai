import re
from ..config import PRConfig
from ..models import MergeInfo
from . import git_utils


class PRTitleBuilder:
    """Builder for generating configurable PR titles.

    The builder uses a format string with tokens that get replaced:
    - %(target_branch) - The target branch name
    - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
    - %(merge_commit_short_sha) - Short SHA of the merge commit (11 chars)

    Example format: "Merge %(merge_commit_short_sha) into %(target_branch)"
    Produces: "Merge abc12345678 into main"

    Usage:
        builder = PRTitleBuilder.from_config(config.pr, merge_info)
        main_title = builder.main_title
        solution_title = builder.solution_title
    """

    # Token pattern for format string - matches %(token_name)
    TOKEN_PATTERN = re.compile(r"%\((\w+)\)")

    def __init__(self, config: PRConfig, merge_info: MergeInfo):
        """Initialize the PR title builder.

        Args:
            config: PRConfig with nested main and solution configs.
            merge_info: MergeInfo with target_branch, merge_commit_sha.
        """
        self._config = config
        self._merge_info = merge_info

    @classmethod
    def from_config(cls, config: PRConfig, merge_info: MergeInfo) -> "PRTitleBuilder":
        """Create a builder from a PRConfig instance.

        Args:
            config: PRConfig with nested main and solution configs.
            merge_info: MergeInfo with target_branch, merge_commit_sha.

        Returns:
            Configured PRTitleBuilder instance.
        """
        return cls(config=config, merge_info=merge_info)

    def _build_title(self, format_str: str) -> str:
        """Build PR title by replacing tokens in format string.

        Args:
            format_str: Format string with %(token) placeholders.

        Returns:
            Formatted PR title with all tokens replaced.
        """
        mi = self._merge_info

        values = {
            "target_branch": mi.target_branch,
            "merge_commit_sha": mi.merge_commit_sha,
            "merge_commit_short_sha": git_utils.short_sha(mi.merge_commit_sha),
        }

        def replace_token(match: re.Match) -> str:
            token = match.group(1)
            if token in values:
                return values[token]
            # Keep unknown tokens as-is for future extensibility
            return match.group(0)

        return self.TOKEN_PATTERN.sub(replace_token, format_str)

    @property
    def main_title(self) -> str:
        """Get the main PR title.

        The main PR is from the main branch to target_branch.
        """
        return self._build_title(self._config.main.title_format)

    @property
    def solution_title(self) -> str:
        """Get the solution PR title.

        The solution PR is from the solution branch to conflict branch.
        """
        return self._build_title(self._config.solution.title_format)
