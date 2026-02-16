from dataclasses import dataclass
from enum import StrEnum
from typing import Optional
import re
from . import git_utils
from ..models import MergeInfo
from ..config import BranchConfig

@dataclass
class ParsedBranchName:
    """Parsed components of a mergai branch name.

    This class represents the result of parsing a branch name that was
    generated using BranchNameBuilder. It extracts the original components
    from the branch name string.

    Attributes:
        target_branch: The original target branch name (e.g., "master", "v8.0")
        target_branch_sha: SHA of target branch (short or full depending on branch format)
        merge_commit_sha: SHA of the merge commit (short or full depending on branch format)
        branch_type: The branch type string (e.g., "main", "conflict", "solution")
        full_name: The full original branch name that was parsed
    """

    target_branch: str
    target_branch_sha: str
    merge_commit_sha: str
    branch_type: str
    full_name: str

    def is_standard_type(self) -> bool:
        """Check if the branch type is one of the standard BranchType values."""
        return self.branch_type in [t.value for t in BranchType]


class BranchType(StrEnum):
    """Standard branch types for merge conflict resolution workflow.

    Attributes:
        MAIN: Main working branch where all work for merging a given commit
              will be merged.
        CONFLICT: Branch containing a merge commit with committed merge markers.
        SOLUTION: Branch with solution attempt(s). PRs are created from solution
                  to conflict branch.
    """

    MAIN = "main"
    CONFLICT = "conflict"
    SOLUTION = "solution"
    TARGET = "target"


class BranchNameBuilder:
    """Builder for generating standardized branch names.

    The builder uses a format string with tokens that get replaced:
    - %(target_branch) - The target branch being merged into (required)
    - %(target_branch_sha) - Full SHA of the target branch (40 chars)
    - %(target_branch_short_sha) - Short SHA of the target branch (11 chars)
    - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
    - %(merge_commit_short_sha) - Short SHA of the merge commit (11 chars)
    - %(type) - Branch type identifier

    The format string must contain:
    - %(target_branch)
    - Either %(merge_commit_sha) or %(merge_commit_short_sha)
    - Either %(target_branch_sha) or %(target_branch_short_sha)

    Example format: "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)"
    Produces: "mergai/main-abc12345678-def09876543/solution"

    The class is designed to be instantiated once with all context information,
    then used multiple times to generate different branch names.

    Usage:
        merge_info = MergeInfo(
            target_branch="main",
            target_branch_sha="def0987654321fedcba0987654321fedcba09876",
            merge_commit_sha="abc1234567890abcdef1234567890abcdef12345",
        )
        builder = BranchNameBuilder(
            name_format="mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)",
            merge_info=merge_info,
        )

        # Get specific branch types via properties
        main_branch = builder.main_branch
        conflict_branch = builder.conflict_branch
        solution_branch = builder.solution_branch

        # Or use methods for more control
        custom_branch = builder.get_branch_name("custom-type")
        typed_branch = builder.get_branch_name_for_type(BranchType.MAIN)
    """

    # Token pattern for format string - matches %(token_name)
    TOKEN_PATTERN = re.compile(r"%\((\w+)\)")

    # Currently supported tokens
    SUPPORTED_TOKENS = {
        "target_branch",
        "target_branch_sha",
        "target_branch_short_sha",
        "merge_commit_sha",
        "merge_commit_short_sha",
        "type",
    }

    # Required token groups - at least one from each group must be present
    REQUIRED_TOKENS = {"target_branch"}
    REQUIRED_MERGE_COMMIT_TOKENS = {"merge_commit_sha", "merge_commit_short_sha"}
    REQUIRED_TARGET_BRANCH_SHA_TOKENS = {"target_branch_sha", "target_branch_short_sha"}

    def __init__(
        self,
        name_format: str,
        merge_info: MergeInfo,
    ):
        """Initialize the branch name builder.

        Args:
            name_format: Format string with %(token) placeholders.
            merge_info: MergeInfo with target_branch, target_branch_sha, merge_commit_sha.

        Raises:
            ValueError: If name_format is missing required tokens.
        """
        self._validate_format(name_format)
        self._name_format = name_format
        self._merge_info = merge_info

    @classmethod
    def _validate_format(cls, name_format: str) -> None:
        """Validate that the format string contains all required tokens.

        Args:
            name_format: The format string to validate.

        Raises:
            ValueError: If required tokens are missing.
        """
        # Extract all tokens from format string
        tokens_in_format = set(cls.TOKEN_PATTERN.findall(name_format))

        # Check for required target_branch token
        if not cls.REQUIRED_TOKENS & tokens_in_format:
            raise ValueError(
                f"Format string must contain %(target_branch). " f"Got: {name_format}"
            )

        # Check for at least one merge commit token
        if not cls.REQUIRED_MERGE_COMMIT_TOKENS & tokens_in_format:
            raise ValueError(
                f"Format string must contain either %(merge_commit_sha) or %(merge_commit_short_sha). "
                f"Got: {name_format}"
            )

        # Check for at least one target branch SHA token
        if not cls.REQUIRED_TARGET_BRANCH_SHA_TOKENS & tokens_in_format:
            raise ValueError(
                f"Format string must contain either %(target_branch_sha) or %(target_branch_short_sha). "
                f"Got: {name_format}"
            )

    @classmethod
    def from_config(
        cls,
        config: BranchConfig,
        merge_info: MergeInfo,
    ) -> "BranchNameBuilder":
        """Create a builder from a BranchConfig instance.

        Args:
            config: BranchConfig with the name_format.
            merge_info: MergeInfo with target_branch, target_branch_sha, merge_commit_sha.

        Returns:
            Configured BranchNameBuilder instance.
        """
        return cls(name_format=config.name_format, merge_info=merge_info)

    @classmethod
    def parse_branch_name(
        cls,
        branch_name: str,
        name_format: str,
    ) -> Optional[ParsedBranchName]:
        """Parse a branch name back into its components.

        This method reverses the branch name generation process, extracting
        the original target_branch, target_branch_sha, merge_commit_sha, and type from a
        branch name that was created using the given format.

        The parsing is done by converting the format string into a regex
        pattern with named capture groups for each token.

        Args:
            branch_name: The branch name to parse
                        (e.g., "mergai/master-abc12345678-def09876543/main")
            name_format: The format string used to generate branch names
                        (e.g., "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)")

        Returns:
            ParsedBranchName if the branch matches the format, None otherwise.

        Example:
            >>> parsed = BranchNameBuilder.parse_branch_name(
            ...     "mergai/master-abc12345678-def09876543/solution",
            ...     "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)"
            ... )
            >>> parsed.target_branch
            'master'
            >>> parsed.merge_commit_sha
            'abc12345678'
            >>> parsed.target_branch_sha
            'def09876543'
            >>> parsed.branch_type
            'solution'
        """
        # Build regex pattern from format string
        # 1. Escape regex special characters in the format
        # 2. Replace tokens with named capture groups

        # First, escape all regex special characters
        pattern = re.escape(name_format)

        # Define capture patterns for each token type
        # - target_branch: non-greedy match of any characters except the delimiter
        #   that follows it in the format (we use .+? and let the rest of the pattern constrain it)
        # - merge_commit_sha / merge_commit_short_sha: hex characters (git SHA)
        # - target_branch_sha / target_branch_short_sha: hex characters (git SHA)
        # - type: word characters and hyphens (for custom types like "attempt-1")
        # Both full and short variants map to the same capture group
        token_patterns = {
            "target_branch": r"(?P<target_branch>.+?)",
            "target_branch_sha": r"(?P<target_branch_sha>[a-f0-9]+)",
            "target_branch_short_sha": r"(?P<target_branch_sha>[a-f0-9]+)",
            "merge_commit_sha": r"(?P<merge_commit_sha>[a-f0-9]+)",
            "merge_commit_short_sha": r"(?P<merge_commit_sha>[a-f0-9]+)",
            "type": r"(?P<type>[\w-]+)",
        }

        # Replace escaped token placeholders with capture groups
        # re.escape converts %(token) to %\(token\)
        for token, capture_pattern in token_patterns.items():
            escaped_placeholder = re.escape(f"%({token})")
            pattern = pattern.replace(escaped_placeholder, capture_pattern)

        # Anchor the pattern to match the entire string
        pattern = f"^{pattern}$"

        # Try to match the branch name
        match = re.match(pattern, branch_name)
        if match is None:
            return None

        return ParsedBranchName(
            target_branch=match.group("target_branch"),
            target_branch_sha=match.group("target_branch_sha"),
            merge_commit_sha=match.group("merge_commit_sha"),
            branch_type=match.group("type"),
            full_name=branch_name,
        )

    @classmethod
    def parse_branch_name_with_config(
        cls,
        branch_name: str,
        config: "BranchConfig",
    ) -> Optional[ParsedBranchName]:
        """Parse a branch name using format from BranchConfig.

        Convenience method that extracts the name_format from config.

        Args:
            branch_name: The branch name to parse.
            config: BranchConfig with the name_format.

        Returns:
            ParsedBranchName if the branch matches the format, None otherwise.
        """
        return cls.parse_branch_name(branch_name, config.name_format)

    def _build_name(self, branch_type: str) -> str:
        """Build branch name by replacing tokens in format string.

        Args:
            branch_type: The type string to use for %(type) token.

        Returns:
            Formatted branch name with all tokens replaced.
        """
        mi = self._merge_info

        if branch_type == BranchType.TARGET:
            return mi.target_branch

        values = {
            "target_branch": mi.target_branch,
            "target_branch_sha": mi.target_branch_sha,
            "target_branch_short_sha": git_utils.short_sha(mi.target_branch_sha),
            "merge_commit_sha": mi.merge_commit_sha,
            "merge_commit_short_sha": git_utils.short_sha(mi.merge_commit_sha),
            "type": branch_type,
        }

        def replace_token(match: re.Match) -> str:
            token = match.group(1)
            if token in values:
                return values[token]
            # Keep unknown tokens as-is for future extensibility
            return match.group(0)

        return self.TOKEN_PATTERN.sub(replace_token, self._name_format)

    def get_branch_name(self, branch_type: str) -> str:
        """Get branch name for a custom type string.

        This method allows using arbitrary type strings beyond the
        standard BranchType enum values.

        Args:
            branch_type: Custom type string for the branch.

        Returns:
            Formatted branch name.
        """
        return self._build_name(branch_type)

    def get_branch_name_for_type(self, branch_type: BranchType) -> str:
        """Get branch name for a standard BranchType.

        Args:
            branch_type: One of the standard BranchType enum values.

        Returns:
            Formatted branch name.
        """
        return self._build_name(branch_type.value)

    @property
    def main_branch(self) -> str:
        """Get the main working branch name.

        The main branch is where all work for merging a given commit
        will be merged.
        """
        return self._build_name(BranchType.MAIN)

    @property
    def conflict_branch(self) -> str:
        """Get the conflict branch name.

        The conflict branch contains a merge commit with committed
        merge markers.
        """
        return self._build_name(BranchType.CONFLICT)

    @property
    def solution_branch(self) -> str:
        """Get the solution branch name.

        The solution branch contains solution attempt(s). PRs are
        created from solution to conflict branch.
        """
        return self._build_name(BranchType.SOLUTION)

    @property
    def target_branch(self) -> str:
        """Get the target branch name.

        The target branch is the branch we're merging into.
        """
        return self._build_name(BranchType.TARGET)