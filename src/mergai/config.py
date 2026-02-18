"""Configuration file support for MergAI.

This module handles loading and parsing the .mergai/config.yaml configuration file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import yaml

from .models import CommitSerializationConfig, ContextSerializationConfig
from .merge_pick_strategies import MergePickStrategy, ImportantFilesStrategy

log = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = ".mergai/config.yaml"
DEFAULT_COMMIT_FIELDS = ["hexsha"]


@dataclass
class ForkConfig:
    """Configuration for the fork subcommand.

    Attributes:
        upstream_url: URL of the upstream repository to sync from.
        upstream_branch: Branch name to use when auto-detecting upstream ref.
        upstream_remote: Name of the git remote for upstream (if not set, derived from URL).
        merge_picks: Configuration for commit prioritization in fork merge-pick.
    """

    upstream_url: Optional[str] = None
    upstream_branch: str = "master"
    upstream_remote: Optional[str] = None
    merge_picks: Optional["MergePicksConfig"] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ForkConfig":
        """Create a ForkConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            ForkConfig instance with values from data.
        """
        merge_picks_data = data.get("merge_picks")
        merge_picks = (
            MergePicksConfig.from_dict(merge_picks_data) if merge_picks_data else None
        )

        return cls(
            upstream_url=data.get("upstream_url"),
            upstream_branch=data.get("upstream_branch", cls.upstream_branch),
            upstream_remote=data.get("upstream_remote"),
            merge_picks=merge_picks,
        )


@dataclass
class ResolveConfig:
    """Configuration for the resolve command.

    Attributes:
        agent: Agent type to use for resolution (e.g., "gemini-cli", "opencode").
        max_attempts: Maximum number of retry attempts for resolution.
    """

    agent: str = "gemini-cli"
    max_attempts: int = 3

    @classmethod
    def from_dict(cls, data: dict) -> "ResolveConfig":
        """Create a ResolveConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            ResolveConfig instance with values from data.
        """
        return cls(
            agent=data.get("agent", cls.agent),
            max_attempts=data.get("max_attempts", cls.max_attempts),
        )


@dataclass
class BranchConfig:
    """Configuration for branch naming.

    The format string uses %(token) syntax for substitution.

    Attributes:
        name_format: Format string for branch names.
            Required tokens:
            - %(target_branch) - The target branch name (required)
            - %(merge_commit_sha) or %(merge_commit_short_sha) - SHA of the merge commit
            - %(target_branch_sha) or %(target_branch_short_sha) - SHA of the target branch

            Optional tokens:
            - %(type) - Branch type (main, conflict, solution, or custom)

            SHA token variants:
            - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
            - %(merge_commit_short_sha) - Short SHA of the merge commit (11 chars)
            - %(target_branch_sha) - Full SHA of the target branch (40 chars)
            - %(target_branch_short_sha) - Short SHA of the target branch (11 chars)
    """

    name_format: str = "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)"

    @classmethod
    def from_dict(cls, data: dict) -> "BranchConfig":
        """Create a BranchConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            BranchConfig instance with values from data.
        """
        return cls(
            name_format=data.get("name_format", cls.name_format),
        )


DEFAULT_COMMIT_FOOTER = "Note: commit created by mergai"


@dataclass
class PRTypeConfig:
    """Configuration for a specific PR type (main or solution).

    The format string uses %(token) syntax for substitution.

    Attributes:
        title_format: Format string for PR titles.

        Available tokens:
        - %(target_branch) - The target branch name
        - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
        - %(merge_commit_short_sha) - Short SHA of the merge commit (11 chars)
    """

    title_format: str = ""

    @classmethod
    def from_dict(cls, data: dict, default_title_format: str = "") -> "PRTypeConfig":
        """Create a PRTypeConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.
            default_title_format: Default title format if not specified.

        Returns:
            PRTypeConfig instance with values from data.
        """
        return cls(
            title_format=data.get("title_format", default_title_format),
        )


# Default title formats
DEFAULT_MAIN_PR_TITLE_FORMAT = "Merge %(merge_commit_short_sha) into %(target_branch)"
DEFAULT_SOLUTION_PR_TITLE_FORMAT = (
    "Resolve conflicts for merge %(merge_commit_short_sha) into %(target_branch)"
)


@dataclass
class PRConfig:
    """Configuration for pull requests.

    Contains separate configuration for main and solution PRs.

    Attributes:
        main: Configuration for main PRs (from main branch to target_branch).
        solution: Configuration for solution PRs (from solution branch to conflict branch).

    Example YAML config:
        pr:
          main:
            title_format: "[MERGE] %(merge_commit_short_sha) -> %(target_branch)"
          solution:
            title_format: "[RESOLVE] Conflicts for %(merge_commit_short_sha) into %(target_branch)"
    """

    main: PRTypeConfig = field(
        default_factory=lambda: PRTypeConfig(title_format=DEFAULT_MAIN_PR_TITLE_FORMAT)
    )
    solution: PRTypeConfig = field(
        default_factory=lambda: PRTypeConfig(
            title_format=DEFAULT_SOLUTION_PR_TITLE_FORMAT
        )
    )

    @classmethod
    def from_dict(cls, data: dict) -> "PRConfig":
        """Create a PRConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            PRConfig instance with values from data.
        """
        main_data = data.get("main", {})
        main_config = PRTypeConfig.from_dict(main_data, DEFAULT_MAIN_PR_TITLE_FORMAT)

        solution_data = data.get("solution", {})
        solution_config = PRTypeConfig.from_dict(
            solution_data, DEFAULT_SOLUTION_PR_TITLE_FORMAT
        )

        return cls(
            main=main_config,
            solution=solution_config,
        )


@dataclass
class CommitConfig:
    """Configuration for commit message generation.

    Controls how commit messages are formatted when MergAI creates commits
    for conflict resolution, merge commits, etc.

    Attributes:
        footer: Footer text appended to all MergAI-generated commit messages.
            Set to empty string to disable the footer.

    Example YAML config:
        commit:
          footer: "Note: commit created by mergai"
    """

    footer: str = DEFAULT_COMMIT_FOOTER

    @classmethod
    def from_dict(cls, data: dict) -> "CommitConfig":
        """Create a CommitConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            CommitConfig instance with values from data.
        """
        return cls(
            footer=data.get("footer", cls.footer),
        )


@dataclass
class FinalizeConfig:
    """Configuration for the finalize command.

    Controls how solution PRs are finalized after being merged into the
    conflict branch.

    Attributes:
        mode: How to finalize the solution PR. Options:
            - 'squash': Squash all commits into a merge commit with combined
                       notes. This creates a clean history with a single merge
                       commit. (Default)
            - 'keep': Validate the repository state and print a summary without
                     modifying any commits. Useful when you want to preserve
                     the individual commit history from the solution PR.
            - 'fast-forward': Remove the GitHub PR merge commit to simulate a
                             fast-forward merge. Keeps the original solution
                             commits with their notes intact. Only removes the
                             merge commit if HEAD is a merge commit without a
                             mergai note and its first parent has a note.

    Example YAML config:
        finalize:
          mode: squash  # or 'keep' or 'fast-forward'
    """

    mode: str = "squash"

    @classmethod
    def from_dict(cls, data: dict) -> "FinalizeConfig":
        """Create a FinalizeConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            FinalizeConfig instance with values from data.
        """
        return cls(
            mode=data.get("mode", cls.mode),
        )


@dataclass
class PromptConfig:
    """Configuration for prompt generation.

    Controls how commits are serialized when generating prompts for AI agents.
    The fields specified here determine what information about commits is
    included in the prompt.

    Attributes:
        commit_fields: List of commit fields to include in prompts.
            Valid values: hexsha, short_sha, author, authored_date, summary,
            message, parents.

    Example YAML config:
        prompt:
          commit_fields:
            - hexsha
            - authored_date
            - summary
            - author
    """

    commit_fields: List[str] = field(
        default_factory=lambda: DEFAULT_COMMIT_FIELDS.copy()
    )

    @classmethod
    def from_dict(cls, data: dict) -> "PromptConfig":
        """Create a PromptConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            PromptConfig instance with values from data.
        """
        return cls(
            commit_fields=data.get("commit_fields", DEFAULT_COMMIT_FIELDS.copy()),
        )

    def to_commit_serialization_config(self) -> CommitSerializationConfig:
        """Convert to CommitSerializationConfig.

        Returns:
            CommitSerializationConfig with fields enabled based on commit_fields.
        """
        return CommitSerializationConfig.from_list(self.commit_fields)

    def to_prompt_serialization_config(self) -> ContextSerializationConfig:
        """Create ContextSerializationConfig for prompt mode.

        Returns:
            ContextSerializationConfig configured for prompt mode with
            commit fields from this config.
        """
        return ContextSerializationConfig.prompt(self.to_commit_serialization_config())


@dataclass
class MergePicksConfig:
    """Configuration for merge-pick strategies.

    Strategies are evaluated in the order they appear in the config list.
    The first matching strategy determines the commit's priority.

    Example YAML config:
        merge_picks:
          most_recent_fallback: true  # Optional: fallback to most recent if no match
          strategies:
            - huge_commit: "num_of_files >= 100 or num_of_lines >= 1000"
            - important_files:
                - BUILD.bazel
                - SConstruct
            - branching_point: true
            - conflict: true

    Available strategies:
        - huge_commit: Prioritize commits based on expression evaluation.
            Uses simpleeval expressions with variables:
            - num_of_files: Number of files changed
            - num_of_lines: Total lines changed (added + deleted)
            - lines_added: Lines added
            - lines_deleted: Lines deleted
            - num_of_dirs: Number of unique directories modified
        - important_files: Prioritize commits touching specific files
        - branching_point: Prioritize commits that are branching points
        - conflict: Prioritize commits that would cause merge conflicts

    Attributes:
        strategies: Ordered list of merge-pick strategies to evaluate.
        most_recent_fallback: If True, select the most recent unmerged commit
            when no other strategy finds a match.
    """

    strategies: List[MergePickStrategy] = field(default_factory=list)
    most_recent_fallback: bool = False

    @classmethod
    def from_dict(cls, data) -> "MergePicksConfig":
        """Parse merge_picks config into strategy instances.

        Args:
            data: Dict or list of strategy definitions from YAML, e.g.:
                {
                    "most_recent_fallback": True,
                    "strategies": [
                        {"huge_commit": "num_of_files > 100 or num_of_lines > 1000"},
                        {"important_files": ["BUILD.bazel"]},
                        {"branching_point": True},
                    ]
                }
                Or legacy list format (strategies only):
                [
                    {"huge_commit": "num_of_files > 100 or num_of_lines > 1000"},
                    {"important_files": ["BUILD.bazel"]},
                    {"branching_point": True},
                ]

        Returns:
            MergePicksConfig with instantiated strategies.
        """
        from .merge_pick_strategies import create_strategy

        # Handle both dict format (new) and list format (legacy)
        if isinstance(data, dict):
            strategies_data = data.get("strategies", [])
            most_recent_fallback = bool(data.get("most_recent_fallback", False))
        elif isinstance(data, list):
            # Legacy format: list of strategies directly
            strategies_data = data
            most_recent_fallback = False
        else:
            return cls()

        strategies = []
        for item in strategies_data:
            if not isinstance(item, dict) or len(item) != 1:
                continue

            strategy_type, strategy_data = next(iter(item.items()))
            strategy = create_strategy(strategy_type, strategy_data)
            if strategy:
                strategies.append(strategy)

        if not strategies:
            # TODO: Verify this approach - should we warn about empty strategies
            # or use a default strategy instead?
            log.warning(
                "No valid strategies in merge_picks config. "
                "No commits will be prioritized."
            )

        return cls(strategies=strategies, most_recent_fallback=most_recent_fallback)


@dataclass
class MergaiConfig:
    """Configuration settings for MergAI.

    All settings are optional and have sensible defaults.

    Attributes:
        fork: Configuration for the fork subcommand (includes merge_picks).
        resolve: Configuration for the resolve command.
        branch: Configuration for branch naming.
        prompt: Configuration for prompt generation.
        commit: Configuration for commit message generation.
        pr: Configuration for pull request titles.
        finalize: Configuration for the finalize command.
        _raw: Raw dictionary data for accessing arbitrary sections.
    """

    fork: ForkConfig = field(default_factory=ForkConfig)
    resolve: ResolveConfig = field(default_factory=ResolveConfig)
    branch: BranchConfig = field(default_factory=BranchConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    pr: PRConfig = field(default_factory=PRConfig)
    finalize: FinalizeConfig = field(default_factory=FinalizeConfig)
    _raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def important_files(self) -> List[str]:
        """Get the list of important files from the merge_picks config, if set."""
        if self.fork and self.fork.merge_picks:
            for strategy in self.fork.merge_picks.strategies:
                if isinstance(strategy, ImportantFilesStrategy):
                    return strategy.config.files
        return []

    def get_section(self, name: str) -> Dict[str, Any]:
        """Get a configuration section by name.

        This allows commands to access their own config sections without
        needing to modify MergaiConfig for each new command.

        Args:
            name: Section name (e.g., "fork", "resolve", "replay").

        Returns:
            Dictionary with the section's configuration, or empty dict if not found.
        """
        return self._raw.get(name, {})

    @classmethod
    def from_dict(cls, data: dict) -> "MergaiConfig":
        """Create a MergaiConfig from a dictionary.

        Unknown keys are stored in _raw for forward compatibility and
        to allow commands to access their own sections.

        Args:
            data: Dictionary with configuration values.

        Returns:
            MergaiConfig instance with values from data, using defaults for missing keys.
        """
        # Parse fork section if present
        fork_data = data.get("fork", {})
        fork_config = ForkConfig.from_dict(fork_data) if fork_data else ForkConfig()

        # Parse resolve section if present
        resolve_data = data.get("resolve", {})
        resolve_config = (
            ResolveConfig.from_dict(resolve_data) if resolve_data else ResolveConfig()
        )

        # Parse branch section if present
        branch_data = data.get("branch", {})
        branch_config = (
            BranchConfig.from_dict(branch_data) if branch_data else BranchConfig()
        )

        # Parse prompt section if present
        prompt_data = data.get("prompt", {})
        prompt_config = (
            PromptConfig.from_dict(prompt_data) if prompt_data else PromptConfig()
        )

        # Parse commit section if present
        commit_data = data.get("commit", {})
        commit_config = (
            CommitConfig.from_dict(commit_data) if commit_data else CommitConfig()
        )

        # Parse pr section if present
        pr_data = data.get("pr", {})
        pr_config = PRConfig.from_dict(pr_data) if pr_data else PRConfig()

        # Parse finalize section if present
        finalize_data = data.get("finalize", {})
        finalize_config = (
            FinalizeConfig.from_dict(finalize_data)
            if finalize_data
            else FinalizeConfig()
        )

        return cls(
            fork=fork_config,
            resolve=resolve_config,
            branch=branch_config,
            prompt=prompt_config,
            commit=commit_config,
            pr=pr_config,
            finalize=finalize_config,
            _raw=data,
        )


def load_config(config_path: Optional[str] = None) -> MergaiConfig:
    """Load configuration from a YAML file.

    If config_path is explicitly provided and the file doesn't exist, raises an error.
    If config_path is None and the default .mergai/config.yaml doesn't exist, returns default config.

    Args:
        config_path: Path to the config file, or None to use the default path.

    Returns:
        MergaiConfig instance with loaded or default values.

    Raises:
        FileNotFoundError: If config_path is explicitly provided but file doesn't exist.
        yaml.YAMLError: If the config file contains invalid YAML.
        ValueError: If the config file contains invalid values.
    """
    explicit_path = config_path is not None
    path = Path(config_path) if config_path else Path(DEFAULT_CONFIG_PATH)

    if not path.exists():
        if explicit_path:
            raise FileNotFoundError(f"Config file not found: {path}")
        # Default path doesn't exist - use default config
        return MergaiConfig()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {path}: {e}")

    # Handle empty file or file with only comments
    if data is None:
        return MergaiConfig()

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping (dictionary)")

    return MergaiConfig.from_dict(data)
