"""Configuration file support for MergAI.

This module handles loading and parsing the .mergai.yaml configuration file.

The config file supports a nested structure where each command/subcommand
can have its own section. Example:

    # Fork command settings
    fork:
      upstream_url: git@github.com:mongodb/mongo.git

    # Resolve command settings
    resolve:
      agent: gemini-cli
      max_attempts: 3
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


DEFAULT_CONFIG_PATH = ".mergai.yaml"


@dataclass
class ForkConfig:
    """Configuration for the fork subcommand.

    Attributes:
        upstream_url: URL of the upstream repository to sync from.
        upstream_branch: Branch name to use when auto-detecting upstream ref.
        upstream_remote: Name of the git remote for upstream (if not set, derived from URL).
        merge_picks: Configuration for commit prioritization in fork pick.
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
class HugeCommitConfig:
    """Configuration for identifying huge commits.

    Attributes:
        min_changed_files: Minimum number of changed files to consider a commit "huge".
        min_changed_lines: Minimum number of changed lines to consider a commit "huge".
    """

    min_changed_files: int = 100
    min_changed_lines: int = 1000

    @classmethod
    def from_dict(cls, data: dict) -> "HugeCommitConfig":
        """Create a HugeCommitConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            HugeCommitConfig instance with values from data.
        """
        return cls(
            min_changed_files=data.get("min_changed_files", cls.min_changed_files),
            min_changed_lines=data.get("min_changed_lines", cls.min_changed_lines),
        )


@dataclass
class MergePicksConfig:
    """Configuration for the merge pick criteria.

    This config controls how commits are prioritized for merging.
    The criteria are evaluated in order: huge_commits, important_files,
    conflict. The first matching criterion determines the priority.

    The design is extensible - new criteria can be added in the future.

    Attributes:
        huge_commits: Config for identifying huge commits to prioritize.
        important_files: List of file paths that, if modified, should prioritize a commit.
        conflict: If true, prioritize commits that would cause merge conflicts
                  (only when no other criterion matches). Not yet fully implemented.
    """

    huge_commits: Optional[HugeCommitConfig] = None
    important_files: List[str] = field(default_factory=list)
    conflict: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "MergePicksConfig":
        """Create a MergePicksConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            MergePicksConfig instance with values from data.
        """
        huge_commits_data = data.get("huge_commits")
        huge_commits = (
            HugeCommitConfig.from_dict(huge_commits_data) if huge_commits_data else None
        )

        return cls(
            huge_commits=huge_commits,
            important_files=data.get("important_files", []),
            conflict=data.get("conflict", False),
        )


@dataclass
class MergaiConfig:
    """Configuration settings for MergAI.

    All settings are optional and have sensible defaults.

    Attributes:
        fork: Configuration for the fork subcommand (includes merge_picks).
        resolve: Configuration for the resolve command.
        _raw: Raw dictionary data for accessing arbitrary sections.
    """

    fork: ForkConfig = field(default_factory=ForkConfig)
    resolve: ResolveConfig = field(default_factory=ResolveConfig)
    _raw: Dict[str, Any] = field(default_factory=dict)

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

        return cls(
            fork=fork_config,
            resolve=resolve_config,
            _raw=data,
        )


def load_config(config_path: Optional[str] = None) -> MergaiConfig:
    """Load configuration from a YAML file.

    If config_path is explicitly provided and the file doesn't exist, raises an error.
    If config_path is None and the default .mergai.yaml doesn't exist, returns default config.

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
