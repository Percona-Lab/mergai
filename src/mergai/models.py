"""Context models for MergAI with configurable serialization.

This module provides dataclasses for managing merge and conflict contexts
with support for three serialization modes:

1. STORAGE: Minimal format for note.json (only SHAs)
2. TEMPLATE: Format for Jinja2 templates (with git.Commit objects)
3. PROMPT: Configurable format for AI prompts (dict with selected fields)

Example usage:
    # Load from note.json
    ctx = ConflictContext.from_dict(note["conflict_context"])
    
    # Bind repo for git operations
    ctx.bind_repo(repo)
    
    # Serialize for storage (default)
    note["conflict_context"] = ctx.to_dict()
    
    # Serialize for templates
    template_data = ctx.to_dict(ContextSerializationConfig.template())
    
    # Serialize for prompts with custom fields
    prompt_config = ContextSerializationConfig.prompt(
        CommitSerializationConfig(include_message=True)
    )
    prompt_data = ctx.to_dict(prompt_config)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union, Tuple, TypeVar
from enum import Enum
import re

T = TypeVar("T", bound="MergaiNote")

if TYPE_CHECKING:
    from git import Repo, Commit


class MarkdownFormat(Enum):
    """Markdown output format styles.
    
    SIMPLE: Plain markdown with just commit hashes (no links).
    PR: GitHub PR-friendly markdown with clickable commit links.
    """
    SIMPLE = "simple"
    PR = "pr"


def _parse_github_url_from_remote(remote_url: str) -> Optional[str]:
    """Parse a GitHub repository URL from a git remote URL.
    
    Supports both SSH and HTTPS formats:
    - git@github.com:owner/repo.git
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo
    
    Args:
        remote_url: Git remote URL (SSH or HTTPS format).
    
    Returns:
        GitHub repository URL (https://github.com/owner/repo) or None if not GitHub.
    """
    # SSH format: git@github.com:owner/repo.git
    ssh_match = re.match(r"git@github\.com:(.+?)(?:\.git)?$", remote_url)
    if ssh_match:
        return f"https://github.com/{ssh_match.group(1)}"
    
    # HTTPS format: https://github.com/owner/repo.git or https://github.com/owner/repo
    https_match = re.match(r"https://github\.com/(.+?)(?:\.git)?$", remote_url)
    if https_match:
        return f"https://github.com/{https_match.group(1)}"
    
    return None


@dataclass
class MarkdownConfig:
    """Configuration for markdown output formatting.
    
    Attributes:
        format: Markdown format style (SIMPLE or PR).
        repo_url: Base GitHub repository URL for generating commit links.
                  If None and format is PR, commit links won't be generated.
    
    Example usage:
        # Simple markdown (no links)
        config = MarkdownConfig.simple()
        
        # PR markdown with auto-detected repo URL
        config = MarkdownConfig.for_pr(repo)
        
        # PR markdown with explicit URL
        config = MarkdownConfig.for_pr_with_url("https://github.com/owner/repo")
    """
    format: MarkdownFormat = MarkdownFormat.SIMPLE
    repo_url: Optional[str] = None
    
    @classmethod
    def simple(cls) -> "MarkdownConfig":
        """Create config for simple markdown (no links)."""
        return cls(format=MarkdownFormat.SIMPLE)
    
    @classmethod
    def for_pr(cls, repo: "Repo", remote_name: str = "origin") -> "MarkdownConfig":
        """Create config for PR markdown with auto-detected repo URL.
        
        Args:
            repo: GitPython Repo instance.
            remote_name: Name of the remote to use for URL detection (default: "origin").
        
        Returns:
            MarkdownConfig with PR format and detected repo URL.
        """
        repo_url = None
        try:
            remote = repo.remote(remote_name)
            for url in remote.urls:
                repo_url = _parse_github_url_from_remote(url)
                if repo_url:
                    break
        except (ValueError, AttributeError):
            pass
        return cls(format=MarkdownFormat.PR, repo_url=repo_url)
    
    @classmethod
    def for_pr_with_url(cls, repo_url: str) -> "MarkdownConfig":
        """Create config for PR markdown with explicit repo URL.
        
        Args:
            repo_url: GitHub repository URL (e.g., "https://github.com/owner/repo").
        
        Returns:
            MarkdownConfig with PR format and specified repo URL.
        """
        # Ensure URL doesn't have trailing slash
        return cls(format=MarkdownFormat.PR, repo_url=repo_url.rstrip("/"))
    
    def get_commit_url(self, sha: str) -> Optional[str]:
        """Get the URL for a commit.
        
        Args:
            sha: Commit SHA (full or short).
        
        Returns:
            Full commit URL or None if repo_url is not set.
        """
        if self.repo_url:
            return f"{self.repo_url}/commit/{sha}"
        return None


class EnhancedCommit:
    """Wrapper around git.Commit that adds additional properties.
    
    This class wraps a git.Commit object and adds:
    - commit_url: URL to the commit on GitHub (if markdown_config is provided)
    - All original git.Commit attributes via delegation
    
    Attributes:
        commit: The underlying git.Commit object.
        markdown_config: Optional MarkdownConfig for URL generation.
    
    Example usage:
        commit = repo.commit("abc123")
        config = MarkdownConfig.for_pr(repo)
        enhanced = EnhancedCommit(commit, config)
        print(enhanced.commit_url)  # https://github.com/owner/repo/commit/abc123...
        print(enhanced.hexsha)  # abc123... (delegated to underlying commit)
    """
    
    def __init__(self, commit: "Commit", markdown_config: Optional[MarkdownConfig] = None):
        """Initialize EnhancedCommit.
        
        Args:
            commit: GitPython Commit object to wrap.
            markdown_config: Optional MarkdownConfig for URL generation.
        """
        self._commit = commit
        self._markdown_config = markdown_config
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying commit."""
        return getattr(self._commit, name)
    
    @property
    def commit_url(self) -> Optional[str]:
        """Get the URL to this commit on GitHub.
        
        Returns:
            Commit URL or None if markdown_config doesn't have repo_url.
        """
        if self._markdown_config:
            return self._markdown_config.get_commit_url(self._commit.hexsha)
        return None
    
    @property
    def short_sha(self) -> str:
        """Get the short SHA (first 11 characters).
        
        Returns:
            Short SHA string.
        """
        return self._commit.hexsha[:11]
    
    def format_sha(self, use_short: bool = False) -> str:
        """Format the SHA as markdown, optionally with link.
        
        Args:
            use_short: If True, use short SHA in display.
        
        Returns:
            Markdown formatted SHA (with link if PR format and repo_url available).
        """
        display_sha = self.short_sha if use_short else self._commit.hexsha
        
        if self._markdown_config and self._markdown_config.format == MarkdownFormat.PR:
            url = self.commit_url
            if url:
                return f"[`{display_sha}`]({url})"
        
        return f"`{display_sha}`"


@dataclass
class MergeInfo:
    """Context information for a merge operation.

    This class holds the basic information about a merge operation,
    including the target branch and the commit being merged. Stores only
    SHAs for persistence, but can hydrate to full git.Commit objects when
    a repo is bound.

    Attributes:
        target_branch: The branch being merged into (e.g., "v8.0", "master").
        target_branch_sha: Full SHA of the target branch HEAD (40 chars).
        merge_commit_sha: Full SHA of the commit being merged (40 chars).
    """

    target_branch: str
    target_branch_sha: str
    merge_commit_sha: str

    # Cached repo reference (not serialized)
    _repo: Optional["Repo"] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict, repo: "Repo" = None) -> "MergeInfo":
        """Create MergeInfo from note.json data.

        Args:
            data: Dictionary from note.json with merge_info data.
            repo: Optional GitPython Repo for resolving commits.

        Returns:
            MergeInfo instance.
        """
        return cls(
            target_branch=data["target_branch"],
            target_branch_sha=data["target_branch_sha"],
            merge_commit_sha=data["merge_commit"],
            _repo=repo,
        )

    def bind_repo(self, repo: "Repo") -> "MergeInfo":
        """Bind a repo for commit resolution. Returns self for chaining.

        Args:
            repo: GitPython Repo instance.

        Returns:
            Self for method chaining.
        """
        self._repo = repo
        return self

    # Git object accessors (lazy)
    @property
    def target_branch_commit(self) -> "Commit":
        """Get the target branch HEAD commit object. Requires repo to be bound."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        return self._repo.commit(self.target_branch_sha)

    @property
    def merge_commit(self) -> "Commit":
        """Get the merge commit object. Requires repo to be bound."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        return self._repo.commit(self.merge_commit_sha)

    def to_dict(self) -> dict:
        """Serialize to dict for storage in note.json.

        Returns:
            Dictionary with only essential fields for storage.
        """
        return {
            "target_branch": self.target_branch,
            "target_branch_sha": self.target_branch_sha,
            "merge_commit": self.merge_commit_sha,
        }


class SerializationMode(Enum):
    """Serialization modes for context objects."""

    STORAGE = "storage"  # For note.json - only SHAs
    TEMPLATE = "template"  # For Jinja2 - includes git.Commit objects
    PROMPT = "prompt"  # For AI prompts - configurable dict fields


@dataclass
class CommitSerializationConfig:
    """Configuration for how commits are serialized in prompt mode.

    Default configuration includes: hexsha, authored_date, summary, author.

    Attributes:
        include_hexsha: Include full 40-char SHA.
        include_short_sha: Include shortened 11-char SHA.
        include_author: Include author dict with name and email.
        include_authored_date: Include Unix timestamp of authoring.
        include_summary: Include first line of commit message.
        include_message: Include full commit message.
        include_parents: Include list of parent commit SHAs.
    """

    include_hexsha: bool = True
    include_short_sha: bool = False
    include_author: bool = True
    include_authored_date: bool = True
    include_summary: bool = True
    include_message: bool = False
    include_parents: bool = False

    @classmethod
    def default(cls) -> "CommitSerializationConfig":
        """Default config: hexsha, authored_date, summary, author."""
        return cls()

    @classmethod
    def full(cls) -> "CommitSerializationConfig":
        """Include all fields."""
        return cls(
            include_hexsha=True,
            include_short_sha=True,
            include_author=True,
            include_authored_date=True,
            include_summary=True,
            include_message=True,
            include_parents=True,
        )

    @classmethod
    def minimal(cls) -> "CommitSerializationConfig":
        """Minimal: just hexsha and summary."""
        return cls(
            include_hexsha=True,
            include_short_sha=False,
            include_author=False,
            include_authored_date=False,
            include_summary=True,
            include_message=False,
            include_parents=False,
        )

    @classmethod
    def from_list(cls, fields: List[str]) -> "CommitSerializationConfig":
        """Create config from a list of field names.

        Args:
            fields: List of field names to include. Valid values:
                    hexsha, short_sha, author, authored_date, summary,
                    message, parents.

        Returns:
            CommitSerializationConfig with specified fields enabled.
        """
        return cls(
            include_hexsha="hexsha" in fields,
            include_short_sha="short_sha" in fields,
            include_author="author" in fields,
            include_authored_date="authored_date" in fields,
            include_summary="summary" in fields,
            include_message="message" in fields,
            include_parents="parents" in fields,
        )

    def to_list(self) -> List[str]:
        """Convert config to list of enabled field names."""
        fields = []
        if self.include_hexsha:
            fields.append("hexsha")
        if self.include_short_sha:
            fields.append("short_sha")
        if self.include_author:
            fields.append("author")
        if self.include_authored_date:
            fields.append("authored_date")
        if self.include_summary:
            fields.append("summary")
        if self.include_message:
            fields.append("message")
        if self.include_parents:
            fields.append("parents")
        return fields


@dataclass
class ContextSerializationConfig:
    """Configuration for context serialization.

    Attributes:
        mode: Serialization mode (STORAGE, TEMPLATE, or PROMPT).
        commit_config: Configuration for commit serialization (only used in PROMPT mode).
    """

    mode: SerializationMode = SerializationMode.STORAGE
    commit_config: CommitSerializationConfig = field(
        default_factory=CommitSerializationConfig.default
    )

    @classmethod
    def storage(cls) -> "ContextSerializationConfig":
        """Create config for storage mode (note.json)."""
        return cls(mode=SerializationMode.STORAGE)

    @classmethod
    def template(cls) -> "ContextSerializationConfig":
        """Create config for template mode (Jinja2 with git.Commit objects)."""
        return cls(mode=SerializationMode.TEMPLATE)

    @classmethod
    def prompt(
        cls, commit_config: CommitSerializationConfig = None
    ) -> "ContextSerializationConfig":
        """Create config for prompt mode (AI prompts with configurable fields).

        Args:
            commit_config: Configuration for commit fields. Defaults to
                          CommitSerializationConfig.default().

        Returns:
            ContextSerializationConfig for prompt mode.
        """
        return cls(
            mode=SerializationMode.PROMPT,
            commit_config=commit_config or CommitSerializationConfig.default(),
        )


def _short_sha(sha: str) -> str:
    """Return shortened SHA (11 chars)."""
    return sha[:11]


def _commit_to_dict(commit: "Commit", config: CommitSerializationConfig) -> Union[dict, str]:
    """Serialize a git Commit based on config.

    Args:
        commit: GitPython Commit object.
        config: Configuration specifying which fields to include.

    Returns:
        Dictionary with requested commit fields, or just the SHA string
        if only hexsha is configured.
    """
    # If only hexsha is configured, return just the SHA string
    if (config.include_hexsha and
        not config.include_short_sha and
        not config.include_author and
        not config.include_authored_date and
        not config.include_summary and
        not config.include_message and
        not config.include_parents):
        return commit.hexsha

    result = {}
    if config.include_hexsha:
        result["hexsha"] = commit.hexsha
    if config.include_short_sha:
        result["short_sha"] = _short_sha(commit.hexsha)
    if config.include_author:
        result["author"] = {"name": commit.author.name, "email": commit.author.email}
    if config.include_authored_date:
        result["authored_date"] = commit.authored_date
    if config.include_summary:
        result["summary"] = commit.summary
    if config.include_message:
        result["message"] = commit.message
    if config.include_parents:
        result["parents"] = [p.hexsha for p in commit.parents]
    return result


@dataclass
class ConflictContext:
    """Context for merge conflicts with configurable serialization.

    Stores conflict information with only SHAs for persistence, but can
    hydrate to full git.Commit objects when needed for templates or prompts.

    Attributes:
        ours_commit_sha: Full SHA of our (HEAD) commit.
        theirs_commit_sha: Full SHA of their (MERGE_HEAD) commit.
        base_commit_sha: Full SHA of the merge base commit.
        files: List of conflicting file paths.
        conflict_types: Dict mapping file paths to conflict type strings.
        diffs: Optional dict mapping file paths to diff strings.
        their_commits_shas: Optional dict mapping file paths to lists of commit SHAs.
    """

    ours_commit_sha: str
    theirs_commit_sha: str
    base_commit_sha: str
    files: List[str]
    conflict_types: Dict[str, str]
    diffs: Optional[Dict[str, str]] = None
    their_commits_shas: Optional[Dict[str, List[str]]] = None

    # Cached repo reference (not serialized)
    _repo: Optional["Repo"] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict, repo: "Repo" = None) -> "ConflictContext":
        """Create ConflictContext from note.json data.

        Args:
            data: Dictionary from note.json with conflict_context data.
            repo: Optional GitPython Repo for resolving commits.

        Returns:
            ConflictContext instance.
        """
        return cls(
            ours_commit_sha=data["ours_commit"],
            theirs_commit_sha=data["theirs_commit"],
            base_commit_sha=data["base_commit"],
            files=data["files"],
            conflict_types=data["conflict_types"],
            diffs=data.get("diffs"),
            their_commits_shas=data.get("their_commits"),
            _repo=repo,
        )

    def bind_repo(self, repo: "Repo") -> "ConflictContext":
        """Bind a repo for commit resolution. Returns self for chaining.

        Args:
            repo: GitPython Repo instance.

        Returns:
            Self for method chaining.
        """
        self._repo = repo
        return self

    # Git object accessors (lazy)
    @property
    def ours_commit(self) -> "Commit":
        """Get the ours (HEAD) commit object. Requires repo to be bound."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        return self._repo.commit(self.ours_commit_sha)

    @property
    def theirs_commit(self) -> "Commit":
        """Get the theirs (MERGE_HEAD) commit object. Requires repo to be bound."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        return self._repo.commit(self.theirs_commit_sha)

    @property
    def base_commit(self) -> "Commit":
        """Get the merge base commit object. Requires repo to be bound."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        return self._repo.commit(self.base_commit_sha)

    def get_their_commits(self, file_path: str) -> List["Commit"]:
        """Get list of their commits for a specific file.

        Args:
            file_path: Path to the conflicting file.

        Returns:
            List of Commit objects. Empty list if no commits or file not found.

        Raises:
            RuntimeError: If repo is not bound.
        """
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        if not self.their_commits_shas or file_path not in self.their_commits_shas:
            return []
        return [self._repo.commit(sha) for sha in self.their_commits_shas[file_path]]

    def to_dict(self, config: ContextSerializationConfig = None) -> dict:
        """Serialize based on configuration.

        Args:
            config: Serialization config. Defaults to storage mode.

        Returns:
            dict suitable for the specified mode:
            - STORAGE: Only SHAs for note.json
            - TEMPLATE: With git.Commit objects for Jinja2
            - PROMPT: With configurable commit dict fields
        """
        if config is None:
            config = ContextSerializationConfig.storage()

        if config.mode == SerializationMode.STORAGE:
            return self._to_storage_dict()
        elif config.mode == SerializationMode.TEMPLATE:
            return self._to_template_dict()
        elif config.mode == SerializationMode.PROMPT:
            return self._to_prompt_dict(config.commit_config)
        else:
            raise ValueError(f"Unknown serialization mode: {config.mode}")

    def _to_storage_dict(self) -> dict:
        """Minimal dict for note.json storage."""
        result = {
            "ours_commit": self.ours_commit_sha,
            "theirs_commit": self.theirs_commit_sha,
            "base_commit": self.base_commit_sha,
            "files": self.files,
            "conflict_types": self.conflict_types,
        }
        if self.diffs:
            result["diffs"] = self.diffs
        if self.their_commits_shas:
            result["their_commits"] = self.their_commits_shas
        return result

    def _to_template_dict(self) -> dict:
        """Dict with git.Commit objects for Jinja2 templates."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")

        result = {
            "ours_commit": self.ours_commit,
            "theirs_commit": self.theirs_commit,
            "base_commit": self.base_commit,
            "files": self.files,
            "conflict_types": self.conflict_types,
        }
        if self.diffs:
            result["diffs"] = self.diffs
        if self.their_commits_shas:
            result["their_commits"] = {
                path: self.get_their_commits(path)
                for path in self.their_commits_shas
            }
        return result

    def _to_prompt_dict(self, commit_config: CommitSerializationConfig) -> dict:
        """Dict with expanded commit info for AI prompts."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")

        result = {
            "ours_commit": _commit_to_dict(self.ours_commit, commit_config),
            "theirs_commit": _commit_to_dict(self.theirs_commit, commit_config),
            "base_commit": _commit_to_dict(self.base_commit, commit_config),
            "files": self.files,
            "conflict_types": self.conflict_types,
        }
        if self.diffs:
            result["diffs"] = self.diffs
        if self.their_commits_shas:
            result["their_commits"] = {
                path: [
                    _commit_to_dict(c, commit_config)
                    for c in self.get_their_commits(path)
                ]
                for path in self.their_commits_shas
            }
        return result


@dataclass
class MergeContext:
    """Context for merge operations with configurable serialization.

    Stores merge information with only SHAs for persistence, but can
    hydrate to full git.Commit objects when needed.

    Attributes:
        merge_commit_sha: Full SHA of the commit being merged.
        merged_commits_shas: List of SHAs for all commits being merged.
        important_files_modified: List of important files modified in the merge.
        auto_merged: Optional dict with auto-merge info (strategy, files).
    """

    merge_commit_sha: str
    merged_commits_shas: List[str]
    important_files_modified: List[str]
    auto_merged: Optional[Dict[str, Any]] = None

    # Cached repo reference (not serialized)
    _repo: Optional["Repo"] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict, repo: "Repo" = None) -> "MergeContext":
        """Create MergeContext from note.json data.

        Args:
            data: Dictionary from note.json with merge_context data.
            repo: Optional GitPython Repo for resolving commits.

        Returns:
            MergeContext instance.
        """
        return cls(
            merge_commit_sha=data["merge_commit"],
            merged_commits_shas=data["merged_commits"],
            important_files_modified=data.get("important_files_modified", []),
            auto_merged=data.get("auto_merged"),
            _repo=repo,
        )

    def bind_repo(self, repo: "Repo") -> "MergeContext":
        """Bind a repo for commit resolution. Returns self for chaining.

        Args:
            repo: GitPython Repo instance.

        Returns:
            Self for method chaining.
        """
        self._repo = repo
        return self

    @property
    def merge_commit(self) -> "Commit":
        """Get the merge commit object. Requires repo to be bound."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        return self._repo.commit(self.merge_commit_sha)

    @property
    def merged_commits(self) -> List["Commit"]:
        """Get list of all merged commit objects. Requires repo to be bound."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")
        return [self._repo.commit(sha) for sha in self.merged_commits_shas]

    def to_dict(self, config: ContextSerializationConfig = None) -> dict:
        """Serialize based on configuration.

        Args:
            config: Serialization config. Defaults to storage mode.

        Returns:
            dict suitable for the specified mode.
        """
        if config is None:
            config = ContextSerializationConfig.storage()

        if config.mode == SerializationMode.STORAGE:
            return self._to_storage_dict()
        elif config.mode == SerializationMode.TEMPLATE:
            return self._to_template_dict()
        elif config.mode == SerializationMode.PROMPT:
            return self._to_prompt_dict(config.commit_config)
        else:
            raise ValueError(f"Unknown serialization mode: {config.mode}")

    def _to_storage_dict(self) -> dict:
        """Minimal dict for note.json storage."""
        result = {
            "merge_commit": self.merge_commit_sha,
            "merged_commits": self.merged_commits_shas,
            "important_files_modified": self.important_files_modified,
        }
        if self.auto_merged:
            result["auto_merged"] = self.auto_merged
        return result

    def _to_template_dict(self) -> dict:
        """Dict with git.Commit objects for Jinja2 templates."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")

        result = {
            "merge_commit": self.merge_commit,
            "merged_commits": self.merged_commits,
            "important_files_modified": self.important_files_modified,
        }
        if self.auto_merged:
            result["auto_merged"] = self.auto_merged
        return result

    def _to_prompt_dict(self, commit_config: CommitSerializationConfig) -> dict:
        """Dict with expanded commit info for AI prompts."""
        if self._repo is None:
            raise RuntimeError("Repo not bound. Call bind_repo() first.")

        result = {
            "merge_commit": _commit_to_dict(self.merge_commit, commit_config),
            "merged_commits": [
                _commit_to_dict(c, commit_config) for c in self.merged_commits
            ],
            "important_files_modified": self.important_files_modified,
        }
        if self.auto_merged:
            result["auto_merged"] = self.auto_merged
        return result


@dataclass
class MergaiNote:
    """A MergAI note containing merge information and optional context data.

    This is the main data structure for storing merge-related information.
    The merge_info and mergai_version fields are required; all other fields are optional.

    Attributes:
        merge_info: Required merge information (target branch, commit SHAs).
        mergai_version: Required version of mergai that created/modified this note.
        conflict_context: Optional context for merge conflicts.
        merge_context: Optional context for successful merges.
        solutions: Optional list of AI-generated solutions.
        pr_comments: Optional list of PR comments.
        user_comment: Optional user-provided comment.
        merge_description: Optional AI-generated merge description.
        note_index: Optional index tracking which commits have which fields.
    """

    merge_info: MergeInfo
    mergai_version: str
    conflict_context: Optional[ConflictContext] = None
    merge_context: Optional[MergeContext] = None
    solutions: Optional[List[dict]] = None
    pr_comments: Optional[List[dict]] = None
    user_comment: Optional[dict] = None  # Dict with user, email, date, body
    merge_description: Optional[dict] = None
    note_index: Optional[List[dict]] = None

    # Cached repo reference (not serialized)
    _repo: Optional["Repo"] = field(default=None, repr=False, compare=False)

    # --- Factory Methods ---

    @classmethod
    def from_dict(cls, data: dict, repo: "Repo" = None) -> "MergaiNote":
        """Create MergaiNote from a note.json dict.

        Args:
            data: Dictionary from note.json.
            repo: Optional GitPython Repo for resolving commits.

        Returns:
            MergaiNote instance with repo bound if provided.

        Raises:
            ValueError: If required field 'mergai_version' is missing.
        """
        if "mergai_version" not in data:
            raise ValueError("Note missing required field 'mergai_version'")

        note = cls(
            merge_info=MergeInfo.from_dict(data["merge_info"], repo),
            mergai_version=data["mergai_version"],
            conflict_context=ConflictContext.from_dict(data["conflict_context"], repo) if "conflict_context" in data else None,
            merge_context=MergeContext.from_dict(data["merge_context"], repo) if "merge_context" in data else None,
            solutions=data.get("solutions"),
            pr_comments=data.get("pr_comments"),
            user_comment=data.get("user_comment"),
            merge_description=data.get("merge_description"),
            note_index=data.get("note_index"),
            _repo=repo,
        )
        return note

    @classmethod
    def create(cls, merge_info: MergeInfo, repo: "Repo" = None) -> "MergaiNote":
        """Create a new MergaiNote with the given merge_info.

        Sets mergai_version to the current version of mergai.

        Args:
            merge_info: Required merge information.
            repo: Optional GitPython Repo for resolving commits.

        Returns:
            New MergaiNote instance with current mergai version.
        """
        from .version import __version__

        return cls(merge_info=merge_info, mergai_version=__version__, _repo=repo)

    @classmethod
    def combine_from_dicts(
        cls: type[T],
        commits_with_notes: List[Tuple[Any, Optional[dict]]],
        repo: "Repo" = None,
    ) -> T:
        """Build a combined note from multiple commit notes.

        Merges all note data from the provided commits into a single note:
        - merge_info: taken from the first commit that has it
        - mergai_version: set to the current mergai version (the version doing the combine)
        - conflict_context: taken from the first commit that has it
        - merge_context: taken from the first commit that has it
        - solutions: all solutions combined into a single array
        - merge_description: taken from the first commit that has it
        - pr_comments: taken from the first commit that has it
        - user_comment: taken from the first commit that has it

        Note: note_index is NOT set by this method. Use set_note_index_for_all_fields()
        after combining to assign all fields to a specific commit.

        Args:
            commits_with_notes: List of (commit, note_dict) tuples. The commit
                can be any object (typically git.Commit); only the note dict is used.
            repo: Optional GitPython Repo for resolving commits.

        Returns:
            A new MergaiNote instance with combined data and current mergai version.
        """
        from .version import __version__

        # Always use current version when combining notes
        combined: Dict[str, Any] = {"mergai_version": __version__}

        for _, git_note in commits_with_notes:
            if git_note is None:
                continue

            # merge_info - take from first commit that has it
            if "merge_info" in git_note and "merge_info" not in combined:
                combined["merge_info"] = git_note["merge_info"]

            # conflict_context - take from first commit that has it
            if "conflict_context" in git_note and "conflict_context" not in combined:
                combined["conflict_context"] = git_note["conflict_context"]

            # merge_context - take from first commit that has it
            if "merge_context" in git_note and "merge_context" not in combined:
                combined["merge_context"] = git_note["merge_context"]

            # solutions - combine all into array
            if "solutions" in git_note:
                if "solutions" not in combined:
                    combined["solutions"] = []
                for solution in git_note["solutions"]:
                    combined["solutions"].append(solution)

            # merge_description - take from first commit that has it
            if "merge_description" in git_note and "merge_description" not in combined:
                combined["merge_description"] = git_note["merge_description"]

            # pr_comments - take from first commit that has it
            if "pr_comments" in git_note and "pr_comments" not in combined:
                combined["pr_comments"] = git_note["pr_comments"]

            # user_comment - take from first commit that has it
            if "user_comment" in git_note and "user_comment" not in combined:
                combined["user_comment"] = git_note["user_comment"]

        return cls.from_dict(combined, repo)

    # --- has_* Properties ---

    @property
    def has_conflict_context(self) -> bool:
        """Check if conflict_context is present."""
        return self.conflict_context is not None

    @property
    def has_merge_context(self) -> bool:
        """Check if merge_context is present."""
        return self.merge_context is not None

    @property
    def has_solutions(self) -> bool:
        """Check if solutions are present."""
        return self.solutions is not None and len(self.solutions) > 0

    @property
    def has_pr_comments(self) -> bool:
        """Check if pr_comments are present."""
        return self.pr_comments is not None and len(self.pr_comments) > 0

    @property
    def has_user_comment(self) -> bool:
        """Check if user_comment is present."""
        return self.user_comment is not None

    @property
    def has_merge_description(self) -> bool:
        """Check if merge_description is present."""
        return self.merge_description is not None

    @property
    def has_note_index(self) -> bool:
        """Check if note_index is present."""
        return self.note_index is not None and len(self.note_index) > 0

    # --- Repo Binding ---

    def bind_repo(self, repo: "Repo") -> "MergaiNote":
        """Bind a repo for commit resolution on all sub-contexts.

        Args:
            repo: GitPython Repo instance.

        Returns:
            Self for method chaining.
        """
        self._repo = repo
        self.merge_info.bind_repo(repo)
        if self.conflict_context:
            self.conflict_context.bind_repo(repo)
        if self.merge_context:
            self.merge_context.bind_repo(repo)
        return self

    # --- Mutation Methods ---

    def set_conflict_context(self, context: ConflictContext) -> "MergaiNote":
        """Set conflict_context.

        Args:
            context: ConflictContext to set.

        Returns:
            Self for method chaining.
        """
        self.conflict_context = context
        if self._repo:
            context.bind_repo(self._repo)
        return self

    def set_merge_context(self, context: MergeContext) -> "MergaiNote":
        """Set merge_context.

        Args:
            context: MergeContext to set.

        Returns:
            Self for method chaining.
        """
        self.merge_context = context
        if self._repo:
            context.bind_repo(self._repo)
        return self

    def add_solution(self, solution: dict) -> int:
        """Add a solution and return its index.

        Args:
            solution: Solution dict to add.

        Returns:
            Index of the added solution.
        """
        if self.solutions is None:
            self.solutions = []
        self.solutions.append(solution)
        return len(self.solutions) - 1

    def set_solution_at(self, index: int, solution: dict) -> "MergaiNote":
        """Set a solution at a specific index.

        Args:
            index: Index to set the solution at.
            solution: Solution dict to set.

        Returns:
            Self for method chaining.

        Raises:
            IndexError: If index is out of range.
        """
        if self.solutions is None:
            raise IndexError("No solutions array exists")
        self.solutions[index] = solution
        return self

    def clear_solutions(self) -> "MergaiNote":
        """Clear all solutions.

        Returns:
            Self for method chaining.
        """
        self.solutions = None
        return self

    def set_pr_comments(self, comments: List[dict]) -> "MergaiNote":
        """Set pr_comments.

        Args:
            comments: List of PR comment dicts.

        Returns:
            Self for method chaining.
        """
        self.pr_comments = comments
        return self

    def set_user_comment(self, comment: dict) -> "MergaiNote":
        """Set user_comment.

        Args:
            comment: User comment dict with user, email, date, body.

        Returns:
            Self for method chaining.
        """
        self.user_comment = comment
        return self

    def set_merge_description(self, description: dict) -> "MergaiNote":
        """Set merge_description.

        Args:
            description: Merge description dict.

        Returns:
            Self for method chaining.
        """
        self.merge_description = description
        return self

    def add_note_index_entry(self, sha: str, fields: List[str]) -> "MergaiNote":
        """Add an entry to note_index.

        Args:
            sha: Commit SHA for the index entry.
            fields: List of field names included in this commit's note.

        Returns:
            Self for method chaining.
        """
        if self.note_index is None:
            self.note_index = []
        self.note_index.append({"sha": sha, "fields": fields})
        return self

    def clear_note_index(self) -> "MergaiNote":
        """Clear the note_index.

        Returns:
            Self for method chaining.
        """
        self.note_index = None
        return self

    def set_note_index_for_all_fields(self, commit_sha: str) -> "MergaiNote":
        """Set note_index to reference all present fields to a single commit.

        This is useful after combining multiple notes into one, where all
        fields should be attributed to a single squashed commit.

        Inspects which fields are present on the note and builds a note_index
        entry that references all of them to the given commit SHA.

        Args:
            commit_sha: The commit SHA to associate all fields with.

        Returns:
            Self for method chaining.
        """
        all_fields = []

        if self.has_conflict_context:
            all_fields.append("conflict_context")

        if self.has_merge_context:
            all_fields.append("merge_context")

        if self.has_solutions:
            for idx in range(len(self.solutions)):
                all_fields.append(f"solutions[{idx}]")

        if self.has_merge_description:
            all_fields.append("merge_description")

        if self.has_pr_comments:
            all_fields.append("pr_comments")

        if self.has_user_comment:
            all_fields.append("user_comment")

        if all_fields:
            self.note_index = [{"sha": commit_sha, "fields": all_fields}]
        else:
            self.note_index = None

        return self

    # --- Drop Methods ---

    def drop_conflict_context(self) -> "MergaiNote":
        """Remove conflict_context.

        Returns:
            Self for method chaining.
        """
        self.conflict_context = None
        return self

    def drop_merge_context(self) -> "MergaiNote":
        """Remove merge_context.

        Returns:
            Self for method chaining.
        """
        self.merge_context = None
        return self

    def drop_pr_comments(self) -> "MergaiNote":
        """Remove pr_comments.

        Returns:
            Self for method chaining.
        """
        self.pr_comments = None
        return self

    def drop_user_comment(self) -> "MergaiNote":
        """Remove user_comment.

        Returns:
            Self for method chaining.
        """
        self.user_comment = None
        return self

    def drop_merge_description(self) -> "MergaiNote":
        """Remove merge_description.

        Returns:
            Self for method chaining.
        """
        self.merge_description = None
        return self

    def drop_solution(self, all: bool = False) -> "MergaiNote":
        """Drop solution(s) from the note.

        Args:
            all: If True, drop all solutions. If False (default), only drop
                 uncommitted solutions (those not in note_index).

        Returns:
            Self for method chaining.
        """
        if not self.has_solutions:
            return self

        if all:
            # Drop all solutions
            self.clear_solutions()
            # Also remove solutions entries from note_index
            if self.has_note_index:
                self.note_index = [
                    entry
                    for entry in self.note_index
                    if not any(
                        f.startswith("solutions[") for f in entry.get("fields", [])
                    )
                ]
                if not self.note_index:
                    self.clear_note_index()
        else:
            # Only drop uncommitted solutions
            committed_indices = self._get_committed_solution_indices()
            if committed_indices:
                # Keep only committed solutions
                self.solutions = [
                    self.solutions[i]
                    for i in sorted(committed_indices)
                    if i < len(self.solutions)
                ]
            else:
                # No committed solutions, drop all
                self.clear_solutions()

        return self

    def _get_committed_solution_indices(self) -> set:
        """Get indices of solutions that have been committed.

        Returns:
            Set of solution indices that are in the note_index.
        """
        import re

        committed = set()
        if not self.has_note_index:
            return committed

        for entry in self.note_index:
            for field in entry.get("fields", []):
                # Match "solutions[N]" pattern
                match = re.match(r"solutions\[(\d+)\]", field)
                if match:
                    committed.add(int(match.group(1)))

        return committed

    def get_uncommitted_solution(self) -> Optional[Tuple[int, dict]]:
        """Get the last uncommitted solution with its index.

        Returns:
            Tuple of (index, solution_dict) or None if no uncommitted solution exists.
        """
        if not self.has_solutions:
            return None

        committed = self._get_committed_solution_indices()
        # Find the last index that is not committed
        for i in range(len(self.solutions) - 1, -1, -1):
            if i not in committed:
                return (i, self.solutions[i])
        return None

    def get_uncommitted_fields(self) -> List[str]:
        """Get list of fields that are present in the note but not in note_index.

        Returns:
            List of field names that have data but are not tracked in note_index.
        """
        uncommitted = []

        # Get all committed fields from note_index
        committed_fields = set()
        if self.has_note_index:
            for entry in self.note_index:
                committed_fields.update(entry.get("fields", []))

        # Check each field
        if self.has_conflict_context and "conflict_context" not in committed_fields:
            uncommitted.append("conflict_context")

        if self.has_merge_context and "merge_context" not in committed_fields:
            uncommitted.append("merge_context")

        if self.has_solutions:
            committed_solution_indices = self._get_committed_solution_indices()
            for idx in range(len(self.solutions)):
                if idx not in committed_solution_indices:
                    uncommitted.append(f"solutions[{idx}]")

        if self.has_merge_description and "merge_description" not in committed_fields:
            uncommitted.append("merge_description")

        if self.has_pr_comments and "pr_comments" not in committed_fields:
            uncommitted.append("pr_comments")

        if self.has_user_comment and "user_comment" not in committed_fields:
            uncommitted.append("user_comment")

        return uncommitted

    def is_fully_committed(self) -> bool:
        """Check if all fields in the note are tracked in note_index.

        Returns:
            True if all present fields have been committed, False otherwise.
        """
        return len(self.get_uncommitted_fields()) == 0

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize to dict for storage in note.json.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        result = {
            "merge_info": self.merge_info.to_dict(),
            "mergai_version": self.mergai_version,
        }
        if self.conflict_context:
            result["conflict_context"] = self.conflict_context.to_dict()
        if self.merge_context:
            result["merge_context"] = self.merge_context.to_dict()
        if self.solutions:
            result["solutions"] = self.solutions
        if self.pr_comments:
            result["pr_comments"] = self.pr_comments
        if self.user_comment:
            result["user_comment"] = self.user_comment
        if self.merge_description:
            result["merge_description"] = self.merge_description
        if self.note_index:
            result["note_index"] = self.note_index
        return result