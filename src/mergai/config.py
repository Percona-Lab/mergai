"""Configuration file support for MergAI.

This module handles loading and parsing the .mergai/config.yml configuration file.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .merge_pick_strategies import ImportantFilesStrategy, MergePickStrategy
from .models import CommitSerializationConfig, ContextSerializationConfig

log = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = ".mergai/config.yml"
DEFAULT_COMMIT_FIELDS = ["hexsha"]


@dataclass
class MergeGateConfig:
    """Deterministic gate controlling *when* a merge happens.

    The gate is a pure go/no-go decision evaluated over already-computed fork
    status + prioritized commits (no AI tokens), so it is safe to run in the
    unprivileged periodic phase. It is mode-agnostic: it only decides whether to
    merge now, not which commit to merge to.

    Attributes:
        min_commits: Merge once at least this many unmerged commits accumulate.
        max_age_days: ...or sooner if the oldest unmerged commit is older than
            this many days.
        max_commits: Never advance more than this many upstream commits in a
            single merge. Defines the candidate window (the oldest
            ``max_commits`` unmerged commits); bounds both the merge batch size
            and the AI prompt size. Commits newer than the window are omitted
            and drained by later merges.
        force_strategies: Merge-pick strategy names that, when any prioritized
            commit matches one, open the gate immediately regardless of count or
            age (e.g. ``conflict``, ``important_files``).
    """

    min_commits: int = 50
    max_age_days: int = 2
    max_commits: int = 150
    force_strategies: list[str] = field(
        default_factory=lambda: ["conflict", "important_files"]
    )

    @classmethod
    def from_dict(cls, data: dict) -> "MergeGateConfig":
        """Create a MergeGateConfig from a dictionary.

        Raises:
            ValueError: If an integer field is non-integer, or
                ``force_strategies`` is neither a string nor a list of strings.
        """

        def _int(name: str, default: int) -> int:
            # `null` (key present, value None) falls back to the default; a
            # non-integer fails fast here rather than later as a TypeError in
            # gate evaluation (e.g. ``commits_behind >= None``). bool is a
            # subclass of int but is almost certainly a YAML mistake here.
            value = data.get(name)
            if value is None:
                return default
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"'merge_gate.{name}' must be an integer, "
                    f"got {type(value).__name__}"
                )
            return value

        # `null` (key present, value None) falls back to the defaults; an
        # explicit empty list still intentionally disables force strategies. A
        # bare string is accepted as a single strategy name so a stray
        # ``force_strategies: conflict`` is not silently split into characters
        # by ``list(...)``.
        raw = data.get("force_strategies")
        if raw is None:
            force_strategies = list(cls().force_strategies)
        elif isinstance(raw, str):
            force_strategies = [raw]
        elif isinstance(raw, list) and all(isinstance(s, str) for s in raw):
            force_strategies = list(raw)
        else:
            raise ValueError(
                "'merge_gate.force_strategies' must be a string or a list of "
                f"strings, got {type(raw).__name__}"
            )
        return cls(
            min_commits=_int("min_commits", cls.min_commits),
            max_age_days=_int("max_age_days", cls.max_age_days),
            max_commits=_int("max_commits", cls.max_commits),
            force_strategies=force_strategies,
        )


@dataclass
class AiPickConfig:
    """Configuration for the AI-assisted merge pick.

    When enabled, the privileged merge phase (``merge-pick --ai``) asks an AI
    agent which upstream commit to merge to, within the gate's candidate
    window. When disabled, the pick is made deterministically.

    Attributes:
        enabled: Whether the AI pick is used. When False, ``merge-pick --plan``
            reports ``mode: deterministic`` and resolves the sha itself.
        agent: Agent descriptor (e.g. ``claude-cli:claude-opus-4-5``), same
            format as ``resolve.agent``. Empty falls back to ``resolve.agent``.
        rules_file: Optional path to a project-specific merge-pick rules file
            (markdown) appended to the built-in system prompt.
        fallback: What to do on agent error / invalid sha: ``deterministic``
            (resilient, the default) or ``error``.
    """

    enabled: bool = False
    agent: str = ""
    rules_file: str = ""
    fallback: str = "deterministic"

    @classmethod
    def from_dict(cls, data: dict) -> "AiPickConfig":
        """Create an AiPickConfig from a dictionary.

        Raises:
            ValueError: If ``fallback`` is not one of ``deterministic`` /
                ``error`` (an unknown value would otherwise be silently treated
                as ``deterministic``).
        """
        fallback = data.get("fallback", cls.fallback)
        if fallback not in ("deterministic", "error"):
            raise ValueError(
                f"Invalid value for ai_pick.fallback: '{fallback}'. "
                "Valid values are: deterministic, error"
            )
        return cls(
            enabled=data.get("enabled", cls.enabled),
            agent=data.get("agent", cls.agent),
            rules_file=data.get("rules_file", cls.rules_file),
            fallback=fallback,
        )


@dataclass
class ForkConfig:
    """Configuration for the fork subcommand.

    Attributes:
        upstream_url: URL of the upstream repository to sync from.
        upstream_branch: Branch name to use when auto-detecting upstream ref.
        upstream_remote: Name of the git remote for upstream (if not set, derived from URL).
        merge_picks: Configuration for commit prioritization in fork merge-pick.
        merge_gate: Deterministic gate controlling when to merge.
        ai_pick: Configuration for the AI-assisted merge pick.
    """

    upstream_url: str | None = None
    upstream_branch: str = "master"
    upstream_remote: str | None = None
    merge_picks: Optional["MergePicksConfig"] = None
    merge_gate: MergeGateConfig = field(default_factory=MergeGateConfig)
    ai_pick: AiPickConfig = field(default_factory=AiPickConfig)

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

        merge_gate_data = data.get("merge_gate")
        merge_gate = (
            MergeGateConfig.from_dict(merge_gate_data)
            if merge_gate_data
            else MergeGateConfig()
        )

        ai_pick_data = data.get("ai_pick")
        ai_pick = (
            AiPickConfig.from_dict(ai_pick_data) if ai_pick_data else AiPickConfig()
        )

        return cls(
            upstream_url=data.get("upstream_url"),
            upstream_branch=data.get("upstream_branch", cls.upstream_branch),
            upstream_remote=data.get("upstream_remote"),
            merge_picks=merge_picks,
            merge_gate=merge_gate,
            ai_pick=ai_pick,
        )


@dataclass
class ProjectConfig:
    """Project identity and wording for AI prompts.

    Substituted into the CI-fix prompts so the prompt text shipped in mergai
    stays project-agnostic and each downstream fork supplies its own terms.

    Attributes:
        name: Full display name of the project/product (optional).
        fork_term: Adjectival name of the fork (downstream) side. Rendered in
            prompts as ``"{fork_term} fork"``, ``"the {fork_term} branch"``,
            ``"{fork_term}-specific"``. The default reads naturally without
            configuration; a fork sets it to its own name (e.g. ``"Percona"``).
        upstream_term: Adjectival name of the upstream side (e.g.
            ``"upstream"``, or a project name like ``"MongoDB"``).
    """

    name: str = ""
    fork_term: str = "downstream"
    upstream_term: str = "upstream"

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectConfig":
        """Create a ProjectConfig from a dictionary."""
        return cls(
            name=data.get("name", cls.name),
            fork_term=data.get("fork_term", cls.fork_term),
            upstream_term=data.get("upstream_term", cls.upstream_term),
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
class ReviewConfig:
    """Configuration for the ``review`` command.

    Controls how ``mergai review fix`` selects which PR review comments to
    act on, which agent generates the fixes, and the wording of the replies
    posted back to each comment. All settings are generic - nothing here is
    specific to any particular downstream repository.

    Attributes:
        agent: Agent descriptor for review fixes (e.g. "gemini-cli",
            "claude-cli:sonnet"). When empty, falls back to
            ``resolve.agent``.
        max_attempts: Per-invocation agent retry budget.
        skip_token: A thread containing this token in any of its comments is
            treated as opt-out and left untouched.
        bot_logins: GitHub logins whose last comment marks a thread as
            already-handled-by-automation. An explicit escape hatch for other
            automation accounts; mergai's own work is tracked durably on the
            note (the addressed review-thread ids from prior ``review_fix``
            solutions), not by inspecting replies, so its own login needs no
            entry here. Default: none.
        process_external: When False (default), threads raised by an author who
            is not trusted (per ``trusted_associations`` / ``trusted_logins``)
            are skipped entirely, and untrusted replies on otherwise-trusted
            threads are dropped from the agent context. Set True to act on every
            author's comments regardless of association.
        trusted_associations: GitHub ``authorAssociation`` values treated as
            trusted to instruct the agent. Defaults to ``OWNER`` / ``MEMBER``
            (repo/org owners and org members) - note this deliberately excludes
            ``COLLABORATOR``, which is an *outside* collaborator. Add it, or use
            ``trusted_logins``, to trust specific outside collaborators. Used
            only when ``process_external`` is False.
        trusted_logins: Explicit allowlist of trusted author logins, in addition
            to ``trusted_associations``. Used only when ``process_external`` is
            False.
        reply_fixed_header: Optional first line of the reply posted to a
            comment the agent addressed. Empty by default - the reply is just
            the agent's note plus the commit reference.
        reply_unfixable_header: Optional first line of the reply posted to a
            comment the agent could not address.
        reply_footer: Optional trailing line appended to every reply.
    """

    agent: str = ""
    max_attempts: int = 3
    skip_token: str = "/mergai skip"
    bot_logins: list[str] = field(default_factory=list)
    process_external: bool = False
    trusted_associations: list[str] = field(default_factory=lambda: ["OWNER", "MEMBER"])
    trusted_logins: list[str] = field(default_factory=list)
    reply_fixed_header: str = ""
    reply_unfixable_header: str = "mergai could not automatically address this comment:"
    reply_footer: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewConfig":
        """Create a ReviewConfig from a dictionary."""
        return cls(
            agent=data.get("agent", cls.agent),
            max_attempts=data.get("max_attempts", cls.max_attempts),
            skip_token=data.get("skip_token", cls.skip_token),
            bot_logins=list(data.get("bot_logins", [])),
            process_external=data.get("process_external", cls.process_external),
            trusted_associations=list(
                data.get("trusted_associations", cls().trusted_associations)
            ),
            trusted_logins=list(data.get("trusted_logins", [])),
            reply_fixed_header=data.get("reply_fixed_header", cls.reply_fixed_header),
            reply_unfixable_header=data.get(
                "reply_unfixable_header", cls.reply_unfixable_header
            ),
            reply_footer=data.get("reply_footer", cls.reply_footer),
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

            Optional tokens:
            - %(type) - Branch type (main, conflict, solution, or custom)

            SHA token variants:
            - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
            - %(merge_commit_short_sha) - Short SHA of the merge commit (11 chars)
        working_prefix: Branch-namespace prefix mergai treats as its own when
            deciding whether to act on a CI workflow run. Should match the
            prefix that ``name_format`` produces.
    """

    name_format: str = "mergai/%(target_branch)-%(merge_commit_short_sha)/%(type)"
    working_prefix: str = "mergai/"

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
            working_prefix=data.get("working_prefix", cls.working_prefix),
        )


DEFAULT_COMMIT_FOOTER = "Note: commit created by mergai"
DEFAULT_CI_FIX_TITLE_FORMAT = (
    "Fix %(workflow) failure for merge commit "
    "%(merge_commit_short_sha) into %(target_branch)"
)
DEFAULT_REVIEW_FIX_TITLE_FORMAT = (
    "Address review comments on PR #%(pr_number) for merge commit "
    "%(merge_commit_short_sha) into %(target_branch)"
)


@dataclass
class PRTypeConfig:
    """Configuration for a specific PR type (main or solution).

    The format string uses %(token) syntax for substitution.

    Attributes:
        title_format: Format string for PR titles.
        labels: Labels always applied to the PR when created.
        labels_on_unresolved: Labels applied only when the PR's note contains
            at least one solution with unresolved conflicts (any file left with
            conflict markers). Typically used to attach skip labels like
            ``ci-skip-format`` so CI only runs on fully-resolved PRs.

        Available tokens for title_format:
        - %(target_branch) - The target branch name
        - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
        - %(merge_commit_short_sha) - Short SHA of the merge commit (11 chars)
    """

    title_format: str = ""
    labels: list[str] = field(default_factory=list)
    labels_on_unresolved: list[str] = field(default_factory=list)

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
            labels=data.get("labels", []),
            labels_on_unresolved=data.get("labels_on_unresolved", []),
        )


# Default title formats
DEFAULT_MAIN_PR_TITLE_FORMAT = "Merge %(merge_commit_short_sha) into %(target_branch)"
DEFAULT_SOLUTION_PR_TITLE_FORMAT = (
    "Resolve conflicts for merge %(merge_commit_short_sha) into %(target_branch)"
)
DEFAULT_SEMANTIC_PR_TITLE_FORMAT = (
    "Resolve semantic conflicts for merge %(merge_commit_short_sha) "
    "into %(target_branch)"
)


@dataclass
class PRConfig:
    """Configuration for pull requests.

    Contains separate configuration for main and solution PRs.

    Attributes:
        main: Configuration for main PRs (from main branch to target_branch).
        solution: Configuration for solution PRs (from solution branch to conflict branch).
        semantic: Configuration for semantic PRs (from semantic branch to main branch).

    Example YAML config:
        pr:
          main:
            title_format: "[MERGE] %(merge_commit_short_sha) -> %(target_branch)"
          solution:
            title_format: "[RESOLVE] Conflicts for %(merge_commit_short_sha) into %(target_branch)"
          semantic:
            title_format: "[SEMANTIC] Fixes for %(merge_commit_short_sha) into %(target_branch)"
    """

    main: PRTypeConfig = field(
        default_factory=lambda: PRTypeConfig(title_format=DEFAULT_MAIN_PR_TITLE_FORMAT)
    )
    solution: PRTypeConfig = field(
        default_factory=lambda: PRTypeConfig(
            title_format=DEFAULT_SOLUTION_PR_TITLE_FORMAT
        )
    )
    semantic: PRTypeConfig = field(
        default_factory=lambda: PRTypeConfig(
            title_format=DEFAULT_SEMANTIC_PR_TITLE_FORMAT
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

        semantic_data = data.get("semantic", {})
        semantic_config = PRTypeConfig.from_dict(
            semantic_data, DEFAULT_SEMANTIC_PR_TITLE_FORMAT
        )

        return cls(
            main=main_config,
            solution=solution_config,
            semantic=semantic_config,
        )


@dataclass
class CommitConfig:
    """Configuration for commit message generation.

    Controls how commit messages are formatted when MergAI creates commits
    for conflict resolution, merge commits, etc.

    Attributes:
        footer: Footer text appended to all MergAI-generated commit messages.
            Set to empty string to disable the footer.
        ci_fix_title_format: Format string for the title of CI-fix commits.
            Uses %(token) syntax. Available tokens:
            - %(workflow) - The failing workflow/check name
            - %(target_branch) - The target branch name
            - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
            - %(merge_commit_short_sha) - Short SHA of the merge commit
        review_fix_title_format: Format string for the title of review-fix
            commits. Uses %(token) syntax. Available tokens:
            - %(pr_number) - The pull request number
            - %(target_branch) - The target branch name
            - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
            - %(merge_commit_short_sha) - Short SHA of the merge commit

    Example YAML config:
        commit:
          footer: "Note: commit created by mergai"
          ci_fix_title_format: "Fix '%(workflow)' failure for merge commit '%(merge_commit_short_sha)' into '%(target_branch)'"
    """

    footer: str = DEFAULT_COMMIT_FOOTER
    ci_fix_title_format: str = DEFAULT_CI_FIX_TITLE_FORMAT
    review_fix_title_format: str = DEFAULT_REVIEW_FIX_TITLE_FORMAT

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
            ci_fix_title_format=data.get(
                "ci_fix_title_format", cls.ci_fix_title_format
            ),
            review_fix_title_format=data.get(
                "review_fix_title_format", cls.review_fix_title_format
            ),
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


# Valid values for merge.describe config
MERGE_DESCRIBE_NEVER = "never"
MERGE_DESCRIBE_ALWAYS = "always"
MERGE_DESCRIBE_SUCCESS = "success"
MERGE_DESCRIBE_CONFLICT = "conflict"
VALID_MERGE_DESCRIBE_VALUES = [
    MERGE_DESCRIBE_NEVER,
    MERGE_DESCRIBE_ALWAYS,
    MERGE_DESCRIBE_SUCCESS,
    MERGE_DESCRIBE_CONFLICT,
]


@dataclass
class MergeConfig:
    """Configuration for the merge command.

    Controls behavior after performing a git merge.

    Attributes:
        describe: When to automatically run the describe command after merge.
            - 'never': Don't run describe (default)
            - 'always': Run describe regardless of merge outcome
            - 'success': Run describe only if merge succeeded (no conflicts)
            - 'conflict': Run describe only if merge resulted in conflicts

    Example YAML config:
        merge:
          describe: never
    """

    describe: str = MERGE_DESCRIBE_NEVER

    @classmethod
    def from_dict(cls, data: dict) -> "MergeConfig":
        """Create a MergeConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            MergeConfig instance with values from data.

        Raises:
            ValueError: If describe value is not valid.
        """
        describe = data.get("describe", cls.describe)
        if describe not in VALID_MERGE_DESCRIBE_VALUES:
            raise ValueError(
                f"Invalid value for merge.describe: '{describe}'. "
                f"Valid values are: {', '.join(VALID_MERGE_DESCRIBE_VALUES)}"
            )
        return cls(
            describe=describe,
        )


@dataclass
class ConflictContextConfig:
    """Configuration for conflict context creation.

    These settings control what information is captured in the conflict_context,
    which is used by AI agents to understand and resolve merge conflicts.

    These settings are used by:
    - The 'mergai merge' command when automatically creating conflict_context
      after merge conflicts are detected
    - The 'mergai context create conflict' command as defaults (CLI flags can
      override these values)

    Attributes:
        use_diffs: Include diffs in the conflict context.
        diff_lines_of_context: Number of context lines around diff hunks.
        use_compressed_diffs: Use compressed diffs to limit size.
        use_their_commits: Include their commits in the conflict context.

    Example YAML config:
        context:
          conflict:
            use_diffs: true
            diff_lines_of_context: 0
            use_compressed_diffs: true
            use_their_commits: true
    """

    use_diffs: bool = True
    diff_lines_of_context: int = 0
    use_compressed_diffs: bool = True
    use_their_commits: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ConflictContextConfig":
        """Create a ConflictContextConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            ConflictContextConfig instance with values from data.
        """
        return cls(
            use_diffs=data.get("use_diffs", cls.use_diffs),
            diff_lines_of_context=data.get(
                "diff_lines_of_context", cls.diff_lines_of_context
            ),
            use_compressed_diffs=data.get(
                "use_compressed_diffs", cls.use_compressed_diffs
            ),
            use_their_commits=data.get("use_their_commits", cls.use_their_commits),
        )

    def with_overrides(
        self,
        use_diffs: bool | None = None,
        diff_lines_of_context: int | None = None,
        use_compressed_diffs: bool | None = None,
        use_their_commits: bool | None = None,
    ) -> "ConflictContextConfig":
        """Create a new config with optional overrides.

        Returns a new ConflictContextConfig where any non-None parameter
        overrides the corresponding value from this config.

        Args:
            use_diffs: Override for use_diffs, or None to keep current value.
            diff_lines_of_context: Override for diff_lines_of_context.
            use_compressed_diffs: Override for use_compressed_diffs.
            use_their_commits: Override for use_their_commits.

        Returns:
            New ConflictContextConfig with overrides applied.
        """
        return ConflictContextConfig(
            use_diffs=use_diffs if use_diffs is not None else self.use_diffs,
            diff_lines_of_context=(
                diff_lines_of_context
                if diff_lines_of_context is not None
                else self.diff_lines_of_context
            ),
            use_compressed_diffs=(
                use_compressed_diffs
                if use_compressed_diffs is not None
                else self.use_compressed_diffs
            ),
            use_their_commits=(
                use_their_commits
                if use_their_commits is not None
                else self.use_their_commits
            ),
        )


@dataclass
class ContextConfig:
    """Configuration for context creation.

    Contains settings for creating various types of merge context.

    Attributes:
        conflict: Configuration for conflict context creation.

    Example YAML config:
        context:
          conflict:
            use_diffs: true
            diff_lines_of_context: 0
    """

    conflict: ConflictContextConfig = field(default_factory=ConflictContextConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "ContextConfig":
        """Create a ContextConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            ContextConfig instance with values from data.
        """
        conflict_data = data.get("conflict", {})
        conflict_config = (
            ConflictContextConfig.from_dict(conflict_data)
            if conflict_data
            else ConflictContextConfig()
        )
        return cls(conflict=conflict_config)


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

    commit_fields: list[str] = field(
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

    strategies: list[MergePickStrategy] = field(default_factory=list)
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
class GitNotesConfig:
    """Configuration for git notes in the config command.

    Controls notes display in git log and marker text content.

    Attributes:
        display: Whether to configure notes.displayRef so mergai markers
            appear in git log output. The ref is always refs/notes/mergai-marker.
        marker_text: Text to display in git log for commits with mergai notes.
            This is the content of the mergai-marker note.

    Example YAML config:
        config:
          git:
            notes:
              display: true
              marker_text: "mergai note available, use `mergai show <commit>` to view it."
    """

    display: bool = True
    marker_text: str = "mergai note available, use `mergai show <commit>` to view it."

    @classmethod
    def from_dict(cls, data: dict) -> "GitNotesConfig":
        """Create a GitNotesConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            GitNotesConfig instance with values from data.
        """
        return cls(
            display=data.get("display", cls.display),
            marker_text=data.get("marker_text", cls.marker_text),
        )


@dataclass
class GitInitConfig:
    """Git configuration settings for the config command.

    Controls what git config values are set when running 'mergai config'.

    Attributes:
        conflictstyle: Value for merge.conflictstyle (default: "diff3").
            Using diff3 provides better conflict context by including the
            base version in conflict markers.
        notes: Configuration for git notes display and marker text.

    Example YAML config:
        config:
          git:
            conflictstyle: diff3
            notes:
              display: true
              marker_text: "MergAI note available"
    """

    conflictstyle: str = "diff3"
    notes: GitNotesConfig = field(default_factory=GitNotesConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "GitInitConfig":
        """Create a GitInitConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            GitInitConfig instance with values from data.
        """
        notes_data = data.get("notes", {})
        return cls(
            conflictstyle=data.get("conflictstyle", cls.conflictstyle),
            notes=GitNotesConfig.from_dict(notes_data),
        )


@dataclass
class CompletionInitConfig:
    """Shell completion configuration for the config command.

    Controls shell completion setup when running 'mergai config'.

    Attributes:
        shell: Shell type for completion (default: "bash").
            Supported values: "bash", "zsh", "fish".

    Example YAML config:
        config:
          completion:
            shell: bash
    """

    shell: str = "bash"

    @classmethod
    def from_dict(cls, data: dict) -> "CompletionInitConfig":
        """Create a CompletionInitConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            CompletionInitConfig instance with values from data.
        """
        return cls(
            shell=data.get("shell", cls.shell),
        )


@dataclass
class InitConfig:
    """Configuration for the config command.

    Controls what gets configured when running 'mergai config'.

    Attributes:
        git: Git configuration settings to apply.
        completion: Shell completion configuration.

    Example YAML config:
        config:
          git:
            conflictstyle: diff3
            notes_display_ref: refs/notes/mergai-marker
          completion:
            shell: bash
    """

    git: GitInitConfig = field(default_factory=GitInitConfig)
    completion: CompletionInitConfig = field(default_factory=CompletionInitConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "InitConfig":
        """Create an InitConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            InitConfig instance with values from data.
        """
        git_data = data.get("git", {})
        completion_data = data.get("completion", {})
        return cls(
            git=GitInitConfig.from_dict(git_data),
            completion=CompletionInitConfig.from_dict(completion_data),
        )


# Valid values for workflow action_type config
WORKFLOW_ACTION_COMMAND = "command"
WORKFLOW_ACTION_RESOLVE = "resolve"
VALID_WORKFLOW_ACTION_TYPES = [WORKFLOW_ACTION_COMMAND, WORKFLOW_ACTION_RESOLVE]


@dataclass
class WorkflowContextConfig:
    """Configuration for extracting failure context from a CI workflow run.

    Resolved at runtime by the ``mergai.ci.context_builders`` factory, which
    maps ``type`` to a concrete builder. Unknown types are rejected there,
    not here, so new builders can be added without touching config parsing.

    Attributes:
        type: Context type (e.g. ``"diff"``, ``"sarif"``, ``"bazel"``).
            Resolved against the context-builder registry at runtime.
        source: Where to read the context from. Currently ``"artifact"``
            (downloaded workflow artifact).
        artifact_name: Names of artifacts to inspect when ``source`` is
            ``"artifact"``. YAML may pass either a single string or a list;
            the value is always normalized to ``list[str]``. Builders that
            expect exactly one artifact (``diff``, ``sarif``) read element
            zero; builders covering multi-job workflows (``bazel``) iterate
            over the list and use whichever artifact the failing job
            actually uploaded.
        code_scanning_check: If true, when the watched workflow_run
            *passes*, also consult GitHub Code Scanning for findings on
            the run's commit. If the latest analysis for the configured
            tool has any results, build a context from that SARIF and
            run the handler. Only meaningful for SARIF-emitting tools
            whose workflow uploads to Code Scanning (e.g. clang-tidy).
        code_scanning_tool_name: Code Scanning tool/driver name to query for
            findings. Defaults to the workflow name; set only when the tool
            name differs from it.

    Example YAML config::

        context:
          type: sarif
          source: artifact
          artifact_name: clang-tidy-results
          code_scanning_check: true

        # Multi-job workflow: each job uploads its own artifact on failure.
        context:
          type: bazel
          source: artifact
          artifact_name:
            - build-failure-artifacts
            - unittest-failure-artifacts
    """

    type: str = "diff"
    source: str = "artifact"
    artifact_name: list[str] = field(default_factory=list)
    code_scanning_check: bool = False
    code_scanning_tool_name: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowContextConfig":
        """Create a WorkflowContextConfig from a dictionary.

        Accepts ``artifact_name`` as a string (legacy single-artifact form)
        or a list of strings; both are normalized to ``list[str]``. Missing
        or ``None`` becomes an empty list.

        Raises:
            ValueError: If ``artifact_name`` is neither a string nor a list
                of strings.
        """
        raw = data.get("artifact_name")
        if raw is None:
            artifact_names: list[str] = []
        elif isinstance(raw, str):
            artifact_names = [raw]
        elif isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            artifact_names = list(raw)
        else:
            raise ValueError(
                "'artifact_name' must be a string or a list of strings, "
                f"got {type(raw).__name__}"
            )

        return cls(
            type=data.get("type", cls.type),
            source=data.get("source", cls.source),
            artifact_name=artifact_names,
            code_scanning_check=data.get(
                "code_scanning_check", cls.code_scanning_check
            ),
            code_scanning_tool_name=data.get("code_scanning_tool_name"),
        )


@dataclass
class WorkflowConfig:
    """Configuration for handling failures of a specific CI workflow.

    Attributes:
        enabled: Whether mergai will attempt fixes for this workflow.
        max_attempts: Cap on the number of fix attempts mergai makes across
            successive workflow runs before giving up and posting a PR comment.
        agent_retries: Retry budget for the AI agent within a single fix
            invocation (``action_type: resolve``). When unset, falls back to
            ``max_attempts``.
        action_type: How to attempt the fix. Either ``"command"`` (run a
            shell command) or ``"resolve"`` (invoke the AI agent via the
            existing ``AgentExecutor`` retry loop).
        command: Shell command to run when ``action_type`` is ``"command"``.
            Receives ``TARGET_BRANCH``, ``PR_NUMBER``, and ``WORKFLOW_NAME``
            as environment variables.
        context: Configuration for extracting failure context for this
            workflow (used to build the AI prompt and/or for logging).

    Example YAML config::

        format:
          enabled: true
          max_attempts: 3
          action_type: command
          command: "git apply ${MERGAI_ARTIFACTS_DIR}/format-results/diff.patch"
          context:
            type: diff
            source: artifact
            artifact_name: format-results
    """

    enabled: bool = False
    max_attempts: int = 3
    agent_retries: int | None = None
    action_type: str = WORKFLOW_ACTION_RESOLVE
    command: str | None = None
    context: WorkflowContextConfig = field(default_factory=WorkflowContextConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowConfig":
        """Create a WorkflowConfig from a dictionary.

        Raises:
            ValueError: If ``action_type`` is not one of
                ``VALID_WORKFLOW_ACTION_TYPES``, or if ``action_type`` is
                ``"command"`` without a non-empty ``command``.
        """
        action_type = data.get("action_type", cls.action_type)
        if action_type not in VALID_WORKFLOW_ACTION_TYPES:
            raise ValueError(
                f"Invalid workflow action_type: '{action_type}'. "
                f"Valid values are: {', '.join(VALID_WORKFLOW_ACTION_TYPES)}"
            )

        command = data.get("command")
        if action_type == WORKFLOW_ACTION_COMMAND and not command:
            raise ValueError(
                "Workflow action_type 'command' requires a non-empty 'command' field"
            )

        context_data = data.get("context", {})
        context = (
            WorkflowContextConfig.from_dict(context_data)
            if context_data
            else WorkflowContextConfig()
        )

        return cls(
            enabled=data.get("enabled", cls.enabled),
            max_attempts=data.get("max_attempts", cls.max_attempts),
            agent_retries=data.get("agent_retries"),
            action_type=action_type,
            command=command,
            context=context,
        )


@dataclass
class WorkflowsConfig:
    """Configuration for CI workflow failure handlers, keyed by workflow name.

    The workflow name is the value of ``name:`` in the GitHub Actions
    workflow file (e.g. ``"format"``, ``"clang-tidy"``). When a ``workflow_run``
    event fires, mergai looks up the matching ``WorkflowConfig`` here.

    Example YAML config::

        workflows:
          format:
            enabled: true
            max_attempts: 3
            action_type: command
            command: "git apply ${MERGAI_ARTIFACTS_DIR}/format-results/diff.patch"
            context:
              type: diff
              source: artifact
              artifact_name: format-results
          clang-tidy:
            enabled: true
            max_attempts: 2
            action_type: resolve
            context:
              type: sarif
              source: artifact
              artifact_name: clang-tidy-results
    """

    workflows: dict[str, WorkflowConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowsConfig":
        """Create a WorkflowsConfig from a dictionary of workflow-name → config."""
        workflows: dict[str, WorkflowConfig] = {}
        for name, wf_data in data.items():
            workflows[name] = WorkflowConfig.from_dict(wf_data or {})
        return cls(workflows=workflows)

    def get(self, name: str) -> WorkflowConfig | None:
        """Look up a workflow's config by name. Returns None if absent."""
        return self.workflows.get(name)


@dataclass
class MergaiConfig:
    """Configuration settings for MergAI.

    All settings are optional and have sensible defaults.

    Attributes:
        project: Project identity/wording substituted into AI prompts.
        fork: Configuration for the fork subcommand (includes merge_picks).
        resolve: Configuration for the resolve command.
        review: Configuration for the review command.
        branch: Configuration for branch naming.
        prompt: Configuration for prompt generation.
        commit: Configuration for commit message generation.
        pr: Configuration for pull request titles.
        finalize: Configuration for the finalize command.
        merge: Configuration for the merge command.
        context: Configuration for context creation (conflict context, etc.).
        config: Configuration for the 'mergai config' command.
        workflows: Configuration for CI workflow failure handlers.
        _raw: Raw dictionary data for accessing arbitrary sections.
    """

    project: ProjectConfig = field(default_factory=ProjectConfig)
    fork: ForkConfig = field(default_factory=ForkConfig)
    resolve: ResolveConfig = field(default_factory=ResolveConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    branch: BranchConfig = field(default_factory=BranchConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    pr: PRConfig = field(default_factory=PRConfig)
    finalize: FinalizeConfig = field(default_factory=FinalizeConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    config: InitConfig = field(default_factory=InitConfig)
    workflows: WorkflowsConfig = field(default_factory=WorkflowsConfig)
    _raw: dict[str, Any] = field(default_factory=dict)

    @property
    def important_files(self) -> list[str]:
        """Get the list of important files from the merge_picks config, if set."""
        if self.fork and self.fork.merge_picks:
            for strategy in self.fork.merge_picks.strategies:
                if isinstance(strategy, ImportantFilesStrategy):
                    return strategy.config.files
        return []

    def get_section(self, name: str) -> dict[str, Any]:
        """Get a configuration section by name.

        This allows commands to access their own config sections without
        needing to modify MergaiConfig for each new command.

        Args:
            name: Section name (e.g., "fork", "resolve", "replay").

        Returns:
            Dictionary with the section's configuration, or empty dict if not found.
        """
        result: dict[str, Any] = self._raw.get(name, {})
        return result

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
        # Parse project section if present
        project_data = data.get("project", {})
        project_config = (
            ProjectConfig.from_dict(project_data) if project_data else ProjectConfig()
        )

        # Parse fork section if present
        fork_data = data.get("fork", {})
        fork_config = ForkConfig.from_dict(fork_data) if fork_data else ForkConfig()

        # Parse resolve section if present
        resolve_data = data.get("resolve", {})
        resolve_config = (
            ResolveConfig.from_dict(resolve_data) if resolve_data else ResolveConfig()
        )

        # Parse review section if present
        review_data = data.get("review", {})
        review_config = (
            ReviewConfig.from_dict(review_data) if review_data else ReviewConfig()
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

        # Parse merge section if present
        merge_data = data.get("merge", {})
        merge_config = (
            MergeConfig.from_dict(merge_data) if merge_data else MergeConfig()
        )

        # Parse context section if present
        context_data = data.get("context", {})
        context_config = (
            ContextConfig.from_dict(context_data) if context_data else ContextConfig()
        )

        # Parse config section if present (for 'mergai config' command)
        config_section_data = data.get("config", {})
        config_config = (
            InitConfig.from_dict(config_section_data)
            if config_section_data
            else InitConfig()
        )

        # Parse workflows section if present
        workflows_data = data.get("workflows", {})
        workflows_config = (
            WorkflowsConfig.from_dict(workflows_data)
            if workflows_data
            else WorkflowsConfig()
        )

        return cls(
            project=project_config,
            fork=fork_config,
            resolve=resolve_config,
            review=review_config,
            branch=branch_config,
            prompt=prompt_config,
            commit=commit_config,
            pr=pr_config,
            finalize=finalize_config,
            merge=merge_config,
            context=context_config,
            config=config_config,
            workflows=workflows_config,
            _raw=data,
        )


def load_config(config_path: str | None = None) -> MergaiConfig:
    """Load configuration from a YAML file.

    If config_path is explicitly provided and the file doesn't exist, raises an error.
    If config_path is None and the default .mergai/config.yml doesn't exist, returns default config.

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
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {path}: {e}") from e

    # Handle empty file or file with only comments
    if data is None:
        return MergaiConfig()

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping (dictionary)")

    return MergaiConfig.from_dict(data)
