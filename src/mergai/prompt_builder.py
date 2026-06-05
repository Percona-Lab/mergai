"""Prompt building utilities for MergAI.

This module provides the PromptBuilder class which encapsulates all logic
for building prompts for AI agents from MergaiNote data.
"""

import json

from . import prompts
from .config import PromptConfig
from .models import MergaiNote
from .utils import util


class PromptBuilder:
    """Builds prompts for AI agents from MergaiNote data.

    This class encapsulates all prompt-building logic, taking a MergaiNote
    and PromptConfig as dependencies. It handles loading system prompts,
    project invariants, and serializing note data for AI consumption.

    Attributes:
        note: The MergaiNote containing merge/conflict data.
        prompt_config: Configuration for prompt serialization.

    Example usage:
        builder = PromptBuilder(note, config.prompt)
        resolve_prompt = builder.build_resolve_prompt()
        describe_prompt = builder.build_describe_prompt(merge_base_sha)
    """

    def __init__(self, note: MergaiNote, prompt_config: PromptConfig):
        """Initialize PromptBuilder.

        Args:
            note: MergaiNote instance with merge data.
            prompt_config: PromptConfig for serialization settings.
        """
        self.note = note
        self.prompt_config = prompt_config

    def build_resolve_prompt(self) -> str:
        """Build the prompt for conflict resolution.

        Constructs a complete prompt by combining:
        - System prompt for resolution
        - Project invariants (if present)
        - Conflict context prompt (if conflict_context exists)
        - PR comments prompt (if PR comments exist)
        - User comment prompt (if user comment exists)
        - Serialized note data as JSON

        Returns:
            The complete prompt string for the AI agent.
        """

        system_prompt_resolve = prompts.load_system_prompt_resolve()
        project_invariants = util.load_if_exists(".mergai/invariants.md")

        prompt = system_prompt_resolve + "\n\n"
        if project_invariants:
            prompt += project_invariants + "\n\n"

        if self.note.has_conflict_context:
            prompt += prompts.load_conflict_context_prompt() + "\n\n"

        if self.note.has_pr_comments:
            prompt += prompts.load_pr_comments_prompt() + "\n\n"

        if self.note.has_user_comment:
            prompt += prompts.load_user_comment_prompt() + "\n\n"

        # Prepare note data for prompt serialization
        # Hydrate conflict_context with configurable commit fields
        note_for_prompt = self._prepare_note_for_prompt()

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(note_for_prompt, indent=2)
        prompt += "\n```\n"

        return prompt

    def build_describe_prompt(
        self,
        merge_base_sha: str,
        verification_feedback: list[dict] | None = None,
    ) -> str:
        """Build the prompt for merge description.

        Constructs a complete prompt by combining:
        - System prompt for description
        - Project invariants (if present)
        - The merge-base SHA (the diff base the agent must use)
        - Verification feedback from a prior attempt (if present)
        - Serialized note data as JSON

        Args:
            merge_base_sha: SHA of the merge-base between the fork tip and the
                merge commit. The agent must diff ``merge_base..merge_commit`` to
                see only what the merge pulls in (see the system prompt).
            verification_feedback: Issues raised by the verifier agent about a
                previous draft. When provided, a "Corrections required" section
                is appended so the regenerated description avoids those claims.

        Returns:
            The complete prompt string for the AI agent.
        """

        system_prompt_describe = prompts.load_system_prompt_describe()
        prompt = system_prompt_describe + "\n\n"

        project_invariants = util.load_if_exists(".mergai/invariants.md")
        if project_invariants:
            prompt += project_invariants + "\n\n"

        prompt += self._render_diff_base(merge_base_sha)

        if verification_feedback:
            prompt += self._render_verification_feedback(verification_feedback)

        # Prepare note data for prompt serialization
        note_for_prompt = self._prepare_note_for_prompt()

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(note_for_prompt, indent=2)
        prompt += "\n```\n"

        return prompt

    def _render_diff_base(self, merge_base_sha: str) -> str:
        """Render the diff-base section naming the exact SHAs to diff.

        Spells out ``git diff <diff_base> <merge_commit>`` with concrete SHAs so
        the agent reads only the merge's incoming changes, not the fork's
        pre-existing divergence from upstream. Callers resolve the diff base
        first; ``describe()`` refuses to run without one, so there is no "no
        diff base" fallback to render here.
        """
        merge_commit_sha = self.note.merge_info.merge_commit_sha
        target_branch_sha = self.note.merge_info.target_branch_sha
        warning = (
            f"`{merge_base_sha}` is the authoritative diff base for this merge "
            + f"and `{merge_commit_sha}` is the merge commit. Use these two SHAs "
            + "exactly as given — do NOT recompute a base with `git merge-base`. "
            + f"Do NOT diff the fork tip (`{target_branch_sha}`) against the "
            + "merge commit — that shows the fork's own customizations as if "
            + "this merge changed them."
        )
        lines = [
            "## Diff base for this merge",
            "",
            "Read the incoming changes with this exact command "
            + "(substitute the file path):",
            "",
            "```",
            f"git diff {merge_base_sha} {merge_commit_sha} -- <file>",
            "```",
            "",
            warning,
            "",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_verification_feedback(issues: list[dict]) -> str:
        """Render verifier issues as a corrections section for the retry prompt."""
        intro = (
            "A fact-check of your previous description against the actual diff "
            + "found the following unsupported claims. Re-read the relevant diffs "
            + "and produce a new description that does NOT repeat them. Drop any "
            + "claim you cannot back with the diff."
        )
        lines = [
            "## Corrections required",
            "",
            intro,
            "",
        ]
        for issue in issues:
            location = issue.get("location", "unknown")
            claim = issue.get("claim", "")
            reason = issue.get("reason", "")
            lines.append(f"- **{location}**: {claim}")
            if reason:
                lines.append(f"  - Why it is wrong: {reason}")
        return "\n".join(lines) + "\n\n"

    def build_describe_verify_prompt(self, draft: dict, merge_base_sha: str) -> str:
        """Build the prompt for fact-checking a draft merge description.

        Combines the verifier system prompt, the diff base, the serialized note
        data, and the draft description the verifier must check against the
        actual diff.

        Args:
            draft: The draft describe response dict (summary / auto_merged /
                review_notes) to fact-check.
            merge_base_sha: SHA of the merge-base; the verifier must diff
                ``merge_base..merge_commit`` for the same reason the drafter does.

        Returns:
            The complete prompt string for the verifier agent.
        """
        prompt = prompts.load_system_prompt_describe_verify() + "\n\n"

        prompt += self._render_diff_base(merge_base_sha)

        note_for_prompt = self._prepare_note_for_prompt()

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(note_for_prompt, indent=2)
        prompt += "\n```\n\n"

        prompt += "## Draft Description To Verify\n\n"
        prompt += "```json\n"
        prompt += json.dumps(draft, indent=2)
        prompt += "\n```\n"

        return prompt

    def _prepare_note_for_prompt(self) -> dict:
        """Prepare note data for prompt serialization.

        Hydrates context fields (conflict_context, merge_context) using the
        configurable prompt serialization settings from config.

        Returns:
            A dict with context fields hydrated for prompt use.
        """
        prompt_serialization_config = (
            self.prompt_config.to_prompt_serialization_config()
        )

        result: dict = {"merge_info": self.note.merge_info.to_dict()}

        if self.note.has_conflict_context and self.note.conflict_context is not None:
            result["conflict_context"] = self.note.conflict_context.to_dict(
                prompt_serialization_config
            )

        if self.note.has_merge_context and self.note.merge_context is not None:
            result["merge_context"] = self.note.merge_context.to_dict(
                prompt_serialization_config
            )

        if self.note.has_pr_comments and self.note.pr_comments is not None:
            result["pr_comments"] = self.note.pr_comments

        if self.note.has_user_comment and self.note.user_comment is not None:
            result["user_comment"] = self.note.user_comment

        return result

    @staticmethod
    def error_to_prompt(error: str) -> str:
        """Convert an error message to a prompt for retry.

        Used during agent retry loops to inform the AI about what went wrong
        with the previous attempt.

        Args:
            error: The error message to convert.

        Returns:
            Formatted prompt string describing the error.
        """
        return f"An error occurred while trying to process the output: {error}"
