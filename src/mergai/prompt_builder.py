"""Prompt building utilities for MergAI.

This module provides the PromptBuilder class which encapsulates all logic
for building prompts for AI agents from MergaiNote data, plus free
functions for prompts that don't depend on a merge note (e.g. CI fixes).
"""

import json
from typing import TYPE_CHECKING

from . import prompts
from .config import PromptConfig
from .models import MergaiNote
from .utils import util

if TYPE_CHECKING:
    from .ci.context_builders.base import WorkflowContext


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
        describe_prompt = builder.build_describe_prompt()
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

    def build_describe_prompt(self) -> str:
        """Build the prompt for merge description.

        Constructs a complete prompt by combining:
        - System prompt for description
        - Project invariants (if present)
        - Serialized note data as JSON

        Returns:
            The complete prompt string for the AI agent.
        """

        system_prompt_describe = prompts.load_system_prompt_describe()
        prompt = system_prompt_describe + "\n\n"

        project_invariants = util.load_if_exists(".mergai/invariants.md")
        if project_invariants:
            prompt += project_invariants + "\n\n"

        # Prepare note data for prompt serialization
        note_for_prompt = self._prepare_note_for_prompt()

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(note_for_prompt, indent=2)
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


def build_ci_fix_preamble() -> str:
    """Build the common prefix shared by all CI-fix prompts.

    System prompt + project invariants + the
    ``ci_fix_context.md`` description of the per-run JSON shape. This
    part doesn't depend on any specific :class:`WorkflowContext`, so
    multi-run renderings (``mergai prompt ci all``) can emit it once
    and follow with one
    :func:`build_ci_fix_run_section` per run.
    """
    system_prompt = prompts.load_system_prompt_ci_fix()
    project_invariants = util.load_if_exists(".mergai/invariants.md")

    parts: list[str] = [system_prompt, "\n\n"]
    if project_invariants:
        parts.extend([project_invariants, "\n\n"])
    parts.extend([prompts.load_ci_fix_context_prompt(), "\n\n"])
    return "".join(parts)


def build_ci_fix_run_section(
    context: "WorkflowContext", *, heading: str = "## CI Fix Context"
) -> str:
    """Build the per-run section: heading + the WorkflowContext as JSON.

    The default heading matches the original single-run prompt shape
    (so the agent sees the same text it always has). Multi-run callers
    pass a per-run heading like ``"## Run 12345 — clang-tidy"`` to
    disambiguate.
    """
    context_dict = {
        "workflow_name": context.workflow_name,
        "run_id": context.run_id,
        "pr_number": context.pr_number,
        "summary": context.summary,
        "files_affected": list(context.files_affected),
        "details": context.details,
    }
    return (
        f"{heading}\n\n" + "```json\n" + json.dumps(context_dict, indent=2) + "\n```\n"
    )


def build_ci_fix_prompt(context: "WorkflowContext") -> str:
    """Build the full single-run CI-fix prompt.

    Mirrors the structure of :meth:`PromptBuilder.build_resolve_prompt`:
    system prompt + project invariants + per-context section + a JSON
    serialization of the input data. The agent is told to write a
    response with the same shape as the resolve flow
    (``resolved``/``unresolved``/``modified``/``summary``/``review_notes``)
    so the post-processing pipeline (validators, commit message, note
    attachment) can be shared.

    This is what :class:`ResolveHandler` feeds to the agent. For the
    inspection command ``mergai prompt ci`` with multi-run targets, see
    :func:`build_ci_fix_preamble` + :func:`build_ci_fix_run_section`.

    Free function rather than a ``PromptBuilder`` method because CI
    fixes are not driven by a merge note — only the optional
    ``.mergai/invariants.md`` is read from the working tree, and the
    rest of the prompt is fully derived from the supplied
    ``WorkflowContext``.
    """
    return build_ci_fix_preamble() + build_ci_fix_run_section(context)
