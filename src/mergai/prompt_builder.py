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
        prompt_serialization_config = self.prompt_config.to_prompt_serialization_config()

        result = {"merge_info": self.note.merge_info.to_dict()}

        if self.note.has_conflict_context:
            result["conflict_context"] = self.note.conflict_context.to_dict(
                prompt_serialization_config
            )

        if self.note.has_merge_context:
            result["merge_context"] = self.note.merge_context.to_dict(
                prompt_serialization_config
            )

        if self.note.has_pr_comments:
            result["pr_comments"] = self.note.pr_comments

        if self.note.has_user_comment:
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
