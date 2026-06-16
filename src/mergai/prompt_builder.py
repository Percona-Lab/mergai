"""Prompt building utilities for MergAI.

This module provides the PromptBuilder class which encapsulates all logic
for building prompts for AI agents from MergaiNote data, plus free
functions for prompts that don't depend on a merge note (e.g. CI fixes).
"""

import json

from . import prompts
from .config import ProjectConfig, PromptConfig
from .models import MergaiNote
from .utils import util


def _project_prompt_context(project_config: ProjectConfig | None) -> dict:
    """Build the render-context for the CI-fix prompt templates.

    Falls back to :class:`ProjectConfig` defaults (neutral wording) when no
    project config is supplied, so the templates always render — never leaving
    raw ``{{ ... }}`` markers in the prompt.
    """
    pc = project_config if project_config is not None else ProjectConfig()
    return {
        "name": pc.name,
        "fork_term": pc.fork_term,
        "upstream_term": pc.upstream_term,
    }


def serialize_note_for_prompt(
    note: MergaiNote,
    prompt_config: PromptConfig,
    *,
    include_solutions: bool = False,
) -> dict:
    """Serialize a merge note's context fields for embedding in a prompt.

    Single source of truth shared by the resolve/describe prompts
    (:meth:`PromptBuilder._prepare_note_for_prompt`) and the CI-fix prompt
    (:func:`build_ci_fix_preamble`). Context fields are hydrated with the
    configurable serialization settings; ``include_solutions`` adds the
    note's prior solutions (prior conflict resolutions and CI fixes), which
    the CI-fix prompt wants but the resolve prompt doesn't.
    """
    serialization_config = prompt_config.to_prompt_serialization_config()
    result: dict = {"merge_info": note.merge_info.to_dict()}
    if note.has_conflict_context and note.conflict_context is not None:
        result["conflict_context"] = note.conflict_context.to_dict(serialization_config)
    if note.has_merge_context and note.merge_context is not None:
        result["merge_context"] = note.merge_context.to_dict(serialization_config)
    if include_solutions and note.has_solutions and note.solutions is not None:
        result["solutions"] = note.solutions
    return result


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

        Thin wrapper over :func:`serialize_note_for_prompt` — the resolve and
        describe prompts don't include prior solutions.
        """
        return serialize_note_for_prompt(self.note, self.prompt_config)

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


def build_ci_fix_preamble(
    note: MergaiNote | None = None,
    prompt_config: PromptConfig | None = None,
    project_config: ProjectConfig | None = None,
) -> str:
    """Build the common prefix shared by all CI-fix prompts.

    Layout:
      1. System prompt for CI fixes
         (``prompts/system_prompt_ci_fix.md``).
      2. Project invariants from ``.mergai/invariants.md`` if present.
      3. ``Merge Context`` section with ``merge_info`` /
         ``merge_context`` / ``conflict_context`` / ``solutions``
         from the merge note —
         only when ``note`` and ``prompt_config`` are both provided.
         Tells the agent it's on a post-merge branch and lets it
         diagnose root cause against the actual merge.
      4. ``CI Fix Context`` description
         (``prompts/ci_fix_context.md``) explaining the per-run JSON
         shape.

    The preamble doesn't depend on any specific
    :class:`WorkflowContext`, so multi-run renderings
    (``mergai prompt ci all``) can emit it once and follow with one
    :func:`build_ci_fix_run_section` per run.
    """
    project_context = _project_prompt_context(project_config)
    system_prompt = prompts.load_system_prompt_ci_fix(project_context)
    project_invariants = util.load_if_exists(".mergai/invariants.md")

    parts: list[str] = [system_prompt, "\n\n"]
    if project_invariants:
        parts.extend([project_invariants, "\n\n"])

    if note is not None and prompt_config is not None:
        merge_data = serialize_note_for_prompt(
            note, prompt_config, include_solutions=True
        )
        if merge_data:
            # merge_context_for_ci_fix.md already opens with a `## Merge Context`
            # heading; the JSON block follows the explanation under it (no second
            # heading).
            parts.extend(
                [
                    prompts.load_merge_context_for_ci_fix_prompt(project_context),
                    "\n\n",
                ]
            )
            parts.append("```json\n")
            parts.append(json.dumps(merge_data, indent=2))
            parts.append("\n```\n\n")

    parts.extend([prompts.load_ci_fix_context_prompt(), "\n\n"])
    return "".join(parts)


def build_ci_fix_run_section(context, *, heading: str = "## CI Fix Context") -> str:
    """Build the per-run section: optional heading + the WorkflowContext JSON.

    Multi-run callers pass a per-run heading like ``"## Run 12345 — clang-tidy"``
    to disambiguate. The single-run prompt passes ``heading=""`` because
    ``ci_fix_context.md`` already supplies the ``## CI Fix Context`` heading;
    an empty heading emits just the JSON block (no duplicate header).
    """
    context_dict = {
        "workflow_name": context.workflow_name,
        "run_id": context.run_id,
        "pr_number": context.pr_number,
        "summary": context.summary,
        "files_affected": list(context.files_affected),
        "artifacts_dir": context.artifacts_dir,
        "details": context.details,
    }
    prefix = f"{heading}\n\n" if heading else ""
    return prefix + "```json\n" + json.dumps(context_dict, indent=2) + "\n```\n"


def build_ci_fix_prompt(
    context,
    note: MergaiNote | None = None,
    prompt_config: PromptConfig | None = None,
    project_config: ProjectConfig | None = None,
) -> str:
    """Build the full single-run CI-fix prompt.

    Mirrors the structure of :meth:`PromptBuilder.build_resolve_prompt`:
    system prompt + project invariants + (optional) merge context +
    per-context section + a JSON serialization of the input data. The
    agent is told to write a response with the same shape as the
    resolve flow
    (``resolved``/``unresolved``/``modified``/``summary``/``review_notes``)
    so the post-processing pipeline (validators, commit message, note
    attachment) can be shared.

    When ``note`` and ``prompt_config`` are both supplied, the prompt
    embeds the merge note (merge_info, merge_context, conflict_context,
    prior solutions, etc.) so the agent can diagnose the CI failure
    against what was just merged in instead of treating the failure as
    an isolated build issue. Always pass them when available.

    This is what :class:`ResolveHandler` feeds to the agent. For the
    inspection command ``mergai prompt ci`` with multi-run targets, see
    :func:`build_ci_fix_preamble` + :func:`build_ci_fix_run_section`.

    Free function rather than a ``PromptBuilder`` method so callers
    that don't have a note (e.g. ``mergai prompt ci`` invoked outside
    a mergai working tree for debugging) can still render the prompt.
    """
    return build_ci_fix_preamble(
        note=note, prompt_config=prompt_config, project_config=project_config
    ) + build_ci_fix_run_section(context, heading="")


def build_review_prompt(
    context,
    note: MergaiNote | None = None,
    prompt_config: PromptConfig | None = None,
    project_config: ProjectConfig | None = None,
) -> str:
    """Build the full prompt for ``mergai review fix``.

    Mirrors :func:`build_ci_fix_prompt`: review system prompt + project
    invariants + (optional) merge context + the ``Review Context`` JSON
    describing the unresolved review threads to address.

    ``context`` is a ``ReviewContext`` (from :mod:`mergai.review.context`)
    exposing a ``threads`` mapping (``{thread_id: {...}}``) that is embedded
    verbatim as the agent's per-thread input. When ``note`` and
    ``prompt_config`` are both supplied, the merge note is embedded so the
    agent can relate a comment to what was merged / resolved.

    Free function (not a ``PromptBuilder`` method) so ``mergai prompt`` and
    tests can render it without a note.
    """
    project_context = _project_prompt_context(project_config)
    system_prompt = prompts.load_system_prompt_review(project_context)
    project_invariants = util.load_if_exists(".mergai/invariants.md")

    parts: list[str] = [system_prompt, "\n\n"]
    if project_invariants:
        parts.extend([project_invariants, "\n\n"])

    if note is not None and prompt_config is not None:
        merge_data = serialize_note_for_prompt(
            note, prompt_config, include_solutions=True
        )
        if merge_data:
            parts.extend(
                [prompts.load_merge_context_for_review_prompt(project_context), "\n\n"]
            )
            parts.append("```json\n")
            parts.append(json.dumps(merge_data, indent=2))
            parts.append("\n```\n\n")

    parts.extend([prompts.load_review_context_prompt(), "\n\n"])
    parts.append("```json\n")
    parts.append(json.dumps(context.threads, indent=2))
    parts.append("\n```\n")
    return "".join(parts)
