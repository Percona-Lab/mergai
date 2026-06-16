"""AI-agent handler for code-review fixes.

Runs the same retry/validation loop as ``mergai resolve`` / ``mergai ci fix``
(via :class:`AgentExecutor`) but over a :class:`ReviewContext` instead of a
merge note or CI failure. On success returns the agent's parsed JSON response;
the caller in ``mergai.commands.review`` wraps it as a ``type: review_fix``
solution and produces the commit.
"""

from __future__ import annotations

import logging

from ..agent_executor import AgentExecutionError, AgentExecutor
from ..app import AppContext
from ..config import ReviewConfig
from ..prompt_builder import build_review_prompt
from .context import ReviewContext

log = logging.getLogger(__name__)


def make_review_validator(executor: AgentExecutor, thread_ids: set[str]):
    """Build a validator for a review-fix response.

    Combines the shared file check (every path under ``resolved`` /
    ``modified`` actually changed on disk) with a coverage check: every
    review thread id must be classified exactly once across ``addressed`` and
    ``unaddressed``, and no unknown thread id may appear. There is no
    conflict-marker check - these are not conflict files.
    """

    def validate(solution: dict) -> str | None:
        file_error = executor.validate_solution_files(solution)
        if file_error:
            return file_error

        response = solution.get("response", {}) or {}
        addressed = set(response.get("addressed", {}) or {})
        unaddressed = set(response.get("unaddressed", {}) or {})

        unknown = (addressed | unaddressed) - thread_ids
        if unknown:
            return (
                "Response references unknown thread id(s): "
                + ", ".join(sorted(unknown))
                + ". Use only the thread ids from the Review Context."
            )

        both = addressed & unaddressed
        if both:
            return (
                "Thread id(s) classified as both addressed and unaddressed: "
                + ", ".join(sorted(both))
                + ". Each thread must appear in exactly one."
            )

        missing = thread_ids - addressed - unaddressed
        if missing:
            return (
                "Every review thread must be classified. Missing thread id(s): "
                + ", ".join(sorted(missing))
                + ". Add each to 'addressed' or 'unaddressed'."
            )
        return None

    return validate


class ReviewHandler:
    """Runs the AI agent over the review context."""

    def __init__(self, app: AppContext, config: ReviewConfig):
        self.app = app
        self.config = config

    def execute(self, context: ReviewContext) -> dict | None:
        """Run the agent against ``context`` and return its parsed solution.

        Returns the agent's JSON response (with the canonical ``response``
        wrapper applied by :class:`AgentExecutor`) on success, or ``None`` if
        the agent could not produce a valid result within the retry budget.

        Embeds the merge note (when available) so the agent can relate a
        comment to what was merged and which conflicts were already resolved.
        """
        note = self.app.note if self.app.has_note else None
        prompt = build_review_prompt(
            context,
            note=note,
            prompt_config=self.app.config.prompt,
            project_config=self.app.config.project,
        )
        # An empty review agent falls back to the resolve agent.
        agent_desc = self.config.agent or None
        agent = self.app.get_agent(agent_desc=agent_desc, yolo=True)
        executor = AgentExecutor(
            agent=agent,
            state_dir=self.app.state.path,
            max_attempts=self.config.max_attempts,
            repo=self.app.repo,
        )

        try:
            return executor.run_with_retry(
                prompt=prompt,
                validator=make_review_validator(executor, context.thread_ids),
            )
        except AgentExecutionError as e:
            log.warning("Agent failed to produce a review fix: %s", e)
            return None
