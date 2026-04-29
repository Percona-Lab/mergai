"""AI-agent handler for CI workflow fixes.

Runs the same retry/validation loop as ``mergai resolve`` (via
:class:`AgentExecutor` + :meth:`AgentExecutor.validate_solution`) but
over a :class:`WorkflowContext` instead of a merge note. On success
returns the agent's parsed JSON response; the caller in
``mergai.commands.ci.handle`` wraps it as a ``type: ci_fix`` solution
on the note and produces the commit.
"""

import logging

from ...agent_executor import AgentExecutionError, AgentExecutor
from ...prompt_builder import build_ci_fix_prompt
from ..context_builders.base import WorkflowContext
from .base import WorkflowHandler

log = logging.getLogger(__name__)


class ResolveHandler(WorkflowHandler):
    """Runs the AI agent over the failure context.

    Reuses :meth:`AgentExecutor.validate_solution` so the response shape
    and post-validation match the conflict-resolution flow (resolved /
    modified files actually have unstaged changes; resolved files have
    no leftover conflict markers).
    """

    def execute(self, context: WorkflowContext) -> dict | None:
        """Run the agent against ``context`` and return its parsed solution.

        Returns the agent's JSON response (with the canonical
        ``response`` wrapper applied by :class:`AgentExecutor`) on
        success, or ``None`` if the agent could not produce a valid
        result within the configured retry budget.

        Embeds the merge note (when available) into the prompt so the
        agent diagnoses the failure against the post-merge state —
        which upstream commits were brought in, which conflicts mergai
        already resolved, and any prior CI fixes — rather than
        treating the failure in isolation.
        """
        note = self.app.note if self.app.has_note else None
        prompt = build_ci_fix_prompt(
            context, note=note, prompt_config=self.app.config.prompt
        )
        agent = self.app.get_agent(yolo=True)
        executor = AgentExecutor(
            agent=agent,
            state_dir=self.app.state.path,
            max_attempts=self.config.max_attempts,
            repo=self.app.repo,
        )

        try:
            return executor.run_with_retry(
                prompt=prompt,
                validator=executor.validate_solution,
            )
        except AgentExecutionError as e:
            log.warning("Agent failed to produce a fix: %s", e)
            return None
