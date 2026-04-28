"""AI-agent handler for CI workflow fixes."""

import logging

from ...agent_executor import AgentExecutionError, AgentExecutor
from ...app import AppContext
from ...config import WorkflowConfig
from ..context_builders.base import WorkflowContext
from .base import WorkflowHandler

log = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    "A CI workflow failed on the current branch and you need to fix the "
    "source files so that the workflow passes next time.\n"
    "\n"
    "Workflow: {workflow_name}\n"
    "PR: #{pr_number}\n"
    "Run ID: {run_id}\n"
    "\n"
    "{summary}\n"
    "\n"
    "## Affected files\n"
    "{files}\n"
    "\n"
    "## Details\n"
    "{details}\n"
    "\n"
    "Please edit the files in the working tree to fix the reported "
    "issues. Do not run any build or test commands yourself — this job "
    "will commit and push your changes, and the CI workflow will rerun "
    "automatically. If you cannot fix an issue, leave the file as-is "
    "and note it in your response.\n"
)


def build_ci_fix_prompt(context: WorkflowContext) -> str:
    """Render the prompt the resolve handler would feed to the AI agent.

    Pulled out as a free function so ``mergai prompt ci`` can render the
    exact same text without instantiating a handler or running the agent.
    """
    files = (
        "\n".join(f"- {p}" for p in context.files_affected)
        if context.files_affected
        else "(none listed)"
    )
    return _PROMPT_TEMPLATE.format(
        workflow_name=context.workflow_name,
        pr_number=context.pr_number,
        run_id=context.run_id,
        summary=context.summary,
        files=files,
        details=context.details or "(no additional details)",
    )


class ResolveHandler(WorkflowHandler):
    """Runs the AI agent over the failure context.

    Reuses the existing :class:`~mergai.agent_executor.AgentExecutor`
    retry loop but with a custom validator that checks the working tree
    is dirty after the agent runs (the agent's JSON response shape is
    not prescribed for CI fixes).
    """

    def __init__(self, app: AppContext, config: WorkflowConfig):
        self.app = app
        self.config = config

    def execute(self, context: WorkflowContext) -> bool:
        prompt = build_ci_fix_prompt(context)
        agent = self.app.get_agent(yolo=True)
        executor = AgentExecutor(
            agent=agent,
            state_dir=self.app.state.path,
            max_attempts=self.config.max_attempts,
            repo=self.app.repo,
        )

        try:
            executor.run_with_retry(prompt=prompt, validator=self._validate_fix)
        except AgentExecutionError as e:
            log.warning("Agent failed to produce a fix: %s", e)
            return False

        return self.app.repo.is_dirty(untracked_files=True)

    def _validate_fix(self, _response: dict) -> str | None:
        """Validator for ``AgentExecutor.run_with_retry``.

        The agent's JSON response shape is unconstrained for CI fixes —
        what matters is that the working tree contains modifications we
        can commit. If it's clean, tell the agent to actually make edits.
        """
        if self.app.repo.is_dirty(untracked_files=True):
            return None
        return (
            "The working tree has no modifications after your run. "
            "You must edit the affected files to fix the CI failure."
        )
