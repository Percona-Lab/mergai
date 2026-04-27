"""Base type for workflow handlers."""

from abc import ABC, abstractmethod

from ..context_builders.base import WorkflowContext


class WorkflowHandler(ABC):
    """Executes a fix for a CI workflow failure given its context.

    Subclasses live in :mod:`mergai.ci.handlers`; one per
    ``WorkflowConfig.action_type`` value:

    - ``command`` → :class:`.command.CommandHandler` (run a shell command)
    - ``resolve`` → :class:`.resolve.ResolveHandler` (run the AI agent)
    """

    @abstractmethod
    def execute(self, context: WorkflowContext) -> bool:
        """Attempt the fix.

        Handlers should write their changes to the working tree. The
        caller (``mergai ci handle``) is responsible for creating the
        commit and pushing. Handlers must NOT commit themselves.

        Args:
            context: The failure context built by the configured
                :class:`~mergai.ci.context_builders.base.WorkflowContextBuilder`.

        Returns:
            True if the fix appears to have been applied (e.g. the
            working tree is dirty), False otherwise. The caller decides
            what to do with a False result (retry vs. give up is driven
            by ``WorkflowConfig.max_attempts`` at the command level).
        """
