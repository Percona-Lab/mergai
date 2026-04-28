"""Base type for workflow handlers."""

from abc import ABC, abstractmethod

from ...app import AppContext
from ...config import WorkflowConfig
from ..context_builders.base import WorkflowContext


class WorkflowHandler(ABC):
    """Executes a fix for a CI workflow failure given its context.

    Subclasses live in :mod:`mergai.ci.handlers`; one per
    ``WorkflowConfig.action_type`` value:

    - ``command`` → :class:`.command.CommandHandler` (run a shell command)
    - ``resolve`` → :class:`.resolve.ResolveHandler` (run the AI agent)
    """

    def __init__(self, app: AppContext, config: WorkflowConfig):
        self.app = app
        self.config = config

    @abstractmethod
    def execute(self, context: WorkflowContext) -> dict | None:
        """Attempt the fix; return the solution dict on success.

        Handlers write their changes to the working tree. The caller
        (``mergai ci handle``) wraps the returned dict as a
        ``type: ci_fix`` entry in the note's ``solutions`` list, builds
        the commit, and attaches the selective git note. Handlers MUST
        NOT commit themselves.

        Args:
            context: The failure context built by the configured
                :class:`~mergai.ci.context_builders.base.WorkflowContextBuilder`.

        Returns:
            A solution dict with the canonical
            ``{"response": {"summary", "resolved", "unresolved",
            "modified", "review_notes"}}`` shape — same as the resolve
            flow — so the post-processing pipeline can be shared.
            ``None`` when the handler did not produce a fix (e.g. the
            agent failed, or a command produced no diff); the caller
            then exits without recording an attempt, leaving the cap
            untouched for the next workflow run.
        """
