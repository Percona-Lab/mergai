"""Handlers that attempt CI-failure fixes.

Dispatch from :class:`mergai.config.WorkflowConfig.action_type` to a
concrete :class:`~.base.WorkflowHandler`. New action types can be
registered by appending to ``_HANDLERS``.
"""

from ...app import AppContext
from ...config import (
    WORKFLOW_ACTION_COMMAND,
    WORKFLOW_ACTION_RESOLVE,
    WorkflowConfig,
)
from .base import WorkflowHandler
from .command import CommandHandler
from .resolve import ResolveHandler

_HANDLERS: dict[str, type[WorkflowHandler]] = {
    WORKFLOW_ACTION_COMMAND: CommandHandler,
    WORKFLOW_ACTION_RESOLVE: ResolveHandler,
}


def get_handler(app: AppContext, config: WorkflowConfig) -> WorkflowHandler:
    """Return a handler for ``config.action_type``.

    Args:
        app: The active :class:`~mergai.app.AppContext`.
        config: The per-workflow config.

    Raises:
        ValueError: If no handler is registered for
            ``config.action_type``. Should not happen in practice —
            :meth:`mergai.config.WorkflowConfig.from_dict` validates
            against :data:`mergai.config.VALID_WORKFLOW_ACTION_TYPES`.
    """
    handler_cls = _HANDLERS.get(config.action_type)
    if handler_cls is None:
        known = ", ".join(sorted(_HANDLERS)) or "(none)"
        raise ValueError(
            f"No handler registered for action_type '{config.action_type}'. "
            f"Known: {known}"
        )
    return handler_cls(app, config)


__all__ = [
    "WorkflowHandler",
    "CommandHandler",
    "ResolveHandler",
    "get_handler",
]
