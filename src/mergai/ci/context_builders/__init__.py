"""Context builders for CI workflow failures.

Looks up a :class:`~.base.WorkflowContextBuilder` by ``type`` string
from :class:`mergai.config.WorkflowContextConfig`. New types can be
registered by appending to ``_BUILDERS``.
"""

from .base import WorkflowContext, WorkflowContextBuilder
from .diff import DiffContextBuilder
from .sarif import SARIFContextBuilder

_BUILDERS: dict[str, type[WorkflowContextBuilder]] = {
    "diff": DiffContextBuilder,
    "sarif": SARIFContextBuilder,
}


def get_context_builder(type_: str) -> WorkflowContextBuilder:
    """Return a context-builder instance for the given context type.

    Args:
        type_: ``WorkflowContextConfig.type`` value (e.g. ``"diff"``,
            ``"sarif"``).

    Raises:
        ValueError: If no builder is registered for ``type_``.
    """
    builder_cls = _BUILDERS.get(type_)
    if builder_cls is None:
        known = ", ".join(sorted(_BUILDERS)) or "(none)"
        raise ValueError(
            f"No context builder registered for type '{type_}'. Known: {known}"
        )
    return builder_cls()


__all__ = [
    "WorkflowContext",
    "WorkflowContextBuilder",
    "DiffContextBuilder",
    "SARIFContextBuilder",
    "get_context_builder",
]
