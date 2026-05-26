"""Context builders for CI workflow failures.

Looks up a :class:`~.base.WorkflowContextBuilder` by ``type`` string
from :class:`mergai.config.WorkflowContextConfig`. New types can be
registered by appending to ``_BUILDERS``.
"""

from ...app import AppContext
from .base import WorkflowContext, WorkflowContextBuilder
from .bazel import BazelContextBuilder
from .diff import DiffContextBuilder
from .sarif import SARIFContextBuilder

_BUILDERS: dict[str, type[WorkflowContextBuilder]] = {
    "diff": DiffContextBuilder,
    "sarif": SARIFContextBuilder,
    "bazel": BazelContextBuilder,
}


def get_context_builder(app: AppContext, type_: str) -> WorkflowContextBuilder:
    """Return a context-builder instance for the given context type.

    Args:
        app: The active :class:`~mergai.app.AppContext`. Builders receive
            it so they can fall back to GitHub-API sources (e.g. job
            logs) when the expected artifact is missing.
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
    return builder_cls(app)


__all__ = [
    "WorkflowContext",
    "WorkflowContextBuilder",
    "DiffContextBuilder",
    "SARIFContextBuilder",
    "BazelContextBuilder",
    "get_context_builder",
]
