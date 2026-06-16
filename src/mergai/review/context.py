"""Building the agent prompt context from actionable review threads.

:class:`ReviewContext` plays the role that ``WorkflowContext`` plays for the
CI-fix flow: a self-contained, JSON-serializable description of the work,
embedded verbatim into the prompt by
:func:`mergai.prompt_builder.build_review_prompt`.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime

from .threads import ReviewComment, ReviewThread, comment_in_scope, is_trusted_author


@dataclass
class ReviewContext:
    """Prompt-ready view of the review threads the agent must address.

    ``threads`` maps each thread id to its file / line / diff hunk and the
    full comment conversation - the exact JSON the agent receives. ``summary``
    is a one-line human description for logging.
    """

    threads: dict[str, dict]
    summary: str

    @property
    def thread_ids(self) -> set[str]:
        return set(self.threads)


def _thread_to_dict(
    thread: ReviewThread, keep: Callable[[ReviewComment], bool]
) -> dict:
    return {
        "path": thread.path,
        "line": thread.line,
        "diff_hunk": thread.diff_hunk,
        "comments": [
            {
                "author": c.author,
                "created_at": c.created_at,
                "body": c.body,
            }
            for c in thread.comments
            if keep(c)
        ],
    }


def build_review_context(
    threads: list[ReviewThread],
    *,
    trusted_associations: AbstractSet[str] = frozenset(),
    trusted_logins: AbstractSet[str] = frozenset(),
    process_external: bool = True,
    cutoff: datetime | None = None,
) -> ReviewContext:
    """Turn actionable :class:`ReviewThread` values into a :class:`ReviewContext`.

    The agent has full repository access and reads file contents itself, so
    the context carries only what GitHub knows and the working tree doesn't:
    the thread anchor (path/line), the diff hunk the reviewer saw, and the
    conversation. Thread ids are the join key back to the agent's response.

    When ``process_external`` is False, comments from untrusted authors are
    dropped from each thread's conversation, so an external account cannot
    smuggle instructions to the agent by replying into a trusted thread. The
    threads themselves are already filtered by their root author in
    :func:`mergai.review.threads.filter_actionable`.

    When ``cutoff`` is set, comments posted or edited after it are likewise
    dropped, so the agent sees the conversation as it stood when the run was
    triggered.
    """

    def keep(c: ReviewComment) -> bool:
        if cutoff is not None and not comment_in_scope(c, cutoff):
            return False
        if process_external:
            return True
        return is_trusted_author(
            c,
            trusted_associations=trusted_associations,
            trusted_logins=trusted_logins,
        )

    mapping = {t.thread_id: _thread_to_dict(t, keep) for t in threads}
    n = len(mapping)
    files = sorted({t.path for t in threads if t.path})
    summary = f"{n} unresolved review thread(s)" + (
        f" across {len(files)} file(s)" if files else ""
    )
    return ReviewContext(threads=mapping, summary=summary)
