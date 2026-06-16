"""Fetching and filtering a PR's review threads.

GitHub exposes thread-level *resolved* / *outdated* state only through the
GraphQL API (the REST review-comments endpoint returns a flat list with no
thread state), so :func:`fetch_review_threads` issues a GraphQL query. The
pure filtering logic (:func:`thread_skip_reason`, :func:`filter_actionable`)
is kept separate and free of any network access so it can be unit-tested with
hand-built :class:`ReviewThread` values.
"""

from __future__ import annotations

import logging
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

# Per-comment fields, shared by the thread-list query and the follow-up
# comment-pagination query via the ``_COMMENT_FIELDS`` token swapped in by
# :func:`_with_comment_fields`. The fields give us the anchor (path/line), the
# diff hunk to show the agent, and the author / databaseId needed to reply over
# REST. A token swap avoids %-formatting / str.format, which clash with the
# query's own braces.
_COMMENT_FIELDS = """
  databaseId
  body
  path
  line
  originalLine
  diffHunk
  createdAt
  lastEditedAt
  authorAssociation
  author { login }
"""


def _with_comment_fields(query: str) -> str:
    """Splice :data:`_COMMENT_FIELDS` into a query's ``_COMMENT_FIELDS`` token."""
    return query.replace("_COMMENT_FIELDS", _COMMENT_FIELDS)


# Pull review threads (and their first page of comments) for a PR. Thread-level
# `isResolved` / `isOutdated` are GraphQL-only.
_REVIEW_THREADS_QUERY = _with_comment_fields("""
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          comments(first: 100) {
            pageInfo { hasNextPage endCursor }
            nodes { _COMMENT_FIELDS }
          }
        }
      }
    }
  }
}
""")

# Follow-up query to drain a single thread's remaining comments. A long-running
# discussion can exceed the 100 comments fetched inline above; without this the
# tail (later replies, opt-out tokens, the real last author) would be dropped
# from both the agent context and the skip filtering.
_THREAD_COMMENTS_QUERY = _with_comment_fields("""
query($id: ID!, $cursor: String) {
  node(id: $id) {
    ... on PullRequestReviewThread {
      comments(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { _COMMENT_FIELDS }
      }
    }
  }
}
""")


@dataclass
class ReviewComment:
    """A single comment within a review thread."""

    database_id: int | None
    author: str
    body: str
    created_at: str
    path: str | None = None
    line: int | None = None
    original_line: int | None = None
    diff_hunk: str | None = None
    author_association: str | None = None
    last_edited_at: str | None = None


@dataclass
class ReviewThread:
    """A PR review thread plus the state needed to filter and reply to it."""

    thread_id: str
    is_resolved: bool
    is_outdated: bool
    comments: list[ReviewComment] = field(default_factory=list)

    @property
    def root_comment(self) -> ReviewComment | None:
        """The first comment - its databaseId anchors a REST reply."""
        return self.comments[0] if self.comments else None

    @property
    def last_comment(self) -> ReviewComment | None:
        return self.comments[-1] if self.comments else None

    @property
    def path(self) -> str | None:
        root = self.root_comment
        return root.path if root else None

    @property
    def line(self) -> int | None:
        root = self.root_comment
        if root is None:
            return None
        return root.line if root.line is not None else root.original_line

    @property
    def diff_hunk(self) -> str | None:
        root = self.root_comment
        return root.diff_hunk if root else None


def _parse_comment(c: dict[str, Any]) -> ReviewComment:
    author = (c.get("author") or {}).get("login") or "unknown"
    return ReviewComment(
        database_id=c.get("databaseId"),
        author=author,
        body=c.get("body") or "",
        created_at=c.get("createdAt") or "",
        path=c.get("path"),
        line=c.get("line"),
        original_line=c.get("originalLine"),
        diff_hunk=c.get("diffHunk"),
        author_association=c.get("authorAssociation"),
        last_edited_at=c.get("lastEditedAt"),
    )


def _parse_thread(node: dict[str, Any]) -> ReviewThread:
    comments = [
        _parse_comment(c) for c in (node.get("comments") or {}).get("nodes") or []
    ]
    return ReviewThread(
        thread_id=node["id"],
        is_resolved=bool(node.get("isResolved")),
        is_outdated=bool(node.get("isOutdated")),
        comments=comments,
    )


def _fetch_remaining_comments(
    requester: Any, thread_id: str, cursor: str | None
) -> list[ReviewComment]:
    """Drain a thread's comments past the first page, starting at ``cursor``."""
    comments: list[ReviewComment] = []
    while cursor is not None:
        _, data = requester.graphql_query(
            _THREAD_COMMENTS_QUERY, {"id": thread_id, "cursor": cursor}
        )
        conn = ((data.get("data") or {}).get("node") or {}).get("comments") or {}
        comments.extend(_parse_comment(c) for c in conn.get("nodes") or [])
        page = conn.get("pageInfo") or {}
        cursor = page.get("endCursor") if page.get("hasNextPage") else None
    return comments


def fetch_review_threads(app: Any, pr_number: int) -> list[ReviewThread]:
    """Fetch all review threads for ``pr_number`` via the GraphQL API.

    ``app`` is an :class:`~mergai.app.AppContext`; it provides ``gh_repo`` (for
    the owner/name and the authenticated requester). Paginates through
    ``reviewThreads`` and, for any thread with more than 100 comments, drains the
    remaining pages so filtering and context see the full conversation.
    """
    owner, name = app.gh_repo.full_name.split("/", 1)
    requester = app.gh_repo._requester  # noqa: SLF001
    threads: list[ReviewThread] = []
    cursor: str | None = None
    while True:
        variables = {
            "owner": owner,
            "name": name,
            "number": pr_number,
            "cursor": cursor,
        }
        _, data = requester.graphql_query(_REVIEW_THREADS_QUERY, variables)
        pull = (data.get("data") or {}).get("repository", {}).get("pullRequest")
        if not pull:
            raise ValueError(f"PR #{pr_number} not found in {app.gh_repo.full_name}.")
        rt = pull.get("reviewThreads") or {}
        for node in rt.get("nodes") or []:
            thread = _parse_thread(node)
            cpage = (node.get("comments") or {}).get("pageInfo") or {}
            if cpage.get("hasNextPage"):
                thread.comments.extend(
                    _fetch_remaining_comments(
                        requester, thread.thread_id, cpage.get("endCursor")
                    )
                )
            threads.append(thread)
        page = rt.get("pageInfo") or {}
        if page.get("hasNextPage"):
            cursor = page.get("endCursor")
        else:
            break
    return threads


def parse_iso8601(value: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp (``...Z``) to an aware datetime.

    Returns ``None`` for an empty or unparseable value.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def comment_in_scope(comment: ReviewComment, cutoff: datetime) -> bool:
    """Whether ``comment`` existed, unedited, at or before ``cutoff``.

    A comment is in scope when it was created at/before the cutoff and has not
    been edited after it. This pins the agent to the comment set as it stood
    when the run was triggered: comments posted - or edited - after the trigger
    are excluded. An unparseable ``created_at`` is treated as out of scope (fail
    safe).
    """
    created = parse_iso8601(comment.created_at)
    if created is None or created > cutoff:
        return False
    edited = parse_iso8601(comment.last_edited_at)
    return edited is None or edited <= cutoff


def scoped_comments(
    comments: list[ReviewComment], cutoff: datetime | None
) -> list[ReviewComment]:
    """The comments in ``comments`` that are in scope for ``cutoff``.

    With no cutoff, returns the list unchanged.
    """
    if cutoff is None:
        return list(comments)
    return [c for c in comments if comment_in_scope(c, cutoff)]


def is_trusted_author(
    comment: ReviewComment,
    *,
    trusted_associations: AbstractSet[str],
    trusted_logins: AbstractSet[str],
) -> bool:
    """Whether ``comment``'s author is trusted to instruct the agent.

    Trusted when the comment's GitHub ``authorAssociation`` is in
    ``trusted_associations`` (e.g. ``OWNER`` / ``MEMBER`` / ``COLLABORATOR``) or
    its author login is in ``trusted_logins`` (an explicit allowlist). A null or
    unknown association is *untrusted* - the gate fails safe.
    """
    if comment.author in trusted_logins:
        return True
    return comment.author_association in trusted_associations


def thread_skip_reason(
    thread: ReviewThread,
    *,
    bot_logins: set[str],
    skip_token: str,
    addressed_ids: AbstractSet[str] = frozenset(),
    trusted_associations: AbstractSet[str] = frozenset(),
    trusted_logins: AbstractSet[str] = frozenset(),
    process_external: bool = True,
    cutoff: datetime | None = None,
) -> str | None:
    """Return why ``thread`` is not actionable, or ``None`` if it is.

    A thread is actionable when it is unresolved, not outdated, carries no
    opt-out token, has comments, hasn't already been addressed by mergai, its
    last comment is not from a configured bot account, and - unless
    ``process_external`` is set - was raised by a trusted author.

    ``addressed_ids`` is the set of thread ids mergai already fixed in a prior
    ``review_fix`` solution (see
    :meth:`MergaiNote.addressed_review_thread_ids`). It is the durable,
    note-backed record of mergai's own work - reloaded from git notes on each
    run - so there is no need to inspect replies on GitHub or assume the
    authenticated token is a bot. ``bot_logins`` stays an explicit escape hatch
    for *other* automation accounts (default: none).

    When ``process_external`` is ``False`` the thread's *root* comment author
    (the reviewer who raised it) must be trusted per
    :func:`is_trusted_author`; otherwise the whole thread is skipped. Untrusted
    *replies* on an otherwise-trusted thread are dropped from the agent context
    separately (see :func:`mergai.review.context.build_review_context`).

    When ``cutoff`` is set, only comments at/before it count (see
    :func:`comment_in_scope`): a thread whose comments are all newer than the
    cutoff is skipped as ``"after cutoff"``, and the opt-out / bot checks run
    against the in-scope comments. (The trust check always uses the thread's
    real root comment - its author is fixed regardless of edits.) This pins the
    run to the comment set as it stood when it was triggered.
    """
    if thread.is_resolved:
        return "resolved"
    if thread.is_outdated:
        return "outdated"
    if not thread.comments:
        return "empty"
    scoped = scoped_comments(thread.comments, cutoff)
    if not scoped:
        return "after cutoff"
    if not process_external:
        # Trust is decided on the thread's actual root (the reviewer who raised
        # it), never on scoped[0]: if the root was edited after the cutoff it
        # falls out of scope, and scoped[0] would then be a later reply - using
        # it would re-root the thread and could misjudge authorship. A root
        # created after the cutoff can't happen here (scoped would be empty,
        # handled above), and an edit never changes a comment's author.
        root = thread.comments[0]
        if not is_trusted_author(
            root,
            trusted_associations=trusted_associations,
            trusted_logins=trusted_logins,
        ):
            return f"external author ({root.author_association or 'unknown'})"
    if thread.thread_id in addressed_ids:
        return "already addressed by mergai"
    if skip_token and any(skip_token in c.body for c in scoped):
        return f"opted out ({skip_token!r})"
    last = scoped[-1]
    if last.author in bot_logins:
        return f"last reply from bot ({last.author})"
    return None


def skip_reason_category(reason: str | None) -> str:
    """Bucket a :func:`thread_skip_reason` value into a short status label.

    ``None`` maps to ``"actionable"``. The descriptive reasons map to stable
    buckets - ``"resolved"`` / ``"outdated"`` / ``"empty"`` / ``"opted-out"`` /
    ``"answered"`` (mergai already replied) / ``"bot"`` / ``"external"`` (raised
    by an untrusted author) / ``"after-cutoff"`` (posted after the trigger) - so
    callers can count and group threads without parsing the human-readable text.
    """
    if reason is None:
        return "actionable"
    if reason.startswith("opted out"):
        return "opted-out"
    if reason.startswith("already addressed by mergai"):
        return "answered"
    if reason.startswith("last reply from bot"):
        return "bot"
    if reason.startswith("external author"):
        return "external"
    if reason.startswith("after cutoff"):
        return "after-cutoff"
    return reason  # "resolved" / "outdated" / "empty"


def filter_actionable(
    threads: list[ReviewThread],
    *,
    bot_logins: set[str],
    skip_token: str,
    addressed_ids: AbstractSet[str] = frozenset(),
    trusted_associations: AbstractSet[str] = frozenset(),
    trusted_logins: AbstractSet[str] = frozenset(),
    process_external: bool = True,
    cutoff: datetime | None = None,
) -> tuple[list[ReviewThread], list[tuple[ReviewThread, str]]]:
    """Split ``threads`` into (actionable, [(skipped, reason), ...])."""
    actionable: list[ReviewThread] = []
    skipped: list[tuple[ReviewThread, str]] = []
    for thread in threads:
        reason = thread_skip_reason(
            thread,
            bot_logins=bot_logins,
            skip_token=skip_token,
            addressed_ids=addressed_ids,
            trusted_associations=trusted_associations,
            trusted_logins=trusted_logins,
            process_external=process_external,
            cutoff=cutoff,
        )
        if reason is None:
            actionable.append(thread)
        else:
            skipped.append((thread, reason))
    return actionable, skipped
