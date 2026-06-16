"""Rendering and posting replies to review threads.

``mergai review fix`` records one reply intent per processed thread on the
note; ``mergai review post`` renders each (a "fixed" note on threads the agent
addressed, an "unfixable" note on the rest) and posts it over REST
(``create_review_comment_reply``) anchored to the thread's root comment.
Threads are never auto-resolved.

Which threads mergai has already addressed is tracked durably on the note (the
``review_fix`` solution's ``response.addressed``, persisted in git notes), not
by inspecting replies - so replies carry no hidden marker.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import ReviewConfig

log = logging.getLogger(__name__)


def _join(parts: list[str]) -> str:
    """Join the non-empty parts of a reply body with blank lines."""
    return "\n\n".join(p for p in parts if p and p.strip())


def render_fixed_reply(config: ReviewConfig, note: str, commit_sha: str | None) -> str:
    """Body for a thread the agent addressed."""
    parts = [config.reply_fixed_header, (note or "").strip()]
    if commit_sha:
        parts.append(f"Addressed in {commit_sha[:11]}.")
    parts.append(config.reply_footer)
    return _join(parts) or "Addressed."


def render_unfixable_reply(config: ReviewConfig, reason: str) -> str:
    """Body for a thread the agent did not change."""
    parts = [config.reply_unfixable_header, (reason or "").strip(), config.reply_footer]
    return _join(parts) or "No change made."


def render_reply_from_record(config: ReviewConfig, record: dict) -> str:
    """Render the reply body for a recorded reply intent.

    Branches on the record's ``outcome``: ``fixed`` renders the
    addressed-reply (with the agent note + commit), anything else renders the
    unfixable-reply (with the reason).
    """
    if record.get("outcome") == "fixed":
        return render_fixed_reply(
            config, record.get("note", ""), record.get("commit_sha")
        )
    return render_unfixable_reply(config, record.get("reason", ""))


def post_reply(pr: Any, comment_id: int | None, body: str) -> tuple[bool, str | None]:
    """Post ``body`` as a reply to the review comment ``comment_id``.

    ``comment_id`` is the REST database id of the thread's root comment.
    Returns ``(posted, url)``: ``(True, html_url)`` when posted - ``url`` is the
    created reply's GitHub URL for traceability (``None`` if the API response
    carries none) - and ``(False, None)`` when there is no id to anchor the
    reply to (logged, not fatal - one bad record shouldn't sink the run).
    """
    if comment_id is None:
        log.warning("Cannot post reply: no root comment id; skipping.")
        return False, None
    reply = pr.create_review_comment_reply(comment_id, body)
    return True, getattr(reply, "html_url", None)
