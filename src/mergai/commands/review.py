"""``mergai review`` click command group.

Drives the code-review handling flow in two decoupled steps:

* ``mergai review fix`` - fetch a PR's review threads, filter to the ones an
  agent should act on, run the agent over the working tree, commit the result
  as a ``type: review_fix`` solution, and **record** (not post) a reply per
  thread on the note.
* ``mergai review post`` - publish the recorded replies to their threads.

Splitting them lets you fix locally, review the commit, push, and only then
post the replies. ``review list`` is a read-only status view. The PR is
auto-detected from the current branch (the CI workflow checks out the PR
branch); ``--pr-number`` overrides for manual / local runs.
"""

from datetime import datetime, timezone

import click

from ..app import AppContext
from ..review.commit import commit_review_fix_solution
from ..review.context import build_review_context
from ..review.handler import ReviewHandler
from ..review.replies import post_reply, render_reply_from_record
from ..review.threads import (
    ReviewThread,
    fetch_review_threads,
    filter_actionable,
    parse_iso8601,
    skip_reason_category,
    thread_skip_reason,
)
from ..solution_types import REVIEW_FIX
from .pr import get_prs_for_current_branch
from .util import ensure_gh_repo


@click.group()
@click.pass_obj
@click.option(
    "--repo",
    "repo",
    type=str,
    required=False,
    envvar="GH_REPO",
    help="The repository where the PR is located.",
)
def review(app: AppContext, repo: str | None):
    """Code-review handling commands."""
    ensure_gh_repo(app, repo)


def _resolve_pr_number(app: AppContext, pr_number: int | None) -> int:
    """Return the PR number to act on, auto-detecting from the branch if needed."""
    if pr_number is not None:
        return pr_number
    prs = get_prs_for_current_branch(app)
    if not prs:
        raise click.ClickException(
            "No open PR found for the current branch. "
            "Pass --pr-number to act on a specific PR."
        )
    if len(prs) > 1:
        numbers = ", ".join(f"#{p.number}" for p in prs)
        raise click.ClickException(
            f"Multiple open PRs match the current branch ({numbers}). "
            "Pass --pr-number to disambiguate."
        )
    return prs[0].number


def _parse_since(since: str | None):
    """Parse the ``--since`` cutoff to an aware datetime, or ``None``.

    Raises a ``ClickException`` on an unparseable value so a bad CI-supplied
    timestamp fails loudly rather than silently disabling the cutoff.
    """
    if not since:
        return None
    cutoff = parse_iso8601(since)
    if cutoff is None:
        raise click.ClickException(
            f"--since: could not parse timestamp {since!r} (expected ISO-8601, "
            "e.g. 2026-06-15T12:00:00Z)."
        )
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise click.ClickException(
            f"--since: timestamp {since!r} has no timezone; include an offset "
            "(e.g. 2026-06-15T12:00:00Z). A naive cutoff cannot be compared to "
            "GitHub's timezone-aware comment timestamps."
        )
    return cutoff


def _addressed_ids(app: AppContext) -> set[str]:
    """Thread ids mergai already addressed in a prior ``review_fix`` solution.

    Durable, note-backed record (reloaded from git notes by ``context init``)
    of mergai's own work, used to skip threads already fixed on a re-run.
    """
    return app.note.addressed_review_thread_ids() if app.has_note else set()


def _ignored_summary(skipped: list[tuple[ReviewThread, str]]) -> str:
    """Clause describing comments ignored for *security* reasons, or "".

    Counts the two security-driven skip categories - threads raised by an
    untrusted author (``external``) and comments posted after the trigger
    (``after-cutoff``) - and names only the non-zero ones, e.g.
    ``"2 from external author(s), 1 posted after the trigger"``. Other skip
    reasons (resolved / outdated / already answered) are routine and excluded.
    """
    n_ext = sum(1 for _, r in skipped if skip_reason_category(r) == "external")
    n_late = sum(1 for _, r in skipped if skip_reason_category(r) == "after-cutoff")
    parts = []
    if n_ext:
        parts.append(f"{n_ext} from external author(s)")
    if n_late:
        parts.append(f"{n_late} posted after the trigger")
    return ", ".join(parts)


def _post_ack(app: AppContext, pr_number: int, message: str, dry_run: bool) -> None:
    """Post a short acknowledgement comment on the PR (best-effort).

    Gives quick feedback on the trigger that mergai ran and what it did. Under
    ``--dry-run`` the message is printed instead of posted.
    """
    if dry_run:
        click.echo(f"[dry-run] would comment on PR #{pr_number}: {message}")
        return
    try:
        app.gh_repo.get_pull(pr_number).create_issue_comment(message)
        click.echo(f"Posted acknowledgement on PR #{pr_number}.")
    except Exception as e:  # noqa: BLE001 - acknowledgement is best-effort
        click.echo(f"warning: could not post acknowledgement: {e}", err=True)


@review.command()
@click.pass_obj
@click.option(
    "--pr-number",
    "-n",
    type=int,
    default=None,
    help="PR number to act on (default: auto-detect from the current branch).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show the actionable threads and planned replies without running the agent.",
)
@click.option(
    "--ack",
    is_flag=True,
    default=False,
    help=(
        "Post a short acknowledgement comment on the PR summarising the outcome "
        "(how many comments were found / addressed), even when there are none. "
        "Use from CI to give quick feedback on the trigger."
    ),
)
@click.option(
    "--since",
    "since",
    type=str,
    default=None,
    envvar="MERGAI_REVIEW_SINCE",
    help=(
        "ISO-8601 cutoff: ignore review comments posted (or edited) after this "
        "time, pinning the run to the comment set as it stood when it was "
        "triggered. Pass the triggering event's timestamp from CI."
    ),
)
def fix(
    app: AppContext,
    pr_number: int | None,
    dry_run: bool,
    ack: bool,
    since: str | None,
) -> None:
    """Generate fixes for unresolved review comments on a PR.

    Fetches the PR's review threads, keeps the ones that are unresolved, not
    outdated, not opted out (``review.skip_token``), and not already answered
    by mergai, then runs the agent once over all of them. On success, commits
    a ``review_fix`` solution and **records** a reply per thread on the note.

    Replies are not posted here - run ``mergai review post`` to publish them.
    This lets you review the commit and push before anything is posted.
    """
    config = app.config.review
    number = _resolve_pr_number(app, pr_number)
    click.echo(f"Handling review comments for PR #{number} in {app.gh_repo.full_name}.")

    trusted_associations = set(config.trusted_associations)
    trusted_logins = set(config.trusted_logins)
    cutoff = _parse_since(since)
    if cutoff is not None:
        click.echo(f"Cutoff: ignoring comments after {since}.")

    threads = fetch_review_threads(app, number)
    actionable, skipped = filter_actionable(
        threads,
        bot_logins=set(config.bot_logins),
        skip_token=config.skip_token,
        addressed_ids=_addressed_ids(app),
        trusted_associations=trusted_associations,
        trusted_logins=trusted_logins,
        process_external=config.process_external,
        cutoff=cutoff,
    )

    click.echo(
        f"Found {len(threads)} review thread(s): "
        f"{len(actionable)} actionable, {len(skipped)} skipped."
    )
    for thread, reason in skipped:
        loc = f"{thread.path}:{thread.line}" if thread.path else "(general)"
        click.echo(f"  skip {loc} [{thread.thread_id}]: {reason}")

    # Surface the security-driven skips (external authors / post-trigger
    # comments) as a one-line rollup, mirrored in the --ack summary below.
    ignored = _ignored_summary(skipped)
    if ignored:
        click.echo(f"Ignored {ignored} (not processed).")

    # Report the actionable count up front, always - even when it is zero.
    click.echo(f"{len(actionable)} review comment(s) to address.")

    if not actionable:
        click.echo("No actionable review comments. Nothing to do.")
        if ack:
            msg = (
                f"mergai review fix: no review comments to address; "
                f"ignored {ignored} (not processed)."
                if ignored
                else "mergai review fix: no review comments to address."
            )
            _post_ack(app, number, msg, dry_run)
        return

    context = build_review_context(
        actionable,
        trusted_associations=trusted_associations,
        trusted_logins=trusted_logins,
        process_external=config.process_external,
        cutoff=cutoff,
    )
    threads_by_id = {t.thread_id: t for t in actionable}

    click.echo(f"Actionable threads ({context.summary}):")
    for tid, data in context.threads.items():
        loc = f"{data['path']}:{data['line']}" if data.get("path") else "(general)"
        click.echo(f"  {loc} [{tid}] ({len(data['comments'])} comment(s))")

    if dry_run:
        click.echo("\n--- dry run: agent prompt context ---")
        from ..prompt_builder import build_review_prompt

        note = app.note if app.has_note else None
        click.echo(
            build_review_prompt(
                context,
                note=note,
                prompt_config=app.config.prompt,
                project_config=app.config.project,
            )
        )
        click.echo("--- end ---")
        click.echo(
            "\nWould run the agent over the above and, on success, record a reply "
            "for each addressed/unaddressed thread (publish with `review post`)."
        )
        return

    handler = ReviewHandler(app, config)
    agent_solution = handler.execute(context)
    if agent_solution is None:
        click.echo(
            "Agent did not produce a valid review fix. Nothing committed; "
            "no replies recorded."
        )
        return

    response = agent_solution.get("response", {}) or {}
    addressed = response.get("addressed", {}) or {}
    unaddressed = response.get("unaddressed", {}) or {}

    # Enrich each thread entry with the comment's id and location so the git
    # note attached to the commit documents *which* review comments it
    # addressed (and which it didn't) - the durable record used to skip them
    # on a re-run and to explain why the commit was made.
    _annotate_threads(addressed, threads_by_id)
    _annotate_threads(unaddressed, threads_by_id)

    commit_sha: str | None = None
    if response.get("resolved") or response.get("modified"):
        solution = {
            "type": REVIEW_FIX,
            "request": {
                "pr_number": number,
                "thread_count": len(actionable),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            **agent_solution,
        }
        solution_idx = app.note.add_solution(solution)
        app.save_note(app.note)
        try:
            commit_review_fix_solution(app, solution_idx)
        except Exception as e:
            # The commit failed, so the solution we just persisted is a phantom:
            # addressed_review_thread_ids() would treat its threads as already
            # addressed on a re-run. Roll it back out of the cache note.
            app.note.drop_solutions_at_indices({solution_idx})
            app.save_note(app.note)
            raise click.ClickException(f"Failed to commit review fix: {e}") from e
        commit_sha = app.repo.head.commit.hexsha
        click.echo(f"Committed review fix {commit_sha[:11]}.")
    else:
        click.echo("Agent made no code changes; recording replies only (no commit).")

    recorded = _record_replies(
        app, number, threads_by_id, addressed, unaddressed, commit_sha
    )
    click.echo(
        f"Recorded {recorded} reply intent(s): "
        f"{len(addressed)} addressed, {len(unaddressed)} unaddressed. "
        f"Run `mergai review post` to publish them."
    )

    if ack:
        total = len(addressed) + len(unaddressed)
        msg = (
            f"mergai review fix: addressed {len(addressed)} of {total} "
            f"review comment(s)."
        )
        if ignored:
            msg = (
                f"mergai review fix: addressed {len(addressed)} of {total} "
                f"review comment(s); ignored {ignored} (not processed)."
            )
        _post_ack(app, number, msg, dry_run)


def _annotate_threads(entries: dict, threads_by_id: dict) -> None:
    """Add each thread's comment id and location to its response entry.

    Mutates ``entries`` (the agent's ``addressed`` / ``unaddressed`` maps) in
    place, keyed by thread id, so the stored solution documents the review
    comment (``comment_id`` is the root review-comment's numeric id, the one in
    its GitHub URL) and its ``path`` / ``line``. Skips keys with no matching
    thread or a non-dict value.
    """
    for tid, info in entries.items():
        thread = threads_by_id.get(tid)
        if thread is None or not isinstance(info, dict):
            continue
        root = thread.root_comment
        info.setdefault("comment_id", root.database_id if root else None)
        info.setdefault("path", thread.path)
        info.setdefault("line", thread.line)


def _record_replies(
    app: AppContext,
    pr_number: int,
    threads_by_id: dict,
    addressed: dict,
    unaddressed: dict,
    commit_sha: str | None,
) -> int:
    """Record one reply intent on the note per addressed / unaddressed thread.

    Each record is self-contained (carries the thread's root comment id, the
    rendered-from fields, and the commit) so ``review post`` can publish it
    without re-fetching the PR's threads. Returns the number recorded.
    """
    now = datetime.now(timezone.utc).isoformat()
    recorded = 0

    def _record(tid: str, outcome: str, **payload) -> None:
        nonlocal recorded
        thread = threads_by_id.get(tid)
        if thread is None:
            return
        root = thread.root_comment
        app.note.add_review_comment(
            {
                "thread_id": tid,
                "comment_id": root.database_id if root else None,
                "pr_number": pr_number,
                "outcome": outcome,
                "path": thread.path,
                "line": thread.line,
                "commit_sha": commit_sha,
                "created_at": now,
                "posted_at": None,
                "posted_comment_url": None,
                **payload,
            }
        )
        recorded += 1

    for tid, info in addressed.items():
        _record(tid, "fixed", note=(info or {}).get("note", ""))
    for tid, info in unaddressed.items():
        _record(tid, "unfixable", reason=(info or {}).get("reason", ""))

    app.save_note(app.note)
    return recorded


def _thread_location(thread: ReviewThread) -> str:
    return f"{thread.path}:{thread.line}" if thread.path else "(general)"


@review.command(name="list")
@click.pass_obj
@click.option(
    "--pr-number",
    "-n",
    type=int,
    default=None,
    help="PR number to inspect (default: auto-detect from the current branch).",
)
@click.option(
    "--actionable-only",
    is_flag=True,
    default=False,
    help="Show only the threads a `review fix` run would act on.",
)
@click.option(
    "--show-comments",
    is_flag=True,
    default=False,
    help="Print each comment's body, not just a per-thread summary.",
)
@click.option(
    "--since",
    "since",
    type=str,
    default=None,
    envvar="MERGAI_REVIEW_SINCE",
    help="ISO-8601 cutoff: classify comments posted after this time as 'after cutoff'.",
)
def list_(
    app: AppContext,
    pr_number: int | None,
    actionable_only: bool,
    show_comments: bool,
    since: str | None,
) -> None:
    """List a PR's review threads and their status (read-only diagnostic).

    Classifies every thread with the same rules ``review fix`` applies -
    actionable, resolved, outdated, or skipped (opted out / already answered
    by the bot) - so you can see exactly what a fix run would act on without
    running the agent or touching the PR.
    """
    config = app.config.review
    number = _resolve_pr_number(app, pr_number)
    threads = fetch_review_threads(app, number)

    if not threads:
        click.echo(f"PR #{number} has no review threads.")
        return

    bot_logins = set(config.bot_logins)
    addressed_ids = _addressed_ids(app)
    trusted_associations = set(config.trusted_associations)
    trusted_logins = set(config.trusted_logins)
    cutoff = _parse_since(since)
    counts: dict[str, int] = {}
    shown = 0
    for thread in threads:
        reason = thread_skip_reason(
            thread,
            bot_logins=bot_logins,
            skip_token=config.skip_token,
            addressed_ids=addressed_ids,
            trusted_associations=trusted_associations,
            trusted_logins=trusted_logins,
            process_external=config.process_external,
            cutoff=cutoff,
        )
        category = skip_reason_category(reason)
        counts[category] = counts.get(category, 0) + 1

        if actionable_only and category != "actionable":
            continue
        shown += 1

        status = "actionable" if reason is None else reason
        last = thread.last_comment
        last_by = f", last by {last.author}" if last else ""
        click.echo(
            f"  [{status}] {_thread_location(thread)} "
            f"[{thread.thread_id}] "
            f"({len(thread.comments)} comment(s){last_by})"
        )
        if show_comments:
            for c in thread.comments:
                body = " ".join(c.body.split())
                if len(body) > 200:
                    body = body[:197] + "..."
                click.echo(f"      {c.author} ({c.created_at}): {body}")

    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    click.echo(f"\nPR #{number}: {len(threads)} thread(s) - {summary}")
    if actionable_only and shown == 0:
        click.echo("No actionable review threads.")


@review.command()
@click.pass_obj
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the replies that would be posted, but don't call GitHub.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-post replies even if they were already posted.",
)
def post(app: AppContext, dry_run: bool, force: bool) -> None:
    """Post the replies recorded by ``review fix`` to their threads.

    For each recorded reply intent, posts a reply on the thread's root comment:
    a "fixed" note (with the commit) on threads the agent addressed, an
    "unfixable" note (with the reason) on the rest. Threads are never
    auto-resolved. No-op when nothing is pending, so it is safe to run
    unconditionally. Records persist in the cache note, so the usual flow is:
    ``review fix`` → review / push → ``review post``.
    """
    config = app.config.review
    records: list[dict] = []
    if app.has_note:
        records = (
            list(app.note.review_comments or [])
            if force
            else app.note.pending_review_comments()
        )

    if not records:
        click.echo("No pending review replies to post.")
        return

    if app.gh is None and not dry_run:
        raise click.ClickException("GitHub auth not available; cannot post replies.")

    now = datetime.now(timezone.utc).isoformat()
    pr_cache: dict[int, object] = {}
    posted_any = False
    posted = 0

    for rec in records:
        tid = rec.get("thread_id")
        if tid is None:
            continue
        pr_number = rec.get("pr_number")
        loc = f"{rec.get('path')}:{rec.get('line')}" if rec.get("path") else "(general)"
        if rec.get("posted_at") is not None and not force:
            click.echo(f"  [{tid}] {loc}: already posted at {rec['posted_at']}; skip.")
            continue

        body = render_reply_from_record(config, rec)
        if dry_run:
            click.echo(f"--- would reply on {loc} [{tid}] (PR #{pr_number}) ---")
            click.echo(body)
            click.echo("--- end ---")
            continue

        if pr_number is None or rec.get("comment_id") is None:
            click.echo(f"  [{tid}] {loc}: missing PR/comment id; cannot post.")
            continue

        if pr_number not in pr_cache:
            pr_cache[pr_number] = app.gh_repo.get_pull(int(pr_number))
        posted_ok, comment_url = post_reply(
            pr_cache[pr_number], rec.get("comment_id"), body
        )
        if posted_ok:
            app.note.mark_review_comment_posted(
                tid, posted_at=now, comment_url=comment_url
            )
            posted += 1
            posted_any = True
            click.echo(f"  [{tid}] {loc}: posted ({rec.get('outcome')}).")

    if posted_any and not dry_run:
        app.save_note(app.note)
    if not dry_run:
        click.echo(f"Posted {posted} {'reply' if posted == 1 else 'replies'}.")


@review.command()
@click.pass_obj
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be dropped without modifying the note.",
)
def sync(app: AppContext, dry_run: bool) -> None:
    """Drop review metadata whose commit is no longer reachable from HEAD.

    After you remove a ``review_fix`` commit (``git reset --hard``, revert, or
    force-push), the cached note still references the gone commit. This prunes:

    \b
      * `review_fix` solutions whose commit is unreachable, and
      * recorded replies (`review_comments`) tied to that commit,

    so a later ``review fix`` / ``review post`` starts clean. Reply records for
    comments that made no commit (``unfixable``) are kept. Replies already
    posted to GitHub are not removed there - this only syncs local metadata.
    """
    if not app.has_note:
        click.echo("No note found; nothing to sync.")
        return

    # Orphaned review_fix solutions (scope to review_fix so we don't touch
    # conflict-resolution / ci_fix solutions - those have their own cleanup
    # via `context drop solution --orphaned`).
    orphaned_idx = [
        i
        for i in app.note.find_orphaned_solution_indices(app.repo)
        if app.note.solutions is not None
        and app.note.solutions[i].get("type") == REVIEW_FIX
    ]
    orphaned_records = app.note.find_orphaned_review_comments(app.repo)

    if not orphaned_idx and not orphaned_records:
        click.echo("Review metadata is in sync; nothing to drop.")
        return

    solutions = app.note.solutions or []
    for i in orphaned_idx:
        sha = (solutions[i].get("commit_sha") or "?")[:11]
        click.echo(f"  review_fix solution[{i}] (commit {sha}) - orphaned")
    for r in orphaned_records:
        loc = f"{r.get('path')}:{r.get('line')}" if r.get("path") else "(general)"
        sha = (r.get("commit_sha") or "?")[:11]
        click.echo(f"  reply [{r.get('thread_id')}] {loc} (commit {sha}) - orphaned")

    if dry_run:
        click.echo(
            f"\nWould drop {len(orphaned_idx)} solution(s) and "
            f"{len(orphaned_records)} reply record(s)."
        )
        return

    app.note.drop_review_comments(orphaned_records)
    if orphaned_idx:
        app.note.drop_solutions_at_indices(set(orphaned_idx))
    app.save_note(app.note)
    click.echo(
        f"Dropped {len(orphaned_idx)} solution(s) and "
        f"{len(orphaned_records)} reply record(s)."
    )
