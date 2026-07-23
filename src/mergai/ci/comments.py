"""Recording and rendering of the PR comments ``mergai ci fix`` produces."""

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal

import click

from ..app import AppContext
from ..config import WorkflowConfig
from ..utils.run_link import append_run_footer
from .context_builders import WorkflowContext

log = logging.getLogger(__name__)


def _render_path_notes(mapping: dict, *, always_colon: bool = False) -> list[str]:
    """Render a ``{path: note}`` mapping as Markdown bullet lines.

    Each entry becomes ``- `path`: note``. When the note is empty the bullet
    is ``- `path``` — unless ``always_colon`` is set, which keeps the trailing
    ``: `` (preserving the unfixable-comment "Unresolved" section's wording).
    """
    out: list[str] = []
    for path, note in mapping.items():
        note_str = note.strip() if isinstance(note, str) else str(note)
        if note_str:
            out.append(f"- `{path}`: {note_str}")
        elif always_colon:
            out.append(f"- `{path}`: ")
        else:
            out.append(f"- `{path}`")
    return out


def _record_ci_comment(
    app: AppContext,
    *,
    outcome: Literal["fixed", "unfixable", "already_resolved"],
    context: WorkflowContext,
    run_id: str,
    attempt_number: int,
    response: dict,
    commit_sha: str | None,
) -> int:
    """Record a postable ci_comment for a fix attempt and save the note.

    Both terminal outcomes of ``ci fix`` that carry agent text record one
    entry here so `mergai ci comment post` can publish an explanation:

    * ``outcome="fixed"`` — the agent changed files; ``commit_sha`` is the
      commit `commit_ci_fix_solution` just created.
    * ``outcome="unfixable"`` — the agent investigated but produced no
      code change; ``commit_sha`` is ``None`` (no commit was made).

    The entry is a self-contained comment payload (it embeds the agent
    ``response`` so the renderer needs no lookup) and lives only in the
    cache note — `ci fix` and the post step share a CI job. Returns the
    index of the appended entry.
    """
    entry = {
        "outcome": outcome,
        "workflow": context.workflow_name,
        "run_id": run_id,
        "pr_number": context.pr_number,
        "attempt_number": attempt_number,
        "context_summary": context.summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "response": response,
        "posted_at": None,
        "posted_comment_url": None,
    }
    idx = app.note.add_ci_comment(entry)
    app.save_note(app.note)
    return idx


def _post_max_attempts_comment(
    app: AppContext, pr_number: int, workflow: str, config: WorkflowConfig
) -> None:
    """Post a PR comment when the per-workflow max-attempts cap is hit."""
    if app.gh is None:
        log.warning("GitHub auth not available; skipping max-attempts PR comment.")
        return
    body = (
        f"mergai gave up auto-fixing the **{workflow}** workflow after "
        f"{config.max_attempts} attempts. Manual intervention required."
    )
    try:
        app.gh_repo.get_pull(pr_number).create_issue_comment(
            append_run_footer(body, app.config.run_link.enabled)
        )
    except Exception as e:  # noqa: BLE001 — best-effort notification
        log.warning("Failed to post PR comment on #%s: %s", pr_number, e)


def _create_pr_comment(
    app: AppContext, pr_number: int, body: str, run_ids: Sequence[str]
) -> Any:
    """Create an issue comment on a PR, wrapping API errors.

    ``run_ids`` are the run ids the comment covers (a summary comment can
    aggregate several); they are named in the error message so a failure points
    at every run in the batch, not just the first.
    """
    try:
        return app.gh_repo.get_pull(int(pr_number)).create_issue_comment(
            append_run_footer(body, app.config.run_link.enabled)
        )
    except Exception as e:  # noqa: BLE001 — wrap external API errors
        runs = ", ".join(run_ids)
        plural = "run" if len(run_ids) == 1 else "runs"
        raise click.ClickException(
            f"Failed to post PR comment for {plural} {runs} on #{pr_number}: {e}"
        ) from e


def _resolve_comments_for_post(
    app: AppContext, target: str, *, include_posted: bool
) -> list[dict]:
    """Return CI comments matching ``target``.

    ``target == "all"`` returns pending comments (or all if
    ``include_posted=True``); a specific run id returns that single
    comment (regardless of posted state — the caller's
    skip-unless-force check applies per entry).
    """
    if not app.has_note or not app.note.ci_comments:
        return []
    if target == "all":
        if include_posted:
            return list(app.note.ci_comments)
        return app.note.pending_ci_comments()
    comment = app.note.get_ci_comment_for_run(target)
    return [comment] if comment is not None else []


def _render_ci_comment(entry: dict) -> str:
    """Format a recorded CI fix attempt as Markdown for a PR comment.

    Branches on ``outcome``: a ``fixed`` entry explains what mergai
    changed; an ``unfixable`` entry explains why it couldn't and what
    needs manual attention. Both render from the agent ``response`` shape
    (``summary`` / ``resolved`` / ``unresolved`` / ``modified`` /
    ``review_notes``).
    """
    outcome = entry.get("outcome", "unfixable")
    workflow = entry.get("workflow", "?")
    run_id = entry.get("run_id", "?")
    attempt = entry.get("attempt_number", "?")
    created_at = entry.get("created_at", "?")
    commit_sha = entry.get("commit_sha")
    response = entry.get("response") or {}
    summary = (response.get("summary") or "").strip()
    review_notes = (response.get("review_notes") or "").strip()

    if outcome == "fixed":
        lines: list[str] = [
            f"### mergai auto-fixed `{workflow}` failure",
            "",
        ]
        if summary:
            lines += [summary, ""]
        changed = {
            **(response.get("resolved") or {}),
            **(response.get("modified") or {}),
        }
        if changed:
            lines += ["**Changed files:**", ""]
            lines += _render_path_notes(changed)
            lines.append("")
        if review_notes:
            lines += ["**Review notes:**", "", review_notes, ""]
        footer = f"_Workflow: `{workflow}` run {run_id} · attempt {attempt}_"
        if commit_sha:
            footer += f"\n_Commit: `{commit_sha[:12]}`_"
        lines.append(footer)
        return "\n".join(lines)

    # outcome == "unfixable"
    unresolved = response.get("unresolved") or {}
    lines = [
        f"### mergai: unable to auto-fix `{workflow}` failure",
        "",
    ]
    if summary:
        lines += [summary, ""]
    if unresolved:
        lines += ["**Unresolved:**", ""]
        lines += _render_path_notes(unresolved, always_colon=True)
        lines.append("")
    if review_notes:
        lines += ["**Review notes:**", "", review_notes, ""]
    lines += [
        f"_Workflow: `{workflow}` run {run_id} · attempt {attempt}_",
        f"_Recorded: {created_at}_",
    ]
    return "\n".join(lines)


def _render_ci_notification(entry: dict) -> str:
    """Render the short PR notification for a CI-fix attempt.

    A terse, one-line notice — which check was fixed in which commit, or that
    it couldn't be fixed. The full per-solution detail lives in the PR body
    (maintained by ``mergai pr update``); this just pings the PR.
    """
    workflow = entry.get("workflow", "?")
    commit_sha = entry.get("commit_sha")
    outcome = entry.get("outcome")
    response = entry.get("response") or {}
    summary = (response.get("summary") or "").strip()
    review_notes = (response.get("review_notes") or "").strip()

    if outcome == "fixed":
        where = f" in commit `{commit_sha[:12]}`" if commit_sha else ""
        return (
            f"The `{workflow}` check was fixed{where}. "
            f"See the PR description for details."
        )

    if outcome == "already_resolved":
        lines = [
            f"No fix needed for the `{workflow}` check — the agent found the "
            "failure already resolved in the current code."
        ]
        if summary:
            lines += ["", summary]
        return "\n".join(lines)

    # unfixable — include the agent's reasoning so reviewers know *why* it
    # could not be fixed, not just that it wasn't.
    lines = [
        f"The `{workflow}` check could not be auto-fixed; it needs manual attention."
    ]
    if summary:
        lines += ["", summary]
    unresolved = response.get("unresolved") or {}
    if unresolved:
        lines += ["", "**Unresolved:**"]
        lines += _render_path_notes(unresolved)
    if review_notes:
        lines += ["", review_notes]
    return "\n".join(lines)


def _render_ci_notification_summary(entries: list[dict]) -> str:
    """Render one PR comment covering every CI-fix attempt in ``entries``.

    A single notification for all checks: each check's status, rendered by
    the existing per-check ``_render_ci_notification`` (reused verbatim so the
    wording stays consistent), one after another. Replaces the old "one comment
    per run", which on a multi-check ``ci fix all`` produced several separate
    PR comments. A single entry renders as just that entry's status.
    """
    return "\n\n".join(_render_ci_notification(entry) for entry in entries) + "\n"
