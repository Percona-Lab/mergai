"""``mergai ci`` click command group.

Entry point invoked by the ``ci-fix`` job in ``.github/workflows/mergai.yml``
on every ``workflow_run.completed`` for a watched CI workflow on a
``mergai/*`` branch. Given a run ID (or ``"all"`` / a workflow name for
manual catch-up), this command:

* Fetches the workflow run (resolves workflow name, head SHA, PR number,
  conclusion) so callers don't have to extract them from the event payload.
* For ``conclusion == "failure"``: downloads the run's artifacts to a
  temp directory and dispatches to the configured context builder
  (which falls back to job logs when the SARIF is missing).
* For ``conclusion == "success"`` and a workflow whose context has
  ``code_scanning_check: true``: queries Code Scanning for findings on
  the run's commit. If any exist, dispatches to the SARIF builder via
  the Code Scanning path; otherwise exits no-op.
* Otherwise exits no-op.

The handler edits the working tree; this command commits the result as
a ``type: ci_fix`` solution on the note. The outer workflow pushes the
new commit so CI re-runs.

Every attempt that yields a usable verdict also records one entry in
``note.ci_comments`` — ``outcome="fixed"`` (with the commit) or
``outcome="unfixable"`` (no commit) — so ``mergai ci comment post`` can
publish an explanation to the PR. Those records live only in the cache
note: ``ci fix`` and the post step share a CI job.

``mergai ci fix <run-id>`` is what GitHub Actions calls. Manually
you can also pass ``"all"`` or a workflow name to process every
unprocessed actionable run on the current branch.
"""

from datetime import datetime, timezone
from typing import Literal

import click
import github

from ..app import AppContext
from ..ci.comments import (
    _create_pr_comment,
    _post_max_attempts_comment,
    _record_ci_comment,
    _render_ci_comment,
    _render_ci_notification_summary,
    _resolve_comments_for_post,
)
from ..ci.dispatch import (
    _resolve_target_runs,
    _run_head_status,
    _take_workflow_runs,
    build_workflow_context_for_run,
)
from ..ci.gate import _aggregate_state, _list_run_status, _watched_runs_for_head
from ..ci.handlers import get_handler
from ..solution_types import CI_FIX
from ..utils.formatters import format_ascii_table
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
def ci(app: AppContext, repo: str | None):
    """CI workflow integration commands."""
    ensure_gh_repo(app, repo)


@ci.command()
@click.pass_obj
@click.argument("target", required=True)
@click.option(
    "--workflow",
    required=False,
    help="Override workflow name (default: from the run).",
)
@click.option(
    "--pr",
    required=False,
    type=int,
    help="Override PR number (default: first PR associated with the run).",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help=(
        "Pre-downloaded artifacts directory (default: download to a temp "
        "directory). Each artifact must already be extracted into a "
        "subdirectory named after it. Useful for manual / offline runs."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Act on runs whose head_sha has been superseded by newer commits "
        "on the branch (the same runs ``ci list`` shows as 'skip — "
        "superseded'). Use when the intervening commits are known to be "
        "unrelated to the failure. Runs whose head_sha is no longer "
        "reachable from HEAD at all (force-pushed away) are still "
        "skipped — those describe code that doesn't exist on this branch."
    ),
)
@click.option(
    "--ack",
    is_flag=True,
    default=False,
    help=(
        "Post a short acknowledgement comment on the PR summarising the "
        "outcome (how many failing checks were found / fixed), even when "
        "there are none. Use from CI to give quick feedback on a "
        "comment-triggered run. Distinct from the detailed per-check summary "
        "produced by `mergai ci comment post`."
    ),
)
def fix(
    app: AppContext,
    target: str,
    workflow: str | None,
    pr: int | None,
    artifacts_dir: str | None,
    force: bool,
    ack: bool,
) -> None:
    """Apply a fix for one or more workflow runs on the current branch.

    \b
    TARGET can be:
      * a numeric run ID — process that specific run
      * "all"            — process every unprocessed actionable run on
                           the current branch (newest first), respecting
                           per-workflow `max_attempts`
      * a workflow name  — like "all" but filtered to that workflow

    For each selected run, builds a `WorkflowContext` via
    `build_workflow_context_for_run`, enforces `max_attempts`,
    dispatches to the configured handler, and commits the result as a
    `type: ci_fix` solution.
    """
    # `check_staleness` only applies at the entry — once a run has been
    # vetted (either by the user passing its run-id explicitly or by
    # `_resolve_target_runs` filtering the listing) the per-iteration
    # head_sha check just gets in our own way: applying a fix for run #1
    # commits something on top of HEAD, which would then disqualify
    # run #2 even though both runs were valid against the same commit at
    # the start of the loop.
    if target.isdigit():
        run_ids = [target]
        check_staleness = not force
    else:
        run_ids = _resolve_target_runs(app, target, force=force)
        check_staleness = False

    # Always report the actionable count, including zero, so a
    # comment-triggered run (`--ack`) gives feedback even when there is
    # nothing to do.
    click.echo(
        f"{len(run_ids)} unprocessed actionable run(s) to address for "
        f"target '{target}'."
    )

    fixed = 0
    for run_id in run_ids:
        outcome = _fix_one_run(
            app,
            run_id,
            workflow_override=workflow,
            pr_override=pr,
            artifacts_dir_override=artifacts_dir,
            check_staleness=check_staleness,
        )
        if outcome == "fixed":
            fixed += 1

    if ack:
        _post_ci_ack(app, pr_override=pr, found=len(run_ids), fixed=fixed)


def _post_ci_ack(
    app: AppContext, *, pr_override: int | None, found: int, fixed: int
) -> None:
    """Post a one-line acknowledgement of a `ci fix` trigger (best-effort).

    Distinct from `mergai ci comment post`: that posts the detailed per-check
    summary; this is a terse acknowledgement of the trigger - how many failing
    checks were found and fixed - so a comment-triggered run gives feedback
    even when there was nothing to do. The PR is `--pr` when given, else the
    single open PR for the current branch.
    """
    if found == 0:
        message = "mergai ci fix: no failing checks to address."
    else:
        message = f"mergai ci fix: fixed {fixed} of {found} failing check(s)."

    if pr_override is not None:
        pr_number = pr_override
    else:
        prs = get_prs_for_current_branch(app)
        if len(prs) != 1:
            click.echo(
                "Skipping --ack comment: could not resolve a single PR for the "
                "current branch (pass --pr).",
                err=True,
            )
            return
        pr_number = prs[0].number

    try:
        app.gh_repo.get_pull(int(pr_number)).create_issue_comment(message)
        click.echo(f"Posted acknowledgement on PR #{pr_number}.")
    except Exception as e:  # noqa: BLE001 - acknowledgement is best-effort
        click.echo(f"warning: could not post acknowledgement: {e}", err=True)


def _fix_one_run(
    app: AppContext,
    run_id: str,
    *,
    workflow_override: str | None,
    pr_override: int | None,
    artifacts_dir_override: str | None,
    check_staleness: bool = True,
) -> Literal["fixed", "already_resolved", "unfixable", "skip"]:
    """Apply a fix for a single workflow run.

    Body of the per-run flow, hoisted out so that the top-level ``fix``
    command can iterate over multiple runs for the ``all`` /
    workflow-name targets.

    Returns the per-run outcome (``"fixed"`` only when a fix was committed)
    so the caller can tally how many checks were fixed for the ``--ack``
    summary.

    Args:
        check_staleness: If True (default; for explicit run-id targets),
            reject the run when its head_sha isn't the current branch
            HEAD. ``fix all`` / ``fix <workflow>`` pass False because
            ``_resolve_target_runs`` has already vetted the runs at the
            top of the loop, and applying a fix for run #1 will move
            HEAD past run #2 even though both were originally valid.
    """
    if check_staleness:
        try:
            run = app.gh_repo.get_workflow_run(int(run_id))
        except github.GithubException as e:
            raise click.ClickException(
                f"Could not fetch workflow run {run_id}: {e}"
            ) from e
        head_status = _run_head_status(app, run)
        if head_status != "current":
            reason = (
                "superseded by newer commits"
                if head_status == "superseded"
                else "head_sha not reachable from HEAD (force-pushed?)"
            )
            click.echo(
                f"Run {run_id} ({run.head_sha[:7]}) is {head_status}: {reason}; "
                f"skipping."
            )
            return "skip"

    with build_workflow_context_for_run(
        app,
        run_id,
        workflow_override=workflow_override,
        pr_override=pr_override,
        artifacts_dir_override=artifacts_dir_override,
    ) as built:
        if built is None:
            return "skip"
        context, config = built

        existing = app.note.get_ci_solution_for_run(run_id)
        if existing is not None:
            click.echo(
                f"Run {run_id} was already processed "
                f"(solution recorded as attempt "
                f"{existing.get('request', {}).get('attempt_number', '?')}); "
                f"nothing to do."
            )
            return "skip"

        # A run judged unfixable / already-resolved leaves a ci_comment but no
        # solution. `_resolve_target_runs` filters those out for `fix all`, but
        # an explicit run id (the GitHub Actions trigger) reaches here directly,
        # so guard against re-invoking the agent for an already-decided run.
        existing_comment = app.note.get_ci_comment_for_run(run_id)
        if existing_comment is not None:
            click.echo(
                f"Run {run_id} was already processed "
                f"(recorded as {existing_comment.get('outcome', '?')!r}); "
                f"nothing to do."
            )
            return "skip"

        prior_solutions = app.note.get_ci_solutions(context.workflow_name)
        attempt_number = len(prior_solutions) + 1
        if attempt_number > config.max_attempts:
            click.echo(
                f"Max attempts ({config.max_attempts}) reached for "
                f"'{context.workflow_name}'; giving up."
            )
            _post_max_attempts_comment(
                app, context.pr_number, context.workflow_name, config
            )
            return "skip"

        click.echo(
            f"Handling '{context.workflow_name}' run {run_id} "
            f"(attempt {attempt_number} of {config.max_attempts}). "
            f"Context: {context.summary}"
        )

        handler = get_handler(app, config)
        agent_solution = handler.execute(context)

        if agent_solution is None:
            click.echo(
                f"Handler did not produce a solution for "
                f"'{context.workflow_name}'. Will retry on the next workflow run."
            )
            return "skip"

        # Agent investigated but produced no code change (empty resolved +
        # modified). The agent itself classifies why via response.status:
        #   * "already_resolved" -> the failure no longer applies to the
        #     current code (e.g. an earlier fix in this run addressed the same
        #     root cause). Not a problem; recorded calmly.
        #   * anything else ("unfixable" / missing) -> needs manual attention.
        # Whether mergai knows it was addressed is the agent's call, not a
        # heuristic. Either way record a ci_comment so the run isn't
        # re-investigated, but don't commit or burn a `max_attempts` slot.
        response = agent_solution.get("response", {}) or {}
        if not response.get("resolved") and not response.get("modified"):
            outcome: Literal["unfixable", "already_resolved"] = (
                "already_resolved"
                if response.get("status") == "already_resolved"
                else "unfixable"
            )
            comment_idx = _record_ci_comment(
                app,
                outcome=outcome,
                context=context,
                run_id=run_id,
                attempt_number=attempt_number,
                response=response,
                commit_sha=None,
            )
            click.echo(
                f"Agent produced no code change for '{context.workflow_name}' "
                f"run {run_id} (outcome={outcome}). "
                f"Recorded comment (ci_comments[{comment_idx}]); "
                f"no commit, no attempt slot consumed. "
                f"Run `mergai ci comment post {run_id}` to publish a PR comment."
            )
            return outcome

        ci_solution = {
            "type": CI_FIX,
            "request": {
                "workflow": context.workflow_name,
                "run_id": run_id,
                "pr_number": context.pr_number,
                "attempt_number": attempt_number,
                "context_summary": context.summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            **agent_solution,
        }
        solution_idx = app.note.add_solution(ci_solution)
        app.save_note(app.note)

        try:
            app.commit_ci_fix_solution(solution_idx)
        except Exception as e:
            # The commit failed, so the solution we just persisted is a phantom:
            # get_ci_solution_for_run() would treat the run as processed and
            # block a retry. Roll it back out of the cache note.
            app.note.drop_solutions_at_indices({solution_idx})
            app.save_note(app.note)
            raise click.ClickException(f"Failed to commit CI fix: {e}") from e

        # Record the fix as a postable ci_comment too, so every attempt —
        # success or not — leaves an explanation for `ci comment post`.
        # The commit exists now; anchor the comment to it for the footer.
        comment_idx = _record_ci_comment(
            app,
            outcome="fixed",
            context=context,
            run_id=run_id,
            attempt_number=attempt_number,
            response=response,
            commit_sha=app.repo.head.commit.hexsha,
        )
        click.echo(
            f"Applied fix for '{context.workflow_name}' run {run_id} "
            f"(attempt {attempt_number} of {config.max_attempts}). "
            f"Recorded comment (ci_comments[{comment_idx}]); "
            f"run `mergai ci comment post {run_id}` to publish a PR comment."
        )
        return "fixed"


@ci.command(name="list")
@click.pass_obj
@click.option(
    "--branch",
    "-b",
    "branch_override",
    required=False,
    help="Branch to list runs for (default: current branch).",
)
@click.option(
    "--limit",
    "-n",
    default=20,
    type=int,
    show_default=True,
    help="Maximum number of recent runs to show.",
)
@click.option(
    "--check-findings/--no-check-findings",
    default=True,
    show_default=True,
    help=(
        "For passing runs whose workflow has 'code_scanning_check', query "
        "Code Scanning to determine the actual finding count. Disable to "
        "skip the extra API calls."
    ),
)
def list_runs(
    app: AppContext,
    branch_override: str | None,
    limit: int,
    check_findings: bool,
) -> None:
    """List recent workflow runs and their mergai status.

    For each configured workflow run on the branch, shows whether
    mergai has already applied a fix (matching ``ci_fix`` solution) or,
    if not, what action mergai would take if the run was handled now.
    """
    repo = app.gh_repo

    branch = branch_override
    if branch is None:
        try:
            branch = app.repo.active_branch.name
        except TypeError as e:
            raise click.ClickException(
                "HEAD is detached; pass --branch explicitly."
            ) from e

    # PyGithub's type stub asks for a Branch object, but the API
    # accepts a branch name string. Pass the string directly.
    runs = repo.get_workflow_runs(branch=branch)  # type: ignore[arg-type]

    rows: list[tuple[str, ...]] = []
    runs_list = _take_workflow_runs(runs, limit)
    for run in runs_list:
        if run.name not in app.config.workflows.workflows:
            continue
        # Drop runs whose head commit isn't reachable from HEAD — they
        # belong to a prior incarnation of this branch (force-pushed
        # away). They're still in GitHub's history for the branch name
        # but they're noise here.
        if _run_head_status(app, run) == "obsolete":
            continue
        status, notes = _list_run_status(app, run, check_findings=check_findings)
        rows.append(
            (
                str(run.id),
                run.name,
                run.conclusion or run.status or "-",
                (run.head_sha or "")[:8],
                status,
                notes,
            )
        )

    if not rows:
        click.echo(
            f"No configured workflow runs found for branch '{branch}' "
            f"(showing first {limit})."
        )
        return

    headers = ("Run ID", "Workflow", "Conclusion", "Head SHA", "Status", "Notes")
    click.echo(format_ascii_table(headers, rows))


@ci.command(name="status")
@click.pass_obj
@click.option(
    "--state",
    "state_only",
    is_flag=True,
    default=False,
    help=(
        "Print only the aggregate state token "
        "(in-progress|success|failure|none) on stdout and nothing else, so "
        "a workflow can do STATE=$(mergai ci status --state)."
    ),
)
def status(app: AppContext, state_only: bool) -> None:
    """Aggregate the watched workflows' state for the current branch HEAD.

    Looks at the latest run of each watched workflow (``format``,
    ``clang-tidy``, ``build-and-test`` from ``.mergai/config.yml``) whose
    ``head_sha`` matches HEAD and reduces them to one token:

    \b
    * in-progress — at least one watched run for HEAD hasn't completed.
    * success     — every watched run for HEAD completed successfully.
    * failure     — all completed, but at least one didn't succeed.
    * none        — no watched runs for HEAD (e.g. all skipped).

    Exits 0 in every case; the state is communicated on stdout. With
    ``--state`` only the bare token is printed (for shell capture).
    """
    runs_by_workflow = _watched_runs_for_head(app)
    state = _aggregate_state(runs_by_workflow)

    if state_only:
        click.echo(state)
        return

    if runs_by_workflow:
        headers = ("Workflow", "Run ID", "Status", "Conclusion")
        rows: list[tuple[str, ...]] = [
            (
                name,
                str(run.id),
                run.status or "-",
                run.conclusion or "-",
            )
            for name, run in sorted(runs_by_workflow.items())
        ]
        click.echo(format_ascii_table(headers, rows))
    else:
        click.echo("No watched workflow runs for the current HEAD.")
    click.echo(f"\nState: {state}")


# ---------------------------------------------------------------------------
# `mergai ci comment` — view and post the explanation `ci fix` records for
# every attempt (a fix it applied, or a failure it couldn't fix).
# ---------------------------------------------------------------------------


@ci.group(name="comment")
def comment() -> None:
    """View and publish the PR comment `ci fix` records for each attempt.

    \b
    Every time `mergai ci fix` invokes the agent and gets a usable
    verdict, it records one comment on the local note:
      * outcome `fixed`     — what mergai changed (a commit was made).
      * outcome `unfixable` — why it couldn't fix the failure (no commit,
                              no `max_attempts` slot consumed).
    `mergai ci comment list` shows recorded comments;
    `mergai ci comment post` publishes one (or all pending) to the PR.
    """


@comment.command(name="list")
@click.pass_obj
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Also print the rendered comment body for each entry.",
)
@click.option(
    "--pending",
    "pending_only",
    is_flag=True,
    default=False,
    help="Show only comments that haven't been posted yet.",
)
def comment_list(app: AppContext, verbose: bool, pending_only: bool) -> None:
    """List recorded CI comments.

    By default lists both pending and posted entries. Use ``--pending``
    to see only what ``mergai ci comment post`` would publish.
    """
    if not app.has_note or not app.note.ci_comments:
        click.echo("No CI comments recorded.")
        return

    entries = app.note.pending_ci_comments() if pending_only else app.note.ci_comments
    if not entries:
        click.echo("No pending CI comments.")
        return

    rows: list[tuple[str, ...]] = []
    for c in entries:
        rows.append(
            (
                str(c.get("run_id", "?")),
                str(c.get("workflow", "?")),
                str(c.get("outcome", "?")),
                str(c.get("pr_number", "?")),
                str(c.get("created_at", "?")),
                str(c.get("posted_at") or "pending"),
            )
        )

    headers = ("Run ID", "Workflow", "Outcome", "PR", "Recorded at", "Posted at")
    click.echo(format_ascii_table(headers, rows))

    if verbose:
        for c in entries:
            click.echo("")
            click.echo(f"--- run {c.get('run_id')} ({c.get('workflow')}) ---")
            click.echo(_render_ci_comment(c))


@comment.command(name="post")
@click.pass_obj
@click.argument("target", required=False, default="all")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the comment that would be posted, but don't call GitHub.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-post even if the comment was already posted.",
)
@click.option(
    "--review-pr",
    type=int,
    default=None,
    help=(
        "Post the notification on this PR instead of the failing run's own "
        "PR. Use when the fix was relocated to a review PR (e.g. a main-branch "
        "failure whose fix lives on a semantic PR)."
    ),
)
def comment_post(
    app: AppContext,
    target: str,
    dry_run: bool,
    force: bool,
    review_pr: int | None,
) -> None:
    """Post a short CI-fix notification to the PR.

    For each recorded attempt, posts a one-line notice: which check was fixed
    in which commit (outcome ``fixed``), or that mergai could not fix it
    (``unfixable``). The full per-solution detail belongs in the PR body,
    maintained by ``mergai pr update``; this command only pings the PR so
    reviewers notice the change (body edits are silent on GitHub).

    \b
    TARGET (optional, default "all"):
      * "all" / omitted   — post every pending notification. No-op when
                            nothing is pending, so this is safe to run
                            unconditionally from CI / a mergai workflow.
      * a numeric run ID  — post the notification for that run.

    ``--review-pr`` redirects the notice to a review PR when the fix was
    relocated off the failing run's own PR.
    """
    comments = _resolve_comments_for_post(app, target, include_posted=force)
    if not comments:
        if target == "all":
            click.echo("No pending CI comments to post.")
        else:
            click.echo(f"No comment found for run '{target}'.")
        return

    if app.gh is None and not dry_run:
        raise click.ClickException("GitHub auth not available; cannot post PR comment.")

    now = datetime.now(timezone.utc).isoformat()

    # Aggregate into one comment per target PR rather than one per run. A
    # single `ci fix all` over multiple checks used to post a separate PR
    # comment for each; group the postable entries and post one summary per
    # PR instead. Grouping by target PR keeps it correct even in the unusual
    # case where entries resolve to different PRs (normally they share one).
    groups: dict[int, list[dict]] = {}
    for c in comments:
        run_id = str(c.get("run_id"))
        if c.get("posted_at") is not None and not force:
            click.echo(
                f"Comment for run {run_id} was already posted at "
                f"{c['posted_at']}; skipping (use --force to re-post)."
            )
            continue

        pr_number = c.get("pr_number")
        target_pr = int(review_pr) if review_pr is not None else pr_number
        if target_pr is None:
            click.echo(f"Run {run_id}: no PR number recorded; cannot post.")
            continue

        groups.setdefault(int(target_pr), []).append(c)

    posted_any = False
    for target_pr, entries in groups.items():
        run_ids = [str(e.get("run_id")) for e in entries]
        body = _render_ci_notification_summary(entries)
        if dry_run:
            click.echo(
                f"--- would post summary for {len(entries)} check(s) "
                f"(runs {', '.join(run_ids)}) on #{target_pr} ---"
            )
            click.echo(body)
            click.echo("--- end ---")
            continue

        posted = _create_pr_comment(app, target_pr, body, run_ids)
        comment_url = getattr(posted, "html_url", None)
        for run_id in run_ids:
            app.note.mark_ci_comment_posted(
                run_id, posted_at=now, comment_url=comment_url
            )
        posted_any = True
        click.echo(f"Posted CI summary for {len(entries)} check(s) on #{target_pr}")

    if posted_any and not dry_run:
        app.save_note(app.note)
