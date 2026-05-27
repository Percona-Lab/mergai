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

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import click
import git
import github

from ..app import AppContext
from ..ci.context_builders import WorkflowContext, get_context_builder
from ..ci.context_builders.sarif import SARIFContextBuilder
from ..ci.handlers import get_handler
from ..config import WorkflowConfig
from ..utils.artifact_downloader import download_workflow_run_artifacts

log = logging.getLogger(__name__)


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
    if repo is None:
        raise click.ClickException(
            "GitHub repository not set. Use --repo or set GH_REPO environment variable."
        )
    app.gh_repo_str = repo


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
def fix(
    app: AppContext,
    target: str,
    workflow: str | None,
    pr: int | None,
    artifacts_dir: str | None,
    force: bool,
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
        if not run_ids:
            click.echo(f"No unprocessed actionable runs found for target '{target}'.")
            return
        click.echo(
            f"Found {len(run_ids)} unprocessed actionable run(s) for target '{target}'."
        )
        check_staleness = False

    for run_id in run_ids:
        _fix_one_run(
            app,
            run_id,
            workflow_override=workflow,
            pr_override=pr,
            artifacts_dir_override=artifacts_dir,
            check_staleness=check_staleness,
        )


def _resolve_target_runs(
    app: AppContext, target: str, *, force: bool = False
) -> list[str]:
    """Resolve ``"all"`` / workflow-name to a list of run IDs to process.

    Lists recent workflow_runs on the current branch, filters out runs
    that are not configured / not actionable / already processed, and
    returns IDs in newest-first order.

    A workflow-name target is just the ``"all"`` filter narrowed to one
    workflow.

    Args:
        force: When True, do not filter out runs whose head_sha is not
            current HEAD. Other actionability filters (workflow enabled,
            mergai/* branch, conclusion type) still apply.
    """
    workflow_filter = None if target == "all" else target
    if (
        workflow_filter is not None
        and workflow_filter not in app.config.workflows.workflows
    ):
        raise click.ClickException(
            f"Unknown workflow '{workflow_filter}'. Configured workflows: "
            + ", ".join(sorted(app.config.workflows.workflows))
            + "."
        )

    try:
        branch = app.repo.active_branch.name
    except TypeError as e:
        raise click.ClickException(
            "HEAD is detached; pass an explicit run ID instead of "
            "'all' / workflow name."
        ) from e

    runs = app.gh_repo.get_workflow_runs(branch=branch)  # type: ignore[arg-type]
    runs_list: list[github.WorkflowRun.WorkflowRun] = list(runs[:50])

    selected: list[str] = []
    for run in runs_list:
        if workflow_filter is not None and run.name != workflow_filter:
            continue
        if run.name not in app.config.workflows.workflows:
            continue
        run_id = str(run.id)
        if app.has_note and app.note.get_ci_solution_for_run(run_id) is not None:
            continue
        if app.has_note and app.note.get_ci_comment_for_run(run_id) is not None:
            continue
        if not _run_is_actionable(app, run, force=force):
            continue
        selected.append(run_id)

    return selected


def _run_is_actionable(
    app: AppContext,
    run: "github.WorkflowRun.WorkflowRun",
    *,
    force: bool = False,
) -> bool:
    """Mirror of the dispatch decision in ``build_workflow_context_for_run``.

    Returns True if mergai would build a context for this run if asked
    to handle it now. Used by ``mergai ci fix all`` /
    ``mergai ci fix <workflow>`` to filter the run list before
    iterating.

    Skips runs whose head commit isn't the current branch HEAD —
    findings on a superseded or force-pushed commit don't necessarily
    apply to the current state, and committing a fix on the wrong base
    is worse than no fix.

    Args:
        force: When True, accept ``superseded`` runs (newer commits on
            the branch since the run, but ``head_sha`` still reachable
            from HEAD). ``obsolete`` runs — whose ``head_sha`` isn't
            reachable from HEAD at all — are still skipped: their work
            was force-pushed away and acting on their findings means
            patching code that no longer exists on this branch. This
            keeps ``ci fix --force`` aligned with ``ci list``, which
            also hides obsolete runs.
    """
    head_status = _run_head_status(app, run)
    if head_status == "obsolete":
        return False
    if not force and head_status != "current":
        return False
    config = app.config.workflows.get(run.name)
    if config is None or not config.enabled:
        return False
    if not (run.head_branch or "").startswith("mergai/"):
        return False
    if run.conclusion == "failure":
        return True
    if run.conclusion == "success" and config.context.code_scanning_check:
        pr_number = _resolve_pr_number(run)
        if pr_number is None:
            return False
        return _code_scanning_has_findings(
            app,
            workflow_name=run.name,
            head_sha=run.head_sha,
            pr_number=pr_number,
        )
    return False


def _failure_note(config: WorkflowConfig) -> str:
    """Describe how `ci fix` would source context for a failed run.

    The note text is shown by ``ci list`` so the user can tell at a
    glance which path mergai would take. The phrasing depends on the
    configured context builder — only the SARIF builder has a log
    fallback, the diff builder doesn't.
    """
    ctx_type = config.context.type
    if ctx_type == "sarif":
        return "failure -> SARIF artifact (log fallback if missing)"
    if ctx_type == "diff":
        return "failure -> diff artifact"
    return f"failure -> {ctx_type} artifact"


def _run_head_status(
    app: AppContext, run: "github.WorkflowRun.WorkflowRun"
) -> Literal["current", "superseded", "obsolete"]:
    """Classify a workflow run's head commit relative to the local branch.

    * ``current``    — ``run.head_sha`` equals the working tree HEAD.
                       Findings still describe the same source the user
                       has checked out, so the run is safe to act on.
    * ``superseded`` — ``run.head_sha`` is an ancestor of HEAD but not
                       equal. Newer commits exist on the branch since
                       the run; its findings may be stale.
    * ``obsolete``   — ``run.head_sha`` isn't reachable from HEAD at
                       all. Typically means the branch was force-pushed
                       (or the SHA was never fetched locally).
    """
    head_sha = app.repo.head.commit.hexsha
    if run.head_sha == head_sha:
        return "current"
    try:
        app.repo.git.merge_base("--is-ancestor", run.head_sha, "HEAD")
        return "superseded"
    except git.GitCommandError:
        # status 1 → not an ancestor; anything else (e.g. unknown SHA) →
        # also obsolete from our perspective. We can't act on a SHA we
        # don't have or can't reach.
        return "obsolete"


def _fix_one_run(
    app: AppContext,
    run_id: str,
    *,
    workflow_override: str | None,
    pr_override: int | None,
    artifacts_dir_override: str | None,
    check_staleness: bool = True,
) -> None:
    """Apply a fix for a single workflow run.

    Body of the per-run flow, hoisted out so that the top-level ``fix``
    command can iterate over multiple runs for the ``all`` /
    workflow-name targets.

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
            return

    with build_workflow_context_for_run(
        app,
        run_id,
        workflow_override=workflow_override,
        pr_override=pr_override,
        artifacts_dir_override=artifacts_dir_override,
    ) as built:
        if built is None:
            return
        context, config = built

        existing = app.note.get_ci_solution_for_run(run_id)
        if existing is not None:
            click.echo(
                f"Run {run_id} was already processed "
                f"(solution recorded as attempt "
                f"{existing.get('request', {}).get('attempt_number', '?')}); "
                f"nothing to do."
            )
            return

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
            return

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
            return

        # "Unable to fix" verdict: agent investigated but produced no
        # code change (empty resolved + modified). Record a ci_comment
        # with outcome="unfixable" so the same run isn't re-investigated
        # next time, but don't commit and don't burn a `max_attempts`
        # slot — the comment is published via `mergai ci comment post`.
        response = agent_solution.get("response", {}) or {}
        if not response.get("resolved") and not response.get("modified"):
            comment_idx = _record_ci_comment(
                app,
                outcome="unfixable",
                context=context,
                run_id=run_id,
                attempt_number=attempt_number,
                response=response,
                commit_sha=None,
            )
            click.echo(
                f"Agent produced no code change for '{context.workflow_name}' "
                f"run {run_id}. Recorded comment (ci_comments[{comment_idx}]); "
                f"no commit, no attempt slot consumed. "
                f"Run `mergai ci comment post {run_id}` to publish a PR comment."
            )
            return

        ci_solution = {
            "type": "ci_fix",
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


def _record_ci_comment(
    app: AppContext,
    *,
    outcome: Literal["fixed", "unfixable"],
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


@contextmanager
def build_workflow_context_for_run(
    app: AppContext,
    run_id: str,
    *,
    workflow_override: str | None = None,
    pr_override: int | None = None,
    artifacts_dir_override: str | None = None,
) -> Iterator[tuple[WorkflowContext, WorkflowConfig] | None]:
    """Resolve a workflow run and build its :class:`WorkflowContext`.

    Yields ``(context, config)`` when the run is actionable, or ``None``
    after echoing a human-readable skip reason. Cleans up any temporary
    artifact directory on exit.

    Shared by ``mergai ci fix`` (which then runs the handler) and
    ``mergai prompt ci`` (which renders the prompt without invoking the
    agent), so both consume the same dispatch decision and the same
    context shape.
    """
    repo = app.gh_repo
    try:
        run = repo.get_workflow_run(int(run_id))
    except github.GithubException as e:
        raise click.ClickException(f"Could not fetch workflow run {run_id}: {e}") from e

    workflow_name = workflow_override or run.name
    pr_number = pr_override if pr_override is not None else _resolve_pr_number(run)
    head_sha = run.head_sha
    head_branch = run.head_branch
    conclusion = run.conclusion  # "success" | "failure" | "cancelled" | ...

    if pr_number is None:
        click.echo("No PR associated with this workflow run; nothing to do.")
        yield None
        return
    if not (head_branch or "").startswith("mergai/"):
        click.echo(
            f"Head branch '{head_branch}' is not a mergai/* branch; nothing to do."
        )
        yield None
        return

    config = app.config.workflows.get(workflow_name)
    if config is None:
        click.echo(f"No configuration for workflow '{workflow_name}'; nothing to do.")
        yield None
        return
    if not config.enabled:
        click.echo(f"Workflow '{workflow_name}' handling is disabled; nothing to do.")
        yield None
        return

    builder_artifacts_dir: str | None = None
    builder_head_sha: str | None = None
    tmp_dir: tempfile.TemporaryDirectory | None = None

    try:
        if conclusion == "failure":
            if artifacts_dir_override:
                builder_artifacts_dir = artifacts_dir_override
            else:
                tmp_dir = tempfile.TemporaryDirectory(prefix="mergai-ci-")
                download_workflow_run_artifacts(run, Path(tmp_dir.name))
                builder_artifacts_dir = tmp_dir.name
        elif conclusion == "success" and config.context.code_scanning_check:
            if not _code_scanning_has_findings(
                app,
                workflow_name=workflow_name,
                head_sha=head_sha,
                pr_number=pr_number,
            ):
                click.echo(
                    f"Workflow '{workflow_name}' passed and Code Scanning has "
                    f"no findings for {head_sha[:7]}; nothing to do."
                )
                yield None
                return
            builder_head_sha = head_sha
        else:
            click.echo(
                f"Run conclusion '{conclusion}' is not actionable for "
                f"'{workflow_name}'; nothing to do."
            )
            yield None
            return

        builder = get_context_builder(app, config.context.type)
        context = builder.build_context(
            config.context,
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            artifacts_dir=builder_artifacts_dir,
            head_sha=builder_head_sha,
        )
        yield context, config
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _resolve_pr_number(run: "github.WorkflowRun.WorkflowRun") -> int | None:
    """Return the first PR number associated with a workflow run, or None."""
    prs = run.pull_requests or []
    if not prs:
        return None
    return prs[0].number


def _code_scanning_has_findings(
    app: AppContext, *, workflow_name: str, head_sha: str, pr_number: int
) -> bool:
    """Return True if Code Scanning has results for this commit + tool.

    Convention: the Code Scanning tool name matches the workflow name
    (true today for ``clang-tidy``). If the tool ever differs, add a
    ``context.code_scanning_tool_name`` config field.
    """
    builder = get_context_builder(app, "sarif")
    if not isinstance(builder, SARIFContextBuilder):  # defensive
        return False
    analysis = builder.find_code_scanning_analysis(
        tool_name=workflow_name, head_sha=head_sha, pr_number=pr_number
    )
    return analysis is not None and analysis.get("results_count", 0) > 0


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
        app.gh_repo.get_pull(pr_number).create_issue_comment(body)
    except Exception as e:  # noqa: BLE001 — best-effort notification
        log.warning("Failed to post PR comment on #%s: %s", pr_number, e)


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
    runs_list: list[github.WorkflowRun.WorkflowRun] = list(runs[:limit])
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
    click.echo(_format_ascii_table(headers, rows))


def _format_ascii_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a plain-ASCII table with `|` columns and `-` rules.

    No Unicode box-drawing characters — works in any terminal /
    log-aggregator without font surprises. Columns are padded to the
    widest cell; the last column is left as-is so long notes don't
    force wide gutters on everything else.
    """
    cols = list(zip(headers, *rows, strict=False))
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def render_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            str(cell).ljust(width) for cell, width in zip(row, widths, strict=False)
        ).rstrip()

    rule = "  ".join("-" * w for w in widths)
    lines = [render_row(headers), rule]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _list_run_status(
    app: AppContext,
    run: "github.WorkflowRun.WorkflowRun",
    *,
    check_findings: bool,
) -> tuple[str, str]:
    """Return ``(status, notes)`` describing what mergai sees for this run.

    Status is one of:
    - ``applied``: a ``type: ci_fix`` solution exists with a matching ``run_id``.
    - ``pending``: not processed and mergai *would* act (failure, or
      success with code_scanning_check + findings present).
    - ``skip``: not processed and mergai would not act (passing run with
      no opt-in, conclusion not actionable, workflow disabled, etc.).
    """
    run_id = str(run.id)
    comment = app.note.get_ci_comment_for_run(run_id) if app.has_note else None
    existing = app.note.get_ci_solution_for_run(run_id) if app.has_note else None
    if existing is not None:
        solutions = app.note.solutions or []
        idx = next(
            (i for i, s in enumerate(solutions) if s is existing),
            -1,
        )
        attempt = existing.get("request", {}).get("attempt_number", "?")
        note = f"solutions[{idx}], attempt {attempt}"
        if comment is not None:
            note += (
                f"; comment posted {comment['posted_at']}"
                if comment.get("posted_at")
                else "; comment pending"
            )
        return "applied", note

    if comment is not None:
        if comment.get("posted_at"):
            return "commented", f"unable to fix; comment posted {comment['posted_at']}"
        return "diagnosed", "agent unable to fix; comment pending"

    config = app.config.workflows.get(run.name)
    if config is None or not config.enabled:
        return "skip", "workflow not enabled in config"

    if not (run.head_branch or "").startswith("mergai/"):
        return "skip", f"head_branch '{run.head_branch}' is not mergai/*"

    head_status = _run_head_status(app, run)
    if head_status == "superseded":
        return "skip", "superseded by newer commits on the branch"
    if head_status == "obsolete":
        return "skip", "head_sha not reachable from HEAD (force-pushed?)"

    # Run hasn't completed yet — neither actionable now nor a reason to
    # give up. `wait` differentiates from `skip` so the user can tell
    # the table is still moving.
    if run.status != "completed":
        return "wait", f"still {run.status}"

    if run.conclusion == "failure":
        return "pending", _failure_note(config)

    if run.conclusion == "success":
        if not config.context.code_scanning_check:
            return "skip", "passed; code_scanning_check not enabled"
        if not check_findings:
            return "pending", "would check Code Scanning"
        pr_number = _resolve_pr_number(run)
        if pr_number is None:
            return "skip", "no associated PR for code scanning lookup"
        if _code_scanning_has_findings(
            app,
            workflow_name=run.name,
            head_sha=run.head_sha,
            pr_number=pr_number,
        ):
            return "pending", "Code Scanning has findings"
        return "skip", "passed; no Code Scanning findings"

    # Completed but with an unusual conclusion (cancelled, timed_out,
    # action_required, neutral, etc.). Surface verbatim — these are
    # rare and worth seeing rather than silently skipping.
    return "skip", f"conclusion '{run.conclusion}'"


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
    click.echo(_format_ascii_table(headers, rows))

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
def comment_post(app: AppContext, target: str, dry_run: bool, force: bool) -> None:
    """Post pending CI comments to their PRs.

    \b
    TARGET (optional, default "all"):
      * "all" / omitted   — post every pending comment. No-op when
                            nothing is pending, so this is safe to run
                            unconditionally from CI / a mergai workflow.
      * a numeric run ID  — post the comment for that run.
    """
    comments = _resolve_comments_for_post(app, target, include_posted=force)
    if not comments:
        if target == "all":
            click.echo("No pending CI comments to post.")
        else:
            click.echo(f"No comment found for run '{target}'.")
        return

    for c in comments:
        run_id = str(c.get("run_id"))
        already = c.get("posted_at") is not None
        if already and not force:
            click.echo(
                f"Comment for run {run_id} was already posted at "
                f"{c['posted_at']}; skipping (use --force to re-post)."
            )
            continue

        body = _render_ci_comment(c)
        if dry_run:
            click.echo(f"--- would post for run {run_id} ---")
            click.echo(body)
            click.echo("--- end ---")
            continue

        pr_number = c.get("pr_number")
        if pr_number is None:
            click.echo(f"Run {run_id}: no PR number recorded; cannot post.")
            continue
        if app.gh is None:
            raise click.ClickException(
                "GitHub auth not available; cannot post PR comment."
            )

        try:
            posted = app.gh_repo.get_pull(int(pr_number)).create_issue_comment(body)
        except Exception as e:  # noqa: BLE001 — wrap external API errors
            raise click.ClickException(
                f"Failed to post PR comment for run {run_id}: {e}"
            ) from e
        comment_url = getattr(posted, "html_url", None)

        app.note.mark_ci_comment_posted(
            run_id,
            posted_at=datetime.now(timezone.utc).isoformat(),
            comment_url=comment_url,
        )
        app.save_note(app.note)
        click.echo(f"Posted comment for run {run_id}: {comment_url or '(no URL)'}")


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
            for path, note in changed.items():
                note_str = note.strip() if isinstance(note, str) else str(note)
                lines.append(f"- `{path}`: {note_str}" if note_str else f"- `{path}`")
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
        for key, note in unresolved.items():
            note_str = note.strip() if isinstance(note, str) else str(note)
            lines.append(f"- `{key}`: {note_str}")
        lines.append("")
    if review_notes:
        lines += ["**Review notes:**", "", review_notes, ""]
    lines += [
        f"_Workflow: `{workflow}` run {run_id} · attempt {attempt}_",
        f"_Recorded: {created_at}_",
    ]
    return "\n".join(lines)
