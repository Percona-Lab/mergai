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

``mergai ci handle <run-id>`` is what GitHub Actions calls. Manually
you can also pass ``"all"`` or a workflow name to process every
unprocessed actionable run on the current branch.
"""

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import click
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
def handle(
    app: AppContext,
    target: str,
    workflow: str | None,
    pr: int | None,
    artifacts_dir: str | None,
) -> None:
    """Handle one or more workflow runs for the current branch.

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
    if target.isdigit():
        run_ids = [target]
    else:
        run_ids = _resolve_target_runs(app, target)
        if not run_ids:
            click.echo(f"No unprocessed actionable runs found for target '{target}'.")
            return
        click.echo(
            f"Found {len(run_ids)} unprocessed actionable run(s) "
            f"for target '{target}'."
        )

    for run_id in run_ids:
        _handle_one_run(
            app,
            run_id,
            workflow_override=workflow,
            pr_override=pr,
            artifacts_dir_override=artifacts_dir,
        )


def _resolve_target_runs(app: AppContext, target: str) -> list[str]:
    """Resolve ``"all"`` / workflow-name to a list of run IDs to process.

    Lists recent workflow_runs on the current branch, filters out runs
    that are not configured / not actionable / already processed, and
    returns IDs in newest-first order.

    A workflow-name target is just the ``"all"`` filter narrowed to one
    workflow.
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
        if not _run_is_actionable(app, run):
            continue
        selected.append(run_id)

    return selected


def _run_is_actionable(app: AppContext, run: "github.WorkflowRun.WorkflowRun") -> bool:
    """Mirror of the dispatch decision in ``build_workflow_context_for_run``.

    Returns True if mergai would build a context for this run if asked
    to handle it now. Used by ``mergai ci handle all`` /
    ``mergai ci handle <workflow>`` to filter the run list before
    iterating.
    """
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


def _handle_one_run(
    app: AppContext,
    run_id: str,
    *,
    workflow_override: str | None,
    pr_override: int | None,
    artifacts_dir_override: str | None,
) -> None:
    """Handle a single workflow run.

    Body of the old single-run ``handle`` command, hoisted out so that
    ``handle`` can iterate over multiple runs for the ``all`` /
    workflow-name targets.
    """
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

    Shared by ``mergai ci handle`` (which then runs the handler) and
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

    from rich.console import Console
    from rich.table import Table

    table = Table(show_lines=False)
    table.add_column("Run ID", no_wrap=True)
    table.add_column("Workflow")
    table.add_column("Conclusion")
    table.add_column("Head SHA", no_wrap=True)
    table.add_column("Status")
    table.add_column("Notes")
    for row in rows:
        table.add_row(*row)
    Console().print(table)


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
    existing = app.note.get_ci_solution_for_run(run_id) if app.has_note else None
    if existing is not None:
        solutions = app.note.solutions or []
        idx = next(
            (i for i, s in enumerate(solutions) if s is existing),
            -1,
        )
        attempt = existing.get("request", {}).get("attempt_number", "?")
        return "applied", f"solutions[{idx}], attempt {attempt}"

    config = app.config.workflows.get(run.name)
    if config is None or not config.enabled:
        return "skip", "workflow not enabled in config"

    if not (run.head_branch or "").startswith("mergai/"):
        return "skip", f"head_branch '{run.head_branch}' is not mergai/*"

    if run.conclusion == "failure":
        return "pending", "failure → artifact + log fallback"

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

    return "skip", f"conclusion '{run.conclusion}'"
