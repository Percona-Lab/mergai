"""``mergai ci`` click command group.

Entry point invoked by the ``ci-fix`` job in ``.github/workflows/mergai.yml``
on every ``workflow_run.completed`` for a watched CI workflow on a
``mergai/*`` branch. Given just a run ID, this command:

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

The handler edits the working tree; this command commits the result.
The outer workflow pushes the new commit so CI re-runs.

The same command is intended to be runnable manually with just
``--run-id``; explicit ``--workflow`` / ``--pr`` / ``--artifacts-dir``
overrides exist for testing or unusual setups.
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
@click.option("--run-id", required=True, help="GitHub workflow run ID.")
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
    run_id: str,
    workflow: str | None,
    pr: int | None,
    artifacts_dir: str | None,
) -> None:
    """Handle a completed CI workflow run for the current branch.

    Builds a :class:`WorkflowContext` via
    :func:`build_workflow_context_for_run`, enforces ``max_attempts``,
    dispatches to the configured handler, and commits the result.
    """
    with build_workflow_context_for_run(
        app,
        run_id,
        workflow_override=workflow,
        pr_override=pr,
        artifacts_dir_override=artifacts_dir,
    ) as built:
        if built is None:
            return
        context, config = built

        prior_attempts = app.note.get_ci_attempts(context.workflow_name)
        attempt_number = len(prior_attempts) + 1
        if attempt_number > config.max_attempts:
            click.echo(
                f"Max attempts ({config.max_attempts}) reached for "
                f"'{context.workflow_name}'; giving up."
            )
            _post_max_attempts_comment(
                app, context.pr_number, context.workflow_name, config
            )
            _record_attempt(
                app,
                workflow=context.workflow_name,
                attempt_number=attempt_number,
                run_id=run_id,
                pr_number=context.pr_number,
                action_type=config.action_type,
                summary="(skipped — max attempts reached)",
                files_affected=[],
                success=False,
                give_up=True,
            )
            return

        click.echo(
            f"Handling '{context.workflow_name}' (attempt {attempt_number} of "
            f"{config.max_attempts}). Context: {context.summary}"
        )

        handler = get_handler(app, config)
        success = handler.execute(context)

        _record_attempt(
            app,
            workflow=context.workflow_name,
            attempt_number=attempt_number,
            run_id=run_id,
            pr_number=context.pr_number,
            action_type=config.action_type,
            summary=context.summary,
            files_affected=context.files_affected,
            success=success,
            give_up=False,
        )

        if not success:
            click.echo(
                f"Handler did not produce any changes for "
                f"'{context.workflow_name}'. Will retry on the next workflow run."
            )
            return

        _commit_fix(app, workflow=context.workflow_name, attempt_number=attempt_number)


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


def _record_attempt(
    app: AppContext,
    *,
    workflow: str,
    attempt_number: int,
    run_id: str,
    pr_number: int,
    action_type: str,
    summary: str,
    files_affected: list[str],
    success: bool,
    give_up: bool,
) -> None:
    """Append the attempt to the note's ``ci_fix_history`` and persist it."""
    attempt = {
        "workflow": workflow,
        "attempt_number": attempt_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "pr_number": pr_number,
        "action_type": action_type,
        "context_summary": summary,
        "files_affected": list(files_affected),
        "success": success,
        "give_up": give_up,
    }
    app.note.add_ci_attempt(attempt)
    app.save_note(app.note)


def _commit_fix(app: AppContext, workflow: str, attempt_number: int) -> None:
    """Stage all changes and commit the CI fix.

    The corresponding mergai note is attached via
    ``add_selective_note(..., ["ci_fix_history"])``, recording only the
    just-added attempt entry.
    """
    repo = app.repo
    work_dir = repo.working_tree_dir
    if work_dir is None:
        raise click.ClickException("Repo has no working tree; cannot commit fix")

    repo.git.add("-A", str(Path(work_dir)))

    if not repo.index.diff("HEAD"):
        # Should not happen — handler returned success because tree was
        # dirty — but guard against subtle edge cases (e.g. all changes
        # in submodules, ignored paths) so we don't create empty commits.
        click.echo("No staged changes after `git add`; skipping commit.")
        return

    message = f"fix({workflow}): automated fix attempt {attempt_number}\n"
    if app.config.commit.footer:
        message += "\n" + app.config.commit.footer

    repo.git.commit("-m", message)

    commit_sha = repo.head.commit.hexsha
    app.add_selective_note(commit_sha, ["ci_fix_history"])
    click.echo(f"Committed CI fix as {commit_sha[:11]}.")
