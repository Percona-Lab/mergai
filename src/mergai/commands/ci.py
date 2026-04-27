"""``mergai ci`` click command group.

Entry point invoked by the ``ci-fix`` job in ``.github/workflows/mergai.yml``
when a CI workflow (``format``, ``clang-tidy``) completes with failure on a
mergai PR. Reads per-workflow config, builds a context from the failure
artifact, runs the configured handler, and commits the fix. The workflow
pushes after this command returns.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import click

from ..app import AppContext
from ..ci.context_builders import get_context_builder
from ..ci.handlers import get_handler
from ..config import WorkflowConfig

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
@click.option("--workflow", required=True, help="Name of the failed workflow.")
@click.option("--run-id", required=True, help="GitHub workflow run ID.")
@click.option("--pr", required=True, type=int, help="PR number.")
@click.option(
    "--artifacts-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Directory containing extracted workflow artifacts.",
)
def handle(
    app: AppContext,
    workflow: str,
    run_id: str,
    pr: int,
    artifacts_dir: str | None,
) -> None:
    """Handle a completed CI workflow failure for the current branch.

    Looks up the workflow's ``WorkflowConfig`` in the mergai config,
    enforces ``max_attempts``, builds a failure context from the
    downloaded artifacts, dispatches to the configured handler, and on
    success creates a commit recording the attempt.
    """
    config = app.config.workflows.get(workflow)
    if config is None:
        click.echo(f"No configuration for workflow '{workflow}'; nothing to do.")
        return

    if not config.enabled:
        click.echo(f"Workflow '{workflow}' handling is disabled; nothing to do.")
        return

    prior_attempts = app.note.get_ci_attempts(workflow)
    attempt_number = len(prior_attempts) + 1
    if attempt_number > config.max_attempts:
        click.echo(
            f"Max attempts ({config.max_attempts}) reached for '{workflow}'; "
            f"giving up."
        )
        _post_max_attempts_comment(app, pr, workflow, config)
        _record_attempt(
            app,
            workflow=workflow,
            attempt_number=attempt_number,
            run_id=run_id,
            pr_number=pr,
            action_type=config.action_type,
            summary="(skipped — max attempts reached)",
            files_affected=[],
            success=False,
            give_up=True,
        )
        return

    click.echo(
        f"Handling '{workflow}' failure (attempt {attempt_number} of "
        f"{config.max_attempts})"
    )

    builder = get_context_builder(config.context.type)
    context = builder.build_context(
        config.context,
        workflow_name=workflow,
        run_id=run_id,
        pr_number=pr,
        artifacts_dir=artifacts_dir,
    )
    click.echo(f"Context: {context.summary}")

    handler = get_handler(app, config)
    success = handler.execute(context)

    _record_attempt(
        app,
        workflow=workflow,
        attempt_number=attempt_number,
        run_id=run_id,
        pr_number=pr,
        action_type=config.action_type,
        summary=context.summary,
        files_affected=context.files_affected,
        success=success,
        give_up=False,
    )

    if not success:
        click.echo(
            f"Handler did not produce any changes for '{workflow}'. "
            f"Will retry on the next workflow run."
        )
        return

    _commit_fix(app, workflow=workflow, attempt_number=attempt_number)


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
