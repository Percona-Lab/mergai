"""Check command group for CI workflow handling.

This module provides commands for checking CI workflow status and fixing
failing checks.
"""

import re
import subprocess

import click

from ..app import AppContext
from ..config import WorkflowConfig
from ..models import CheckContext
from ..utils import git_utils
from ..utils.github_workflows import (
    WorkflowRunInfo,
    WorkflowStatusSummary,
    format_status_icon,
    get_workflow_runs_for_pr,
)


def _get_repo_from_remote(app: AppContext) -> str | None:
    """Get GitHub repository (owner/repo) from git remote.

    Args:
        app: Application context with repo.

    Returns:
        Repository string in 'owner/repo' format, or None if not detectable.
    """
    try:
        remote = app.repo.remote("origin")
        for url in remote.urls:
            # SSH format: git@github.com:owner/repo.git
            ssh_match = re.match(r"git@github\.com:(.+?)(?:\.git)?$", url)
            if ssh_match:
                return ssh_match.group(1)

            # HTTPS format: https://github.com/owner/repo.git
            https_match = re.match(r"https://github\.com/(.+?)(?:\.git)?$", url)
            if https_match:
                return https_match.group(1)
    except (ValueError, AttributeError):
        pass
    return None


def _get_pr_number_for_branch(app: AppContext, repo_str: str) -> int | None:
    """Get PR number for the current branch.

    Args:
        app: Application context with GitHub client.
        repo_str: Repository string in 'owner/repo' format.

    Returns:
        PR number if found, None otherwise.
    """
    current_branch = git_utils.get_current_branch(app.repo)

    try:
        # Query for open PRs with this head branch
        pulls = app.gh_repo.get_pulls(
            state="open", head=f"{repo_str.split('/')[0]}:{current_branch}"
        )
        for pr in pulls:
            if pr.head.ref == current_branch:
                return pr.number
    except Exception:
        pass

    return None


def _resolve_repo_and_pr(
    app: AppContext, repo_str: str | None, pr_number: int | None
) -> tuple[str, int]:
    """Resolve repository and PR number, auto-detecting if not provided.

    Args:
        app: Application context.
        repo_str: Explicit repo string or None to auto-detect.
        pr_number: Explicit PR number or None to auto-detect.

    Returns:
        Tuple of (repo_str, pr_number).

    Raises:
        click.ClickException: If unable to detect repo or PR.
    """
    # Resolve repo
    if repo_str is None:
        repo_str = _get_repo_from_remote(app)
        if repo_str is None:
            raise click.ClickException(
                "Could not detect GitHub repository from git remote. "
                "Please specify --repo explicitly."
            )

    app.gh_repo_str = repo_str

    # Resolve PR number
    if pr_number is None:
        pr_number = _get_pr_number_for_branch(app, repo_str)
        if pr_number is None:
            raise click.ClickException(
                "Could not find an open PR for the current branch. "
                "Please specify --pr explicitly."
            )

    return repo_str, pr_number


@click.group()
def check():
    """CI check handling commands."""
    pass


@check.command()
@click.pass_obj
@click.option(
    "--repo",
    "repo_str",
    required=False,
    envvar="GH_REPO",
    help="GitHub repository (owner/repo). Auto-detected from git remote if not specified.",
)
@click.option(
    "--pr",
    "pr_number",
    required=False,
    type=int,
    help="Pull request number. Auto-detected from current branch if not specified.",
)
def status(app: AppContext, repo_str: str | None, pr_number: int | None):
    """Check status of configured CI workflows for a PR.

    Queries GitHub API to check the status of all configured workflows.

    Exit codes:
        0 - All workflows complete (ready to process)
        1 - Not all workflows complete (still waiting)
        2 - Error occurred
    """
    # Resolve repo and PR number
    try:
        repo_str, pr_number = _resolve_repo_and_pr(app, repo_str, pr_number)
    except click.ClickException as e:
        click.echo(str(e))
        raise SystemExit(2)

    # Get configured workflows
    workflow_configs = app.config.checks.get_enabled_workflows()
    if not workflow_configs:
        click.echo("No workflows configured in checks.workflows")
        raise SystemExit(2)

    workflow_names = [w.name for w in workflow_configs]
    click.echo(f"Checking workflows for PR #{pr_number}: {', '.join(workflow_names)}")

    try:
        summary = get_workflow_runs_for_pr(app.gh_repo, pr_number, workflow_names)
    except Exception as e:
        click.echo(f"Error querying GitHub API: {e}")
        raise SystemExit(2)

    # Print status table
    click.echo(f"\nWorkflow Status for PR #{pr_number}:")
    _print_status_table(summary, workflow_names)

    # Report summary
    if summary.missing_workflows:
        click.echo(
            f"\nMissing workflows (no runs found): {', '.join(summary.missing_workflows)}"
        )

    if not summary.all_complete:
        incomplete = summary.get_incomplete_workflows()
        if incomplete:
            click.echo(
                f"\nNot all workflows complete. Waiting for: "
                f"{', '.join(w.workflow_name for w in incomplete)}"
            )
        else:
            click.echo("\nNot all workflows complete. Waiting...")
        raise SystemExit(1)

    # All complete
    failed = summary.get_failed_workflows()
    if failed:
        click.echo(f"\nAll workflows complete. {len(failed)} failure(s) to process.")
    else:
        click.echo("\nAll workflows complete. All passed!")

    raise SystemExit(0)


@check.command()
@click.pass_obj
@click.option(
    "--repo",
    "repo_str",
    required=False,
    envvar="GH_REPO",
    help="GitHub repository (owner/repo). Auto-detected from git remote if not specified.",
)
@click.option(
    "--pr",
    "pr_number",
    required=False,
    type=int,
    help="Pull request number. Auto-detected from current branch if not specified.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Force re-fix even if max attempts reached",
)
@click.option(
    "-y",
    "--yolo",
    is_flag=True,
    default=False,
    help="Enable YOLO mode for agent (auto-approve)",
)
@click.argument("workflow", required=False)
def fix(
    app: AppContext,
    repo_str: str | None,
    pr_number: int | None,
    force: bool,
    yolo: bool,
    workflow: str | None,
):
    """Fix failing CI checks.

    Processes failing workflows and attempts to fix them using the configured
    resolution strategy (command or agent). Each fix creates a separate commit.

    If WORKFLOW is specified, only that workflow is processed.
    Otherwise, all failing workflows are processed.
    """
    # Resolve repo and PR number
    try:
        repo_str, pr_number = _resolve_repo_and_pr(app, repo_str, pr_number)
    except click.ClickException as e:
        click.echo(str(e))
        raise SystemExit(1)

    # Get configured workflows
    all_configs = app.config.checks.get_enabled_workflows()
    if not all_configs:
        click.echo("No workflows configured in checks.workflows")
        raise SystemExit(1)

    # Filter to specific workflow if requested
    if workflow:
        configs = [c for c in all_configs if c.name == workflow]
        if not configs:
            click.echo(f"Workflow '{workflow}' not found in configuration")
            raise SystemExit(1)
    else:
        configs = all_configs

    workflow_names = [w.name for w in configs]

    # Get workflow status
    try:
        summary = get_workflow_runs_for_pr(app.gh_repo, pr_number, workflow_names)
    except Exception as e:
        click.echo(f"Error querying GitHub API: {e}")
        raise SystemExit(1)

    # Check if all complete
    if not summary.all_complete:
        click.echo("Not all workflows have completed yet. Cannot process.")
        _print_status_table(summary, workflow_names)
        raise SystemExit(1)

    # Get failing workflows
    failed_runs = summary.get_failed_workflows()
    if not failed_runs:
        click.echo("No failing workflows to fix!")
        raise SystemExit(0)

    click.echo(f"Found {len(failed_runs)} failing workflow(s) to process:")
    for run in failed_runs:
        click.echo(f"  - {run.workflow_name}: {run.run_url}")

    # Process each failing workflow
    fixed_count = 0
    skipped_count = 0
    error_count = 0

    for run in failed_runs:
        config = app.config.checks.get_workflow(run.workflow_name)
        if config is None:
            click.echo(f"\nSkipping {run.workflow_name}: not in configuration")
            skipped_count += 1
            continue

        click.echo(f"\nProcessing: {run.workflow_name}")

        try:
            result = _process_failing_workflow(app, config, run, force, yolo)
            if result:
                fixed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            click.echo(f"  Error: {e}")
            error_count += 1

    # Summary
    click.echo(
        f"\nSummary: {fixed_count} fixed, {skipped_count} skipped, {error_count} errors"
    )

    if error_count > 0:
        raise SystemExit(1)
    if fixed_count == 0:
        click.echo("No changes made.")
        raise SystemExit(0)

    click.echo(f"\n{fixed_count} commit(s) created. Ready to push.")
    raise SystemExit(0)


def _print_status_table(summary: WorkflowStatusSummary, workflow_names: list[str]):
    """Print a formatted status table for workflows."""
    max_name_len = max(len(name) for name in workflow_names)
    # Icon width matches the width used in format_status_icon (default 11)
    icon_width = 11

    for name in workflow_names:
        if name in summary.workflows:
            run = summary.workflows[name]
            icon = format_status_icon(run.status)
            click.echo(f"  {name:<{max_name_len}}  {icon}  {run.run_url}")
        elif name in summary.missing_workflows:
            icon = f"[{'WARN':<{icon_width - 2}}]"
            click.echo(f"  {name:<{max_name_len}}  {icon}  no runs found")
        else:
            icon = f"[{'?':<{icon_width - 2}}]"
            click.echo(f"  {name:<{max_name_len}}  {icon}  unknown")


def _process_failing_workflow(
    app: AppContext,
    config: WorkflowConfig,
    run: WorkflowRunInfo,
    force: bool,
    yolo: bool,
) -> bool:
    """Process a single failing workflow.

    Args:
        app: Application context.
        config: Workflow configuration.
        run: Workflow run info.
        force: Force re-fix even if max attempts reached.
        yolo: Enable YOLO mode for agent.

    Returns:
        True if a fix was applied (commit created), False otherwise.
    """
    # Check existing context and attempt count
    existing_context = app.note.get_check_context(config.name)

    if existing_context:
        attempt_count = existing_context.attempt_count
        if attempt_count >= config.max_attempts and not force:
            click.echo(
                f"  Skipping: max attempts ({config.max_attempts}) reached. "
                f"Use --force to override."
            )
            return False
    else:
        attempt_count = 0

    # Create or update check context
    check_ctx = existing_context or CheckContext(
        workflow_name=config.name,
        current_run_url=run.run_url,
    )
    check_ctx.current_run_url = run.run_url

    click.echo(f"  Resolution type: {config.resolution.type}")
    click.echo(f"  Attempt: {attempt_count + 1}/{config.max_attempts}")

    # Execute resolution strategy
    if config.resolution.type == "command":
        result = _fix_with_command(app, config, run, check_ctx)
    else:
        result = _fix_with_agent(app, config, run, check_ctx, yolo)

    if result is None:
        return False

    summary, files_modified = result

    # Commit the changes
    _commit_check_fix(app, config, run, check_ctx, summary, files_modified)

    return True


def _fix_with_command(
    app: AppContext,
    config: WorkflowConfig,
    run: WorkflowRunInfo,
    check_ctx: CheckContext,
) -> tuple[str, list[str]] | None:
    """Fix a check using a shell command.

    Returns:
        Tuple of (summary, files_modified) or None if no changes.
    """
    command = config.resolution.command
    if not command:
        raise ValueError(f"No command configured for workflow '{config.name}'")

    click.echo(f"  Running: {command}")

    # Run the command
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=app.repo.working_tree_dir,
    )

    if result.returncode != 0:
        click.echo(f"  Command failed with exit code {result.returncode}")
        if result.stderr:
            click.echo(f"  stderr: {result.stderr[:500]}")
        # Record the failed attempt
        check_ctx.add_attempt(
            workflow_run_url=run.run_url,
            resolution_type="command",
            summary=f"Command failed: {command}",
            success=False,
        )
        app.note.set_check_context(check_ctx)
        app.save_note(app.note)
        return None

    # Check for changes
    if not app.repo.is_dirty():
        click.echo("  No changes made by command")
        return None

    # Get modified files
    files_modified = [item.a_path for item in app.repo.index.diff(None)]
    click.echo(f"  Modified {len(files_modified)} file(s)")

    summary = f"Ran command: {command}"
    return summary, files_modified


def _fix_with_agent(
    app: AppContext,
    config: WorkflowConfig,
    run: WorkflowRunInfo,
    check_ctx: CheckContext,
    yolo: bool,
) -> tuple[str, list[str]] | None:
    """Fix a check using an AI agent.

    Returns:
        Tuple of (summary, files_modified) or None if no changes.
    """
    from ..agent_executor import AgentExecutionError, AgentExecutor

    # Build the prompt
    prompt = app.prompt_builder.build_check_fix_prompt(
        workflow_name=config.name,
        workflow_run_url=run.run_url,
        previous_attempts=check_ctx.attempts,
    )

    agent = app.get_agent(yolo=yolo)
    executor = AgentExecutor(
        agent=agent,
        state_dir=app.state.path,
        max_attempts=1,  # Single attempt per invocation
        repo=app.repo,
    )

    try:
        result = executor.run_with_retry(
            prompt=prompt,
            validator=_validate_check_fix_result,
        )
    except AgentExecutionError as e:
        click.echo(f"  Agent failed: {e}")
        check_ctx.add_attempt(
            workflow_run_url=run.run_url,
            resolution_type="agent",
            summary=f"Agent failed: {e}",
            success=False,
        )
        app.note.set_check_context(check_ctx)
        app.save_note(app.note)
        return None

    # Check for changes
    if not app.repo.is_dirty():
        click.echo("  No changes made by agent")
        return None

    # Get modified files
    files_modified = [item.a_path for item in app.repo.index.diff(None)]
    summary = result.get("response", {}).get("summary", "Agent fix applied")

    click.echo(f"  Modified {len(files_modified)} file(s)")
    return summary, files_modified


def _validate_check_fix_result(result: dict) -> str | None:
    """Validate the agent's check fix result.

    Args:
        result: The result dict from the agent.

    Returns:
        None if valid, error message if invalid.
    """
    response = result.get("response", {})
    if "summary" not in response:
        return "Missing 'summary' field in response"
    return None


def _commit_check_fix(
    app: AppContext,
    config: WorkflowConfig,
    run: WorkflowRunInfo,
    check_ctx: CheckContext,
    summary: str,
    files_modified: list[str],
):
    """Commit the check fix and update notes.

    Args:
        app: Application context.
        config: Workflow configuration.
        run: Workflow run info.
        check_ctx: Check context being updated.
        summary: Summary of the fix.
        files_modified: List of modified file paths.
    """
    attempt_num = check_ctx.attempt_count + 1

    # Stage all modified files
    for file_path in files_modified:
        app.repo.index.add([file_path])

    # Build commit message
    message = f"Fix {config.name} check (attempt {attempt_num})\n\n"
    message += f"{summary}\n\n"
    message += "Modified files:\n"
    for file_path in files_modified:
        message += f"    {file_path}\n"
    message += f"\nWorkflow run: {run.run_url}\n\n"
    message += app.commit_footer

    # Create commit
    app.repo.index.commit(message)
    click.echo(f"  Created commit: {app.repo.head.commit.hexsha[:11]}")

    # Record the attempt
    check_ctx.add_attempt(
        workflow_run_url=run.run_url,
        resolution_type=config.resolution.type,
        summary=summary,
        success=True,  # Committed successfully
    )

    # Update note
    app.note.set_check_context(check_ctx)
    app.save_note(app.note)

    # Attach note to commit
    app.add_selective_note(
        app.repo.head.commit.hexsha,
        [f"check_contexts[{config.name}]"],
    )
