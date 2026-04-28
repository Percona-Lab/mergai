"""``mergai prompt`` command group — render AI-agent prompts for inspection.

Three subtypes share one entry point:

* ``prompt resolve`` — the conflict-resolution prompt built from the
  current merge note (was the top-level ``mergai prompt``).
* ``prompt describe`` — the merge-description prompt (was
  ``mergai merge-prompt``).
* ``prompt ci --run-id <id>`` — the CI-fix prompt that ``ci handle``
  would feed to the agent for a given workflow run, rendered without
  invoking the agent. Useful for inspecting what the agent will see and
  for iterating on the template / context builder.
"""

import click

from ..app import AppContext
from ..ci.handlers.resolve import build_ci_fix_prompt
from ..utils import util
from .ci import build_workflow_context_for_run


@click.group()
def prompt():
    """Render AI-agent prompts for inspection or testing."""


@prompt.command(name="resolve")
@click.pass_obj
def prompt_resolve(app: AppContext) -> None:
    """Print the conflict-resolution prompt built from the current merge note."""
    if not app.has_note:
        click.echo("No note found. Please prepare the context first.")
        click.echo("Use `mergai context init` to add merge context.")
        raise click.exceptions.Exit(1)
    util.print_or_page(app.prompt_builder.build_resolve_prompt(), format="markdown")


@prompt.command(name="describe")
@click.pass_obj
def prompt_describe(app: AppContext) -> None:
    """Print the merge-description prompt built from the current merge note."""
    if not app.has_note:
        click.echo("No note found. Please prepare the context first.")
        click.echo("Use `mergai context init` to add merge context.")
        raise click.exceptions.Exit(1)
    util.print_or_page(app.prompt_builder.build_describe_prompt(), format="markdown")


@prompt.command(name="ci")
@click.pass_obj
@click.option(
    "--repo",
    "repo",
    type=str,
    required=False,
    envvar="GH_REPO",
    help="The repository where the run lives.",
)
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
        "subdirectory named after it."
    ),
)
def prompt_ci(
    app: AppContext,
    repo: str | None,
    run_id: str,
    workflow: str | None,
    pr: int | None,
    artifacts_dir: str | None,
) -> None:
    """Print the CI-fix prompt that ``ci handle`` would build for a workflow run.

    Runs the same orchestration as ``ci handle`` (resolve run → pick
    artifact / Code Scanning / log-fallback → build WorkflowContext) but
    stops before invoking the agent and prints the rendered prompt
    instead.
    """
    if repo is None:
        raise click.ClickException(
            "GitHub repository not set. Use --repo or set GH_REPO environment variable."
        )
    app.gh_repo_str = repo

    with build_workflow_context_for_run(
        app,
        run_id,
        workflow_override=workflow,
        pr_override=pr,
        artifacts_dir_override=artifacts_dir,
    ) as built:
        if built is None:
            return
        context, _config = built
        util.print_or_page(build_ci_fix_prompt(context), format="markdown")
