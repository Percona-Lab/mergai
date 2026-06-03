"""``mergai prompt`` command group — render AI-agent prompts for inspection.

Three subtypes share one entry point:

* ``prompt resolve`` — the conflict-resolution prompt built from the
  current merge note (was the top-level ``mergai prompt``).
* ``prompt describe`` — the merge-description prompt (was
  ``mergai merge-prompt``).
* ``prompt ci <target>`` — the CI-fix prompt(s) that ``ci fix``
  would feed to the agent. Same positional ``target`` shape as
  ``ci fix`` (numeric run id, ``"all"``, or a workflow name).
  Renders without invoking the agent.
"""

import click

from ..app import AppContext
from ..ci.dispatch import _resolve_target_runs, build_workflow_context_for_run
from ..prompt_builder import (
    build_ci_fix_preamble,
    build_ci_fix_prompt,
    build_ci_fix_run_section,
)
from ..utils import util
from .util import ensure_gh_repo


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
    # The describe prompt is grounded in the merge-base..merge-commit diff,
    # so resolve the diff base first (same as AppContext.describe()).
    merge_base_sha = app.resolve_merge_diff_base()
    if merge_base_sha is None:
        click.echo(
            "Could not resolve a diff base for "
            f"{app.note.merge_info.merge_commit_sha} "
            "(unrelated histories or no boundary parent); "
            "cannot build a merge description prompt."
        )
        raise click.exceptions.Exit(1)
    util.print_or_page(
        app.prompt_builder.build_describe_prompt(merge_base_sha=merge_base_sha),
        format="markdown",
    )


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
        "subdirectory named after it."
    ),
)
def prompt_ci(
    app: AppContext,
    repo: str | None,
    target: str,
    workflow: str | None,
    pr: int | None,
    artifacts_dir: str | None,
) -> None:
    """Print the CI-fix prompt(s) that ``ci fix`` would build.

    \b
    TARGET can be:
      * a numeric run ID — print the prompt for that run
      * "all"            — print prompts for every unprocessed
                           actionable run on the current branch
                           (same filter as `ci fix all`)
      * a workflow name  — like "all" but filtered to that workflow

    Runs the same orchestration as ``ci fix`` (resolve run → pick
    artifact / Code Scanning / log fallback → build WorkflowContext)
    but stops before invoking the agent and prints the rendered prompt
    instead. Multiple prompts are separated by a header line.
    """
    ensure_gh_repo(app, repo)

    if target.isdigit():
        run_ids = [target]
    else:
        run_ids = _resolve_target_runs(app, target)
        if not run_ids:
            click.echo(f"No unprocessed actionable runs found for target '{target}'.")
            return

    # Embed the merge note when available so the prompt actually shows
    # the agent it's a post-merge state. `prompt ci <run-id>` for an
    # arbitrary run outside a mergai working tree still works — the
    # preamble just skips the merge section in that case.
    note = app.note if app.has_note else None
    prompt_config = app.config.prompt
    project_config = app.config.project

    # Single run renders as the agent would see it (full prompt). Multi
    # run shares the system prompt + invariants + merge context +
    # context-format description across runs and emits one per-run
    # section per run, so the output is actually useful for inspection
    # and not wall-of-text repetition.
    if len(run_ids) == 1:
        with build_workflow_context_for_run(
            app,
            run_ids[0],
            workflow_override=workflow,
            pr_override=pr,
            artifacts_dir_override=artifacts_dir,
        ) as built:
            if built is None:
                return
            context, _config = built
            util.print_or_page(
                build_ci_fix_prompt(
                    context,
                    note=note,
                    prompt_config=prompt_config,
                    project_config=project_config,
                ),
                format="markdown",
            )
        return

    sections: list[str] = []
    for run_id in run_ids:
        with build_workflow_context_for_run(
            app,
            run_id,
            workflow_override=workflow,
            pr_override=pr,
            artifacts_dir_override=artifacts_dir,
        ) as built:
            if built is None:
                continue
            context, _config = built
            heading = f"## Run {run_id} — {context.workflow_name}"
            sections.append(build_ci_fix_run_section(context, heading=heading))

    if not sections:
        return
    util.print_or_page(
        build_ci_fix_preamble(
            note=note, prompt_config=prompt_config, project_config=project_config
        )
        + "\n".join(sections),
        format="markdown",
    )
