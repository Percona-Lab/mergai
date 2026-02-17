"""Merge command for mergai.

This module provides the merge command that performs a git merge
using the SHA from merge_info.
"""

import click
from git.exc import GitCommandError
from ..app import AppContext
from ..utils import git_utils


# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFLICT = 1
EXIT_ERROR = 2


@click.command()
@click.pass_obj
@click.option(
    "--no-context",
    "no_context",
    is_flag=True,
    default=False,
    help="Skip creating merge_context and conflict_context after merge.",
)
@click.option(
    "-f",
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing merge_context and conflict_context.",
)
def merge(app: AppContext, no_context: bool, force: bool):
    """Perform a git merge using the commit from merge_info.

    Executes 'git merge --no-commit --no-ff <sha>' where <sha> is
    the merge_commit from the initialized merge context.

    This command requires merge_info to be initialized first
    (via 'mergai context init').

    By default, after merge:
    - On success: Creates merge_context with auto-merged files info
    - On conflict: Creates both merge_context and conflict_context

    Use --no-context to skip automatic context creation.
    Use --force to overwrite existing contexts.

    \b
    Exit codes:
        0: Merge completed successfully (no conflicts)
        1: Merge resulted in conflicts
        2: Error (missing merge_info, invalid commit, etc.)

    \b
    Examples:
        mergai context init abc1234 --target v8.0
        mergai merge
        mergai merge --no-context  # skip context creation
        mergai merge --force       # overwrite existing contexts

    \b
    Script usage:
        mergai merge
        case $? in
            0) echo "Merge successful" ;;
            1) echo "Merge has conflicts" ;;
            2) echo "Error occurred" ;;
        esac
    """
    # Load note to get merge_info
    if not app.has_note:
        click.echo("Error: merge_info not found. Run 'mergai context init' first.")
        raise SystemExit(EXIT_ERROR)

    merge_commit_sha = app.note.merge_info.merge_commit_sha

    if not merge_commit_sha:
        click.echo("Error: merge_commit not found in merge_info.")
        raise SystemExit(EXIT_ERROR)

    # Pre-merge validation: check if contexts exist when force is not set
    # This prevents leaving the repo in a merging state if context creation would fail
    if not no_context and not force:
        if app.note.has_merge_context:
            click.echo(
                "Error: merge_context already exists. "
                "Use -f/--force to overwrite or --no-context to skip context creation."
            )
            raise SystemExit(EXIT_ERROR)
        if app.note.has_conflict_context:
            click.echo(
                "Error: conflict_context already exists. "
                "Use -f/--force to overwrite or --no-context to skip context creation."
            )
            raise SystemExit(EXIT_ERROR)

    # Execute git merge --no-commit --no-ff <sha>
    try:
        output = app.repo.git.merge("--no-commit", "--no-ff", merge_commit_sha)

        # Print git merge output first
        if output:
            click.echo(output)

        # Merge succeeded without conflicts
        parsed = git_utils.parse_git_merge_output(output, repo=app.repo)

        # Print mergai summary
        click.echo("")
        click.echo("--- mergai summary ---")
        click.echo("Merge successful (no conflicts).")

        if not no_context:
            try:
                app.create_merge_context(
                    force=force,
                    auto_merged_files=parsed.auto_merged_files,
                    merge_strategy=parsed.strategy,
                )
                click.echo("Created merge_context.")
                if parsed.strategy:
                    click.echo(f"  strategy: {parsed.strategy}")
                if parsed.auto_merged_files:
                    click.echo(f"  auto_merged: {len(parsed.auto_merged_files)} files")
            except Exception as e:
                click.echo(f"Warning: Failed to create merge_context: {e}")

        raise SystemExit(EXIT_SUCCESS)

    except GitCommandError as e:
        # GitCommandError is raised when merge has conflicts
        # Git returns exit code 1 for conflicts
        if e.status == 1:
            # Parse output from stderr (where git writes merge info)
            output = e.stderr if e.stderr else (e.stdout if e.stdout else "")

            # Print git merge output first
            if output:
                click.echo(output)

            parsed = git_utils.parse_git_merge_output(output, repo=app.repo)

            # Print mergai summary
            click.echo("")
            click.echo("--- mergai summary ---")
            click.echo("Merge has conflicts.")

            # Print conflicting files
            if parsed.conflicting_files:
                click.echo("Conflicting files:")
                for file_path, conflict_type in parsed.conflicting_files.items():
                    click.echo(f"  {file_path} ({conflict_type})")
            else:
                # Fallback: get conflicts from git index if parsing didn't find them
                try:
                    blobs_map = app.repo.index.unmerged_blobs()
                    if blobs_map:
                        click.echo("Conflicting files:")
                        for file_path in blobs_map.keys():
                            click.echo(f"  {file_path}")
                except Exception:
                    pass

            if not no_context:
                # Create merge context with auto-merged files
                try:
                    app.create_merge_context(
                        force=force,
                        auto_merged_files=parsed.auto_merged_files,
                        merge_strategy=parsed.strategy,
                    )
                    click.echo("Created merge_context.")
                    if parsed.strategy:
                        click.echo(f"  strategy: {parsed.strategy}")
                    if parsed.auto_merged_files:
                        click.echo(f"  auto_merged: {len(parsed.auto_merged_files)} files")
                except Exception as ctx_err:
                    click.echo(f"Warning: Failed to create merge_context: {ctx_err}")

                # Create conflict context
                try:
                    if not git_utils.is_merge_conflict_style_diff3(app.repo):
                        click.echo(
                            "Warning: Git is not configured to use diff3 for merges. "
                            "Consider setting 'merge.conflictstyle' to 'diff3'."
                        )
                    app.create_conflict_context(
                        use_diffs=True,
                        diff_lines_of_context=0,
                        use_compressed_diffs=True,
                        use_their_commits=True,
                        force=force,
                    )
                    click.echo("Created conflict_context.")
                except Exception as ctx_err:
                    click.echo(f"Warning: Failed to create conflict_context: {ctx_err}")

            raise SystemExit(EXIT_CONFLICT)
        else:
            # Other git errors (e.g., invalid ref, not a git repo)
            click.echo(f"Error: Git merge failed with status {e.status}")
            if e.stderr:
                click.echo(e.stderr)
            raise SystemExit(EXIT_ERROR)

    except Exception as e:
        # Any other unexpected error
        click.echo(f"Error: {e}")
        raise SystemExit(EXIT_ERROR)
