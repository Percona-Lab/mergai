"""Context management commands for merge workflows.

This module provides commands for creating and managing merge context:

- init: Initialize merge context with commit and target branch info
- create conflict: Create context for merge conflicts
- create merge: Create context for successful automatic merges
"""

import click
from typing import Optional

from ..app import AppContext
from .. import git_utils
from ..util import BranchNameBuilder


CREATE_CONFLICT_CONTEXT_FLAGS = [
    click.option(
        "--use-diffs/--no-diffs",
        "use_diffs",
        default=True,
        help="Include diffs in the conflict context.",
    ),
    click.option(
        "--diff-lines-of-context",  # TODO: requires use_diffs=True
        "diff_lines_of_context",
        default=0,
        type=int,
        help="Number of lines of context to include in diffs.",
    ),
    click.option(
        "--use-compressed-diffs/--no-compressed-diffs",
        "use_compressed_diffs",
        default=True,
        help="Use compressed diffs to limit size.",
    ),
    click.option(
        "--use-their-commits/--no-their-commits",
        "use_their_commits",
        default=True,
        help="Include their commits in the conflict context.",
    ),
]


def conflict_context_flags(func):
    """Decorator to add conflict context flags to a command."""
    for option in reversed(CREATE_CONFLICT_CONTEXT_FLAGS):
        func = option(func)
    return func


@click.group()
def context():
    """Commands for managing merge context.

    These commands help create and manage context information
    for merge operations, including conflict resolution context
    and successful merge tracking.

    \b
    Subcommands:
    - init: Initialize merge context with commit and target branch
    - create conflict: Create context for merge conflicts
    - create merge: Create context for successful automatic merges
    """
    pass


@context.command()
@click.pass_obj
@click.argument("commit", type=str, required=False, default=None)
@click.option(
    "--target",
    "-t",
    type=str,
    default=None,
    help="Target branch name (default: extracted from current branch or current branch name)",
)
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing merge_info.",
)
def init(
    app: AppContext,
    commit: Optional[str],
    target: Optional[str],
    force: bool,
):
    """Initialize merge context with commit and target branch info.

    Prepares the note.json with information about the merge operation
    including the merging commit and target branch.

    COMMIT is the commit SHA or ref being merged. If omitted, the command will:
    1. If on a mergai branch, extract commit info from the branch name
    2. Scan commits for existing mergai notes and rebuild note.json from them

    This command should be run before starting a merge operation to
    establish the context for subsequent merge-related commands.

    \b
    Examples:
        mergai context init abc1234
        mergai context init abc1234 --target v8.0
        mergai context init HEAD
        mergai context init              # rebuild from commits or use branch info
        mergai context init -f           # force overwrite existing merge_info
    """
    # For init, we only use branch parsing for defaults (not note.json)
    # since we're creating/overwriting merge_info
    current_branch = git_utils.get_current_branch(app.repo)
    parsed = BranchNameBuilder.parse_branch_name_with_config(
        current_branch, app.config.branch
    )

    # If no commit argument provided and no target, try to rebuild from commits
    if commit is None and target is None:
        if parsed is not None:
            # On a mergai branch - can use branch info or rebuild
            # Try to rebuild from commits first
            try:
                note = app.rebuild_note_from_commits()
                app.save_note(note)

                merge_info = note.get("merge_info", {})
                click.echo("Context created from commit notes:")
                click.echo(f"  target_branch: {merge_info.get('target_branch', 'N/A')}")
                click.echo(f"  target_branch_sha: {merge_info.get('target_branch_sha', 'N/A')}")
                click.echo(f"  merge_commit: {merge_info.get('merge_commit', 'N/A')}")
                if "solutions" in note:
                    committed = len(app._get_committed_solution_indices(note))
                    total = len(note["solutions"])
                    click.echo(f"  solutions: {total} ({committed} committed)")
                if "conflict_context" in note:
                    click.echo("  conflict_context: present")
                if "merge_context" in note:
                    click.echo("  merge_context: present")
                return
            except click.ClickException:
                # Fall back to using branch info
                pass
        else:
            # Not on a mergai branch, try to rebuild from commits
            try:
                note = app.rebuild_note_from_commits()
                app.save_note(note)

                merge_info = note.get("merge_info", {})
                click.echo("Rebuilt note.json from commit notes:")
                click.echo(f"  target_branch: {merge_info.get('target_branch', 'N/A')}")
                click.echo(f"  target_branch_sha: {merge_info.get('target_branch_sha', 'N/A')}")
                click.echo(f"  merge_commit: {merge_info.get('merge_commit', 'N/A')}")
                if "solutions" in note:
                    committed = len(app._get_committed_solution_indices(note))
                    total = len(note["solutions"])
                    click.echo(f"  solutions: {total} ({committed} committed)")
                if "conflict_context" in note:
                    click.echo("  conflict_context: present")
                if "merge_context" in note:
                    click.echo("  merge_context: present")
                return
            except click.ClickException as e:
                raise click.ClickException(
                    f"Cannot rebuild context from commits: {e.message}\n\n"
                    "COMMIT is required when not on a mergai branch and no commit notes exist."
                )

    # Resolve commit to full SHA
    if commit is not None:
        try:
            resolved = app.repo.commit(commit)
            merge_commit_sha = resolved.hexsha
        except Exception as e:
            raise click.ClickException(f"Invalid commit reference '{commit}': {e}")
    elif parsed is not None:
        # Resolve the SHA from branch name to full SHA
        try:
            resolved = app.repo.commit(parsed.merge_commit_sha)
            merge_commit_sha = resolved.hexsha
        except Exception as e:
            raise click.ClickException(
                f"Cannot resolve commit from branch name '{parsed.merge_commit_sha}': {e}"
            )
    else:
        raise click.ClickException(
            "COMMIT is required when not on a mergai branch."
        )

    # Resolve target branch
    if target is not None:
        target_branch = target
    elif parsed is not None:
        target_branch = parsed.target_branch
    else:
        target_branch = current_branch

    # Resolve target branch SHA
    # Uses fallback to origin/ for remote-only branches (common in CI)
    try:
        target_branch_sha = git_utils.resolve_ref_sha(app.repo, target_branch)
    except ValueError as e:
        raise click.ClickException(str(e))

    note = app.load_or_create_note()

    if "merge_info" in note and not force:
        raise click.ClickException(
            "merge_info already exists in the note. Use -f/--force to overwrite."
        )

    merge_info = {
        "target_branch": target_branch,
        "target_branch_sha": target_branch_sha,
        "merge_commit": merge_commit_sha,
    }

    note["merge_info"] = merge_info
    app.save_note(note)

    click.echo("Initialized merge context:")
    click.echo(f"  target_branch: {target_branch}")
    click.echo(f"  target_branch_sha: {target_branch_sha}")
    click.echo(f"  merge_commit: {merge_commit_sha}")


@context.group()
def create():
    """Create various types of merge context.

    \b
    Subcommands:
    - conflict: Create context for merge conflicts
    - merge: Create context for successful automatic merges
    """
    pass


@create.command(name="conflict")
@click.pass_obj
@conflict_context_flags
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing saved conflict context.",
)
def create_conflict(
    app: AppContext,
    use_diffs: bool,
    diff_lines_of_context: int,
    use_compressed_diffs: bool,
    use_their_commits: bool,
    force: bool,
):
    """Create conflict context from current merge state.

    Captures information about the current merge conflict including
    conflicting files, diffs, and commit information.

    This command should be run when a merge has resulted in conflicts
    and you want to capture the context for AI-assisted resolution.

    \b
    Examples:
        mergai context create conflict
        mergai context create conflict --no-diffs
        mergai context create conflict -f
        mergai context create conflict --diff-lines-of-context 3
    """
    try:
        if not git_utils.is_merge_conflict_style_diff3(app.repo):
            click.echo(
                "Warning: Git is not configured to use diff3 for merges. "
                "It's recommended to set 'merge.conflictstyle' to 'diff3' "
                "for better conflict resolution context.\n\n"
                "You can set it globally with:\n"
                "  git config --global merge.conflictstyle diff3\n"
                "or locally in the repository with:\n"
                "  git config merge.conflictstyle diff3"
            )
        app.create_conflict_context(
            use_diffs,
            diff_lines_of_context,
            use_compressed_diffs,
            use_their_commits,
            force,
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@create.command(name="merge")
@click.pass_obj
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing merge context.",
)
def create_merge(app: AppContext, force: bool):
    """Create context for a successful automatic merge.

    Captures the list of commits being merged and identifies which
    important files (from config) were modified. This is useful for
    AI agents to generate merge descriptions and summaries.

    This command requires merge_info to be initialized first
    (via 'mergai context init').

    \b
    Examples:
        mergai context init abc1234 --target v8.0
        mergai context create merge
        mergai context create merge -f  # force overwrite
    """
    try:
        context = app.create_merge_context(force)
        click.echo("Created merge context:")
        click.echo(f"  merge_commit: {context['merge_commit']}")
        click.echo(f"  merged_commits: {len(context['merged_commits'])} commits")
        if context["important_files_modified"]:
            click.echo(
                f"  important_files_modified: {', '.join(context['important_files_modified'])}"
            )
        else:
            click.echo("  important_files_modified: (none)")
    except Exception as e:
        raise click.ClickException(str(e))


@context.command()
@click.pass_obj
@click.argument(
    "part",
    type=click.Choice(["conflict", "solution", "pr_comments", "user_comment", "merge_info", "merge_context"]),
    required=False,
    default=None,
)
@click.option(
    "--all",
    "drop_all_solutions",
    is_flag=True,
    default=False,
    help="When dropping solution, drop all solutions including committed ones.",
)
def drop(app: AppContext, part: Optional[str], drop_all_solutions: bool):
    """Drop all or part of the stored context.

    Without arguments, drops all context (removes the entire note).
    With an argument, drops only the specified part of the context.

    \b
    Parts that can be dropped:
    - conflict: The conflict context (file diffs, conflict markers, etc.)
    - solution: The generated/stored solution(s). By default only drops uncommitted
                solutions. Use --all to drop all solutions including committed ones.
    - pr_comments: PR comments added to the context
    - user_comment: User-provided comments
    - merge_info: Merge initialization info (target branch, commit)
    - merge_context: Merge context (list of merged commits, important files)

    \b
    Examples:
        mergai context drop              # drops everything
        mergai context drop conflict     # drops only conflict_context
        mergai context drop solution     # drops only uncommitted solution
        mergai context drop solution --all  # drops all solutions
        mergai context drop pr_comments  # drops only PR comments
    """
    try:
        if part is None:
            app.drop_all()
            click.echo("Dropped all context.")
        elif part == "conflict":
            app.drop_conflict_context()
            click.echo("Dropped conflict context.")
        elif part == "solution":
            app.drop_solution(all=drop_all_solutions)
            if drop_all_solutions:
                click.echo("Dropped all solutions.")
            else:
                click.echo("Dropped uncommitted solutions.")
        elif part == "pr_comments":
            app.drop_pr_comments()
            click.echo("Dropped PR comments.")
        elif part == "user_comment":
            app.drop_user_comment()
            click.echo("Dropped user comment.")
        elif part == "merge_info":
            app.drop_merge_info()
            click.echo("Dropped merge info.")
        elif part == "merge_context":
            app.drop_merge_context()
            click.echo("Dropped merge context.")
        else:
            raise Exception(f"Invalid part: {part}")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
