"""Context management commands for merge workflows.

This module provides commands for creating and managing merge context:

- init: Initialize merge context with commit and target branch info
- create conflict: Create context for merge conflicts
- create merge: Create context for successful automatic merges
"""

import click
from ..app import AppContext
from .. import git_utils


CONTEXT_FLAGS = [
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
    for option in reversed(CONTEXT_FLAGS):
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
def init(app: AppContext):
    """Initialize merge context with commit and target branch info.

    Prepares the note.json with information about the merge operation
    including the merging commit and target branch.

    This command should be run before starting a merge operation to
    establish the context for subsequent merge-related commands.
    """
    # TODO: Implement to add merging commit and target branch to note.json
    # Design should allow easy extension for additional metadata in the future
    # Suggested fields for note.json:
    #   - merge_info.merging_commit: the commit being merged
    #   - merge_info.target_branch: the branch being merged into
    #   - merge_info.timestamp: when the merge was initiated
    #   - merge_info.* : extensible for future metadata
    raise click.ClickException(
        "Not implemented yet.\n\n"
        "TODO: Initialize merge context by storing:\n"
        "  - merging commit reference\n"
        "  - target branch name\n"
        "  - (extensible for future metadata)"
    )


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
def create_merge(app: AppContext):
    """Create context for a successful automatic merge commit.

    Captures information about a merge that completed without conflicts,
    useful for tracking merge history and context.

    This command should be run after a successful automatic merge
    to record the merge context for future reference.
    """
    # TODO: Implement context creation for successful automatic merges
    # Suggested implementation:
    #   - Verify HEAD is a merge commit
    #   - Extract parent commits information
    #   - Record merged branches/refs
    #   - Store merge metadata (timestamp, author, etc.)
    #   - Save to note.json in a format compatible with conflict context
    raise click.ClickException(
        "Not implemented yet.\n\n"
        "TODO: Create context for successful automatic merge commit:\n"
        "  - capture merge commit information\n"
        "  - record merged branches/commits\n"
        "  - store merge metadata"
    )
