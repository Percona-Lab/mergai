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
    for option in reversed(CONTEXT_FLAGS):
        func = option(func)
    return func


@click.command()
@click.pass_obj
@conflict_context_flags
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing saved conflict context.",
)
def create_conflict_context(
    app: AppContext,
    use_diffs: bool,
    diff_lines_of_context: int,
    use_compressed_diffs: bool,
    use_their_commits: bool,
    force: bool,
):
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
