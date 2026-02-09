import click
from ..app import AppContext


@click.group()
@click.pass_obj
def commit(app: AppContext):
    """Commit-related commands."""
    pass


@commit.command()
@click.pass_obj
def solution(app: AppContext):
    """Commit the solution."""
    try:
        app.commit_solution()
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@commit.command()
@click.pass_obj
def conflict(app: AppContext):
    """Commit the conflict."""
    try:
        app.commit_conflict()
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)

@commit.command()
@click.pass_obj
def merge(app: AppContext):
    """Commit the merge."""
    try:
        app.commit_merge()
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@commit.command()
@click.pass_obj
def squash(app: AppContext):
    """Squash solution commits into the merge commit.

    This command squashes all commits from HEAD back to the merge commit
    (defined in merge_info) into a single merge commit. The resulting commit
    will have two parents (target_branch_sha and merge_commit) and will contain:

    - A combined commit message with summaries from all solution commits
    - A combined git note merging all notes from squashed commits

    This is typically used in CI workflows after solution PRs are merged into
    the conflict branch.
    """
    try:
        app.squash_to_merge()
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)