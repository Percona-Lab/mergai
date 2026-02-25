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


@commit.command()
@click.pass_obj
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Re-sync commits that already have notes attached (overwrite existing).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be synced without making changes.",
)
def sync(app: AppContext, force: bool, dry_run: bool):
    """Sync human commits to the note.

    Scans commits from HEAD to target_branch_sha and creates solution entries
    for commits that don't have solutions attached. This is useful when humans
    make additional commits to fix unresolved conflicts or modify files after
    the AI resolution.

    Each human commit becomes a separate solution entry with:
    - Summary from the commit message's first line
    - Resolved files (files modified without conflict markers)
    - Unresolved files (files that still contain conflict markers)
    - Author information (name, email, type: "human")

    The command attaches git notes to each synced commit, similar to how
    'mergai commit solution' works for AI solutions.

    \b
    Examples:
        mergai commit sync              # Sync untracked human commits
        mergai commit sync --force      # Re-sync all human commits
        mergai commit sync --dry-run    # Preview what would be synced
    """
    try:
        app.sync_human_commits(force=force, dry_run=dry_run)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
