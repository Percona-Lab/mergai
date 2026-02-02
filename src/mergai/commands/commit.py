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
