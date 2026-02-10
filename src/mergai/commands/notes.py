import click
from ..app import AppContext


@click.group()
@click.pass_obj
def notes(app: AppContext):
    """Manage notes."""
    pass


@notes.command()
@click.pass_obj
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Force update (overwrite local notes)",
)
@click.argument("remote", default="origin")
def update(app: AppContext, remote: str, force: bool):
    try:
        refspec = "refs/notes/mergai*:refs/notes/mergai*"
        if force:
            refspec = "+" + refspec
        app.get_repo().git.fetch(remote, refspec)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@notes.command()
@click.pass_obj
@click.argument("remote", default="origin")
def push(app: AppContext, remote: str):
    try:
        app.get_repo().git.push(remote, "refs/notes/mergai*:refs/notes/mergai*")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
