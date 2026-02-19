import click
from ..app import AppContext
from .context import conflict_context_flags


@click.command()
@click.pass_obj
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing saved conflict context, conflict prompt and solution.",
)
@click.option(
    "-y/--yolo",
    "yolo",
    is_flag=True,
    default=False,
    help="Enable YOLO mode.",
)
def resolve(
    app: AppContext,
    force: bool,
    yolo: bool,
):
    try:
        app.resolve(force, yolo)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
