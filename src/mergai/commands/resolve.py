import click
from ..app import AppContext
from .. import util
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
    "--use-history/--no-history",
    "use_history",
    is_flag=True,
    default=False,
    help="Include commit history in the prompt.",
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
    use_history: bool,
    yolo: bool,
):
    try:
        app.resolve(force, use_history, yolo, max_attempts=app.config.resolve.max_attempts)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
