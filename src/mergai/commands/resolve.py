import click

from ..app import AppContext


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
@click.option(
    "--agent",
    "-a",
    "agent",
    type=str,
    default=None,
    help="Override the agent:model to use (e.g., 'gemini-cli:gemini-2.5-pro').",
)
def resolve(
    app: AppContext,
    force: bool,
    yolo: bool,
    agent: str | None,
):
    try:
        app.resolve(force, yolo, agent_desc=agent)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
