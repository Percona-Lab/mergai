import click

from ..app import AppContext


@click.command()
@click.pass_obj
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing merge description.",
)
@click.option(
    "--agent",
    "-a",
    "agent",
    type=str,
    default=None,
    help="Override the agent:model to use (e.g., 'gemini-cli:gemini-2.5-pro').",
)
def describe(
    app: AppContext,
    force: bool,
    agent: str | None,
):
    """Generate a description of the merge based on the note context.

    This command uses an AI agent to analyze the merge context and generate
    a description without modifying any files. The description is stored
    in the note as 'merge_description'.
    """
    click.echo(
        click.style("WARNING: ", fg="yellow")
        + "The 'describe' command is experimental and may change in future versions."
    )
    click.echo("")
    try:
        app.describe(
            force, max_attempts=app.config.resolve.max_attempts, agent_desc=agent
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
