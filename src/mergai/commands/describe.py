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
    "-y/--yolo",
    "yolo",
    is_flag=True,
    default=False,
    help=(
        "Enable YOLO mode. Required for the agent to inspect the real diff "
        "(run git) while describing; the no-file-modification check still guards "
        "against repo changes."
    ),
)
@click.option(
    "--verify/--no-verify",
    "verify",
    default=True,
    help=(
        "Run a second agent pass that fact-checks the description against the "
        "actual diff and regenerates it when unsupported claims are found "
        "(enabled by default)."
    ),
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
    yolo: bool,
    verify: bool,
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
            force,
            max_attempts=app.config.resolve.max_attempts,
            agent_desc=agent,
            yolo=yolo,
            verify=verify,
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
