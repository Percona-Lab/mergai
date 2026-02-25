import click

from ..app import AppContext


@click.command()
@click.pass_obj
@click.option(
    "--mode",
    type=click.Choice(["squash", "keep", "fast-forward"], case_sensitive=False),
    default=None,
    help="Override finalize mode from config.",
)
def finalize(app: AppContext, mode: str):
    """Finalize the solution PR after merging.

    This command is typically run after a solution PR is merged into the
    conflict branch. Its behavior depends on the finalize.mode config:

    \b
    - squash: Squash all commits from HEAD to the merge commit into a single
              merge commit with combined notes. This creates a clean history
              with a single merge commit. (Default behavior)

    \b
    - keep: Validate the repository state and print a summary without
            modifying any commits. Useful when you want to preserve the
            individual commit history from the solution PR.

    \b
    - fast-forward: Remove the GitHub PR merge commit to simulate a fast-forward
                    merge. Keeps the original solution commits with their mergai
                    notes intact. Only acts if HEAD is a merge commit without a
                    mergai note and its first parent has a note attached.

    The mode can be overridden with the --mode option.

    \b
    Examples:
        mergai finalize                     # Use mode from config (default: squash)
        mergai finalize --mode squash       # Force squash mode
        mergai finalize --mode keep         # Force keep mode (validate only)
        mergai finalize --mode fast-forward # Remove PR merge commit
    """
    try:
        app.finalize(mode=mode)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
