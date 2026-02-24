import click
import logging
from .app import AppContext
from .config import load_config, DEFAULT_CONFIG_PATH
from .version import __version__
import git

LOG_FORMAT = "[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s"

from dotenv import load_dotenv

load_dotenv()


def register_commands(cli):
    from .commands.context import context

    cli.add_command(context)

    from .commands.resolve import resolve

    cli.add_command(resolve)

    from .commands.describe import describe

    cli.add_command(describe)

    from .commands.pr import pr

    cli.add_command(pr)

    from .commands.util import show, status, log, prompt, comment, merge_prompt

    cli.add_command(show)
    cli.add_command(status)
    cli.add_command(log)
    cli.add_command(prompt)
    cli.add_command(comment)
    cli.add_command(merge_prompt)

    from .commands.commit import commit

    cli.add_command(commit)

    from .commands.notes import notes

    cli.add_command(notes)

    from .commands.fork import fork

    cli.add_command(fork)

    from .commands.branch import branch

    cli.add_command(branch)

    from .commands.merge import merge

    cli.add_command(merge)

    from .commands.finalize import finalize

    cli.add_command(finalize)

    from .commands.config_cmd import config

    cli.add_command(config)


@click.group()
@click.version_option(version=__version__, prog_name="mergai")
@click.pass_obj
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(),
    default=None,
    help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
)
@click.option(
    "--repo-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
    help="Path to the git repository",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity level",
)
def cli(app: AppContext, config_path: str, repo_path: str = ".", verbose: int = 0):
    try:
        app.config = load_config(config_path)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Error loading config: {e}")
    app.repo = git.Repo(repo_path)
    logging.basicConfig(
        level=max(logging.WARNING - (10 * verbose), logging.DEBUG),
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():

    register_commands(cli)
    cli(obj=AppContext())


if __name__ == "__main__":
    main()
