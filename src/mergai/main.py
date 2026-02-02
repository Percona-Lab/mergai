import click
import logging
from .app import AppContext
from .config import load_config
import git

LOG_FORMAT = "[%(levelname)s] %(message)s"

from dotenv import load_dotenv

load_dotenv()


def register_commands(cli):
    from .commands.context import context

    cli.add_command(context)

    from .commands.resolve import resolve

    cli.add_command(resolve)

    from .commands.utils import get_merge_conflicts

    cli.add_command(get_merge_conflicts)

    from .commands.replay import replay

    cli.add_command(replay)

    from .commands.pr import pr

    cli.add_command(pr)

    from .commands.notes import show, status, log, prompt, drop, comment

    cli.add_command(show)
    cli.add_command(status)
    cli.add_command(log)
    cli.add_command(prompt)
    cli.add_command(drop)
    cli.add_command(comment)

    from .commands.commit import commit

    cli.add_command(commit)

    from .commands.repo import (
        get_merge_conflict,
        cherry_pick_solution,
        finalize,
        add_note,
        update,
        push,
    )
    cli.add_command(get_merge_conflict)
    cli.add_command(cherry_pick_solution)
    cli.add_command(finalize)
    cli.add_command(add_note)
    cli.add_command(update)
    cli.add_command(push)

    from .commands.fork import fork
    cli.add_command(fork)

    from .commands.branch import branch
    cli.add_command(branch)

@click.group()
@click.pass_obj
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(),
    default=None,
    help="Path to config file (default: .mergai.yaml)",
)
@click.option(
    "--repo-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
    help="Path to the git repository",
)
def cli(app: AppContext, config_path: str, repo_path: str = "."):
    try:
        app.config = load_config(config_path)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Error loading config: {e}")
    app.repo = git.Repo(repo_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )
    register_commands(cli)
    cli(obj=AppContext())


if __name__ == "__main__":
    main()
