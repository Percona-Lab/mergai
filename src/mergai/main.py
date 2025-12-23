import click
import logging
from .app import AppContext
import git

LOG_FORMAT = "[%(levelname)s] %(message)s"

from dotenv import load_dotenv

load_dotenv()


def register_commands(cli):
    from .commands.conflict_context import create_conflict_context

    cli.add_command(create_conflict_context)

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

    from .commands.repo import (
        commit,
        commit_conflict,
        get_merge_conflict,
        cherry_pick_solution,
        finalize,
        add_note,
        update,
        push,
    )

    cli.add_command(commit)
    cli.add_command(commit_conflict)
    cli.add_command(get_merge_conflict)
    cli.add_command(cherry_pick_solution)
    cli.add_command(finalize)
    cli.add_command(add_note)
    cli.add_command(update)
    cli.add_command(push)


@click.group()
@click.pass_obj
@click.option(
    "--repo-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
    help="Path to the git repository",
)
def cli(app: AppContext, repo_path: str = "."):
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
