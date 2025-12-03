import click
from ..app import AppContext
from .. import git_utils


@click.command()
@click.pass_obj
@click.argument("revision", default="HEAD")
@click.option(
    "--max-count",
    default=0,
    help="Maximum number of merge commits to retrieve. 0 means no limit.",
)
def get_merge_conflicts(app: AppContext, revision, max_count):
    for commit in git_utils.get_merge_conflicts(app.repo, revision, max_count):
        print(f"{commit.hexsha} {commit.parents[0].hexsha} {commit.parents[1].hexsha}")
