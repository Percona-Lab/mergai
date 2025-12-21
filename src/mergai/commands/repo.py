import click
from ..app import AppContext
from .. import git_utils


@click.command()
@click.pass_obj
def commit(app: AppContext):
    try:
        app.commit()
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
def commit_conflict(app: AppContext):
    try:
        app.commit_conflict()
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
@click.argument("revision", default="HEAD")
def get_merge_conflict(app: AppContext, revision: str):
    (commit, conflict_context) = app.get_merge_conflict(revision)

    if not conflict_context:
        click.echo("No conflict context found in any commit notes.")
        exit(1)

    if not commit:
        ours = conflict_context["ours_commit"]["hexsha"]
        theirs = conflict_context["theirs_commit"]["hexsha"]
        base = conflict_context["base_commit"]["hexsha"]
        click.echo(
            f"No merge commit found matching the conflict context (ours: {ours}, theirs: {theirs}, base: {base})."
        )
        exit(1)

    click.echo(commit.hexsha)


@click.command()
@click.pass_obj
def finalize(app: AppContext):
    (commit, conflict_context) = app.get_merge_conflict("HEAD")
    if not commit:
        click.echo("No merge commit found with conflict context.")
        exit(1)

    def get_merge_commit_message(commit, conflict_context) -> str:
        theirs = commit.parents[1]
        message = f"MergaAI: Merge commit '{theirs}'\n\n"
        message += "Conflicts:\n"
        for path in conflict_context["files"]:
            message += f"        {path}\n"
        message += "\nModified:\n"
        message += "        TODO\n"
        message += "\n"
        # TODO: add more details
        message += "MergAI-Note: TODO\n"
        message += "Reviewed-by: TODO\n"
        message += "Approved-by: TODO\n"

        return message

    message = get_merge_commit_message(commit, conflict_context)

    app.get_repo().git.reset("--soft", commit)
    app.get_repo().git.commit("--amend", "-m", message)
