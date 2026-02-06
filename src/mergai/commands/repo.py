import click
from ..app import AppContext
from .. import git_utils


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
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Replace uncommitted solution if exists",
)
@click.argument("commit", required=True)
def cherry_pick_solution(app: AppContext, commit: str, force: bool):
    """Copy a solution from another commit's note into the current note.

    The solution is appended to the solutions array. If there's already an
    uncommitted solution and --force is used, it will be replaced.

    COMMIT is the commit SHA or ref to copy the solution from.
    """
    try:
        note = app.read_note(commit)
        if not note:
            click.echo(f"No note found for commit {commit}.")
            exit(1)

        solution = note.get("solution")
        if not solution:
            click.echo(f"No solution found in commit note for {commit}.")
            exit(1)

        cur_note = app.load_or_create_note()

        # Migrate legacy solution field
        cur_note = app._migrate_solution_to_solutions(cur_note)

        # Check for uncommitted solution
        uncommitted_idx = app._get_uncommitted_solution_index(cur_note)
        if uncommitted_idx is not None and not force:
            click.echo(
                "An uncommitted solution already exists in the current note. Use --force to replace it."
            )
            exit(1)

        # Initialize solutions array if needed
        if "solutions" not in cur_note:
            cur_note["solutions"] = []

        if uncommitted_idx is not None and force:
            # Replace the uncommitted solution
            cur_note["solutions"][uncommitted_idx] = solution
            click.echo(f"Replaced uncommitted solution (index {uncommitted_idx}) with solution from {commit}.")
        else:
            # Append new solution
            cur_note["solutions"].append(solution)
            click.echo(f"Added solution from {commit} as solutions[{len(cur_note['solutions']) - 1}].")

        app.save_note(cur_note)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Force update (overwrite local notes)",
)
@click.argument("remote", default="origin")
def update(app: AppContext, remote: str, force: bool):
    try:
        refspec = "refs/notes/mergai*:refs/notes/mergai*"
        if force:
            refspec = "+" + refspec
        app.get_repo().git.fetch(remote, refspec)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
@click.argument("remote", default="origin")
def push(app: AppContext, remote: str):
    try:
        app.get_repo().git.push(remote, "refs/notes/mergai*:refs/notes/mergai*")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing note",
)
@click.argument("ref", default="HEAD")
def add_note(app: AppContext, ref: str, force: bool):
    try:
        commit = app.get_repo().commit(ref)
        note = app.read_note(ref)
        if note is not None and not force:
            click.echo(
                f"A note already exists for commit {commit}. Use --force to overwrite."
            )
            exit(1)

        app.add_note(commit)
        app.drop_all()
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


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
