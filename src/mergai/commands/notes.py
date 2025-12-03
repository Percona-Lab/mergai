import click
import json
from .. import util
from ..app import AppContext, convert_note


@click.command()
@click.pass_obj
@click.option(
    "--summary",
    "show_summary",
    is_flag=True,
    default=False,
    show_default=False,
    help="Show summary information.",
)
@click.option(
    "--raw",
    "show_raw",
    is_flag=True,
    default=False,
    show_default=False,
    help="Show raw note data.",
)
@click.option(
    "--solution",
    "show_solution",
    is_flag=True,
    default=False,
    show_default=False,
    help="Show the conflict solution.",
)
@click.option(
    "--context",
    "show_context",
    is_flag=True,
    default=False,
    show_default=False,
    help="Show the context.",
)
@click.option(
    "--pr-comments",
    "show_pr_comments",
    is_flag=True,
    default=False,
    show_default=False,
    help="Show the PR comments.",
)
@click.option(
    "--prompt",
    "show_prompt",
    is_flag=True,
    default=False,
    show_default=False,
    help="Show the prompt.",
)
@click.option("--pretty", is_flag=True, help="Pretty-print the JSON output.")
@click.option(
    "--format",
    type=click.Choice(["json", "markdown"]),
    default="markdown",
    help="Output format.",
)
@click.argument(
    "commit",
    type=str,
    required=False,
    default=None,
)
# TODO: refactor
def show(
    app: AppContext,
    show_summary: bool,
    show_solution: bool,
    show_context: bool,
    show_prompt: bool,
    show_pr_comments: bool,
    show_raw: bool,
    pretty: bool,
    format: str,
    commit: str,
):
    try:
        commit = "HEAD" if commit is None else commit

        note = app.read_note(commit)
        show_summary = (
            not (
                show_solution
                or show_context
                or show_prompt
                or show_pr_comments
                or show_raw
            )
            or show_summary
        )

        output_str = ""

        if show_raw:
            import json

            json_str = json.dumps(note, indent=2 if pretty else None)
            if format == "markdown":
                raise Exception("Raw output is only available in JSON format.")

            output_str += json_str + "\n"

        if show_summary:
            # TODO: improve summary format, move to separate util function
            output_str += "# Summary\n"
            output_str += util.commit_note_to_summary_str(
                app.repo.commit(commit), note, format=format, pretty=pretty
            )

        if show_context:
            context = note.get("conflict_context")
            if not context:
                raise Exception("No context found in the note.")

            output_str += util.conflict_context_to_str(context, format, pretty=pretty)

        if show_pr_comments:
            pr_comments = note.get("pr_comments")
            if not pr_comments:
                raise Exception("No PR comments found in the note.")

            output_str += util.pr_comments_to_str(pr_comments, format, pretty=pretty)

        if show_solution:
            solution = note.get("solution")
            if not solution:
                raise Exception("No solution found in the note.")

            output_str += util.conflict_solution_to_str(solution, format, pretty=pretty)

        if output_str:
            util.print_or_page(output_str, format=format)

    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


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
@click.option("--pretty", is_flag=True, help="Pretty-print the JSON output.")
@click.option(
    "--format",
    type=click.Choice(["json", "markdown"]),
    default="markdown",
    help="Output format.",
)
def status(app: AppContext, format: str, pretty: bool):
    try:
        if not app.state.note_exists():
            click.echo("No note found in the state store.")
            exit(0)
        note = app.state.load_note()
        util.print_or_page(
            convert_note(note, format=format, pretty=pretty), format=format
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
def log(app: AppContext):
    try:
        notes = app.get_notes()
        output_str = ""
        for idx, (commit, note) in enumerate(notes):
            output_str += util.commit_note_to_summary_str(commit, note, format="text")

        util.print_or_page(output_str, format="text")

    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
@click.option(
    "--use-history/--no-history",
    "use_history",
    is_flag=True,
    default=False,
    help="Include commit history in the prompt.",
)
def prompt(app: AppContext, use_history: bool):
    try:
        note = app.load_note()
        if note is None:
            click.echo("No note found. Please prepare the context first.")
            click.echo("Use `mergai create-conflict-context` to add conflict context.")
            click.echo(
                "Use `mergai pr-add-comments-to-context` to add PR comments to the context."
            )
            exit(1)
        prompt = app.build_prompt(note, use_history=use_history)
        util.print_or_page(prompt, format="markdown")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
@click.argument(
    "choice",
    type=click.Choice(["all", "solution", "context", "pr_comments"]),
    required=False,
    default="all",
)
def drop(app: AppContext, choice: str):
    try:
        # click.echo(f"Dropping {choice}...")
        if choice == "all":
            app.drop_all()
        elif choice == "solution":
            app.drop_solution()
        elif choice == "context":
            app.drop_conflict_context()
        elif choice == "pr_comments":
            app.drop_pr_comments()
        else:
            raise Exception(f"Invalid choice: {choice}")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
    pass
