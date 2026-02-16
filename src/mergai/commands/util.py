import click
import json
from git import Commit
from .. import git_utils
from .. import util
from ..app import AppContext, convert_note
from ..models import ConflictContext


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
    "--user-comment",
    "show_user_comment",
    is_flag=True,
    default=False,
    show_default=False,
    help="Show the user comment.",
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
    type=click.Choice(["json", "markdown", "text"]),
    default="text",
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
    show_user_comment: bool,
    show_raw: bool,
    pretty: bool,
    format: str,
    commit: str,
):
    try:
        commit = "HEAD" if commit is None else commit

        note = app.get_note_from_commit(commit)
        if not note:
            raise Exception(f"No note found for commit {commit}.")
        show_summary = (
            not (
                show_solution
                or show_context
                or show_prompt
                or show_pr_comments
                or show_user_comment
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

        # TODO: add support for MergeContext
        if show_context:
            context_dict = note.get("conflict_context")
            if not context_dict:
                raise Exception("No context found in the note.")

            context = ConflictContext.from_dict(context_dict, app.repo)
            output_str += util.conflict_context_to_str(context, format, pretty=pretty)

        if show_pr_comments:
            pr_comments = note.get("pr_comments")
            if not pr_comments:
                raise Exception("No PR comments found in the note.")

            output_str += util.pr_comments_to_str(pr_comments, format, pretty=pretty)

        if show_user_comment:
            user_comment = note.get("user_comment")
            if not user_comment:
                raise Exception("No user comment found in the note.")

            output_str += util.user_comment_to_str(user_comment, format, pretty=pretty)

        if show_solution:
            # Handle both legacy "solution" and new "solutions" array
            if "solutions" in note:
                solutions = note["solutions"]
                if not solutions:
                    raise Exception("No solutions found in the note.")
                for idx, solution in enumerate(solutions):
                    if len(solutions) > 1:
                        if format == "markdown":
                            output_str += f"## Solution {idx + 1}\n\n"
                        else:
                            output_str += f"=== Solution {idx + 1} ===\n"
                    output_str += util.conflict_solution_to_str(
                        solution, format, pretty=pretty
                    )
            elif "solution" in note:
                solution = note["solution"]
                output_str += util.conflict_solution_to_str(
                    solution, format, pretty=pretty
                )
            else:
                raise Exception("No solution found in the note.")

        if output_str:
            util.print_or_page(output_str, format=format)

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
            convert_note(note, format=format, repo=app.repo, pretty=pretty),
            format=format,
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
@click.argument("ref", type=str, required=False, default="HEAD")
def log(app: AppContext, ref: str):
    def format_commit(commit: Commit) -> str:
        output_str = ""
        output_str += f"commit: {commit.hexsha}\n"
        output_str += f"Author: {commit.author.name} <{commit.author.email}>\n"
        output_str += f"Date:   {commit.authored_datetime}\n"
        output_str += "Content:\n"
        output_str += "  (no note)\n"
        output_str += (
            f"Message:\n    {commit.message.strip().replace('\n', '\n    ')}\n"
        )
        return output_str

    output_str = ""
    merge_commit = None
    for commit in app.repo.iter_commits(ref):
        note = app.try_get_note_from_commit(commit)
        if not merge_commit and note and note.merge_info:
            merge_commit = note.merge_info.merge_commit_sha

        if note:
            output_str += util.commit_note_to_summary_str(
                commit, note.to_dict(), format="text"
            )
        else:
            output_str += format_commit(commit)

        # Show log until the merge commit is reached
        if git_utils.is_merge_commit_parent(commit, merge_commit):
            break

    util.print_or_page(output_str, format="text")


@click.command()
@click.pass_obj
def prompt(app: AppContext):
    try:
        note = app.load_note()
        if note is None:
            click.echo("No note found. Please prepare the context first.")
            click.echo("Use `mergai create-conflict-context` to add conflict context.")
            click.echo(
                "Use `mergai pr-add-comments-to-context` to add PR comments to the context."
            )
            exit(1)
        prompt = app.build_resolve_prompt(note)
        util.print_or_page(prompt, format="markdown")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
def merge_prompt(app: AppContext):
    try:
        note = app.load_note()
        if note is None:
            click.echo("No note found. Please prepare the context first.")
            click.echo("Use `mergai create-conflict-context` to add conflict context.")
            click.echo(
                "Use `mergai pr-add-comments-to-context` to add PR comments to the context."
            )
            exit(1)
        prompt = app.build_describe_prompt(note)
        util.print_or_page(prompt, format="markdown")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


COMMENT_FILE_TEMPLATE = """\

# MergAI comment
#
# Please write your comment below. Lines starting with '#' will be ignored.
# An empty comment will abort the operation.
#
# TODO:
# - add support for having comments per file
"""


def strip_comment_lines(edited: str) -> str:
    lines = edited.splitlines()
    stripped_lines = [line for line in lines if not line.strip().startswith("#")]
    return "\n".join(stripped_lines).strip()


def now_utc_iso() -> str:
    from datetime import datetime
    from datetime import timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_cur_comment(c: dict) -> str:
    if not c:
        return ""
    return f"# Date: {c.get('date')}\n# User: {c.get('user', '')} [{c.get('email')}]\n\n{c.get('body') or ''}\n"


def get_comment_from_cli(body: str, file: str) -> str:
    parts = []
    if body:
        parts.append(body)
    if file:
        with open(file, "r") as f:
            parts.append("```")
            parts.append(f.read())
            parts.append("```")
    stripped = "\n".join(parts).strip()
    return stripped


@click.command()
@click.pass_obj
@click.option(
    "--file",
    type=click.Path(exists=True),
    help="Path to a file containing the comment.",
)
@click.option(
    "-f/--force", "force", is_flag=True, default=False, help="Force overwrite."
)
@click.argument("body", required=False)
def comment(app: AppContext, file: str, force: bool, body: str):
    note = app.load_or_create_note()
    # TODO: support multiple comments
    user_comment = note.get("user_comment", "")
    cur_comment = ""
    if user_comment:
        cur_comment = get_cur_comment(user_comment)

    if (body or file) and user_comment and not force:
        raise click.ClickException(
            "Comment already exists. Use -f/--force to overwrite."
        )

    if body or file:
        stripped = get_comment_from_cli(body, file)
    else:
        edited = click.edit(cur_comment + COMMENT_FILE_TEMPLATE + "\n", extension=".sh")
        if edited is None:
            raise click.ClickException("No comment provided, aborting.")
        stripped = strip_comment_lines(edited)

    if not stripped:
        click.echo("Empty comment, cancelling.")
        exit(0)

    comment = {
        "user": app.repo.git.config("user.name"),
        "email": app.repo.git.config("user.email"),
        "date": now_utc_iso(),
        "body": stripped,
    }

    note["user_comment"] = comment

    app.state.save_note(note)
