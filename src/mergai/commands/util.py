import json

import click
import git

from ..app import AppContext
from ..models import ConflictContext, MergaiNote, MergeContext, MergeInfo
from ..utils import formatters, git_utils, util
from ..utils.output import OutputFormat, format_option


@click.command()
@click.pass_obj
@click.option(
    "--summary",
    "show_summary",
    is_flag=True,
    default=False,
    help="Show summary information.",
)
@click.option(
    "--raw",
    "show_raw",
    is_flag=True,
    default=False,
    help="Show raw note data (JSON only).",
)
@click.option(
    "--merge-info",
    "show_merge_info",
    is_flag=True,
    default=False,
    help="Show merge info (target branch, merge commit).",
)
@click.option(
    "--merge-context",
    "show_merge_context",
    is_flag=True,
    default=False,
    help="Show merge context (merged commits, auto-merged files).",
)
@click.option(
    "--conflict-context",
    "show_conflict_context",
    is_flag=True,
    default=False,
    help="Show conflict context (base/ours/theirs commits, conflicted files).",
)
@click.option(
    "--solution",
    "solution_index",
    type=int,
    default=None,
    help="Show a specific solution by index (0-based).",
)
@click.option(
    "--solutions",
    "show_solutions",
    is_flag=True,
    default=False,
    help="Show all solutions.",
)
@click.option(
    "--pr-comments",
    "show_pr_comments",
    is_flag=True,
    default=False,
    help="Show the PR comments.",
)
@click.option(
    "--user-comment",
    "show_user_comment",
    is_flag=True,
    default=False,
    help="Show the user comment.",
)
@click.option(
    "--merge-description",
    "show_merge_description",
    is_flag=True,
    default=False,
    help="Show the merge description.",
)
@click.option(
    "--prompt",
    "show_prompt",
    is_flag=True,
    default=False,
    help="Show the prompt.",
)
@format_option(default=OutputFormat.TEXT)
@click.argument(
    "commit",
    type=str,
    required=False,
    default=None,
)
def show(
    app: AppContext,
    show_summary: bool,
    show_raw: bool,
    show_merge_info: bool,
    show_merge_context: bool,
    show_conflict_context: bool,
    solution_index: int,
    show_solutions: bool,
    show_pr_comments: bool,
    show_user_comment: bool,
    show_merge_description: bool,
    show_prompt: bool,
    format: str,
    commit: str,
):
    """Show note data from a commit.

    By default shows a summary. Use flags to show specific sections.

    \b
    Examples:
        mergai show                      # Show summary from HEAD
        mergai show abc123               # Show summary from specific commit
        mergai show --merge-info         # Show merge info
        mergai show --conflict-context   # Show conflict context
        mergai show --solutions          # Show all solutions
        mergai show --solution 0         # Show first solution
        mergai show --solution 1 --md    # Show second solution in markdown
    """
    try:
        commit = "HEAD" if commit is None else commit

        note = app.get_note_from_commit(commit)
        if not note:
            raise Exception(f"No note found for commit {commit}.")

        # Determine if any specific section was requested
        has_specific_request = (
            show_merge_info
            or show_merge_context
            or show_conflict_context
            or solution_index is not None
            or show_solutions
            or show_pr_comments
            or show_user_comment
            or show_merge_description
            or show_prompt
            or show_raw
        )

        # Show summary by default if no specific section requested
        show_summary = show_summary or not has_specific_request

        output_str = ""

        if show_raw:
            json_str = json.dumps(note, indent=2)
            if format == "markdown":
                raise Exception("Raw output is only available in text/JSON format.")
            output_str += json_str + "\n"

        if show_summary:
            if format == "markdown":
                output_str += "# Summary\n"
            output_str += formatters.commit_note_to_summary_str(
                app.repo.commit(commit), note, format=format, pretty=True
            )

        if show_merge_info:
            merge_info_dict = note.get("merge_info")
            if not merge_info_dict:
                raise Exception("No merge info found in the note.")
            merge_info = MergeInfo.from_dict(merge_info_dict, app.repo)
            output_str += formatters.merge_info_to_str(merge_info, format)

        if show_merge_context:
            merge_context_dict = note.get("merge_context")
            if not merge_context_dict:
                raise Exception("No merge context found in the note.")
            merge_context = MergeContext.from_dict(merge_context_dict, app.repo)
            output_str += formatters.merge_context_to_str(merge_context, format)

        if show_conflict_context:
            conflict_context_dict = note.get("conflict_context")
            if not conflict_context_dict:
                raise Exception("No conflict context found in the note.")
            conflict_context = ConflictContext.from_dict(
                conflict_context_dict, app.repo
            )
            output_str += formatters.conflict_context_to_str(
                conflict_context, format, pretty=True
            )

        if show_pr_comments:
            pr_comments = note.get("pr_comments")
            if not pr_comments:
                raise Exception("No PR comments found in the note.")
            output_str += formatters.pr_comments_to_str(
                pr_comments, format, pretty=True
            )

        if show_user_comment:
            user_comment = note.get("user_comment")
            if not user_comment:
                raise Exception("No user comment found in the note.")
            output_str += formatters.user_comment_to_str(
                user_comment, format, pretty=True
            )

        if show_merge_description:
            merge_description = note.get("merge_description")
            if not merge_description:
                raise Exception("No merge description found in the note.")
            output_str += formatters.merge_description_to_str(
                merge_description, format, pretty=True
            )

        if solution_index is not None:
            # Show a specific solution by index
            solutions = note.get("solutions", [])
            if not solutions:
                raise Exception("No solutions found in the note.")
            elif solution_index < 0 or solution_index >= len(solutions):
                raise Exception(
                    f"Solution index {solution_index} out of range. Available: 0-{len(solutions)-1}"
                )
            else:
                output_str += formatters.conflict_solution_to_str(
                    solutions[solution_index], format, pretty=True
                )

        if show_solutions:
            # Show all solutions
            solutions = note.get("solutions", [])
            if not solutions:
                raise Exception("No solutions found in the note.")
            else:
                for idx, solution in enumerate(solutions):
                    if len(solutions) > 1:
                        if format == "markdown":
                            output_str += f"## Solution {idx}\n\n"
                        else:
                            output_str += f"=== Solution {idx} ===\n"
                    output_str += formatters.conflict_solution_to_str(
                        solution, format, pretty=True
                    )

        if output_str:
            util.print_or_page(output_str, format=format)

    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


def convert_note_to_text_summary(note: dict) -> str:
    """Convert a note to a text summary format.

    Shows what fields are present in note.json and their basic info,
    including the note_index which tracks which commits have which fields.
    Also shows commit status - whether all fields are committed to git notes.

    Args:
        note: The note dict from note.json.

    Returns:
        Text summary of the note contents.
    """
    output = []

    output.append("Note Summary:")
    output.append("")

    # Merge Info
    if "merge_info" in note:
        mi = note["merge_info"]
        output.append("  Merge Info:")
        output.append(f"    Target Branch: {mi.get('target_branch', 'unknown')}")
        if mi.get("target_branch_sha"):
            output.append(
                f"    Target Branch SHA: {mi.get('target_branch_sha', 'unknown')[:11]}"
            )
        output.append(f"    Merge Commit: {mi.get('merge_commit', 'unknown')[:11]}")
        output.append("")

    # Merge Context
    if "merge_context" in note:
        mc = note["merge_context"]
        output.append("  Merge Context:")
        output.append(f"    Merged Commits: {len(mc.get('merged_commits', []))}")
        auto_merged = mc.get("auto_merged", {})
        if auto_merged:
            output.append(f"    Auto-Merged Files: {len(auto_merged.get('files', []))}")
            if auto_merged.get("strategy"):
                output.append(f"    Strategy: {auto_merged.get('strategy')}")
        important = mc.get("important_files_modified", [])
        if important:
            output.append(f"    Important Files Modified: {len(important)}")
        output.append("")

    # Conflict Context
    if "conflict_context" in note:
        cc = note["conflict_context"]
        output.append("  Conflict Context:")
        # Handle both SHA strings and dict format
        base = cc.get("base_commit", "")
        ours = cc.get("ours_commit", "")
        theirs = cc.get("theirs_commit", "")
        base_sha = base if isinstance(base, str) else base.get("hexsha", str(base))
        ours_sha = ours if isinstance(ours, str) else ours.get("hexsha", str(ours))
        theirs_sha = (
            theirs if isinstance(theirs, str) else theirs.get("hexsha", str(theirs))
        )
        output.append(f"    Base Commit: {base_sha[:11]}")
        output.append(f"    Ours Commit: {ours_sha[:11]}")
        output.append(f"    Theirs Commit: {theirs_sha[:11]}")
        output.append(f"    Conflicted Files: {len(cc.get('conflict_types', {}))}")
        output.append("")

    # Solutions
    if "solutions" in note:
        solutions = note["solutions"]
        output.append(f"  Solutions ({len(solutions)}):")
        for idx, sol in enumerate(solutions):
            response = sol.get("response", {})
            resolved = len(response.get("resolved", {}))
            unresolved = len(response.get("unresolved", {}))
            modified = len(response.get("modified", {}))
            total = resolved + unresolved

            # Author info
            if "agent_info" in sol:
                agent = sol["agent_info"]
                author = (
                    f"{agent.get('agent_type', 'unknown')} v{agent.get('version', '?')}"
                )
            elif "author" in sol:
                name = sol["author"].get("name", "unknown")
                email = sol["author"].get("email", "")
                author = f"{name} <{email}>" if email else name
            else:
                author = "unknown"

            output.append(f"    [{idx}] By: {author}")
            # Build stats line: "Resolved: X/Y" + optional ", Unresolved: Z" + optional ", Modified: M"
            stats_line = f"        Resolved: {resolved}/{total}"
            if unresolved > 0:
                stats_line += f", Unresolved: {unresolved}"
            if modified > 0:
                stats_line += f", Modified: {modified}"
            output.append(stats_line)
            if sol.get("commit_sha"):
                output.append(f"        Commit: {sol['commit_sha'][:11]}")
        output.append("")

    # PR Comments
    if "pr_comments" in note:
        comments = note["pr_comments"]
        output.append(f"  PR Comments ({len(comments)}):")
        stats = formatters.get_comments_stats(comments)
        for user, count in stats.items():
            output.append(f"    {user}: {count} comment(s)")
        output.append("")

    # User Comment
    if "user_comment" in note:
        uc = note["user_comment"]
        output.append("  User Comment:")
        output.append(f"    By: {uc.get('user', 'unknown')} <{uc.get('email', '')}>")
        output.append(f"    Date: {uc.get('date', 'unknown')}")
        output.append("")

    # Merge Description
    if "merge_description" in note:
        md = note["merge_description"]
        response = md.get("response", {})
        output.append("  Merge Description:")
        output.append(f"    Auto-Merged Files: {len(response.get('auto_merged', {}))}")
        if md.get("agent_info"):
            agent = md["agent_info"]
            output.append(
                f"    Agent: {agent.get('agent_type', 'unknown')} v{agent.get('version', '?')}"
            )
        output.append("")

    # Note Index and Commit Status - tracks which commits have which fields
    # Check commit status using MergaiNote model
    mergai_note = MergaiNote.from_dict(note)
    uncommitted_fields = mergai_note.get_uncommitted_fields()

    if "note_index" in note:
        ni = note["note_index"]
        output.append(f"  Note Index ({len(ni)} entries):")
        for entry in ni:
            sha = entry.get("sha", "unknown")[:11]
            fields = entry.get("fields", [])
            output.append(f"    {sha}: {', '.join(fields)}")
        output.append("")

    # Show commit status
    if uncommitted_fields:
        output.append("  Uncommitted Changes:")
        for field in uncommitted_fields:
            output.append(f"    - {field}")
        output.append("")
        output.append("  Run 'mergai save' to commit these changes to git notes.")
        output.append("")
    else:
        output.append("  Commit Status: All fields committed")
        output.append("")

    # MergAI Version (at the end)
    if "mergai_version" in note:
        output.append(f"  note created with mergai {note['mergai_version']}")
        output.append("")

    return "\n".join(output)


def convert_note(
    note: dict,
    format: str,
    repo: git.Repo | None = None,
    pretty: bool = False,
    show_context: bool = True,
    show_solution: bool = True,
    show_pr_comments: bool = True,
    show_user_comment: bool = True,
    show_summary: bool = True,
    show_merge_info: bool = True,
    show_merge_context: bool = True,
    show_merge_description: bool = True,
) -> str:
    """Convert a note to the specified format.

    Args:
        note: The note dict from note.json.
        format: Output format ('json', 'markdown', or 'text').
        repo: Optional GitPython Repo for hydrating contexts in markdown format.
        pretty: If True, format JSON with indentation.
        show_context: Include conflict_context in output.
        show_solution: Include solutions in output.
        show_pr_comments: Include PR comments in output.
        show_user_comment: Include user comment in output.
        show_summary: Include summary in output.
        show_merge_info: Include merge_info in output.
        show_merge_context: Include merge_context in output.
        show_merge_description: Include merge_description in output.

    Returns:
        Formatted string representation of the note.
    """
    if format == "json":
        return json.dumps(note, indent=2 if pretty else None) + "\n"

    elif format == "text":
        # Text format shows a summary of what's in the note
        return convert_note_to_text_summary(note)

    elif format == "markdown":
        # Markdown format shows full details
        output_str = ""
        if show_merge_info and "merge_info" in note:
            merge_info = MergeInfo.from_dict(note["merge_info"], repo)
            output_str += formatters.merge_info_to_markdown(merge_info) + "\n"
        if show_merge_context and "merge_context" in note:
            merge_ctx = MergeContext.from_dict(note["merge_context"], repo)
            output_str += formatters.merge_context_to_markdown(merge_ctx) + "\n"
        if show_context and "conflict_context" in note:
            conflict_ctx = ConflictContext.from_dict(note["conflict_context"], repo)
            output_str += formatters.conflict_context_to_markdown(conflict_ctx) + "\n"
        if show_pr_comments and "pr_comments" in note:
            output_str += formatters.pr_comments_to_markdown(note["pr_comments"]) + "\n"
        if show_user_comment and "user_comment" in note:
            output_str += (
                formatters.user_comment_to_markdown(note["user_comment"]) + "\n"
            )
        if show_solution and note.get("solutions"):
            output_str += (
                formatters.solutions_to_markdown(note.get("solutions", [])) + "\n"
            )
        if show_merge_description and "merge_description" in note:
            output_str += (
                formatters.merge_description_to_markdown(note["merge_description"])
                + "\n"
            )

        return output_str

    return str(note)


@click.command()
@click.pass_obj
@format_option(default=OutputFormat.TEXT)
def status(app: AppContext, format: str):
    """Show current note status from the state store.

    Text format (default): Shows a summary of what's in note.json
    Markdown format: Shows full details of all note sections
    JSON format: Shows raw note.json content (pretty-printed)
    """
    if not app.has_note:
        click.echo("No note found in the state store.")
        exit(0)
    note_dict = app.note.to_dict()
    util.print_or_page(
        convert_note(note_dict, format=format, repo=app.repo, pretty=True),
        format=format,
    )


@click.command()
@click.pass_obj
@click.argument("ref", type=str, required=False, default="HEAD")
def log(app: AppContext, ref: str):
    """Show commits with notes in git-log style (text format only)."""
    output_str = ""
    merge_commit = None
    for commit in app.repo.iter_commits(ref):
        note = app.try_get_note_from_commit(commit)
        if not merge_commit and note and note.merge_info:
            merge_commit = note.merge_info.merge_commit_sha

        if note:
            output_str += formatters.commit_note_to_summary_str(
                commit, note.to_dict(), format="text"
            )
        else:
            output_str += formatters.commit_to_summary_str(commit)

        # Show log until the merge commit is reached
        if merge_commit is not None and git_utils.is_merge_commit_parent(
            commit, merge_commit
        ):
            break

    util.print_or_page(output_str, format="text")


@click.command()
@click.pass_obj
def prompt(app: AppContext):
    try:
        if not app.has_note:
            click.echo("No note found. Please prepare the context first.")
            click.echo("Use `mergai create-conflict-context` to add conflict context.")
            click.echo(
                "Use `mergai pr-add-comments-to-context` to add PR comments to the context."
            )
            exit(1)
        prompt_text = app.prompt_builder.build_resolve_prompt()
        util.print_or_page(prompt_text, format="markdown")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@click.command()
@click.pass_obj
def merge_prompt(app: AppContext):
    try:
        if not app.has_note:
            click.echo("No note found. Please prepare the context first.")
            click.echo("Use `mergai create-conflict-context` to add conflict context.")
            click.echo(
                "Use `mergai pr-add-comments-to-context` to add PR comments to the context."
            )
            exit(1)
        prompt_text = app.prompt_builder.build_describe_prompt()
        util.print_or_page(prompt_text, format="markdown")
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
    from datetime import datetime, timezone

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
        with open(file) as f:
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
    # TODO: support multiple comments
    cur_comment = ""
    if app.note.has_user_comment and app.note.user_comment is not None:
        cur_comment = get_cur_comment(app.note.user_comment)

    if (body or file) and app.note.has_user_comment and not force:
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

    comment_dict = {
        "user": app.repo.git.config("user.name"),
        "email": app.repo.git.config("user.email"),
        "date": now_utc_iso(),
        "body": stripped,
    }

    app.note.set_user_comment(comment_dict)

    app.save_note(app.note)
