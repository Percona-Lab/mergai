import io
import os
import shutil
import subprocess
import sys
import json
import git

from typing import Optional
from . import git_utils
from jinja2 import Template
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from datetime import datetime, timezone
from ..models import (
    MergeInfo,
    MergeContext,
    ConflictContext,
    ContextSerializationConfig,
    MarkdownConfig,
    MarkdownFormat,
    EnhancedCommit,
)

from ..config import BranchConfig


def gh_auth_token() -> str:
    import os

    token = os.getenv("GITHUB_TOKEN")
    if token is not None:
        return token
    token = os.getenv("GH_TOKEN")
    if token is not None:
        return token

    try:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except:
        token = None

    return token


GITHUB_MD_THEME = Theme(
    {
        "markdown.h1": "bold",
        "markdown.h2": "bold underline",
        "markdown.h3": "bold",
        "markdown.h4": "",
        "markdown.h5": "",
        "markdown.h6": "",
        "markdown.item.bullet": "dim",
        "markdown.link": "blue underline",
        "markdown.code": "dim",
        "markdown.code_block": "",
        "markdown.table.border": "dim",
    }
)


def get_rich_markdown(text: str) -> str:
    """Render markdown in GitHub-like terminal style."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=shutil.get_terminal_size((80, 20)).columns,
        theme=GITHUB_MD_THEME,
    )

    md = Markdown(
        text,
        code_theme="github-dark",  # GitHub-style code fences
        justify="left",
        inline_code_lexer="text",
    )

    console.print(md)
    return buf.getvalue()


def print_or_page(text: str, format: str = "text"):
    """Print text to stdout, using a pager if the output is a terminal."""
    term_height = shutil.get_terminal_size((80, 20)).lines
    lines = text.count("\n") + 1

    if sys.stdout.isatty() and format == "markdown":
        text = get_rich_markdown(text)

    if not sys.stdout.isatty() or lines + 4 <= term_height:
        print(text)
        return

    pager = os.environ.get("PAGER", "less -FRSX")

    proc = subprocess.Popen(pager, shell=True, stdin=subprocess.PIPE)

    try:
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()


def render_from_template(template_str: str, **kwargs) -> str:
    template = Template(template_str)
    return template.render(**kwargs)


MERGE_INFO_MARKDOWN_TEMPLATE = """\
# Merge Info

- **Target Branch:** `{{ merge_info.target_branch }}`
{%- if merge_info.target_branch_sha %}
- **Target Branch SHA:** {{ format_sha(merge_info.target_branch_sha) }}
{%- endif %}
- **Merge Commit:** {{ format_sha(merge_info.merge_commit_sha) }}
"""


def _create_format_sha_func(markdown_config: Optional[MarkdownConfig] = None):
    """Create a format_sha function for Jinja2 templates.

    Args:
        markdown_config: Optional MarkdownConfig for PR-style links.

    Returns:
        A function that formats SHA as markdown (with or without links).
    """

    def format_sha(sha: str, use_short: bool = False) -> str:
        """Format a SHA as markdown, optionally with link.

        Args:
            sha: Commit SHA (full or short).
            use_short: If True, display short SHA.

        Returns:
            Markdown formatted SHA.
        """
        display_sha = sha[:11] if use_short else sha

        if markdown_config and markdown_config.format == MarkdownFormat.PR:
            url = markdown_config.get_commit_url(sha)
            if url:
                return f"[`{display_sha}`]({url})"

        return f"`{display_sha}`"

    return format_sha


def merge_info_to_markdown(
    merge_info: MergeInfo, markdown_config: Optional[MarkdownConfig] = None
) -> str:
    """Convert a MergeInfo object to markdown format.

    Args:
        merge_info: MergeInfo object with merge operation details.
        markdown_config: Optional MarkdownConfig for PR-style formatting with links.

    Returns:
        Markdown formatted string.
    """
    format_sha = _create_format_sha_func(markdown_config)
    return render_from_template(
        MERGE_INFO_MARKDOWN_TEMPLATE, merge_info=merge_info, format_sha=format_sha
    )


def merge_info_to_str(
    merge_info: MergeInfo,
    format: str,
    pretty: bool = False,
    markdown_config: Optional[MarkdownConfig] = None,
):
    """Convert a MergeInfo object to the specified format.

    Args:
        merge_info: MergeInfo object with merge operation details.
        format: Output format ('json' or 'markdown').
        pretty: If True, format JSON with indentation.
        markdown_config: Optional MarkdownConfig for PR-style markdown formatting.

    Returns:
        Formatted string representation.
    """
    if format == "json":
        return (
            json.dumps(merge_info.to_dict(), default=str, indent=2 if pretty else None)
            + "\n"
        )
    elif format == "markdown":
        return merge_info_to_markdown(merge_info, markdown_config) + "\n"

    return str(merge_info)


CONFLICT_CONTEXT_MARKDOWN_TEMPLATE = """\
# Conflict Context

- **Base Commit:** {{ format_sha(conflict_context.base_commit.hexsha) }}
- **Ours Commit:** {{ format_sha(conflict_context.ours_commit.hexsha) }}
- **Theirs Commit:** {{ format_sha(conflict_context.theirs_commit.hexsha) }}

## Conflicted Files

| Path | Conflict Type |
|------|---------------|
{%- for path, conflict_type in conflict_context.conflict_types.items() %}
| `{{ path }}` | {{ conflict_type }} |
{%- endfor %}


{%- if conflict_context.get('their_commits') %}

<details>
<summary>Conflict details</summary>

## Their Commits

| Path | Commit | Message |
|------|--------|---------|
{%- for path, commits in conflict_context.their_commits.items() %}
{%- for commit in commits %}
| `{{ path }}` | {{ format_sha(commit.hexsha) }} | {{ commit.message.split('\n')[0] }} |
{%- endfor %}
{%- endfor %}

</details>
{%- endif %}
"""


def conflict_context_to_markdown(
    conflict_context: ConflictContext, markdown_config: Optional[MarkdownConfig] = None
) -> str:
    """Convert ConflictContext to markdown.

    Args:
        conflict_context: ConflictContext object (must have repo bound).
        markdown_config: Optional MarkdownConfig for PR-style formatting with links.

    Returns:
        Markdown formatted string.
    """

    if not isinstance(conflict_context, ConflictContext):
        raise TypeError(
            f"Expected ConflictContext, got {type(conflict_context).__name__}. "
            "Use ConflictContext.from_dict() to create from dict."
        )

    template_data = conflict_context.to_dict(ContextSerializationConfig.template())
    format_sha = _create_format_sha_func(markdown_config)
    return render_from_template(
        CONFLICT_CONTEXT_MARKDOWN_TEMPLATE,
        conflict_context=template_data,
        format_sha=format_sha,
    )


def conflict_context_to_str(
    context: ConflictContext,
    format: str,
    pretty: bool = False,
    markdown_config: Optional[MarkdownConfig] = None,
) -> str:
    """Convert ConflictContext to string in specified format.

    Args:
        context: ConflictContext object.
        format: Output format ('json' or 'markdown').
        pretty: If True, format JSON with indentation.
        markdown_config: Optional MarkdownConfig for PR-style markdown formatting.

    Returns:
        Formatted string representation.
    """

    if not isinstance(context, ConflictContext):
        raise TypeError(
            f"Expected ConflictContext, got {type(context).__name__}. "
            "Use ConflictContext.from_dict() to create from dict."
        )

    if format == "json":
        return (
            json.dumps(context.to_dict(), default=str, indent=2 if pretty else None)
            + "\n"
        )
    elif format == "markdown":
        return conflict_context_to_markdown(context, markdown_config) + "\n"

    return str(context.to_dict())


# TODO: the session section should be improved
CONFLICT_SOLUTION_MARKDOWN_TEMPLATE = """# Conflict Solution
## Solution Summary

{{ solution.response.summary }}

## Resolved files

{%- if solution.response.resolved | length == 0 %}
No files were resolved.
{%- else %}
| File Path | Resolution |
|-----------|------------|
{%- for file_path, resolution in solution.response.resolved.items() %}
| `{{ file_path }}` | {{ resolution }} |
{%- endfor %}
{%- endif %}

## Unresolved files

{%- if solution.response.unresolved | length == 0 %}
All conflicts have been resolved.
{%- else %}
| File Path | Issue |
|-----------|------------|
{%- for file_path, issue in solution.response.unresolved.items() %}
| `{{ file_path }}` | {{ issue }} |
{%- endfor %}
{%- endif %}

## Review Notes

{{ solution.response.review_notes if solution.response.review_notes else "No review notes provided." }}

{%- if solution.stats %}
## Stats

{%- if solution.stats.models | length > 0 %}
### Models

| Model | Input tokens | Output tokens | Cached tokens | Thoughts tokens | Tool tokens | Total tokens |
|-------|--------------|------------------|---------------|-----------------|-------------|--------------|
{%- for model, stat in solution.stats.models.items() %}
| {{ model }} | {{ stat.tokens.input }} | {{ stat.tokens.output }} | {{ stat.tokens.cached }} | {{ stat.tokens.thoughts }} | {{ stat.tokens.tool }} | {{ stat.tokens.total }} | ${{ "%.6f"|format(0.0) }} |
{%- endfor %}
{%- endif %}

{%- endif %}

{%- if solution.agent_info %}
## Agent Info

Executed with '{{ solution.agent_info.agent_type }}' agent, version '{{ solution.agent_info.version }}'.

{%- endif %}

{%- if solution.session %}
## Session

- ID: `{{ solution.session.sessionId }}`
{%- if solution.session.projectHash %}
- Project Hash: `{{ solution.session.projectHash }}`
{%- endif %}
{%- if solution.session.startTime %}
- Started: {{ solution.session.startTime }}
{%- endif %}
{%- if solution.session.lastUpdated %}
- Last Updated: {{ solution.session.lastUpdated }}
{%- endif %}

{%- if solution.session.messages and solution.session.messages | length > 0 %}
### Messages

{%- for message in solution.session.messages %}
- {{ message.timestamp or "unknown time" }} - {{ message.type|capitalize if message.type else "Message" }}{% if message.model %} ({{ message.model }}){% endif %}{% if message.id %} [`{{ message.id }}`]{% endif %}

```text
{{ (message.content if message.content else "No content provided.") | indent(4, first=True) }}
```

{%- if message.thoughts %}
{%- for thought in message.thoughts %}
    - Thought: **{{ thought.subject }}**{% if thought.timestamp %} ({{ thought.timestamp }}){% endif %}

{{ (thought.description if thought.description else "") | indent(8, first=True) }}

{% endfor %}
{%- endif %}

{%- if message.tokens %}
    - Tokens: {% for key, value in message.tokens.items() %}{{ key }}={{ value }}{% if not loop.last %}, {% endif %}{% endfor %}
{%- endif %}

{%- endfor %}
{%- else %}
No session messages available.
{%- endif %}
{%- endif %}
"""


# TODO:
# - add tools stats
# - add support for dynamic columns so that each model can have different columns if neccessary (e.g. gemini vs chat-gpt)
# - add support for total
# - add cost estimation
def conflict_solution_to_markdown(solution: dict) -> str:
    return render_from_template(CONFLICT_SOLUTION_MARKDOWN_TEMPLATE, solution=solution)


SOLUTION_MARKDOWN_TEMPLATE = """\

## Solution Summary

{{ solution.response.summary }}

## Resolved Files

{%- if solution.response.resolved | length == 0 %}
No files were resolved.
{%- else %}
| File Path | Resolution |
|-----------|------------|
{%- for file_path, resolution in solution.response.resolved.items() %}
| `{{ file_path }}` | {{ resolution }} |
{%- endfor %}
{%- endif %}

## Unresolved Files

{%- if solution.response.unresolved | length == 0 %}
All conflicts have been resolved.
{%- else %}
| File Path | Issue |
|-----------|-------|
{%- for file_path, issue in solution.response.unresolved.items() %}
| `{{ file_path }}` | {{ issue }} |
{%- endfor %}
{%- endif %}

## Review Notes

{{ solution.response.review_notes if solution.response.review_notes else "No review notes provided." }}

<details>
<summary>Agent Stats</summary>

{%- if solution.agent_info %}

**Agent:** {{ solution.agent_info.agent_type }} (version {{ solution.agent_info.version }})
{%- endif %}

{%- if solution.stats and solution.stats.models | length > 0 %}

| Model | Input | Output | Cached | Thoughts | Tool | Total |
|-------|-------|--------|--------|----------|------|-------|
{%- for model, stat in solution.stats.models.items() %}
| {{ model }} | {{ stat.tokens.input }} | {{ stat.tokens.output }} | {{ stat.tokens.cached }} | {{ stat.tokens.thoughts }} | {{ stat.tokens.tool }} | {{ stat.tokens.total }} |
{%- endfor %}
{%- endif %}

</details>
"""


def solution_to_markdown(solution: dict) -> str:
    """Convert solution data to markdown formatted for PR body.

    This format is optimized for GitHub PR descriptions with:
    - Clear sections for summary, resolved files, and unresolved files
    - Review notes for developers
    - Stats hidden in a collapsible section

    Args:
        solution: Solution dict with response, stats, and agent_info.

    Returns:
        Markdown formatted string suitable for PR body.
    """
    return render_from_template(SOLUTION_MARKDOWN_TEMPLATE, solution=solution)


SOLUTIONS_MARKDOWN_TEMPLATE = """\
{%- for solution in solutions %}
## Solution {{ loop.index }}{% if loop.length == 1 %}{% endif %}
{{ solution_to_markdown(solution) }}
{%- endfor %}
"""


def solutions_to_markdown(solutions: list) -> str:
    md = "# Solution"
    md += "\n"

    if len(solutions) == 1:
        md += solution_to_markdown(solutions[0])
    else:
        md += render_from_template(
            SOLUTIONS_MARKDOWN_TEMPLATE,
            solutions=solutions,
            solution_to_markdown=solution_to_markdown,
        )

    return md


def conflict_solution_to_str(solution: dict, format: str, pretty: bool = False):
    if format == "json":
        return json.dumps(solution, default=str, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        return conflict_solution_to_markdown(solution) + "\n"
    return str(solution)


PR_COMMENTS_TEMPLATE = """# Pull Request Comments
{%- for comment_id, comment in comments.items() %}
{%- if comment.path %}
- `{{ comment.created_at }}`: review comment from `{{ comment.user }}` at `{{ comment.path }}:{{ comment.line_str }}`:
{%- else %}
- `{{ comment.created_at }}`: general comment from `{{ comment.user }}`:
{%- endif %}

{{ comment.body }}

{%- endfor %}
"""


def pr_comments_to_markdown(comments: dict) -> str:
    return render_from_template(PR_COMMENTS_TEMPLATE, comments=comments)


def pr_comments_to_str(comments: dict, format: str, pretty: bool = False):
    if format == "json":
        return json.dumps(comments, default=str, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        return pr_comments_to_markdown(comments) + "\n"
    return str(comments)


USER_COMMENT_TEMPLATE = """\
# User Comment
{{ user_comment.date }}: comment from `{{ user_comment.user }} <{{ user_comment.email }}>`:

{{ user_comment.body }}
"""


def user_comment_to_markdown(user_comment: dict) -> str:
    return render_from_template(USER_COMMENT_TEMPLATE, user_comment=user_comment)


def user_comment_to_str(user_comment: dict, format: str, pretty: bool = False):
    if format == "json":
        return (
            json.dumps(user_comment, default=str, indent=2 if pretty else None) + "\n"
        )
    elif format == "markdown":
        return user_comment_to_markdown(user_comment) + "\n"

    return str(user_comment)


MERGE_CONTEXT_MARKDOWN_TEMPLATE = """\
# Merge Context

- **Number of merged commits:** {{ merge_context.merged_commits | length }}
- **Auto-merged files:** {{ merge_context.auto_merged.files | length if merge_context.auto_merged and merge_context.auto_merged.files else 0 }}

{%- if merge_context.important_files_modified | length > 0 %}

## Important Files Modified

{%- for file_path in merge_context.important_files_modified %}
- `{{ file_path }}`
{%- endfor %}
{%- endif %}

{%- if merge_context.auto_merged %}

## Auto-Merged Files
{% if merge_context.auto_merged.files | length == 0 %}
No files were auto-merged.
{%- else %}
{%- for file_path in merge_context.auto_merged.files %}
- `{{ file_path }}`
{%- endfor %}
{%- endif %}
{% if merge_context.auto_merged.strategy %}
**Merged with strategy:** {{ merge_context.auto_merged.strategy }}
{%- endif %}

{%- endif %}

<details>
<summary>Show details</summary>

| Commit | Summary |
|------------|---------|
{%- for commit in merge_context.merged_commits %}
| {{ format_sha(commit.hexsha) }} | {{ commit.summary }} |
{%- endfor %}

</details>
"""


def merge_context_to_markdown(
    merge_context: MergeContext, markdown_config: Optional[MarkdownConfig] = None
) -> str:
    """Convert MergeContext to markdown.

    Args:
        merge_context: MergeContext object.
        markdown_config: Optional MarkdownConfig for PR-style formatting with links.

    Returns:
        Markdown formatted string.
    """

    if not isinstance(merge_context, MergeContext):
        raise TypeError(
            f"Expected MergeContext, got {type(merge_context).__name__}. "
            "Use MergeContext.from_dict() to create from dict."
        )

    format_sha = _create_format_sha_func(markdown_config)
    return render_from_template(
        MERGE_CONTEXT_MARKDOWN_TEMPLATE,
        merge_context=merge_context,
        format_sha=format_sha,
    )


def merge_context_to_str(
    merge_context: MergeContext,
    format: str,
    pretty: bool = False,
    markdown_config: Optional[MarkdownConfig] = None,
) -> str:
    """Convert MergeContext to string in specified format.

    Args:
        merge_context: MergeContext object.
        format: Output format ('json' or 'markdown').
        pretty: If True, format JSON with indentation.
        markdown_config: Optional MarkdownConfig for PR-style markdown formatting.

    Returns:
        Formatted string representation.
    """

    if not isinstance(merge_context, MergeContext):
        raise TypeError(
            f"Expected MergeContext, got {type(merge_context).__name__}. "
            "Use MergeContext.from_dict() to create from dict."
        )

    if format == "json":
        return (
            json.dumps(
                merge_context.to_dict(), default=str, indent=2 if pretty else None
            )
            + "\n"
        )
    elif format == "markdown":
        return merge_context_to_markdown(merge_context, markdown_config) + "\n"

    return str(merge_context.to_dict())


MERGE_DESCRIPTION_MARKDOWN_TEMPLATE = """\
# Merge Description

## Summary

{{ merge_description.response.summary }}

## Auto-Merged Files

{%- if merge_description.response.auto_merged | length == 0 %}
No files were auto-merged.
{%- else %}
| File Path | Description |
|-----------|-------------|
{%- for file_path, description in merge_description.response.auto_merged.items() %}
| `{{ file_path }}` | {{ description }} |
{%- endfor %}
{%- endif %}

## Review Notes

{{ merge_description.response.review_notes if merge_description.response.review_notes else "No review notes provided." }}

{%- if merge_description.stats %}

## Stats

{%- if merge_description.stats.models | length > 0 %}
### Models

| Model | Input tokens | Output tokens | Cached tokens | Thoughts tokens | Tool tokens | Total tokens |
|-------|--------------|------------------|---------------|-----------------|-------------|--------------|
{%- for model, stat in merge_description.stats.models.items() %}
| {{ model }} | {{ stat.tokens.input }} | {{ stat.tokens.output }} | {{ stat.tokens.cached }} | {{ stat.tokens.thoughts }} | {{ stat.tokens.tool }} | {{ stat.tokens.total }} |
{%- endfor %}
{%- endif %}
{%- endif %}

{%- if merge_description.agent_info %}

## Agent Info

Executed with '{{ merge_description.agent_info.agent_type }}' agent, version '{{ merge_description.agent_info.version }}'.
{%- endif %}
"""


def merge_description_to_markdown(merge_description: dict) -> str:
    return render_from_template(
        MERGE_DESCRIPTION_MARKDOWN_TEMPLATE, merge_description=merge_description
    )


def merge_description_to_str(
    merge_description: dict, format: str, pretty: bool = False
):
    if format == "json":
        return (
            json.dumps(merge_description, default=str, indent=2 if pretty else None)
            + "\n"
        )
    elif format == "markdown":
        return merge_description_to_markdown(merge_description) + "\n"

    return str(merge_description)


def load_if_exists(filename: str) -> str:
    if not os.path.exists(filename):
        return ""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def commit_note_to_summary_markdown(commit: git.Commit, note: dict) -> str:
    output_str = f"- Commit: `{commit.hexsha}`\n"
    output_str += f"- Author: {commit.author.name} <{commit.author.email}>\n"
    output_str += f"- Date:   {commit.authored_datetime}\n"
    output_str += (
        f"- Message:\n\n    {commit.message.strip().replace('\n', '\n    ')}\n"
    )
    output_str += f"- Content:\n"
    if "merge_info" in note:
        output_str += (
            f"  - Merge Info (use mergai show --merge-info to see the merge info.)\n"
        )
    if "merge_context" in note:
        output_str += f"  - Merge Context (use mergai show --merge-context to see the merge context.)\n"
    if "conflict_context" in note:
        output_str += f"  - Conflict Context (use mergai show --context to see the conflict context.)\n"
    if "pr_comments" in note:
        output_str += (
            f"  - PR Comments (use mergai show --pr-comments to see the PR comments.)\n"
        )

    # Handle both legacy "solution" and new "solutions" array
    if "solutions" in note:
        count = len(note["solutions"])
        output_str += f"  - Solutions ({count}) (use mergai show --solution to see the conflict solutions.)\n"
    elif "solution" in note:
        output_str += (
            f"  - Solution (use mergai show --solution to see the conflict solution.)\n"
        )

    if "merge_description" in note:
        output_str += f"  - Merge Description (use mergai show --merge-description to see the merge description.)\n"

    output_str += "\n"

    return output_str


def commit_note_to_summary_json(
    commit: git.Commit, note: dict, pretty: bool = False
) -> str:
    summary = {
        "commit": commit.hexsha,
        "author": {
            "name": commit.author.name,
            "email": commit.author.email,
        },
        "date": str(commit.authored_datetime),
        "message": commit.message.strip(),
        "content": {},
    }

    if "merge_info" in note:
        summary["content"]["merge_info"] = True
        summary["merge_info"] = note["merge_info"]
    if "merge_context" in note:
        summary["content"]["merge_context"] = True
    if "conflict_context" in note:
        summary["content"]["conflict_context"] = True
    if "pr_comments" in note:
        summary["content"]["pr_comments"] = True
    # Handle both legacy "solution" and new "solutions" array
    if "solutions" in note:
        summary["content"]["solutions"] = len(note["solutions"])
    elif "solution" in note:
        summary["content"]["solution"] = True
    if "user_comment" in note:
        summary["content"]["user_comment"] = True
        summary["user_comment"] = note["user_comment"]
    if "merge_description" in note:
        summary["content"]["merge_description"] = True

    return json.dumps(summary, indent=2 if pretty else None) + "\n"


def get_comments_stats(comments: dict) -> dict:
    stats = {}
    for comment in comments.values():
        stats.setdefault(comment["user"], 0)
        stats[comment["user"]] += 1
    return stats


def get_solution_stats(solution: dict) -> dict:
    stats = {}
    resolved_count = len(solution.get("response", {}).get("resolved", {}))
    unresolved_count = len(solution.get("response", {}).get("unresolved", {}))
    stats["resolved_files"] = resolved_count
    stats["unresolved_files"] = unresolved_count
    stats["total_files"] = resolved_count + unresolved_count
    return stats


def commit_note_to_summary_text(commit: git.Commit, note: dict) -> str:
    output_str = f"commit: {commit.hexsha}\n"
    output_str += f"Author: {commit.author.name} <{commit.author.email}>\n"
    output_str += f"Date:   {commit.authored_datetime}\n"
    output_str += f"Content:\n"
    if "merge_info" in note:
        output_str += f"  - Merge Info\n"
        merge_info = note["merge_info"]
        output_str += (
            f"    - Target Branch: {merge_info.get('target_branch', 'unknown')}\n"
        )
        output_str += (
            f"    - Merge Commit: {merge_info.get('merge_commit', 'unknown')}\n"
        )
    if "merge_context" in note:
        output_str += f"  - Merge Context\n"
        merge_context = note["merge_context"]
        output_str += (
            f"    - Merge Commit: {merge_context.get('merge_commit', 'unknown')}\n"
        )
        output_str += (
            f"    - Merged Commits: {len(merge_context.get('merged_commits', []))}\n"
        )
        important_files = merge_context.get("important_files_modified", [])
        if important_files:
            output_str += f"    - Important Files Modified: {len(important_files)}\n"
    if "conflict_context" in note:
        output_str += f"  - Conflict Context\n"
        conflict_context = note["conflict_context"]
        # Handle both storage format (SHA strings) and legacy format (dicts/objects)
        base_commit = conflict_context["base_commit"]
        ours_commit = conflict_context["ours_commit"]
        theirs_commit = conflict_context["theirs_commit"]
        base_sha = (
            base_commit
            if isinstance(base_commit, str)
            else base_commit.get("hexsha", base_commit)
        )
        ours_sha = (
            ours_commit
            if isinstance(ours_commit, str)
            else ours_commit.get("hexsha", ours_commit)
        )
        theirs_sha = (
            theirs_commit
            if isinstance(theirs_commit, str)
            else theirs_commit.get("hexsha", theirs_commit)
        )
        output_str += f"    - Base Commit: {base_sha}\n"
        output_str += f"    - Ours Commit: {ours_sha}\n"
        output_str += f"    - Theirs Commit: {theirs_sha}\n"
    if "pr_comments" in note:
        output_str += f"  - PR Comments (total: {len(note['pr_comments'])})\n"
        stats = get_comments_stats(note["pr_comments"])
        for user, count in stats.items():
            if user != "total_comments":
                output_str += f"    - {user}: {count} comment(s)\n"

    # Handle both legacy "solution" and new "solutions" array
    if "solutions" in note:
        output_str += f"  - Solutions ({len(note['solutions'])})\n"
        for idx, solution in enumerate(note["solutions"]):
            stats = get_solution_stats(solution)
            output_str += f"    [{idx}] Resolved Files: {stats['resolved_files']}/{stats['total_files']}"
            if stats["unresolved_files"] > 0:
                output_str += f", Unresolved: {stats['unresolved_files']}"
            output_str += "\n"
    elif "solution" in note:
        output_str += f"  - Solution\n"
        stats = get_solution_stats(note["solution"])
        output_str += (
            f"    - Resolved Files: {stats['resolved_files']}/{stats['total_files']}\n"
        )
        if stats["unresolved_files"] > 0:
            output_str += f"    - Unresolved Files: {stats['unresolved_files']}/{stats['total_files']}\n"

    if "merge_description" in note:
        output_str += f"  - Merge Description\n"
        merge_desc = note["merge_description"]
        response = merge_desc.get("response", {})
        auto_merged_count = len(response.get("auto_merged", {}))
        output_str += f"    - Auto-Merged Files: {auto_merged_count}\n"

    if "user_comment" in note:
        output_str += f"\n"
        output_str += f"User Comment:"
        output_str += f" {note['user_comment'].get('user', 'unknown')}"
        output_str += f" <{note['user_comment'].get('email', 'unknown')}>"
        output_str += f" at {note['user_comment'].get('date', 'unknown')}\n"
        output_str += f"{note['user_comment'].get('body', '')}\n"
        output_str += f"\n"

    output_str += f"Message:\n    {commit.message.strip().replace('\n', '\n    ')}\n"

    output_str += "\n"
    return output_str


def commit_note_to_summary_str(
    commit: git.Commit, note: dict, format: str, pretty: bool = False
) -> str:
    if format == "markdown":
        return commit_note_to_summary_markdown(commit, note)
    elif format == "json":
        return commit_note_to_summary_json(commit, note, pretty)
    else:
        return commit_note_to_summary_text(commit, note)


def commit_to_summary_str(commit: git.Commit) -> str:
    output_str = f"commit: {commit.hexsha}\n"
    output_str += f"Author: {commit.author.name} <{commit.author.email}>\n"
    output_str += f"Date:   {commit.authored_datetime}\n"
    output_str += "Content:\n"
    output_str += "  (no note)\n"
    output_str += (
        f"Message:\n    {commit.message.strip().replace('\n', '\n    ')}\n"
    )
    output_str += "\n"
    return output_str

def format_commit_info(commit, indent: str = "  ") -> str:
    """Format a commit for display."""
    if not commit:
        return f"{indent}(none)"

    authored_date = datetime.fromtimestamp(commit.authored_date, tz=timezone.utc)
    date_str = authored_date.strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"{indent}SHA:    {git_utils.short_sha(commit.hexsha)}",
        f"{indent}Date:   {date_str}",
        f"{indent}Author: {commit.author.name} <{commit.author.email}>",
        f"{indent}Title:  {commit.summary}",
    ]
    return "\n".join(lines)


def format_commit_info_oneline(commit) -> str:
    """Format a commit in a single line."""
    if not commit:
        return "(none)"

    authored_date = datetime.fromtimestamp(commit.authored_date, tz=timezone.utc)
    date_str = authored_date.strftime("%Y-%m-%d")

    return f"{git_utils.short_sha(commit.hexsha)} {date_str} {commit.summary}"


def format_number(n: int) -> str:
    """Format a number with thousand separators."""
    return f"{n:,}"


