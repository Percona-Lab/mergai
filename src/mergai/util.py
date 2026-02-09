import io
import os
import re
import shutil
import subprocess
import sys
import json
import git
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Optional
from . import git_utils
from jinja2 import Template
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from datetime import datetime, timezone

if TYPE_CHECKING:
    from .config import BranchConfig

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


CONFLICT_CONTEXT_MARKDOWN_TEMPLATE = """# Conflict Context

## SHA Details

- base: {{ context.base_commit.hexsha }}
- ours: {{ context.ours_commit.hexsha }}
- theirs: {{ context.theirs_commit.hexsha }}

## Conflict Details

| path | conflict type |
|------|---------------|
{%- for path, conflict_type in context.conflict_types.items() %}
| `{{ path }}` | {{ conflict_type }} |
{%- endfor %}

{%- if context.get('their_commits') %}
## Their Commits
{%- for path, commits in context.their_commits.items() %}
- `{{ path }}`
{%- for commit in commits %}

```text
commit {{ commit.hexsha }}
Author: {{ commit.author.name }} <{{ commit.author.email }}>
Date:   {{ commit.authored_datetime }}

{{ commit.message | indent(4, first=True) }}
```

{% endfor %}
{%- endfor %}
{%- endif %}

{%- if context.get('diffs') %}
## Diffs
{%- for path, conflict_data in context.diffs.items() %}
- `{{ path }}`

```diff
{{ conflict_data }}
```

{%- endfor %}
{%- endif %}
"""


def conflict_context_to_str(context: dict, format, pretty: bool = False):
    if format == "json":
        return json.dumps(context, default=str, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        return conflict_context_to_markdown(context) + "\n"

    return str(context)


def render_from_template(template_str: str, context: dict) -> str:
    template = Template(template_str)
    return template.render(context=context)


def conflict_context_to_markdown(context: dict) -> str:
    return render_from_template(CONFLICT_CONTEXT_MARKDOWN_TEMPLATE, context)


# TODO: the session section should be improved
CONFLICT_SOLUTION_MARKDOWN_TEMPLATE = """# Conflict Solution
## Solution Summary

{{ context.response.summary }}

## Resolved files

{%- if context.response.resolved | length == 0 %}
No files were resolved.
{%- else %}
| File Path | Resolution |
|-----------|------------|
{%- for file_path, resolution in context.response.resolved.items() %}
| `{{ file_path }}` | {{ resolution }} |
{%- endfor %}
{%- endif %}

## Unresolved files

{%- if context.response.unresolved | length == 0 %}
All conflicts have been resolved.
{%- else %}
| File Path | Issue |
|-----------|------------|
{%- for file_path, issue in context.response.unresolved.items() %}
| `{{ file_path }}` | {{ issue }} |
{%- endfor %}
{%- endif %}

## Review Notes

{{ context.response.review_notes if context.response.review_notes else "No review notes provided." }}

{%- if context.stats %}
## Stats

{%- if context.stats.models | length > 0 %}
### Models

| Model | Input tokens | Output tokens | Cached tokens | Thoughts tokens | Tool tokens | Total tokens |
|-------|--------------|------------------|---------------|-----------------|-------------|--------------|
{%- for model, stat in context.stats.models.items() %}
| {{ model }} | {{ stat.tokens.input }} | {{ stat.tokens.output }} | {{ stat.tokens.cached }} | {{ stat.tokens.thoughts }} | {{ stat.tokens.tool }} | {{ stat.tokens.total }} | ${{ "%.6f"|format(0.0) }} |
{%- endfor %}
{%- endif %}

{%- endif %}

{%- if context.agent_info %}
## Agent Info

Executed with '{{ context.agent_info.agent_type }}' agent, version '{{ context.agent_info.version }}'.

{%- endif %}

{%- if context.session %}
## Session

- ID: `{{ context.session.sessionId }}`
{%- if context.session.projectHash %}
- Project Hash: `{{ context.session.projectHash }}`
{%- endif %}
{%- if context.session.startTime %}
- Started: {{ context.session.startTime }}
{%- endif %}
{%- if context.session.lastUpdated %}
- Last Updated: {{ context.session.lastUpdated }}
{%- endif %}

{%- if context.session.messages and context.session.messages | length > 0 %}
### Messages

{%- for message in context.session.messages %}
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
    return render_from_template(CONFLICT_SOLUTION_MARKDOWN_TEMPLATE, solution)


SOLUTION_PR_BODY_TEMPLATE = """\
## Solution Summary

{{ context.response.summary }}

## Resolved Files

{%- if context.response.resolved | length == 0 %}
No files were resolved.
{%- else %}
| File Path | Resolution |
|-----------|------------|
{%- for file_path, resolution in context.response.resolved.items() %}
| `{{ file_path }}` | {{ resolution }} |
{%- endfor %}
{%- endif %}

## Unresolved Files

{%- if context.response.unresolved | length == 0 %}
All conflicts have been resolved.
{%- else %}
| File Path | Issue |
|-----------|-------|
{%- for file_path, issue in context.response.unresolved.items() %}
| `{{ file_path }}` | {{ issue }} |
{%- endfor %}
{%- endif %}

## Review Notes

{{ context.response.review_notes if context.response.review_notes else "No review notes provided." }}

<details>
<summary>Agent Stats</summary>

{%- if context.agent_info %}

**Agent:** {{ context.agent_info.agent_type }} (version {{ context.agent_info.version }})
{%- endif %}

{%- if context.stats and context.stats.models | length > 0 %}

| Model | Input | Output | Cached | Thoughts | Tool | Total |
|-------|-------|--------|--------|----------|------|-------|
{%- for model, stat in context.stats.models.items() %}
| {{ model }} | {{ stat.tokens.input }} | {{ stat.tokens.output }} | {{ stat.tokens.cached }} | {{ stat.tokens.thoughts }} | {{ stat.tokens.tool }} | {{ stat.tokens.total }} |
{%- endfor %}
{%- endif %}

</details>
"""


def solution_pr_body_to_markdown(solution: dict) -> str:
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
    return render_from_template(SOLUTION_PR_BODY_TEMPLATE, solution)


CONFLICT_RESOLUTION_PR_BODY_TEMPLATE = """\
# Conflict Resolution

## Conflict Context

- **Base Commit:** `{{ conflict_context.base_commit.hexsha }}`
- **Ours Commit:** `{{ conflict_context.ours_commit.hexsha }}`
- **Theirs Commit:** `{{ conflict_context.theirs_commit.hexsha }}`

### Conflicted Files

| Path | Conflict Type |
|------|---------------|
{%- for path, conflict_type in conflict_context.conflict_types.items() %}
| `{{ path }}` | {{ conflict_type }} |
{%- endfor %}

{%- for solution in solutions %}

## Solution {{ loop.index }}{% if loop.length == 1 %}{% endif %}

### Summary

{{ solution.response.summary }}

### Resolved Files

{%- if solution.response.resolved | length == 0 %}
No files were resolved.
{%- else %}
| File Path | Resolution |
|-----------|------------|
{%- for file_path, resolution in solution.response.resolved.items() %}
| `{{ file_path }}` | {{ resolution }} |
{%- endfor %}
{%- endif %}

### Unresolved Files

{%- if solution.response.unresolved | length == 0 %}
All conflicts have been resolved.
{%- else %}
| File Path | Issue |
|-----------|-------|
{%- for file_path, issue in solution.response.unresolved.items() %}
| `{{ file_path }}` | {{ issue }} |
{%- endfor %}
{%- endif %}

{%- if solution.response.review_notes %}

### Review Notes

{{ solution.response.review_notes }}
{%- endif %}

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
{%- endfor %}
"""


def conflict_resolution_pr_body_to_markdown(conflict_context: dict, solutions: list) -> str:
    """Convert conflict context and solutions to markdown for PR body.

    This is used when creating a main PR after conflict resolution,
    where merge_context doesn't exist but conflict_context and solutions do.

    Args:
        conflict_context: The conflict context dict with commit info and conflicted files.
        solutions: List of solution dicts, each with response, stats, and agent_info.

    Returns:
        Markdown formatted string suitable for PR body.
    """
    template = Template(CONFLICT_RESOLUTION_PR_BODY_TEMPLATE)
    return template.render(conflict_context=conflict_context, solutions=solutions)


def conflict_solution_to_str(solution: dict, format: str, pretty: bool = False):
    if format == "json":
        return json.dumps(solution, default=str, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        return conflict_solution_to_markdown(solution) + "\n"
    return str(solution)


PR_COMMENTS_TEMPLATE = """# Pull Request Comments
{%- for comment_id, comment in context.items() %}
{%- if comment.path %}
- `{{ comment.created_at }}`: review comment from `{{ comment.user }}` at `{{ comment.path }}:{{ comment.line_str }}`:
{%- else %}
- `{{ comment.created_at }}`: general comment from `{{ comment.user }}`:
{%- endif %}

{{ comment.body }}

{%- endfor %}
"""


def pr_comments_to_markdown(comments: dict) -> str:
    return render_from_template(PR_COMMENTS_TEMPLATE, comments)


def pr_comments_to_str(comments: dict, format: str, pretty: bool = False):
    if format == "json":
        return json.dumps(comments, default=str, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        return pr_comments_to_markdown(comments) + "\n"
    return str(comments)


USER_COMMENT_TEMPLATE = """\
# User Comment
{{ context.date }}: comment from `{{ context.user }} <{{ context.email }}>`:

{{ context.body }}
"""


def user_comment_to_markdown(user_comment: dict) -> str:
    return render_from_template(USER_COMMENT_TEMPLATE, user_comment)


def user_comment_to_str(user_comment: dict, format: str, pretty: bool = False):
    if format == "json":
        return (
            json.dumps(user_comment, default=str, indent=2 if pretty else None) + "\n"
        )
    elif format == "markdown":
        return user_comment_to_markdown(user_comment) + "\n"

    return str(user_comment)


MERGE_INFO_MARKDOWN_TEMPLATE = """\
# Merge Info

- **Target Branch:** `{{ context.target_branch }}`
{%- if context.target_branch_sha %}
- **Target Branch SHA:** `{{ context.target_branch_sha }}`
{%- endif %}
- **Merge Commit:** `{{ context.merge_commit }}`
"""


def merge_info_to_markdown(merge_info: dict) -> str:
    return render_from_template(MERGE_INFO_MARKDOWN_TEMPLATE, merge_info)


def merge_info_to_str(merge_info: dict, format: str, pretty: bool = False):
    if format == "json":
        return json.dumps(merge_info, default=str, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        return merge_info_to_markdown(merge_info) + "\n"

    return str(merge_info)


MERGE_CONTEXT_MARKDOWN_TEMPLATE = """\
# Merge Context

- **Merge Commit:** `{{ context.merge_commit }}`
- **Timestamp:** {{ context.timestamp }}

## Merged Commits

{%- if context.merged_commits | length == 0 %}
No commits in this merge.
{%- else %}
| # | Commit SHA |
|---|------------|
{%- for commit_sha in context.merged_commits %}
| {{ loop.index }} | `{{ commit_sha }}` |
{%- endfor %}
{%- endif %}

{%- if context.important_files_modified | length > 0 %}

## Important Files Modified

{%- for file_path in context.important_files_modified %}
- `{{ file_path }}`
{%- endfor %}
{%- endif %}
"""


def merge_context_to_markdown(merge_context: dict) -> str:
    return render_from_template(MERGE_CONTEXT_MARKDOWN_TEMPLATE, merge_context)


def merge_context_to_str(merge_context: dict, format: str, pretty: bool = False):
    if format == "json":
        return (
            json.dumps(merge_context, default=str, indent=2 if pretty else None) + "\n"
        )
    elif format == "markdown":
        return merge_context_to_markdown(merge_context) + "\n"

    return str(merge_context)


MERGE_DESCRIPTION_MARKDOWN_TEMPLATE = """\
# Merge Description

## Summary

{{ context.response.summary }}

## Auto-Merged Files

{%- if context.response.auto_merged | length == 0 %}
No files were auto-merged.
{%- else %}
| File Path | Description |
|-----------|-------------|
{%- for file_path, description in context.response.auto_merged.items() %}
| `{{ file_path }}` | {{ description }} |
{%- endfor %}
{%- endif %}

## Review Notes

{{ context.response.review_notes if context.response.review_notes else "No review notes provided." }}

{%- if context.stats %}

## Stats

{%- if context.stats.models | length > 0 %}
### Models

| Model | Input tokens | Output tokens | Cached tokens | Thoughts tokens | Tool tokens | Total tokens |
|-------|--------------|------------------|---------------|-----------------|-------------|--------------|
{%- for model, stat in context.stats.models.items() %}
| {{ model }} | {{ stat.tokens.input }} | {{ stat.tokens.output }} | {{ stat.tokens.cached }} | {{ stat.tokens.thoughts }} | {{ stat.tokens.tool }} | {{ stat.tokens.total }} |
{%- endfor %}
{%- endif %}
{%- endif %}

{%- if context.agent_info %}

## Agent Info

Executed with '{{ context.agent_info.agent_type }}' agent, version '{{ context.agent_info.version }}'.
{%- endif %}
"""


def merge_description_to_markdown(merge_description: dict) -> str:
    return render_from_template(MERGE_DESCRIPTION_MARKDOWN_TEMPLATE, merge_description)


def merge_description_to_str(merge_description: dict, format: str, pretty: bool = False):
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
        output_str += f"  - Merge Info (use mergai show --merge-info to see the merge info.)\n"
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
        output_str += (
            f"  - Solutions ({count}) (use mergai show --solution to see the conflict solutions.)\n"
        )
    elif "solution" in note:
        output_str += (
            f"  - Solution (use mergai show --solution to see the conflict solution.)\n"
        )

    if "merge_description" in note:
        output_str += (
            f"  - Merge Description (use mergai show --merge-description to see the merge description.)\n"
        )

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
        output_str += f"    - Target Branch: {merge_info.get('target_branch', 'unknown')}\n"
        output_str += f"    - Merge Commit: {merge_info.get('merge_commit', 'unknown')}\n"
    if "merge_context" in note:
        output_str += f"  - Merge Context\n"
        merge_context = note["merge_context"]
        output_str += f"    - Merge Commit: {merge_context.get('merge_commit', 'unknown')}\n"
        output_str += f"    - Merged Commits: {len(merge_context.get('merged_commits', []))}\n"
        important_files = merge_context.get("important_files_modified", [])
        if important_files:
            output_str += f"    - Important Files Modified: {len(important_files)}\n"
    if "conflict_context" in note:
        output_str += f"  - Conflict Context\n"
        conflict_context = note["conflict_context"]
        output_str += (
            f"    - Base Commit: {conflict_context['base_commit']['hexsha']}\n"
        )
        output_str += (
            f"    - Ours Commit: {conflict_context['ours_commit']['hexsha']}\n"
        )
        output_str += (
            f"    - Theirs Commit: {conflict_context['theirs_commit']['hexsha']}\n"
        )
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

def format_commit_info(commit, indent: str = "  ") -> str:
    """Format a commit for display."""
    if not commit:
        return f"{indent}(none)"
    
    authored_date = datetime.fromtimestamp(
        commit.authored_date, tz=timezone.utc
    )
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
    
    authored_date = datetime.fromtimestamp(
        commit.authored_date, tz=timezone.utc
    )
    date_str = authored_date.strftime("%Y-%m-%d")
    
    return f"{git_utils.short_sha(commit.hexsha)} {date_str} {commit.summary}"

def format_number(n: int) -> str:
    """Format a number with thousand separators."""
    return f"{n:,}"


@dataclass
class ParsedBranchName:
    """Parsed components of a mergai branch name.

    This class represents the result of parsing a branch name that was
    generated using BranchNameBuilder. It extracts the original components
    from the branch name string.

    Attributes:
        target_branch: The original target branch name (e.g., "master", "v8.0")
        target_branch_sha: SHA of target branch (short or full depending on branch format)
        merge_commit_sha: SHA of the merge commit (short or full depending on branch format)
        branch_type: The branch type string (e.g., "main", "conflict", "solution")
        full_name: The full original branch name that was parsed
    """

    target_branch: str
    target_branch_sha: str
    merge_commit_sha: str
    branch_type: str
    full_name: str

    def is_standard_type(self) -> bool:
        """Check if the branch type is one of the standard BranchType values."""
        return self.branch_type in [t.value for t in BranchType]


class BranchType(StrEnum):
    """Standard branch types for merge conflict resolution workflow.

    Attributes:
        MAIN: Main working branch where all work for merging a given commit
              will be merged.
        CONFLICT: Branch containing a merge commit with committed merge markers.
        SOLUTION: Branch with solution attempt(s). PRs are created from solution
                  to conflict branch.
    """

    MAIN = "main"
    CONFLICT = "conflict"
    SOLUTION = "solution"


class BranchNameBuilder:
    """Builder for generating standardized branch names.

    The builder uses a format string with tokens that get replaced:
    - %(target_branch) - The target branch being merged into (required)
    - %(target_branch_sha) - Full SHA of the target branch (40 chars)
    - %(target_branch_short_sha) - Short SHA of the target branch (11 chars)
    - %(merge_commit_sha) - Full SHA of the merge commit (40 chars)
    - %(merge_commit_short_sha) - Short SHA of the merge commit (11 chars)
    - %(type) - Branch type identifier

    The format string must contain:
    - %(target_branch)
    - Either %(merge_commit_sha) or %(merge_commit_short_sha)
    - Either %(target_branch_sha) or %(target_branch_short_sha)

    Example format: "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)"
    Produces: "mergai/main-abc12345678-def09876543/solution"

    The class is designed to be instantiated once with all context information,
    then used multiple times to generate different branch names.

    Usage:
        builder = BranchNameBuilder(
            name_format="mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)",
            target_branch="main",
            merge_commit_sha="abc1234567890abcdef1234567890abcdef12345",
            target_branch_sha="def0987654321fedcba0987654321fedcba09876"
        )

        # Get specific branch types via properties
        main_branch = builder.main_branch
        conflict_branch = builder.conflict_branch
        solution_branch = builder.solution_branch

        # Or use methods for more control
        custom_branch = builder.get_branch_name("custom-type")
        typed_branch = builder.get_branch_name_for_type(BranchType.MAIN)
    """

    # Token pattern for format string - matches %(token_name)
    TOKEN_PATTERN = re.compile(r"%\((\w+)\)")

    # Currently supported tokens
    SUPPORTED_TOKENS = {
        "target_branch",
        "target_branch_sha",
        "target_branch_short_sha",
        "merge_commit_sha",
        "merge_commit_short_sha",
        "type",
    }

    # Required token groups - at least one from each group must be present
    REQUIRED_TOKENS = {"target_branch"}
    REQUIRED_MERGE_COMMIT_TOKENS = {"merge_commit_sha", "merge_commit_short_sha"}
    REQUIRED_TARGET_BRANCH_SHA_TOKENS = {"target_branch_sha", "target_branch_short_sha"}

    def __init__(
        self,
        name_format: str,
        target_branch: str,
        merge_commit_sha: str,
        target_branch_sha: str,
    ):
        """Initialize the branch name builder.

        Args:
            name_format: Format string with %(token) placeholders.
            target_branch: Name of the target branch.
            merge_commit_sha: Full SHA of the merge commit (40 chars).
            target_branch_sha: Full SHA of the target branch (40 chars).

        Raises:
            ValueError: If name_format is missing required tokens.
        """
        self._validate_format(name_format)
        self._name_format = name_format
        self._target_branch = target_branch
        self._merge_commit_sha = merge_commit_sha
        self._target_branch_sha = target_branch_sha

    @classmethod
    def _validate_format(cls, name_format: str) -> None:
        """Validate that the format string contains all required tokens.

        Args:
            name_format: The format string to validate.

        Raises:
            ValueError: If required tokens are missing.
        """
        # Extract all tokens from format string
        tokens_in_format = set(cls.TOKEN_PATTERN.findall(name_format))

        # Check for required target_branch token
        if not cls.REQUIRED_TOKENS & tokens_in_format:
            raise ValueError(
                f"Format string must contain %(target_branch). "
                f"Got: {name_format}"
            )

        # Check for at least one merge commit token
        if not cls.REQUIRED_MERGE_COMMIT_TOKENS & tokens_in_format:
            raise ValueError(
                f"Format string must contain either %(merge_commit_sha) or %(merge_commit_short_sha). "
                f"Got: {name_format}"
            )

        # Check for at least one target branch SHA token
        if not cls.REQUIRED_TARGET_BRANCH_SHA_TOKENS & tokens_in_format:
            raise ValueError(
                f"Format string must contain either %(target_branch_sha) or %(target_branch_short_sha). "
                f"Got: {name_format}"
            )

    @classmethod
    def from_config(
        cls,
        config: "BranchConfig",
        target_branch: str,
        merge_commit_sha: str,
        target_branch_sha: str,
    ) -> "BranchNameBuilder":
        """Create a builder from a BranchConfig instance.

        Args:
            config: BranchConfig with the name_format.
            target_branch: Name of the target branch.
            merge_commit_sha: Full SHA of the merge commit.
            target_branch_sha: Full SHA of the target branch.

        Returns:
            Configured BranchNameBuilder instance.
        """
        return cls(
            name_format=config.name_format,
            target_branch=target_branch,
            merge_commit_sha=merge_commit_sha,
            target_branch_sha=target_branch_sha,
        )

    @classmethod
    def parse_branch_name(
        cls,
        branch_name: str,
        name_format: str,
    ) -> Optional[ParsedBranchName]:
        """Parse a branch name back into its components.

        This method reverses the branch name generation process, extracting
        the original target_branch, target_branch_sha, merge_commit_sha, and type from a
        branch name that was created using the given format.

        The parsing is done by converting the format string into a regex
        pattern with named capture groups for each token.

        Args:
            branch_name: The branch name to parse
                        (e.g., "mergai/master-abc12345678-def09876543/main")
            name_format: The format string used to generate branch names
                        (e.g., "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)")

        Returns:
            ParsedBranchName if the branch matches the format, None otherwise.

        Example:
            >>> parsed = BranchNameBuilder.parse_branch_name(
            ...     "mergai/master-abc12345678-def09876543/solution",
            ...     "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)"
            ... )
            >>> parsed.target_branch
            'master'
            >>> parsed.merge_commit_sha
            'abc12345678'
            >>> parsed.target_branch_sha
            'def09876543'
            >>> parsed.branch_type
            'solution'
        """
        # Build regex pattern from format string
        # 1. Escape regex special characters in the format
        # 2. Replace tokens with named capture groups

        # First, escape all regex special characters
        pattern = re.escape(name_format)

        # Define capture patterns for each token type
        # - target_branch: non-greedy match of any characters except the delimiter
        #   that follows it in the format (we use .+? and let the rest of the pattern constrain it)
        # - merge_commit_sha / merge_commit_short_sha: hex characters (git SHA)
        # - target_branch_sha / target_branch_short_sha: hex characters (git SHA)
        # - type: word characters and hyphens (for custom types like "attempt-1")
        # Both full and short variants map to the same capture group
        token_patterns = {
            "target_branch": r"(?P<target_branch>.+?)",
            "target_branch_sha": r"(?P<target_branch_sha>[a-f0-9]+)",
            "target_branch_short_sha": r"(?P<target_branch_sha>[a-f0-9]+)",
            "merge_commit_sha": r"(?P<merge_commit_sha>[a-f0-9]+)",
            "merge_commit_short_sha": r"(?P<merge_commit_sha>[a-f0-9]+)",
            "type": r"(?P<type>[\w-]+)",
        }

        # Replace escaped token placeholders with capture groups
        # re.escape converts %(token) to %\(token\)
        for token, capture_pattern in token_patterns.items():
            escaped_placeholder = re.escape(f"%({token})")
            pattern = pattern.replace(escaped_placeholder, capture_pattern)

        # Anchor the pattern to match the entire string
        pattern = f"^{pattern}$"

        # Try to match the branch name
        match = re.match(pattern, branch_name)
        if match is None:
            return None

        return ParsedBranchName(
            target_branch=match.group("target_branch"),
            target_branch_sha=match.group("target_branch_sha"),
            merge_commit_sha=match.group("merge_commit_sha"),
            branch_type=match.group("type"),
            full_name=branch_name,
        )

    @classmethod
    def parse_branch_name_with_config(
        cls,
        branch_name: str,
        config: "BranchConfig",
    ) -> Optional[ParsedBranchName]:
        """Parse a branch name using format from BranchConfig.

        Convenience method that extracts the name_format from config.

        Args:
            branch_name: The branch name to parse.
            config: BranchConfig with the name_format.

        Returns:
            ParsedBranchName if the branch matches the format, None otherwise.
        """
        return cls.parse_branch_name(branch_name, config.name_format)

    def _build_name(self, branch_type: str) -> str:
        """Build branch name by replacing tokens in format string.

        Args:
            branch_type: The type string to use for %(type) token.

        Returns:
            Formatted branch name with all tokens replaced.
        """
        values = {
            "target_branch": self._target_branch,
            "target_branch_sha": self._target_branch_sha,
            "target_branch_short_sha": git_utils.short_sha(self._target_branch_sha),
            "merge_commit_sha": self._merge_commit_sha,
            "merge_commit_short_sha": git_utils.short_sha(self._merge_commit_sha),
            "type": branch_type,
        }

        def replace_token(match: re.Match) -> str:
            token = match.group(1)
            if token in values:
                return values[token]
            # Keep unknown tokens as-is for future extensibility
            return match.group(0)

        return self.TOKEN_PATTERN.sub(replace_token, self._name_format)

    def get_branch_name(self, branch_type: str) -> str:
        """Get branch name for a custom type string.

        This method allows using arbitrary type strings beyond the
        standard BranchType enum values.

        Args:
            branch_type: Custom type string for the branch.

        Returns:
            Formatted branch name.
        """
        return self._build_name(branch_type)

    def get_branch_name_for_type(self, branch_type: BranchType) -> str:
        """Get branch name for a standard BranchType.

        Args:
            branch_type: One of the standard BranchType enum values.

        Returns:
            Formatted branch name.
        """
        return self._build_name(branch_type.value)

    @property
    def main_branch(self) -> str:
        """Get the main working branch name.

        The main branch is where all work for merging a given commit
        will be merged.
        """
        return self._build_name(BranchType.MAIN)

    @property
    def conflict_branch(self) -> str:
        """Get the conflict branch name.

        The conflict branch contains a merge commit with committed
        merge markers.
        """
        return self._build_name(BranchType.CONFLICT)

    @property
    def solution_branch(self) -> str:
        """Get the solution branch name.

        The solution branch contains solution attempt(s). PRs are
        created from solution to conflict branch.
        """
        return self._build_name(BranchType.SOLUTION)