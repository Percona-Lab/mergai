import io
import os
import shutil
import subprocess
import sys
import json
import git
from jinja2 import Template
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme


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
    if "conflict_context" in note:
        output_str += f"  - Conflict Context (use mergai show --context to see the conflict context.)\n"
    if "pr_comments" in note:
        output_str += (
            f"  - PR Comments (use mergai show --pr-comments to see the PR comments.)\n"
        )

    if "solution" in note:
        output_str += (
            f"  - Solution (use mergai show --solution to see the conflict solution.)\n"
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

    if "conflict_context" in note:
        summary["content"]["conflict_context"] = True
    if "pr_comments" in note:
        summary["content"]["pr_comments"] = True
    if "solution" in note:
        summary["content"]["solution"] = True
    if "user_comment" in note:
        summary["content"]["user_comment"] = True
        summary["user_comment"] = note["user_comment"]

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

    if "solution" in note:
        output_str += f"  - Solution\n"
        stats = get_solution_stats(note["solution"])
        output_str += (
            f"    - Resolved Files: {stats['resolved_files']}/{stats['total_files']}\n"
        )
        if stats["unresolved_files"] > 0:
            output_str += f"    - Unresolved Files: {stats['unresolved_files']}/{stats['total_files']}\n"

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
