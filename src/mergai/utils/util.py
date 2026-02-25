"""General utilities for mergai.

This module provides general-purpose utilities. Formatting functions
have been moved to formatters.py.
"""

import io
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from jinja2 import Template
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

from . import git_utils


def gh_auth_token() -> str | None:
    import os

    token = os.getenv("GITHUB_TOKEN")
    if token is not None:
        return token
    token = os.getenv("GH_TOKEN")
    if token is not None:
        return token

    try:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except Exception:
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
    """Render markdown in GitHub-like terminal style.

    Note: This function is kept for potential future use but is no longer
    called by print_or_page() by default.
    """
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
    """Print text to stdout, using a pager if the output is a terminal.

    Note: Markdown is output as raw text (no Rich rendering).
    Users can pipe to their preferred markdown renderer if desired.

    Args:
        text: The text to print.
        format: The format of the text (unused, kept for backward compatibility).
    """
    term_height = shutil.get_terminal_size((80, 20)).lines
    lines = text.count("\n") + 1

    # Output raw text - no Rich markdown rendering
    if not sys.stdout.isatty() or lines + 4 <= term_height:
        print(text)
        return

    pager = os.environ.get("PAGER", "less -FRSX")

    proc = subprocess.Popen(pager, shell=True, stdin=subprocess.PIPE)

    try:
        if proc.stdin is not None:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()


def render_from_template(template_str: str, **kwargs) -> str:
    """Render a Jinja2 template string with the given context.

    Note: For new code, prefer using utils.templates.render_template()
    which loads templates from files.
    """
    template = Template(template_str)
    return template.render(**kwargs)


def load_if_exists(filename: str) -> str:
    """Load file contents if the file exists, otherwise return empty string."""
    if not os.path.exists(filename):
        return ""
    with open(filename, encoding="utf-8") as f:
        return f.read()


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
