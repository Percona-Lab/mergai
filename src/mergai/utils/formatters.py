"""Formatters for mergai data types.

This module provides functions to convert mergai data structures
to various output formats (text, markdown, JSON).
"""

import json
from typing import Optional, Callable
import git

from .output import OutputFormat
from .templates import render_template, get_jinja_env
from ..models import (
    MergeInfo,
    MergeContext,
    ConflictContext,
    ContextSerializationConfig,
    MarkdownConfig,
    MarkdownFormat,
)


# =============================================================================
# Helper Functions
# =============================================================================


def _create_format_sha_func(markdown_config: Optional[MarkdownConfig] = None) -> Callable[[str, bool], str]:
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


def format_solution_author(solution: dict) -> str:
    """Format the author/agent attribution for a solution.

    Args:
        solution: Solution dict that may contain agent_info or author fields.

    Returns:
        Formatted author string for display in markdown.
        - For AI: "Agent (gemini-cli v1.2.3)"
        - For Human: "John Doe <john@example.com>"
        - Fallback: "Unknown"
    """
    if "agent_info" in solution:
        agent = solution["agent_info"]
        agent_type = agent.get("agent_type", "unknown")
        version = agent.get("version", "unknown")
        return f"Agent ({agent_type} v{version})"
    elif "author" in solution:
        author = solution["author"]
        name = author.get("name", "Unknown")
        email = author.get("email", "")
        if email:
            return f"{name} <{email}>"
        return name
    return "Unknown"


def is_human_solution(solution: dict) -> bool:
    """Check if a solution was created by a human (not an agent).

    Args:
        solution: Solution dict that may contain agent_info or author fields.

    Returns:
        True if the solution has an author with type "human", False otherwise.
    """
    return "author" in solution and solution.get("author", {}).get("type") == "human"


def get_comments_stats(comments: dict) -> dict:
    """Get statistics about PR comments grouped by user.

    Args:
        comments: Dict of comment_id -> comment data.

    Returns:
        Dict mapping user names to comment counts.
    """
    stats = {}
    for comment in comments.values():
        stats.setdefault(comment["user"], 0)
        stats[comment["user"]] += 1
    return stats


def get_solution_stats(solution: dict) -> dict:
    """Get statistics about a solution's resolved/unresolved files.

    Args:
        solution: Solution dict with response containing resolved/unresolved.

    Returns:
        Dict with resolved_files, unresolved_files, and total_files counts.
    """
    stats = {}
    resolved_count = len(solution.get("response", {}).get("resolved", {}))
    unresolved_count = len(solution.get("response", {}).get("unresolved", {}))
    stats["resolved_files"] = resolved_count
    stats["unresolved_files"] = unresolved_count
    stats["total_files"] = resolved_count + unresolved_count
    return stats


# =============================================================================
# MergeInfo Formatters
# =============================================================================


def merge_info_to_text(merge_info: MergeInfo) -> str:
    """Convert MergeInfo to text format."""
    return render_template("text", "merge_info", merge_info=merge_info)


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
    return render_template("markdown", "merge_info", merge_info=merge_info, format_sha=format_sha)


def merge_info_to_str(
    merge_info: MergeInfo,
    format: str,
    pretty: bool = False,
    markdown_config: Optional[MarkdownConfig] = None,
) -> str:
    """Convert a MergeInfo object to the specified format.

    Args:
        merge_info: MergeInfo object with merge operation details.
        format: Output format ('json', 'markdown', or 'text').
        pretty: If True, format JSON with indentation.
        markdown_config: Optional MarkdownConfig for PR-style markdown formatting.

    Returns:
        Formatted string representation.
    """
    if format == OutputFormat.JSON.value or format == "json":
        return json.dumps(merge_info.to_dict(), default=str, indent=2 if pretty else None) + "\n"
    elif format == OutputFormat.MARKDOWN.value or format == "markdown":
        return merge_info_to_markdown(merge_info, markdown_config) + "\n"
    else:
        return merge_info_to_text(merge_info) + "\n"


# =============================================================================
# ConflictContext Formatters
# =============================================================================


def conflict_context_to_text(conflict_context: ConflictContext) -> str:
    """Convert ConflictContext to text format."""
    if not isinstance(conflict_context, ConflictContext):
        raise TypeError(
            f"Expected ConflictContext, got {type(conflict_context).__name__}. "
            "Use ConflictContext.from_dict() to create from dict."
        )
    template_data = conflict_context.to_dict(ContextSerializationConfig.template())
    return render_template("text", "conflict_context", conflict_context=template_data)


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
    return render_template("markdown", "conflict_context", conflict_context=template_data, format_sha=format_sha)


def conflict_context_to_str(
    context: ConflictContext,
    format: str,
    pretty: bool = False,
    markdown_config: Optional[MarkdownConfig] = None,
) -> str:
    """Convert ConflictContext to string in specified format.

    Args:
        context: ConflictContext object.
        format: Output format ('json', 'markdown', or 'text').
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

    if format == OutputFormat.JSON.value or format == "json":
        return json.dumps(context.to_dict(), default=str, indent=2 if pretty else None) + "\n"
    elif format == OutputFormat.MARKDOWN.value or format == "markdown":
        return conflict_context_to_markdown(context, markdown_config) + "\n"
    else:
        return conflict_context_to_text(context) + "\n"


# =============================================================================
# MergeContext Formatters
# =============================================================================


def merge_context_to_text(merge_context: MergeContext) -> str:
    """Convert MergeContext to text format."""
    if not isinstance(merge_context, MergeContext):
        raise TypeError(
            f"Expected MergeContext, got {type(merge_context).__name__}. "
            "Use MergeContext.from_dict() to create from dict."
        )
    return render_template("text", "merge_context", merge_context=merge_context)


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
    return render_template("markdown", "merge_context", merge_context=merge_context, format_sha=format_sha)


def merge_context_to_str(
    merge_context: MergeContext,
    format: str,
    pretty: bool = False,
    markdown_config: Optional[MarkdownConfig] = None,
) -> str:
    """Convert MergeContext to string in specified format.

    Args:
        merge_context: MergeContext object.
        format: Output format ('json', 'markdown', or 'text').
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

    if format == OutputFormat.JSON.value or format == "json":
        return json.dumps(merge_context.to_dict(), default=str, indent=2 if pretty else None) + "\n"
    elif format == OutputFormat.MARKDOWN.value or format == "markdown":
        return merge_context_to_markdown(merge_context, markdown_config) + "\n"
    else:
        return merge_context_to_text(merge_context) + "\n"


# =============================================================================
# Solution Formatters
# =============================================================================


def conflict_solution_to_text(solution: dict) -> str:
    """Convert solution data to text format."""
    return render_template("text", "solution", solution=solution)


def conflict_solution_to_markdown(solution: dict) -> str:
    """Convert solution data to markdown format for terminal display."""
    return render_template("markdown", "solution", solution=solution)


def conflict_solution_to_str(solution: dict, format: str, pretty: bool = False) -> str:
    """Convert solution to string in specified format.

    Args:
        solution: Solution dict.
        format: Output format ('json', 'markdown', or 'text').
        pretty: If True, format JSON with indentation.

    Returns:
        Formatted string representation.
    """
    if format == OutputFormat.JSON.value or format == "json":
        return json.dumps(solution, default=str, indent=2 if pretty else None) + "\n"
    elif format == OutputFormat.MARKDOWN.value or format == "markdown":
        return conflict_solution_to_markdown(solution) + "\n"
    else:
        return conflict_solution_to_text(solution) + "\n"


def solution_to_markdown(solution: dict) -> str:
    """Convert solution data to markdown formatted for PR body.

    This format is optimized for GitHub PR descriptions with:
    - Clear sections for summary, resolved files, and unresolved files
    - Author/agent attribution
    - Review notes for developers
    - Stats hidden in a collapsible section (for AI solutions)
    - Simple file lists for human solutions (no description column)

    Args:
        solution: Solution dict with response, stats/author, and agent_info/author.

    Returns:
        Markdown formatted string suitable for PR body.
    """
    return render_template(
        "markdown",
        "solution_pr",
        solution=solution,
        format_solution_author=format_solution_author,
        is_human_solution=is_human_solution,
    )


def solutions_to_text(solutions: list) -> str:
    """Convert a list of solutions to text format."""
    return render_template("text", "solutions", solutions=solutions)


def solutions_to_markdown(solutions: list) -> str:
    """Convert a list of solutions to markdown.

    Handles both AI solutions (with agent_info) and human solutions (with author).

    Args:
        solutions: List of solution dicts.

    Returns:
        Markdown formatted string with all solutions.
    """
    md = "# Solutions"
    md += "\n"

    if len(solutions) == 1:
        md += "\n## Solution 1\n"
        md += solution_to_markdown(solutions[0])
    else:
        for i, solution in enumerate(solutions, 1):
            md += f"\n## Solution {i}\n"
            md += solution_to_markdown(solution)

    return md


# =============================================================================
# PR Comments Formatters
# =============================================================================


def pr_comments_to_text(comments: dict) -> str:
    """Convert PR comments to text format."""
    return render_template("text", "pr_comments", comments=comments)


def pr_comments_to_markdown(comments: dict) -> str:
    """Convert PR comments to markdown format."""
    return render_template("markdown", "pr_comments", comments=comments)


def pr_comments_to_str(comments: dict, format: str, pretty: bool = False) -> str:
    """Convert PR comments to string in specified format.

    Args:
        comments: Dict of comment_id -> comment data.
        format: Output format ('json', 'markdown', or 'text').
        pretty: If True, format JSON with indentation.

    Returns:
        Formatted string representation.
    """
    if format == OutputFormat.JSON.value or format == "json":
        return json.dumps(comments, default=str, indent=2 if pretty else None) + "\n"
    elif format == OutputFormat.MARKDOWN.value or format == "markdown":
        return pr_comments_to_markdown(comments) + "\n"
    else:
        return pr_comments_to_text(comments) + "\n"


# =============================================================================
# User Comment Formatters
# =============================================================================


def user_comment_to_text(user_comment: dict) -> str:
    """Convert user comment to text format."""
    return render_template("text", "user_comment", user_comment=user_comment)


def user_comment_to_markdown(user_comment: dict) -> str:
    """Convert user comment to markdown format."""
    return render_template("markdown", "user_comment", user_comment=user_comment)


def user_comment_to_str(user_comment: dict, format: str, pretty: bool = False) -> str:
    """Convert user comment to string in specified format.

    Args:
        user_comment: User comment dict with date, user, email, body.
        format: Output format ('json', 'markdown', or 'text').
        pretty: If True, format JSON with indentation.

    Returns:
        Formatted string representation.
    """
    if format == OutputFormat.JSON.value or format == "json":
        return json.dumps(user_comment, default=str, indent=2 if pretty else None) + "\n"
    elif format == OutputFormat.MARKDOWN.value or format == "markdown":
        return user_comment_to_markdown(user_comment) + "\n"
    else:
        return user_comment_to_text(user_comment) + "\n"


# =============================================================================
# Merge Description Formatters
# =============================================================================


def merge_description_to_text(merge_description: dict) -> str:
    """Convert merge description to text format."""
    return render_template("text", "merge_description", merge_description=merge_description)


def merge_description_to_markdown(merge_description: dict) -> str:
    """Convert merge description to markdown format."""
    return render_template("markdown", "merge_description", merge_description=merge_description)


def merge_description_to_str(merge_description: dict, format: str, pretty: bool = False) -> str:
    """Convert merge description to string in specified format.

    Args:
        merge_description: Merge description dict.
        format: Output format ('json', 'markdown', or 'text').
        pretty: If True, format JSON with indentation.

    Returns:
        Formatted string representation.
    """
    if format == OutputFormat.JSON.value or format == "json":
        return json.dumps(merge_description, default=str, indent=2 if pretty else None) + "\n"
    elif format == OutputFormat.MARKDOWN.value or format == "markdown":
        return merge_description_to_markdown(merge_description) + "\n"
    else:
        return merge_description_to_text(merge_description) + "\n"


# =============================================================================
# Commit Summary Formatters
# =============================================================================


def commit_note_to_summary_text(commit: git.Commit, note: dict) -> str:
    """Convert commit and note to text summary format.

    This function generates a detailed text summary including all note
    sections with their statistics.

    Args:
        commit: GitPython Commit object.
        note: Note dict from note.json.

    Returns:
        Formatted text string.
    """
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
            author = format_solution_author(solution)
            output_str += f"    [{idx}] {author}: Resolved Files: {stats['resolved_files']}/{stats['total_files']}"
            if stats["unresolved_files"] > 0:
                output_str += f", Unresolved: {stats['unresolved_files']}"
            output_str += "\n"
    elif "solution" in note:
        output_str += f"  - Solution\n"
        stats = get_solution_stats(note["solution"])
        author = format_solution_author(note["solution"])
        output_str += f"    - {author}: Resolved Files: {stats['resolved_files']}/{stats['total_files']}\n"
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

    output_str += f"Message:\n    {commit.message.strip().replace(chr(10), chr(10) + '    ')}\n"

    output_str += "\n"
    return output_str


def commit_note_to_summary_markdown(commit: git.Commit, note: dict) -> str:
    """Convert commit and note to markdown summary format.

    Args:
        commit: GitPython Commit object.
        note: Note dict from note.json.

    Returns:
        Formatted markdown string.
    """
    output_str = f"- Commit: `{commit.hexsha}`\n"
    output_str += f"- Author: {commit.author.name} <{commit.author.email}>\n"
    output_str += f"- Date:   {commit.authored_datetime}\n"
    output_str += f"- Message:\n\n    {commit.message.strip().replace(chr(10), chr(10) + '    ')}\n"
    output_str += f"- Content:\n"
    if "merge_info" in note:
        output_str += f"  - Merge Info (use mergai show --merge-info to see the merge info.)\n"
    if "merge_context" in note:
        output_str += f"  - Merge Context (use mergai show --merge-context to see the merge context.)\n"
    if "conflict_context" in note:
        output_str += f"  - Conflict Context (use mergai show --context to see the conflict context.)\n"
    if "pr_comments" in note:
        output_str += f"  - PR Comments (use mergai show --pr-comments to see the PR comments.)\n"

    # Handle both legacy "solution" and new "solutions" array
    if "solutions" in note:
        count = len(note["solutions"])
        output_str += f"  - Solutions ({count}) (use mergai show --solution to see the conflict solutions.)\n"
        for idx, solution in enumerate(note["solutions"]):
            author = format_solution_author(solution)
            output_str += f"    [{idx}] {author}\n"
    elif "solution" in note:
        author = format_solution_author(note["solution"])
        output_str += f"  - Solution by {author} (use mergai show --solution to see the conflict solution.)\n"

    if "merge_description" in note:
        output_str += f"  - Merge Description (use mergai show --merge-description to see the merge description.)\n"

    output_str += "\n"

    return output_str


def commit_note_to_summary_json(commit: git.Commit, note: dict, pretty: bool = False) -> str:
    """Convert commit and note to JSON summary format.

    Args:
        commit: GitPython Commit object.
        note: Note dict from note.json.
        pretty: If True, format with indentation.

    Returns:
        JSON formatted string.
    """
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


def commit_note_to_summary_str(
    commit: git.Commit, note: dict, format: str, pretty: bool = False
) -> str:
    """Convert commit and note to summary string in specified format.

    Args:
        commit: GitPython Commit object.
        note: Note dict from note.json.
        format: Output format ('json', 'markdown', or 'text').
        pretty: If True, format JSON with indentation.

    Returns:
        Formatted string representation.
    """
    if format == OutputFormat.MARKDOWN.value or format == "markdown":
        return commit_note_to_summary_markdown(commit, note)
    elif format == OutputFormat.JSON.value or format == "json":
        return commit_note_to_summary_json(commit, note, pretty)
    else:
        return commit_note_to_summary_text(commit, note)


def commit_to_summary_str(commit: git.Commit) -> str:
    """Convert a commit (without note) to summary string.

    Args:
        commit: GitPython Commit object.

    Returns:
        Formatted text string.
    """
    output_str = f"commit: {commit.hexsha}\n"
    output_str += f"Author: {commit.author.name} <{commit.author.email}>\n"
    output_str += f"Date:   {commit.authored_datetime}\n"
    output_str += "Content:\n"
    output_str += "  (no note)\n"
    output_str += f"Message:\n    {commit.message.strip().replace(chr(10), chr(10) + '    ')}\n"
    output_str += "\n"
    return output_str
