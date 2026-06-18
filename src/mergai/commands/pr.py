import json
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote, urlencode

import click
from github import GithubException
from github import PullRequest as GithubPullRequest

from ..app import AppContext
from ..models import MarkdownConfig
from ..utils import formatters, git_utils, util
from ..utils.branch_name_builder import (
    BranchNameBuilder,
    BranchType,
    ParsedBranchName,
)
from ..utils.output import OutputFormat, format_option
from .util import ensure_gh_repo


def _parse_labels_option(labels_arg: str | None, config_labels: list[str]) -> list[str]:
    """Parse the --labels argument and combine with config labels.

    Behavior:
    - None (not specified): Return config_labels as-is
    - "label1,label2" (no +/- prefix): Override - return only these labels
    - "+label1,-label2,label3": Modify config_labels
        - +label1: Add label1 to config_labels
        - -label2: Remove label2 from config_labels
        - label3 (no prefix in modifier mode): Treat as +label3

    Args:
        labels_arg: The --labels argument value, or None if not specified.
        config_labels: Labels from the config file.

    Returns:
        Final list of labels to apply.
    """
    if labels_arg is None:
        return list(config_labels)

    parts = [p.strip() for p in labels_arg.split(",") if p.strip()]

    if not parts:
        return list(config_labels)

    # Check if any part has +/- prefix (modifier mode)
    has_modifiers = any(p.startswith("+") or p.startswith("-") for p in parts)

    if not has_modifiers:
        # Override mode: return exactly what was specified
        return parts

    # Modifier mode: start with config labels and modify
    result = set(config_labels)
    for part in parts:
        if part.startswith("-"):
            result.discard(part[1:])  # Remove (if exists)
        elif part.startswith("+"):
            result.add(part[1:])  # Add
        else:
            result.add(part)  # No prefix in modifier mode = add

    return list(result)


def _build_pr_url(
    repo_str: str,
    title: str,
    body: str,
    head: str,
    base: str,
    labels: list[str] | None = None,
) -> str:
    """Build a GitHub URL for creating a PR with pre-filled information.

    Args:
        repo_str: Repository in 'owner/repo' format.
        title: PR title.
        body: PR body/description.
        head: Source branch name.
        base: Target branch name.
        labels: Optional list of labels to apply to the PR.

    Returns:
        GitHub compare URL with query parameters for PR creation.
    """
    # GitHub compare URL format: https://github.com/{owner}/{repo}/compare/{base}...{head}
    # Query params: expand=1 (opens PR form), title, body, labels
    base_url = f"https://github.com/{repo_str}/compare/{quote(base, safe='')}...{quote(head, safe='')}"
    params = {
        "expand": "1",  # Automatically expand the PR creation form
        "title": title,
        "body": body,
    }
    if labels:
        params["labels"] = ",".join(labels)
    return f"{base_url}?{urlencode(params)}"


class PRBodyTooLongError(click.ClickException):
    """GitHub rejected the PR because its body exceeds the 65,536 char limit.

    Raised by :func:`_create_pr` so callers can optionally retry with
    ``skip_commit_list=True`` (dropping the per-merged-commit table, which is
    the usual cause of an oversized body).
    """


def _create_pr(
    app: AppContext,
    title: str,
    body: str,
    head: str,
    base: str,
    dry_run: bool = False,
    url_only: bool = False,
    labels: list[str] | None = None,
):

    if url_only:
        url = _build_pr_url(app.gh_repo.full_name, title, body, head, base, labels)
        click.echo(f"Open this URL to create the PR:\n\n{url}")
        return None

    labels_str = ", ".join(labels) if labels else "(none)"
    click.echo(
        f"Creating PR:\n"
        f"    repo: {app.gh_repo.full_name}\n"
        f"    from: {head}\n"
        f"      to: {base}\n"
        f"   title: {title}\n"
        f"  labels: {labels_str}"
    )

    if dry_run:
        click.echo("--- body ---")
        util.print_or_page(body, format="markdown")
        click.echo("--- end ---")
        return None

    try:
        pr = app.gh_repo.create_pull(title=title, body=body, head=head, base=base)
        if labels:
            pr.add_to_labels(*labels)
        click.echo(f"PR created: {pr.html_url}")
        return pr
    except GithubException as e:
        if e.status == 422:
            data = e.data if isinstance(e.data, dict) else {}
            errors = data.get("errors") or []
            fields = {err.get("field") for err in errors if isinstance(err, dict)}
            if fields and fields <= {"base", "head"}:
                raise click.ClickException(
                    "GitHub rejected the PR: branch(es) not found on remote. Push your branches first."
                ) from e
            if "body" in fields:
                # The merged-commits table can push the body past GitHub's
                # 65,536 character limit. Surface a typed error so callers
                # opting into --skip-commit-list-on-failure can retry without it.
                raise PRBodyTooLongError(_format_github_error(e)) from e
        raise click.ClickException(_format_github_error(e)) from e


def _format_github_error(e: GithubException) -> str:
    """Format a GithubException with details from response payload.

    GitHub's 422 validation errors carry the actual reason in
    `errors[*].message` (or `errors[*].code`), not in the top-level
    `message` field. Surface those so callers see *why* validation failed
    (e.g. "Body is too long (maximum is 65536 characters)") instead of the
    generic "Validation Failed".
    """
    data = e.data if isinstance(e.data, dict) else {}
    top = data.get("message") or str(e)
    parts: list[str] = []
    for err in data.get("errors") or []:
        if not isinstance(err, dict):
            continue
        # Prefer human message; fall back to field+code.
        msg = err.get("message")
        if not msg:
            field = err.get("field")
            code = err.get("code")
            if field and code:
                msg = f"{field}: {code}"
            elif code:
                msg = code
            elif field:
                msg = field
        if msg:
            parts.append(msg)
    detail = "; ".join(parts)
    if detail:
        return f"GitHub API error ({e.status}): {top} — {detail}"
    return f"GitHub API error ({e.status}): {top}"


def _build_solutions_pr_body(app: AppContext, skip_commit_list: bool = False) -> str:
    markdown_config = MarkdownConfig.for_pr(app.repo)

    body = formatters.merge_info_to_markdown(app.note.merge_info, markdown_config)
    body += "\n\n"
    if app.note.has_merge_context and app.note.merge_context is not None:
        body += formatters.merge_context_to_markdown(
            app.note.merge_context,
            markdown_config,
            include_commit_list=not skip_commit_list,
        )
        body += "\n\n"
    if app.note.has_merge_description and app.note.merge_description is not None:
        body += formatters.merge_description_to_markdown(app.note.merge_description)
        body += "\n\n"
    if app.note.has_conflict_context and app.note.conflict_context is not None:
        body += formatters.conflict_context_to_markdown(
            app.note.conflict_context, markdown_config
        )
        body += "\n\n"
    if app.note.has_solutions and app.note.solutions is not None:
        body += formatters.solutions_to_markdown(app.note.solutions)
    body += "\n\n"
    body += f"---\n\n*note created with mergai {app.note.mergai_version}*\n"

    return body


def _build_merge_pr_body(app: AppContext, skip_commit_list: bool = False) -> str:
    markdown_config = MarkdownConfig.for_pr(app.repo)

    body = formatters.merge_info_to_markdown(app.note.merge_info, markdown_config)
    body += "\n\n"
    if app.note.has_merge_context and app.note.merge_context is not None:
        body += formatters.merge_context_to_markdown(
            app.note.merge_context,
            markdown_config,
            include_commit_list=not skip_commit_list,
        )
        body += "\n\n"
    if app.note.has_merge_description and app.note.merge_description is not None:
        body += formatters.merge_description_to_markdown(app.note.merge_description)
        body += "\n\n"
    body += f"---\n\n*note created with mergai {app.note.mergai_version}*\n"

    return body


def _create_solution_pr(
    app: AppContext,
    dry_run: bool,
    url_only: bool = False,
    skip_body: bool = False,
    skip_commit_list: bool = False,
    labels: list[str] | None = None,
) -> None:
    """Create a PR from the current branch (with existing solution commits) to the conflict branch."""

    if (
        not app.note.has_solutions
        or app.note.solutions is None
        or len(app.note.solutions) == 0
    ):
        raise click.ClickException("No solutions found. Run 'mergai resolve' first.")

    if app.note.get_uncommitted_solution() is not None:
        raise click.ClickException(
            "You have uncommitted solution(s). Run 'mergai commit solution' first."
        )

    title = app.pr_titles.solution_title

    body = (
        ""
        if skip_body
        else _build_solutions_pr_body(app, skip_commit_list=skip_commit_list)
    )

    _create_pr(
        app,
        title,
        body,
        app.branches.solution_branch,
        app.branches.conflict_branch,
        dry_run=dry_run,
        url_only=url_only,
        labels=labels,
    )


def _build_main_pr_body(app: AppContext, skip_commit_list: bool = False) -> str:
    """Build PR body for main PR from merge_context or conflict resolution data."""
    # If we have solutions (from any source - AI, human, or synced), include them
    if app.note.has_solutions:
        return _build_solutions_pr_body(app, skip_commit_list=skip_commit_list)

    # No solutions - use merge PR body if we have merge_context
    if app.note.has_merge_context:
        return _build_merge_pr_body(app, skip_commit_list=skip_commit_list)

    raise click.ClickException(
        "No merge_context or solutions found. "
        "Run 'mergai context create merge' for non-conflict merges, "
        "or run 'mergai resolve' to generate solutions."
    )


def _create_main_pr(
    app: AppContext,
    dry_run: bool,
    url_only: bool = False,
    skip_body: bool = False,
    skip_commit_list: bool = False,
    labels: list[str] | None = None,
) -> None:
    """Create a PR from the main branch to target_branch (merge or conflict resolution)."""

    title = app.pr_titles.main_title

    body = (
        "" if skip_body else _build_main_pr_body(app, skip_commit_list=skip_commit_list)
    )

    _create_pr(
        app,
        title,
        body,
        app.branches.main_branch,
        app.branches.target_branch,
        dry_run=dry_run,
        url_only=url_only,
        labels=labels,
    )


def _build_semantic_pr_body(app: AppContext, skip_commit_list: bool = False) -> str:
    """Build PR body for a semantic PR (semantic-conflict fixes against main).

    Semantic-conflict fixes are recorded as solutions (typically ``ci_fix``)
    in the note, so the solutions body - which renders merge info, contexts and
    the per-solution summaries - is the right view for reviewing them.
    """
    return _build_solutions_pr_body(app, skip_commit_list=skip_commit_list)


def _create_semantic_pr(
    app: AppContext,
    dry_run: bool,
    url_only: bool = False,
    skip_body: bool = False,
    skip_commit_list: bool = False,
    labels: list[str] | None = None,
) -> None:
    """Create a PR from the semantic branch to the main branch.

    Semantic conflicts surface after a clean merge (failing build/tests). Their
    fixes live on the semantic branch and are reviewed against the main branch
    before being squashed into the merge commit by finalize.
    """

    title = app.pr_titles.semantic_title

    body = (
        ""
        if skip_body
        else _build_semantic_pr_body(app, skip_commit_list=skip_commit_list)
    )

    _create_pr(
        app,
        title,
        body,
        app.branches.semantic_branch,
        app.branches.main_branch,
        dry_run=dry_run,
        url_only=url_only,
        labels=labels,
    )


@dataclass(frozen=True)
class _PRKind:
    """Per-PR-kind dispatch table entry.

    Maps a PR type to how it is created, how its body is built, and which
    ``app.branches`` attributes give its expected head/base branches. Adding a
    new PR kind is a single entry in ``_PR_KINDS`` rather than a new arm in
    every ``if pr_type == ...`` chain.
    """

    create: Callable[..., None]
    body_builder: Callable[[AppContext], str]
    head_attr: str
    base_attr: str


_PR_KINDS: dict[str, _PRKind] = {
    "main": _PRKind(
        _create_main_pr, _build_main_pr_body, "main_branch", "target_branch"
    ),
    "solution": _PRKind(
        _create_solution_pr,
        _build_solutions_pr_body,
        "solution_branch",
        "conflict_branch",
    ),
    "semantic": _PRKind(
        _create_semantic_pr, _build_semantic_pr_body, "semantic_branch", "main_branch"
    ),
}


def _pr_kind(pr_type: str) -> _PRKind:
    """Resolve a PR-type string to its dispatch entry (defaults to ``main``)."""
    return _PR_KINDS.get(pr_type.lower(), _PR_KINDS["main"])


@click.group()
@click.pass_obj
@click.option(
    "--repo",
    "repo",
    type=str,
    required=False,
    envvar="GH_REPO",
    help="The repository where the PR is located.",
)
def pr(app: AppContext, repo: str | None):
    ensure_gh_repo(app, repo)


@pr.command()
@click.pass_obj
@click.option("--dry-run", is_flag=True, default=False, help="Dry run the PR creation.")
@click.option(
    "--url-only",
    is_flag=True,
    default=False,
    help="Print a URL to create the PR manually on GitHub instead of creating it via API.",
)
@click.option(
    "--skip-body",
    is_flag=True,
    default=False,
    help="Skip creating a body for the PR (create with empty body).",
)
@click.option(
    "--skip-commit-list",
    is_flag=True,
    default=False,
    help=(
        "Omit the per-merged-commit table from the PR body. "
        "Use this when the body would otherwise exceed GitHub's 65,536 "
        "character limit (typical for merges with hundreds of commits)."
    ),
)
@click.option(
    "--skip-commit-list-on-failure",
    is_flag=True,
    default=False,
    help=(
        "If PR creation fails because the body exceeds GitHub's 65,536 "
        "character limit, retry once automatically with --skip-commit-list."
    ),
)
@click.option(
    "--labels",
    "labels_arg",
    type=str,
    default=None,
    help=(
        "Labels to apply to the PR. Overrides config labels by default. "
        "Use +label to add to config labels, -label to remove from config labels. "
        "Examples: --labels=urgent,review (override), --labels=+urgent,-automated (modify)."
    ),
)
@click.option(
    "--no-labels",
    is_flag=True,
    default=False,
    help="Skip applying any labels (ignores config labels).",
)
@click.argument(
    "pr_type", type=click.Choice(["main", "solution", "semantic"], case_sensitive=False)
)
def create(
    app: AppContext,
    pr_type: str,
    dry_run: bool,
    url_only: bool,
    skip_body: bool,
    skip_commit_list: bool,
    skip_commit_list_on_failure: bool,
    labels_arg: str | None,
    no_labels: bool,
):
    """Create a pull request.

    PR_TYPE specifies which type of PR to create:

    \b
    - main: Creates a PR from the main branch (created with 'mergai branch create main')
      against the target_branch from merge_info. Auto-detects the merge scenario:

      \b
      1. No conflict: Uses merge_info and merge_context for the PR body.
         Requires 'mergai context create merge' to have been run.

      \b
      2. Conflict resolution: When merge_context is not available but
         conflict_context and solutions are present (after conflicts were
         resolved and squashed), uses those for the PR body instead.

    - solution: Creates a PR from the current branch (typically a solution branch)
      against the conflict branch. Uses the solution data from note for title and body.
      The PR body includes solution summary, resolved/unresolved files, review notes,
      and agent stats (hidden in a collapsible section).

    \b
    - semantic: Creates a PR from the semantic branch against the main branch.
      Use this for semantic-conflict fixes - a clean merge whose result fails to
      build/test - so the fixes can be reviewed before finalize squashes them
      into the merge commit. Uses the note's solutions for title and body.

    \b
    Options:
        --dry-run   Show what would be created without actually creating the PR.
        --url-only  Print a GitHub URL to create the PR manually. When you open
                    this URL, GitHub will show the PR creation form with all
                    fields pre-filled (title, body, branches). You can review
                    and edit everything before submitting.
        --skip-body Skip creating a body for the PR (create with empty body).
        --skip-commit-list
                    Omit the per-merged-commit table from the PR body. Use
                    this when a normal `pr create` fails GitHub validation
                    because the body exceeds the 65,536 character limit.
        --skip-commit-list-on-failure
                    Retry once with --skip-commit-list if PR creation fails
                    because the body exceeds GitHub's 65,536 character limit.
        --labels    Labels to apply to the PR. By default, uses labels from config.
                    Specifying labels without +/- prefix overrides config labels.
                    Use +label to add, -label to remove from config labels.
        --no-labels Skip applying any labels (ignores config labels).

    \b
    Examples:
        mergai pr create main            # Create PR from main branch to target_branch
        mergai pr create solution        # Create PR from solution branch to conflict branch
        mergai pr create main --url-only # Get URL to create PR manually on GitHub
        mergai pr create main --skip-body # Create PR with empty body
        mergai pr create main --skip-commit-list  # Drop merged-commits table from body
        mergai pr create main --skip-commit-list-on-failure  # Auto-retry if body too long
        mergai pr create main --labels=urgent,review  # Override config labels
        mergai pr create main --labels=+urgent,-auto  # Add/remove from config labels
        mergai pr create main --no-labels             # Create PR without any labels
    """
    if dry_run and url_only:
        raise click.ClickException("Cannot use --dry-run and --url-only together.")

    if no_labels and labels_arg is not None:
        raise click.ClickException("Cannot use --no-labels and --labels together.")

    # Get config labels based on PR type (config attrs are named per kind)
    pr_type_config = getattr(app.config.pr, pr_type.lower(), app.config.pr.main)

    config_labels = list(pr_type_config.labels)
    if app.note.has_unresolved_conflicts:
        config_labels.extend(
            lbl
            for lbl in pr_type_config.labels_on_unresolved
            if lbl not in config_labels
        )

    # Compute final labels
    if no_labels:
        final_labels: list[str] = []
    else:
        final_labels = _parse_labels_option(labels_arg, config_labels)

    def _dispatch(skip_list: bool) -> None:
        _pr_kind(pr_type).create(
            app, dry_run, url_only, skip_body, skip_list, final_labels
        )

    try:
        _dispatch(skip_commit_list)
    except PRBodyTooLongError:
        # Already dropping the commit list -> nothing left to trim; re-raise.
        if not skip_commit_list_on_failure or skip_commit_list:
            raise
        click.echo(
            "PR body exceeds GitHub's 65,536 character limit; retrying without "
            "the merged-commits table (--skip-commit-list).",
            err=True,
        )
        _dispatch(True)


def _head_filter(app: AppContext, branch: str) -> str:
    """Return the ``OWNER:branch`` head qualifier GitHub's pulls API expects.

    ``GET /pulls?head=`` wants ``OWNER:ref``; a bare branch name can raise a
    422 or silently fail to filter server-side. Callers still re-check
    ``pr.head.ref`` locally as defense in depth.
    """
    owner = app.gh_repo.full_name.split("/")[0]
    return f"{owner}:{branch}"


def get_prs_for_current_branch(app: AppContext) -> list[GithubPullRequest.PullRequest]:
    branch = git_utils.get_current_branch(app.repo)
    pulls = app.gh_repo.get_pulls(sort="created", head=_head_filter(app, branch))
    return list(filter(lambda pr: pr.head.ref == branch, pulls))


def _resolve_open_pr_for_type(app: AppContext, pr_type: str):
    """Return the open PR whose head is the branch for ``pr_type``, or None.

    Resolves the branch name for the given type from the note/config and
    returns the first matching open PR. Also re-checks ``head.ref`` locally.
    """
    branch = app.branches.get_branch_name(pr_type)
    pulls = app.gh_repo.get_pulls(
        state="open", sort="created", head=_head_filter(app, branch)
    )
    matches = [p for p in pulls if p.head.ref == branch]
    return matches[0] if matches else None


def _resolve_pr_type_arg(app: AppContext, pr_type: str | None) -> str:
    """Return ``pr_type`` lowercased, auto-detecting from the branch if None."""
    if pr_type is not None:
        return pr_type.lower()
    detected = _detect_pr_type_from_branch(app)
    if detected is None:
        raise click.ClickException(
            "Cannot auto-detect PR type from current branch. "
            "Please specify 'main', 'solution', or 'semantic' explicitly."
        )
    return detected


@pr.command()
@click.pass_obj
@click.argument(
    "pr_type",
    type=click.Choice(["main", "solution", "semantic"], case_sensitive=False),
    required=False,
)
def number(app: AppContext, pr_type: str | None):
    """Print the number of the open PR for a branch type.

    PR_TYPE is auto-detected from the current branch when omitted. Prints
    nothing and exits 0 when no open PR exists, so callers can write:

    \b
        SEMANTIC_PR=$(mergai pr --repo owner/name number semantic)
    """
    pr_type = _resolve_pr_type_arg(app, pr_type)
    pr = _resolve_open_pr_for_type(app, pr_type)
    if pr is not None:
        click.echo(pr.number)


@pr.command()
@click.pass_obj
@click.option("--body", required=True, help="Comment body (markdown).")
@click.option(
    "--allow-missing",
    is_flag=True,
    default=False,
    help="Warn and exit 0 if no open PR exists for the type, instead of failing.",
)
@click.argument(
    "pr_type",
    type=click.Choice(["main", "solution", "semantic"], case_sensitive=False),
    required=False,
)
def comment(app: AppContext, pr_type: str | None, body: str, allow_missing: bool):
    """Post a comment on the open PR for a branch type.

    Resolves the open PR whose head is the branch for PR_TYPE (auto-detected
    from the current branch when omitted) and posts BODY as a comment. With
    --allow-missing, a missing PR is a warning + no-op (exit 0) rather than an
    error -- handy for best-effort cross-PR notifications from CI.

    \b
        mergai pr --repo owner/name comment main --body "Fixes opened in #123"
    """
    pr_type = _resolve_pr_type_arg(app, pr_type)
    pr = _resolve_open_pr_for_type(app, pr_type)
    if pr is None:
        msg = f"No open {pr_type} PR found."
        if allow_missing:
            click.echo(f"warning: {msg} Skipping comment.", err=True)
            return
        raise click.ClickException(msg)
    pr.create_issue_comment(body)
    click.echo(f"Commented on PR #{pr.number}: {pr.html_url}")


def _sha_matches(parsed_sha: str, requested_sha: str) -> bool:
    """Whether a parsed branch SHA matches a requested SHA by prefix.

    The branch carries a short SHA (``merge_commit_short_sha``, 11 chars by
    default) while ``--sha`` may be passed as a full or short SHA, so match if
    either is a prefix of the other. SHAs are compared case-insensitively.
    """
    parsed_sha = parsed_sha.lower()
    requested_sha = requested_sha.lower()
    return parsed_sha.startswith(requested_sha) or requested_sha.startswith(parsed_sha)


_STATE_TO_GH = {"open": "open", "closed": "closed", "merged": "closed", "all": "all"}


def _branch_carries_full_sha(name_format: str) -> bool:
    """Whether the branch name embeds the full merge SHA (not the short form).

    ``%(merge_commit_sha)`` is not a substring of ``%(merge_commit_short_sha)``,
    so this cleanly distinguishes the two configured formats.
    """
    return "%(merge_commit_sha)" in name_format


def _resolve_full_merge_sha(
    app: AppContext, pr: GithubPullRequest.PullRequest, parsed: ParsedBranchName
) -> str | None:
    """Best-effort full merge commit SHA for a PR.

    Prefers the SHA recorded in the PR's mergai note (read from its head
    commit); falls back to the branch-encoded SHA when the configured branch
    format already embeds the full SHA. Returns ``None`` when neither yields a
    full SHA (note not fetched and the branch only carries the short SHA).
    """
    note = app.try_get_note_from_commit(pr.head.sha)
    if note is not None:
        return note.merge_info.merge_commit_sha
    if _branch_carries_full_sha(app.config.branch.name_format):
        return parsed.merge_commit_sha
    return None


def _emit_missing_note_hint(enriched) -> None:
    """Warn (to stderr) if any listed PR's full merge SHA couldn't be resolved.

    The full merge SHA comes from the note, which lives in ``refs/notes/mergai``
    and is not fetched by default - so suggest updating notes. PRs whose branch
    format already embeds the full SHA never count here.
    """
    missing = sum(1 for _, _, full_sha in enriched if full_sha is None)
    if missing == 0:
        return
    click.echo(
        f"hint: {missing} PR(s) have no local mergai note; their full merge SHA is "
        "unavailable. Update notes with: "
        "git fetch origin 'refs/notes/mergai:refs/notes/mergai'",
        err=True,
    )


@pr.command("list")
@click.pass_obj
@click.option(
    "--sha",
    "sha",
    type=str,
    default=None,
    help="Only list PRs for this picked upstream SHA (matched by prefix).",
)
@click.option(
    "--state",
    "state",
    type=click.Choice(list(_STATE_TO_GH), case_sensitive=False),
    default="open",
    show_default=True,
    help="Which PR states to include.",
)
@click.option(
    "--type",
    "pr_type",
    type=click.Choice(
        ["main", "conflict", "solution", "semantic"], case_sensitive=False
    ),
    default=None,
    help="Only list PRs whose head branch is of this type.",
)
@click.option(
    "--quiet",
    "-q",
    "quiet",
    is_flag=True,
    default=False,
    help="Print only PR numbers, one per line; nothing when none match.",
)
@format_option(default=OutputFormat.TEXT)
def list_prs(
    app: AppContext,
    sha: str | None,
    state: str,
    pr_type: str | None,
    quiet: bool,
    format: str,
):
    """List mergai-managed pull requests.

    Only PRs whose head branch parses as a mergai branch name (per the
    configured branch name_format) are listed; arbitrary repo PRs are never
    shown. Lists open PRs by default. A run with no matches still exits 0
    (printing nothing under --quiet), so callers can capture output in a shell
    variable; GitHub API errors still surface as failures.

    \b
        # is any merge in progress?
        mergai pr --repo owner/name list -q
        # is this same pick in progress?
        mergai pr --repo owner/name list --sha <SHA> -q
    """
    pulls = app.gh_repo.get_pulls(state=_STATE_TO_GH[state], sort="created")

    matches = []
    for pr in pulls:
        parsed = BranchNameBuilder.parse_branch_name_with_config(
            pr.head.ref, app.config.branch
        )
        if parsed is None:
            continue
        if state == "merged" and pr.merged_at is None:
            continue
        if sha is not None and not _sha_matches(parsed.merge_commit_sha, sha):
            continue
        if pr_type is not None and parsed.branch_type != pr_type.lower():
            continue
        matches.append((pr, parsed))

    if quiet:
        for pr, _ in matches:
            click.echo(pr.number)
        return

    # Resolve the full merge commit SHA for the human/JSON output only - the
    # quiet path above never needs it.
    enriched = [
        (pr, parsed, _resolve_full_merge_sha(app, pr, parsed)) for pr, parsed in matches
    ]

    if format == OutputFormat.JSON.value:
        click.echo(
            json.dumps(
                [
                    {
                        "number": pr.number,
                        "url": pr.html_url,
                        "title": pr.title,
                        "state": pr.state,
                        "merged": pr.merged_at is not None,
                        "head": pr.head.ref,
                        "base": pr.base.ref,
                        "target_branch": parsed.target_branch,
                        "branch_merge_commit_sha": parsed.merge_commit_sha,
                        "merge_commit_sha": full_sha,
                        "type": parsed.branch_type,
                    }
                    for pr, parsed, full_sha in enriched
                ],
                indent=2,
                default=str,
            )
        )
        _emit_missing_note_hint(enriched)
        return

    if not enriched:
        click.echo("No matching PRs.")
        return
    for pr, _, full_sha in enriched:
        show_prs([pr])
        click.echo(f"  Merge SHA  : {full_sha or '(note unavailable)'}")
    _emit_missing_note_hint(enriched)


def show_prs(prs):
    for pr in prs:
        click.echo(f"#{pr.number}: ({pr.html_url})")
        click.echo(f"  Title      : {pr.title}")
        click.echo(f"  Created at : {pr.created_at}")
        click.echo(f"  Author     : {pr.user.login}")
        click.echo(f"  Head       : {pr.head.ref}")
        click.echo(f"  Base       : {pr.base.ref}")
        click.echo(f"  State      : {pr.state}")


@pr.command()
@click.pass_obj
@click.option(
    "--pr-number",
    "-n",
    type=int,
    default=None,
    help="PR number to show directly instead of searching by branch.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force show even if PR doesn't match current branch context.",
)
def show(app: AppContext, pr_number: int | None, force: bool):
    """Show pull request details.

    Without arguments, shows open PRs for the current branch.
    With --pr-number, shows details for that specific PR.

    \b
    Options:
        --pr-number, -n   PR number to show directly.
        --force, -f       Show the PR even if it doesn't match the current branch.

    \b
    Examples:
        mergai pr --repo owner/name show              # Show PRs for current branch
        mergai pr --repo owner/name show -n 123       # Show PR #123
        mergai pr --repo owner/name show -n 123 -f    # Show PR #123 even if branch mismatch
    """
    try:
        if pr_number is not None:
            # Fetch PR directly by number
            try:
                pr = app.gh_repo.get_pull(pr_number)
            except GithubException as e:
                msg = (
                    e.data.get("message", str(e))
                    if isinstance(e.data, dict)
                    else str(e)
                )
                raise click.ClickException(
                    f"Failed to fetch PR #{pr_number}: {msg}"
                ) from e

            # Validate context (only branch check for show)
            warnings = _validate_pr_context(app, pr)
            if warnings:
                if not force:
                    msg = "\n".join(f"  - {w}" for w in warnings)
                    raise click.ClickException(
                        f"PR #{pr_number} context mismatch:\n{msg}\n\nUse --force to show anyway."
                    )
                for w in warnings:
                    click.echo(f"Warning: {w}", err=True)

            show_prs([pr])
            return

        # Original behavior: show PRs for current branch
        prs = get_prs_for_current_branch(app)
        if len(prs) == 0:
            click.echo("No open pull requests found for the current branch.")
            exit(0)

        if len(prs) > 1:
            click.echo("Multiple open pull requests found for the current branch:")

        show_prs(prs)
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


def _detect_pr_type_from_branch(app: AppContext) -> str | None:
    """Detect PR type from current branch name.

    Parses the current branch name using the branch config format to determine
    if it's a 'main' or 'solution' branch.

    Args:
        app: AppContext with config and repo.

    Returns:
        'main' or 'solution' if detected, None otherwise.
    """
    current_branch = git_utils.get_current_branch(app.repo)
    parsed = BranchNameBuilder.parse_branch_name_with_config(
        current_branch, app.config.branch
    )

    if parsed is None:
        return None

    if parsed.branch_type == BranchType.MAIN.value:
        return "main"
    elif parsed.branch_type == BranchType.SOLUTION.value:
        return "solution"
    elif parsed.branch_type == BranchType.SEMANTIC.value:
        return "semantic"

    return None


def _validate_pr_context(
    app: AppContext,
    pr: GithubPullRequest.PullRequest,
    pr_type: str | None = None,
) -> list[str]:
    """Validate that a PR matches the expected context from the note.

    Checks if the PR's head and base branches match the expected branches
    based on the PR type and the current mergai note.

    Args:
        app: AppContext with config, repo, and note.
        pr: The GitHub PR to validate.
        pr_type: The PR type ('main' or 'solution'). If None, validation
                 only checks against the current git branch.

    Returns:
        List of warning messages. Empty list if PR matches context.
    """
    warnings = []

    # Check against current git branch
    current_branch = git_utils.get_current_branch(app.repo)
    if pr.head.ref != current_branch:
        warnings.append(
            f"PR head branch '{pr.head.ref}' doesn't match current branch '{current_branch}'"
        )

    # If we have a pr_type and note, validate against expected branches
    if pr_type is not None:
        try:
            kind = _pr_kind(pr_type)
            expected_head = getattr(app.branches, kind.head_attr)
            expected_base = getattr(app.branches, kind.base_attr)

            if pr.head.ref != expected_head:
                warnings.append(
                    f"PR head branch '{pr.head.ref}' doesn't match expected '{expected_head}' for {pr_type} PR"
                )
            if pr.base.ref != expected_base:
                warnings.append(
                    f"PR base branch '{pr.base.ref}' doesn't match expected '{expected_base}' for {pr_type} PR"
                )
        except click.ClickException:
            # Note or branches not available - skip branch validation
            warnings.append(
                "Cannot validate PR branches: mergai note not found or incomplete"
            )

    return warnings


def _find_pr_for_branch(
    app: AppContext, head_branch: str, base_branch: str
) -> GithubPullRequest.PullRequest | None:
    """Find an open PR from head_branch to base_branch.

    Args:
        app: AppContext with GitHub repo.
        head_branch: Source branch name.
        base_branch: Target branch name.

    Returns:
        PullRequest if found, None otherwise.
    """
    pulls = app.gh_repo.get_pulls(
        state="open", head=_head_filter(app, head_branch), base=base_branch
    )
    for pr in pulls:
        if pr.head.ref == head_branch and pr.base.ref == base_branch:
            return pr
    return None


@pr.command()
@click.pass_obj
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show the new body without updating the PR.",
)
@click.option(
    "--pr-number",
    "-n",
    type=int,
    default=None,
    help="PR number to update directly instead of searching by branch.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force update even if PR doesn't match current context.",
)
@click.argument(
    "pr_type",
    type=click.Choice(["main", "solution", "semantic"], case_sensitive=False),
    required=False,
)
def update(
    app: AppContext,
    pr_number: int | None,
    pr_type: str | None,
    dry_run: bool,
    force: bool,
):
    """Update an existing pull request's body.

    Without arguments, finds the PR by branch names and auto-detects the PR type.
    With --pr-number, updates that specific PR directly.

    PR_TYPE specifies which type of body to generate ('main', 'solution', or
    'semantic'). If not provided, it is auto-detected from the current branch.

    The PR body is rebuilt using the current note data, including any solutions
    added after the PR was created (e.g., human solutions from 'mergai commit sync').

    \b
    Arguments:
        PR_TYPE       Optional type of PR body to generate ('main' or 'solution').

    \b
    Options:
        --dry-run       Show the new body without updating the PR.
        --pr-number, -n PR number to update directly.
        --force, -f     Update even if PR doesn't match current context.

    \b
    Examples:
        mergai pr --repo owner/name update              # Auto-detect PR from branch
        mergai pr --repo owner/name update main         # Update main PR body
        mergai pr --repo owner/name update solution     # Update solution PR body
        mergai pr --repo owner/name update -n 123       # Update PR #123
        mergai pr --repo owner/name update -n 123 main  # Update PR #123 with main body
        mergai pr --repo owner/name update -n 123 -f    # Update PR #123 even if mismatch
        mergai pr --repo owner/name update --dry-run    # Preview new body
    """
    # Auto-detect PR type if not provided
    if pr_type is None:
        pr_type = _detect_pr_type_from_branch(app)
        if pr_type is None:
            raise click.ClickException(
                "Cannot auto-detect PR type from current branch. "
                "Please specify 'main', 'solution', or 'semantic' explicitly."
            )
        click.echo(f"Auto-detected PR type: {pr_type}")

    # Determine branches and body based on PR type
    kind = _pr_kind(pr_type)
    head_branch = getattr(app.branches, kind.head_attr)
    base_branch = getattr(app.branches, kind.base_attr)
    body = kind.body_builder(app)

    # If PR number is provided, fetch directly by number
    pr: GithubPullRequest.PullRequest | None = None
    if pr_number is not None:
        try:
            pr = app.gh_repo.get_pull(pr_number)
        except GithubException as e:
            msg = e.data.get("message", str(e)) if isinstance(e.data, dict) else str(e)
            raise click.ClickException(f"Failed to fetch PR #{pr_number}: {msg}") from e

        # Validate context
        warnings = _validate_pr_context(app, pr, pr_type)
        if warnings:
            if not force:
                msg = "\n".join(f"  - {w}" for w in warnings)
                raise click.ClickException(
                    f"PR #{pr_number} context mismatch:\n{msg}\n\nUse --force to update anyway."
                )
            for w in warnings:
                click.echo(f"Warning: {w}", err=True)

        head_branch = pr.head.ref
        base_branch = pr.base.ref
    else:
        # Original behavior: find PR by branches
        if dry_run:
            click.echo(f"Would update PR from {head_branch} to {base_branch}")
            click.echo("--- new body ---")
            util.print_or_page(body, format="markdown")
            click.echo("--- end ---")
            return

        pr = _find_pr_for_branch(app, head_branch, base_branch)
        if pr is None:
            raise click.ClickException(
                f"No open PR found from '{head_branch}' to '{base_branch}'. "
                f"Create one first with 'mergai pr create {pr_type}'."
            )

    # Handle dry-run for PR number case
    if dry_run:
        click.echo(f"Would update PR #{pr.number} from {head_branch} to {base_branch}")
        click.echo("--- new body ---")
        util.print_or_page(body, format="markdown")
        click.echo("--- end ---")
        return

    click.echo(f"Updating PR #{pr.number}: {pr.title}")
    click.echo(f"  from: {head_branch}")
    click.echo(f"    to: {base_branch}")

    try:
        pr.edit(body=body)
        click.echo(f"PR body updated: {pr.html_url}")
    except GithubException as e:
        msg = e.data.get("message", str(e)) if isinstance(e.data, dict) else str(e)
        raise click.ClickException(f"GitHub API error: {msg}") from e
