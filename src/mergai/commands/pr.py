import click
from ..app import AppContext
from ..models import MarkdownConfig
from github import PullRequest as GithubPullRequest
from github import GithubException
from ..utils import git_utils
from ..utils import formatters
from ..utils.branch_name_builder import BranchNameBuilder, BranchType
from typing import Optional, List
from urllib.parse import urlencode, quote

from mergai import app


def _parse_labels_option(labels_arg: Optional[str], config_labels: List[str]) -> List[str]:
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
    labels: Optional[List[str]] = None,
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


def _create_pr(
    app: AppContext,
    title: str,
    body: str,
    head: str,
    base: str,
    dry_run: bool = False,
    url_only: bool = False,
    labels: Optional[List[str]] = None,
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
        click.echo(body)
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
        msg = e.data.get("message", str(e)) if isinstance(e.data, dict) else str(e)
        raise click.ClickException(f"GitHub API error: {msg}") from e


def _build_solutions_pr_body(app: AppContext) -> str:
    markdown_config = MarkdownConfig.for_pr(app.repo)

    body = formatters.merge_info_to_markdown(app.note.merge_info, markdown_config)
    body += "\n\n"
    body += formatters.merge_context_to_markdown(app.note.merge_context, markdown_config)
    body += "\n\n"
    if app.note.has_conflict_context:
        body += formatters.conflict_context_to_markdown(app.note.conflict_context, markdown_config)
        body += "\n\n"
    body += formatters.solutions_to_markdown(app.note.solutions)
    body += "\n\n"
    body += f"---\n\n*note created with mergai {app.note.mergai_version}*\n"

    return body


def _build_merge_pr_body(app: AppContext) -> str:
    markdown_config = MarkdownConfig.for_pr(app.repo)

    body = formatters.merge_info_to_markdown(app.note.merge_info, markdown_config)
    body += "\n\n"
    body += formatters.merge_context_to_markdown(app.note.merge_context, markdown_config)
    body += "\n\n"
    body += f"---\n\n*note created with mergai {app.note.mergai_version}*\n"

    return body


def _create_solution_pr(
    app: AppContext,
    dry_run: bool,
    url_only: bool = False,
    skip_body: bool = False,
    labels: Optional[List[str]] = None,
) -> None:
    """Create a PR from the current branch (with existing solution commits) to the conflict branch."""

    if not app.note.has_solutions or len(app.note.solutions) == 0:
        raise click.ClickException("No solutions found. Run 'mergai resolve' first.")

    if app.note.get_uncommitted_solution() is not None:
        raise click.ClickException(
            "You have uncommitted solution(s). Run 'mergai commit solution' first."
        )

    title = app.pr_titles.solution_title

    body = "" if skip_body else _build_solutions_pr_body(app)

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


def _build_main_pr_body(app: AppContext) -> str:
    """Build PR body for main PR from merge_context or conflict resolution data."""
    # If we have solutions (from any source - AI, human, or synced), include them
    if app.note.has_solutions:
        return _build_solutions_pr_body(app)

    # No solutions - use merge PR body if we have merge_context
    if app.note.has_merge_context:
        return _build_merge_pr_body(app)

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
    labels: Optional[List[str]] = None,
) -> None:
    """Create a PR from the main branch to target_branch (merge or conflict resolution)."""

    title = app.pr_titles.main_title

    body = "" if skip_body else _build_main_pr_body(app)

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
def pr(app: AppContext, repo: Optional[str]):
    if repo is None:
        raise click.ClickException(
            "GitHub repository not set. Use --repo or set GH_REPO environment variable."
        )

    app.gh_repo_str = repo
    pass


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
    "pr_type", type=click.Choice(["main", "solution"], case_sensitive=False)
)
def create(
    app: AppContext,
    pr_type: str,
    dry_run: bool,
    url_only: bool,
    skip_body: bool,
    labels_arg: Optional[str],
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
    Options:
        --dry-run   Show what would be created without actually creating the PR.
        --url-only  Print a GitHub URL to create the PR manually. When you open
                    this URL, GitHub will show the PR creation form with all
                    fields pre-filled (title, body, branches). You can review
                    and edit everything before submitting.
        --skip-body Skip creating a body for the PR (create with empty body).
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
        mergai pr create main --labels=urgent,review  # Override config labels
        mergai pr create main --labels=+urgent,-auto  # Add/remove from config labels
        mergai pr create main --no-labels             # Create PR without any labels
    """
    if dry_run and url_only:
        raise click.ClickException("Cannot use --dry-run and --url-only together.")

    if no_labels and labels_arg is not None:
        raise click.ClickException("Cannot use --no-labels and --labels together.")

    # Get config labels based on PR type
    if pr_type.lower() == "solution":
        config_labels = app.config.pr.solution.labels
    else:
        config_labels = app.config.pr.main.labels

    # Compute final labels
    if no_labels:
        final_labels: List[str] = []
    else:
        final_labels = _parse_labels_option(labels_arg, config_labels)

    if pr_type.lower() == "solution":
        _create_solution_pr(app, dry_run, url_only, skip_body, final_labels)
    else:
        _create_main_pr(app, dry_run, url_only, skip_body, final_labels)


def get_prs_for_current_branch(app: AppContext) -> List[GithubPullRequest.PullRequest]:
    # TODO: the head should include the repo owner
    pulls = app.gh_repo.get_pulls(
        sort="created", head=git_utils.get_current_branch(app.repo)
    )
    return list(
        filter(lambda pr: pr.head.ref == git_utils.get_current_branch(app.repo), pulls)
    )


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
def show(app: AppContext):
    try:
        prs = get_prs_for_current_branch(app)
        if len(prs) == 0:
            click.echo("No open pull requests found for the current branch.")
            exit(0)

        if len(prs) > 1:
            click.echo("Multiple open pull requests found for the current branch:")

        else:
            show_prs(prs)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


def _detect_pr_type_from_branch(app: AppContext) -> Optional[str]:
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

    return None


def _find_pr_for_branch(
    app: AppContext, head_branch: str, base_branch: str
) -> Optional[GithubPullRequest.PullRequest]:
    """Find an open PR from head_branch to base_branch.

    Args:
        app: AppContext with GitHub repo.
        head_branch: Source branch name.
        base_branch: Target branch name.

    Returns:
        PullRequest if found, None otherwise.
    """
    pulls = app.gh_repo.get_pulls(state="open", head=head_branch, base=base_branch)
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
@click.argument(
    "pr_type",
    type=click.Choice(["main", "solution"], case_sensitive=False),
    required=False,
)
def update(app: AppContext, pr_type: Optional[str], dry_run: bool):
    """Update an existing pull request's body.

    PR_TYPE specifies which PR to update. If not provided, it will be auto-detected
    from the current branch name:
    - If on a 'main' branch -> updates the main PR
    - If on a 'solution' branch -> updates the solution PR

    The PR body is rebuilt using the current note data, including any solutions
    added after the PR was created (e.g., human solutions from 'mergai commit sync').

    \b
    Options:
        --dry-run   Show the new body without updating the PR.

    \b
    Examples:
        mergai pr --repo owner/name update           # Auto-detect PR type from branch
        mergai pr --repo owner/name update main      # Update main PR body
        mergai pr --repo owner/name update solution  # Update solution PR body
        mergai pr --repo owner/name update --dry-run # Preview new body
    """
    # Auto-detect PR type if not provided
    if pr_type is None:
        pr_type = _detect_pr_type_from_branch(app)
        if pr_type is None:
            raise click.ClickException(
                "Cannot auto-detect PR type from current branch. "
                "Please specify 'main' or 'solution' explicitly."
            )
        click.echo(f"Auto-detected PR type: {pr_type}")

    # Determine branches based on PR type
    if pr_type.lower() == "solution":
        head_branch = app.branches.solution_branch
        base_branch = app.branches.conflict_branch
        body = _build_solutions_pr_body(app)
    else:  # main
        head_branch = app.branches.main_branch
        base_branch = app.branches.target_branch
        body = _build_main_pr_body(app)

    if dry_run:
        click.echo(f"Would update PR from {head_branch} to {base_branch}")
        click.echo("--- new body ---")
        click.echo(body)
        click.echo("--- end ---")
        return

    # Find the PR
    pr = _find_pr_for_branch(app, head_branch, base_branch)
    if pr is None:
        raise click.ClickException(
            f"No open PR found from '{head_branch}' to '{base_branch}'. "
            f"Create one first with 'mergai pr create {pr_type}'."
        )

    click.echo(f"Updating PR #{pr.number}: {pr.title}")
    click.echo(f"  from: {head_branch}")
    click.echo(f"    to: {base_branch}")

    try:
        pr.edit(body=body)
        click.echo(f"PR body updated: {pr.html_url}")
    except GithubException as e:
        msg = e.data.get("message", str(e)) if isinstance(e.data, dict) else str(e)
        raise click.ClickException(f"GitHub API error: {msg}") from e
