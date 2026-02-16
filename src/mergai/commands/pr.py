import click
from ..app import AppContext
from ..models import MarkdownConfig
from github import PullRequest as GithubPullRequest
from github import GithubException
from ..utils import git_utils
from ..utils import util
from typing import Optional, List
from urllib.parse import urlencode, quote


def _build_pr_url(repo_str: str, title: str, body: str, head: str, base: str) -> str:
    """Build a GitHub URL for creating a PR with pre-filled information.

    Args:
        repo_str: Repository in 'owner/repo' format.
        title: PR title.
        body: PR body/description.
        head: Source branch name.
        base: Target branch name.

    Returns:
        GitHub compare URL with query parameters for PR creation.
    """
    # GitHub compare URL format: https://github.com/{owner}/{repo}/compare/{base}...{head}
    # Query params: expand=1 (opens PR form), title, body
    base_url = f"https://github.com/{repo_str}/compare/{quote(base, safe='')}...{quote(head, safe='')}"
    params = {
        "expand": "1",  # Automatically expand the PR creation form
        "title": title,
        "body": body,
    }
    return f"{base_url}?{urlencode(params)}"


def _create_pr(
    app: AppContext,
    title: str,
    body: str,
    head: str,
    base: str,
    dry_run: bool = False,
    url_only: bool = False,
):

    if url_only:
        url = _build_pr_url(app.gh_repo.full_name, title, body, head, base)
        click.echo(f"Open this URL to create the PR:\n\n{url}")
        return None

    click.echo(
        f"Creating PR:\n"
        f"    repo: {app.gh_repo.full_name}\n"
        f"    from: {head}\n"
        f"      to: {base}\n"
        f"   title: {title}"
    )

    if dry_run:
        click.echo("--- body ---")
        click.echo(body)
        click.echo("--- end ---")
        return None

    try:
        pr = app.gh_repo.create_pull(title=title, body=body, head=head, base=base)
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

    body = util.merge_info_to_markdown(app.note.merge_info, markdown_config)
    body += "\n\n"
    body += util.merge_context_to_markdown(app.note.merge_context, markdown_config)
    body += "\n\n"
    body += util.conflict_context_to_markdown(app.note.conflict_context, markdown_config)
    body += "\n\n"
    body += util.solutions_to_markdown(app.note.solutions)
    body += "\n\n"

    return body


def _build_merge_pr_body(app: AppContext) -> str:
    markdown_config = MarkdownConfig.for_pr(app.repo)

    body = util.merge_info_to_markdown(app.note.merge_info, markdown_config)
    body += "\n\n"
    body += util.merge_context_to_markdown(app.note.merge_context, markdown_config)
    body += "\n\n"

    return body


def _create_solution_pr(
    app: AppContext, dry_run: bool, url_only: bool = False, skip_body: bool = False
) -> None:
    """Create a PR from the current branch (with existing solution commits) to the conflict branch."""

    merge_info = app.note.merge_info
    branches = app.branches

    if not app.check_all_solutions_committed():
        raise click.ClickException(
            "You have uncommitted solution(s). Run 'mergai commit solution' first."
        )

    if not app.note.has_solutions or len(app.note.solutions) == 0:
        raise click.ClickException("No solutions found. Run 'mergai resolve' first.")

    merge_commit_short = git_utils.short_sha(merge_info.merge_commit_sha)
    # TODO: title format from config
    title = f"Resolve conflicts for merge {merge_commit_short} into {app.branches.target_branch}"

    body = "" if skip_body else _build_solutions_pr_body(app)

    _create_pr(
        app,
        title,
        body,
        branches.solution_branch,
        branches.conflict_branch,
        dry_run=dry_run,
        url_only=url_only,
    )


def _build_main_pr_body(app: AppContext) -> str:
    """Build PR body for main PR from merge_context or conflict resolution data."""
    if app.note.has_conflict_context and app.note.has_solutions:
        return _build_solutions_pr_body(app)

    if app.note.has_merge_context:
        return _build_merge_pr_body(app)

    if app.note.has_conflict_context and not app.note.has_solutions:
        raise click.ClickException(
            "Found conflict_context but no solutions. "
            "Run 'mergai resolve' to generate a solution first."
        )

    if app.note.has_solutions and not app.note.has_conflict_context:
        raise click.ClickException(
            "Found solutions but no conflict_context. "
            "Run 'mergai context create conflict' first."
        )
    raise click.ClickException(
        "No merge_context or conflict resolution data found. "
        "Run 'mergai context create merge' for non-conflict merges, "
        "or ensure conflict_context and solutions are present for conflict resolutions."
    )


def _create_main_pr(
    app: AppContext, dry_run: bool, url_only: bool = False, skip_body: bool = False
) -> None:
    """Create a PR from the main branch to target_branch (merge or conflict resolution)."""

    merge_commit_short = git_utils.short_sha(app.note.merge_info.merge_commit_sha)
    # TODO: title format from config
    title = f"Merge {merge_commit_short} into {app.branches.target_branch}"

    body = "" if skip_body else _build_main_pr_body(app)

    _create_pr(
        app,
        title,
        body,
        app.branches.main_branch,
        app.branches.target_branch,
        dry_run=dry_run,
        url_only=url_only,
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
@click.argument(
    "pr_type", type=click.Choice(["main", "solution"], case_sensitive=False)
)
def create(
    app: AppContext, pr_type: str, dry_run: bool, url_only: bool, skip_body: bool
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

    \b
    Examples:
        mergai pr create main            # Create PR from main branch to target_branch
        mergai pr create solution        # Create PR from solution branch to conflict branch
        mergai pr create main --url-only # Get URL to create PR manually on GitHub
        mergai pr create main --skip-body # Create PR with empty body
    """
    if dry_run and url_only:
        raise click.ClickException("Cannot use --dry-run and --url-only together.")

    if pr_type.lower() == "solution":
        _create_solution_pr(app, dry_run, url_only, skip_body)
    else:
        _create_main_pr(app, dry_run, url_only, skip_body)


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
