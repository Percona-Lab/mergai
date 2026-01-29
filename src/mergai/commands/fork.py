import click
from typing import Optional, List
from dataclasses import dataclass
from git import Commit
from ..app import AppContext
from .. import git_utils
from ..config import MergePicksConfig
from ..strategies import StrategyResult, StrategyContext
from ..util import format_number, format_commit_info, format_commit_info_oneline

@click.group()
def fork():
    """Commands for managing forks and syncing with upstream repositories."""
    pass


@fork.command()
@click.pass_obj
@click.argument(
    "upstream_url",
    type=str,
    required=False,
    default=None,
    metavar="UPSTREAM-URL",
)
def init(app: AppContext, upstream_url: Optional[str]):
    """Initialize upstream remote for fork management.

    Adds a git remote for the upstream repository (if not already present)
    and fetches it. Uses fork.upstream_url from config unless overridden
    by the UPSTREAM-URL argument.
    """
    # Resolve URL (argument overrides config)
    url = upstream_url or app.config.fork.upstream_url

    if url is None:
        click.echo("Error: No upstream URL provided and fork.upstream_url not configured.", err=True)
        click.echo("Either provide UPSTREAM-URL argument or set fork.upstream_url in .mergai.yaml", err=True)
        raise SystemExit(1)

    # Determine desired remote name from config, default to "upstream"
    desired_remote_name = app.config.fork.upstream_remote or "upstream"

    # Check if a remote with the desired name already exists
    existing_remote = None
    for remote in app.repo.remotes:
        if remote.name == desired_remote_name:
            existing_remote = remote
            break

    if existing_remote is not None:
        # Remote with desired name exists - check if URL matches
        existing_urls = list(existing_remote.urls)
        if url in existing_urls:
            click.echo(f"Remote '{desired_remote_name}' already configured with URL: {url}")
            remote_name = desired_remote_name
        else:
            # Remote exists but with different URL
            click.echo(f"Error: Remote '{desired_remote_name}' exists but has different URL: {existing_urls[0]}", err=True)
            click.echo(f"Expected URL: {url}", err=True)
            raise SystemExit(1)
    else:
        # Check if any other remote has this URL
        remote_name = git_utils.find_remote_by_url(app.repo, url)

        if remote_name is not None:
            click.echo(f"Remote '{remote_name}' already configured with URL: {url}")
            if remote_name != desired_remote_name:
                click.echo(f"Note: Using existing remote '{remote_name}' instead of configured name '{desired_remote_name}'")
        else:
            # No remote found, create new one with desired name
            remote_name = desired_remote_name
            click.echo(f"Adding remote '{remote_name}' with URL: {url}")
            try:
                app.repo.create_remote(remote_name, url)
            except Exception as e:
                click.echo(f"Error: Failed to create remote: {e}", err=True)
                raise SystemExit(1)

    # Fetch the remote
    click.echo(f"Fetching '{remote_name}'...")
    try:
        app.repo.remotes[remote_name].fetch()
        click.echo(f"Done. Remote '{remote_name}' is ready.")
    except Exception as e:
        click.echo(f"Error: Failed to fetch remote: {e}", err=True)
        raise SystemExit(1)


@fork.command()
@click.pass_obj
@click.argument(
    "upstream_ref",
    type=str,
    required=False,
    default=None,
    metavar="UPSTREAM-REF",
)
@click.argument(
    "fork_ref",
    type=str,
    required=False,
    default="HEAD",
    metavar="FORK-REF",
)
@click.option(
    "--list", "-l",
    "list_commits",
    is_flag=True,
    default=False,
    help="List unmerged commits",
)
def status(
    app: AppContext,
    upstream_ref: Optional[str],
    fork_ref: str,
    list_commits: bool,
):
    # If upstream_ref not provided, try to derive from config
    if upstream_ref is None:
        upstream_url = app.config.fork.upstream_url

        if upstream_url is None:
            click.echo("Error: No UPSTREAM-REF provided and fork.upstream_url not configured.", err=True)
            click.echo("Either provide UPSTREAM-REF argument or set fork.upstream_url in .mergai.yaml", err=True)
            raise SystemExit(1)

        # Find remote matching the URL
        remote_name = git_utils.find_remote_by_url(app.repo, upstream_url)

        if remote_name is None:
            click.echo(f"Error: No remote found matching upstream_url: {upstream_url}", err=True)
            click.echo("Hint: Run 'mergai fork init' to add and fetch the upstream remote.", err=True)
            raise SystemExit(1)

        # Build upstream_ref from remote name and branch
        upstream_branch = app.config.fork.upstream_branch
        upstream_ref = f"{remote_name}/{upstream_branch}"

    try:
        status = git_utils.get_fork_status(app.repo, upstream_ref, fork_ref)
    except Exception as e:
        click.echo(f"Error: Failed to get fork status: {e}", err=True)
        raise SystemExit(1)
    
    output_lines = []

    # Header with upstream info
    output_lines.append(f"Upstream URL:    {app.config.fork.upstream_url or '(not configured)'}")
    output_lines.append(f"Upstream branch: {app.config.fork.upstream_branch}")
    output_lines.append(f"Upstream ref:    {upstream_ref}")
    output_lines.append("")
    output_lines.append(f"Status: {"up to date" if status.is_up_to_date else "diverged"}")
    
    # Divergence estimate
    if not status.is_up_to_date:
        output_lines.append("Divergence:")
        output_lines.append(f"  Commits behind:   {format_number(status.commits_behind)}")
        output_lines.append(f"  Days behind:      {status.days_behind}")
        date_range = status.unmerged_date_range
        if date_range:
            first_date_str = date_range[0].strftime("%Y-%m-%d")
            last_date_str = date_range[1].strftime("%Y-%m-%d")
            output_lines.append(f"  Date range:       {first_date_str} to {last_date_str}")
        
        output_lines.append(f"  Files affected:   {format_number(status.files_affected)}")
        output_lines.append(f"  Total additions:  +{format_number(status.total_additions)}")
        output_lines.append(f"  Total deletions:  -{format_number(status.total_deletions)}")
        output_lines.append("")

        # Last merged commit
        output_lines.append("Last merged commit:")
        output_lines.append(format_commit_info(status.last_merged_commit))
        output_lines.append("")
        
        # First unmerged commit
        output_lines.append("First unmerged commit:")
        output_lines.append(format_commit_info(status.first_unmerged_commit))
        output_lines.append("")
    
    # Optional commit listing
    if list_commits and not status.is_up_to_date:
        total_commits = len(status.unmerged_commits)
        index_width = len(str(total_commits))
        
        output_lines.append("Unmerged commits:")
        output_lines.append("")
        for i, commit in enumerate(reversed(status.unmerged_commits), 1):
            
            commit_info = format_commit_info_oneline(commit)
            output_lines.append(
                f"{i:{index_width}d}: {commit_info}"
            )
    
    # Output via pager if listing commits, otherwise print directly
    output = "\n".join(output_lines)
    if list_commits:
        click.echo_via_pager(output)
    else:
        click.echo(output)


@dataclass
class PrioritizedCommit:
    """A commit that matched a priority strategy.

    Attributes:
        commit: The git commit object.
        strategy_name: Name of the strategy that matched.
        result: The strategy result with match details.
    """

    commit: Commit
    strategy_name: str
    result: StrategyResult


def get_prioritized_commits(
    repo,
    unmerged_commits: List[Commit],
    config: MergePicksConfig,
    context: StrategyContext,
) -> List[PrioritizedCommit]:
    """Evaluate all unmerged commits and return those matching priority strategies.

    Strategies are evaluated in the order they appear in the config.
    The first matching strategy for each commit determines its priority.

    Args:
        repo: GitPython Repo object.
        unmerged_commits: List of unmerged commits (in reverse chronological order).
        config: MergePicksConfig with ordered priority strategies.
        context: Strategy context with upstream_ref, fork_ref, etc.

    Returns:
        List of PrioritizedCommit objects for commits that matched strategies,
        in chronological order (oldest first).
    """
    prioritized = []

    # Process in chronological order (oldest first)
    for commit in reversed(unmerged_commits):
        for strategy in config.strategies:
            result = strategy.check(repo, commit, context)
            if result is not None:
                prioritized.append(
                    PrioritizedCommit(
                        commit=commit,
                        strategy_name=strategy.name,
                        result=result,
                    )
                )
                break  # First matching strategy wins

    return prioritized


@fork.command()
@click.pass_obj
@click.argument(
    "upstream_ref",
    type=str,
    required=False,
    default=None,
    metavar="UPSTREAM-REF",
)
@click.argument(
    "fork_ref",
    type=str,
    required=False,
    default="HEAD",
    metavar="FORK-REF",
)
@click.option(
    "--next",
    "-n",
    "next_only",
    is_flag=True,
    default=False,
    help="Print only the hash of the next commit to merge",
)
def pick(
    app: AppContext,
    upstream_ref: Optional[str],
    fork_ref: str,
    next_only: bool,
):
    """Suggest commits to merge based on configured priority strategies.

    Analyzes unmerged commits and lists those that match priority strategies
    from the fork.merge_picks config section. Strategies are evaluated in
    the order they appear in the config.

    \b
    Available strategies:
    - huge_commit: Commits with many changed files/lines
    - important_files: Commits touching specific important files
    - branching_point: Commits that are branching points (multiple children)
    - conflict: (not yet implemented) Commits that would cause merge conflicts

    Use --next/-n to get just the hash of the recommended next commit.
    Output is designed to be easily parseable in scripts.

    \b
    Examples:
        mergai fork pick                    # List prioritized commits
        mergai fork pick --next             # Get next commit hash
        mergai fork pick mongodb/master     # Use specific upstream ref
    """
    # Resolve upstream_ref (same logic as status command)
    if upstream_ref is None:
        upstream_url = app.config.fork.upstream_url

        if upstream_url is None:
            click.echo(
                "Error: No UPSTREAM-REF provided and fork.upstream_url not configured.",
                err=True,
            )
            click.echo(
                "Either provide UPSTREAM-REF argument or set fork.upstream_url in .mergai.yaml",
                err=True,
            )
            raise SystemExit(1)

        # Find remote matching the URL
        remote_name = git_utils.find_remote_by_url(app.repo, upstream_url)

        if remote_name is None:
            click.echo(
                f"Error: No remote found matching upstream_url: {upstream_url}",
                err=True,
            )
            click.echo(
                "Hint: Run 'mergai fork init' to add and fetch the upstream remote.",
                err=True,
            )
            raise SystemExit(1)

        # Build upstream_ref from remote name and branch
        upstream_branch = app.config.fork.upstream_branch
        upstream_ref = f"{remote_name}/{upstream_branch}"

    # Get fork status to obtain unmerged commits
    try:
        fork_status = git_utils.get_fork_status(app.repo, upstream_ref, fork_ref)
    except Exception as e:
        click.echo(f"Error: Failed to get fork status: {e}", err=True)
        raise SystemExit(1)

    if fork_status.is_up_to_date:
        # No unmerged commits - nothing to do
        # For --next, output nothing (success with empty output)
        # TODO: Consider whether --next should fall back to first unmerged commit
        # if no priority commits are found, or return error/empty as currently
        return

    # Get merge_picks config from fork section
    merge_picks_config = app.config.fork.merge_picks
    if merge_picks_config is None:
        # No merge_picks config - use empty config (no strategies will match)
        merge_picks_config = MergePicksConfig()

    # Create strategy context
    context = StrategyContext(
        upstream_ref=upstream_ref,
        fork_ref=fork_ref,
    )

    # Get prioritized commits
    prioritized = get_prioritized_commits(
        app.repo,
        fork_status.unmerged_commits,
        merge_picks_config,
        context,
    )

    if next_only:
        # Output only the hash of the first prioritized commit
        if prioritized:
            click.echo(prioritized[0].commit.hexsha)
        # If no prioritized commits, output nothing (success)
        # TODO: Consider fallback to first unmerged commit if no priority match
        return

    # List all prioritized commits
    if not prioritized:
        # No output if no commits match criteria (for script parseability)
        return

    # Output format matching 'fork status -l' style
    total_commits = len(prioritized)
    index_width = len(str(total_commits))

    output_lines = []
    for i, pc in enumerate(prioritized, 1):
        commit_info = format_commit_info_oneline(pc.commit)
        strategy_tag = f"[{pc.strategy_name}]"
        details_tag = f"({pc.result.format_short()})"
        output_lines.append(f"{i:{index_width}d}: {commit_info} {strategy_tag} {details_tag}")

    click.echo("\n".join(output_lines))
