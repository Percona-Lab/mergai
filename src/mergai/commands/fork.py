import click
import logging
from typing import Optional, List
from dataclasses import dataclass
from git import Commit
from ..app import AppContext
from ..utils import git_utils
from ..config import MergePicksConfig, DEFAULT_CONFIG_PATH
from ..merge_pick_strategies import MergePickCommit, MergePickStrategyContext
from ..utils.util import format_number, format_commit_info, format_commit_info_oneline, print_or_page

log = logging.getLogger(__name__)

def resolve_upstream_ref(
    app: AppContext,
    upstream_ref: Optional[str],
) -> str:
    """Resolve upstream ref from argument or config.

    If upstream_ref is provided, returns it unchanged.
    Otherwise, derives it from fork.upstream_url and fork.upstream_branch config.

    Args:
        app: Application context with repo and config.
        upstream_ref: Explicit upstream ref, or None to derive from config.

    Returns:
        Resolved upstream ref string (e.g., "upstream/master").

    Raises:
        SystemExit: If config is missing or remote not found.
    """
    if upstream_ref is not None:
        return upstream_ref

    upstream_url = app.config.fork.upstream_url

    if upstream_url is None:
        click.echo("Error: No UPSTREAM-REF provided and fork.upstream_url not configured.", err=True)
        click.echo(f"Either provide UPSTREAM-REF argument or set fork.upstream_url in {DEFAULT_CONFIG_PATH}", err=True)
        raise SystemExit(1)

    remote_name = git_utils.find_remote_by_url(app.repo, upstream_url)

    if remote_name is None:
        click.echo(f"Error: No remote found matching upstream_url: {upstream_url}", err=True)
        click.echo("Hint: Run 'mergai fork init' to add and fetch the upstream remote.", err=True)
        raise SystemExit(1)

    upstream_branch = app.config.fork.upstream_branch
    return f"{remote_name}/{upstream_branch}"


def _needs_commit_stats(config: MergePicksConfig) -> bool:
    """Check if any strategy needs commit stats."""
    for strategy in config.strategies:
        if strategy.name in ("huge_commit", "important_files"):
            return True
    return False


def _needs_branching_points(config: MergePicksConfig) -> bool:
    """Check if any strategy needs branching point data."""
    for strategy in config.strategies:
        if strategy.name == "branching_point":
            return True
    return False


def get_prioritized_commits(
    repo,
    unmerged_commit_shas: List[str],
    config: MergePicksConfig,
    context: MergePickStrategyContext,
    limit: Optional[int] = None,
) -> List[MergePickCommit]:
    """Evaluate unmerged commits and return those matching priority strategies.

    This function uses batch operations to efficiently compute data needed
    by strategies, avoiding per-commit git calls.

    Strategies are evaluated in the order they appear in the config.
    The first matching strategy for each commit determines its priority.

    If most_recent_fallback is enabled and no strategies match any commits,
    the most recent unmerged commit is returned as a fallback.

    Args:
        repo: GitPython Repo object.
        unmerged_commit_shas: List of unmerged commit SHAs (newest first).
        config: MergePicksConfig with ordered priority strategies.
        context: Strategy context with upstream_ref, fork_ref, etc.
        limit: Optional maximum number of matches to return. If specified,
            evaluation stops early once the limit is reached.

    Returns:
        List of MergePickCommit objects for commits that matched strategies,
        in chronological order (oldest first).
    """
    if not unmerged_commit_shas:
        return []

    # Pre-populate batch caches for performance
    # This avoids per-commit git calls in the strategy loop
    if _needs_commit_stats(config):
        log.debug(f"Batch loading commit stats for {len(unmerged_commit_shas)} commits")
        context.commit_stats_cache = git_utils.get_batch_commit_stats(repo, unmerged_commit_shas)
        log.debug(f"Loaded stats for {len(context.commit_stats_cache)} commits")

    if _needs_branching_points(config) and context.upstream_ref:
        # Get the oldest commit SHA (last in list since list is newest-first)
        oldest_sha = unmerged_commit_shas[-1]
        # Get the parent of oldest commit as the base for branching point detection
        try:
            oldest_commit = repo.commit(oldest_sha)
            if oldest_commit.parents:
                base_sha = oldest_commit.parents[0].hexsha
            else:
                base_sha = oldest_sha
        except Exception:
            base_sha = oldest_sha
        
        log.debug(f"Batch loading branching points from {base_sha[:11]} to {context.upstream_ref}")
        context.branching_points_cache = git_utils.get_batch_branching_points(
            repo, base_sha, context.upstream_ref
        )
        log.debug(f"Found {len(context.branching_points_cache)} branching points")

    prioritized = []

    # Process in chronological order (oldest first)
    # Commits are stored newest-first, so we reverse
    for sha in reversed(unmerged_commit_shas):
        commit = repo.commit(sha)
        for strategy in config.strategies:
            result = strategy.check(repo, commit, context)
            if result is not None:
                prioritized.append(
                    MergePickCommit(
                        commit=commit,
                        strategy_name=strategy.name,
                        result=result,
                    )
                )
                if limit is not None and len(prioritized) >= limit:
                    return prioritized  # Early return when limit reached
                break  # First matching strategy wins

    # Fallback to most recent commit if enabled and no matches found
    if not prioritized and config.most_recent_fallback and unmerged_commit_shas:
        from ..merge_pick_strategies.most_recent import MostRecentResult

        # Most recent = first in unmerged_commit_shas (newest first)
        most_recent_commit = repo.commit(unmerged_commit_shas[0])
        prioritized.append(
            MergePickCommit(
                commit=most_recent_commit,
                strategy_name="most_recent",
                result=MostRecentResult(),
            )
        )

    return prioritized

def build_status_summary(
    app: AppContext,
    fork_status,
    upstream_ref: str,
) -> List[str]:
    """Build the status summary output lines.

    Args:
        app: Application context with config.
        fork_status: ForkStatus object with divergence info.
        upstream_ref: Resolved upstream ref string.

    Returns:
        List of formatted output lines for the status summary.
    """
    output_lines = []

    # Header with upstream info
    output_lines.append(f"Upstream URL:    {app.config.fork.upstream_url or '(not configured)'}")
    output_lines.append(f"Upstream branch: {app.config.fork.upstream_branch}")
    output_lines.append(f"Upstream ref:    {upstream_ref}")
    output_lines.append("")
    output_lines.append(f"Status: {'up to date' if fork_status.is_up_to_date else 'diverged'}")

    # Divergence estimate
    if not fork_status.is_up_to_date:
        output_lines.append("Divergence:")
        output_lines.append(f"  Commits behind:   {format_number(fork_status.commits_behind)}")
        output_lines.append(f"  Days behind:      {fork_status.days_behind}")
        date_range = fork_status.unmerged_date_range
        if date_range:
            first_date_str = date_range[0].strftime("%Y-%m-%d")
            last_date_str = date_range[1].strftime("%Y-%m-%d")
            output_lines.append(f"  Date range:       {first_date_str} to {last_date_str}")

        output_lines.append(f"  Files affected:   {format_number(fork_status.files_affected)}")
        output_lines.append(f"  Total additions:  +{format_number(fork_status.total_additions)}")
        output_lines.append(f"  Total deletions:  -{format_number(fork_status.total_deletions)}")
        output_lines.append("")

        # Last merged commit
        output_lines.append("Last merged commit:")
        output_lines.append(format_commit_info(fork_status.last_merged_commit))
        output_lines.append("")

        # First unmerged commit
        output_lines.append("First unmerged commit:")
        output_lines.append(format_commit_info(fork_status.first_unmerged_commit))
        output_lines.append("")

    return output_lines

def format_commit_list(
    commits: List[Commit],
    prioritized_commits: Optional[List[MergePickCommit]] = None,
    show_all: bool = True,
    show_prefix: bool = True,
) -> List[str]:
    """Format a list of commits for display.

    Args:
        commits: List of commits in chronological order (oldest first).
        prioritized_commits: Optional list of prioritized commits with strategy info.
        show_all: If True, show all commits; if False, only show prioritized commits.
        show_prefix: If True, show '*N:' prefix; if False, show only commit info with tags.

    Returns:
        List of formatted output lines.
    """
    # Build a lookup dict from commit SHA to MergePickCommit
    picks_by_sha = {}
    if prioritized_commits:
        picks_by_sha = {pc.commit.hexsha: pc for pc in prioritized_commits}

    # Determine which commits to show
    if show_all:
        display_commits = commits
    else:
        # Only show prioritized commits
        display_commits = [pc.commit for pc in prioritized_commits] if prioritized_commits else []

    if not display_commits:
        return []

    total_commits = len(display_commits)
    index_width = len(str(total_commits))

    output_lines = []
    for i, commit in enumerate(display_commits, 1):
        commit_info = format_commit_info_oneline(commit)
        pc = picks_by_sha.get(commit.hexsha)

        if pc:
            # This is a merge pick - add strategy info (with colors)
            strategy_tag = click.style(f" [{pc.strategy_name}]", fg="cyan")
            details_tag = click.style(f" ({pc.result.format_short()})", fg="bright_black")
        else:
            strategy_tag = ""
            details_tag = ""

        if show_prefix:
            if pc:
                prefix = click.style("*", fg="yellow", bold=True)
            else:
                prefix = " "
            output_lines.append(
                f"{prefix}{i:{index_width}d}: {commit_info}{strategy_tag}{details_tag}"
            )
        else:
            output_lines.append(f"{commit_info}{strategy_tag}{details_tag}")

    return output_lines

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
        click.echo(f"Either provide UPSTREAM-URL argument or set fork.upstream_url in {DEFAULT_CONFIG_PATH}", err=True)
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
@click.option(
    "--show-merge-picks", "-p",
    "show_merge_picks",
    is_flag=True,
    default=False,
    help="Show merge picks. With -l, marks picks in the commit list.",
)
def status(
    app: AppContext,
    upstream_ref: Optional[str],
    fork_ref: str,
    list_commits: bool,
    show_merge_picks: bool,
):
    log.info(f"getting fork status for:")
    log.info(f"  upstream_ref:{upstream_ref}")
    log.info(f"  fork_ref={fork_ref}")
    log.info(f"  list_commits={list_commits}")
    log.info(f"  show_merge_picks={show_merge_picks}")

    upstream_ref = resolve_upstream_ref(app, upstream_ref)

    try:
        fork_status = git_utils.get_fork_status(app.repo, upstream_ref, fork_ref)
    except Exception as e:
        click.echo(f"Error: Failed to get fork status: {e}", err=True)
        raise SystemExit(1)

    # Get prioritized commits if -p is specified
    prioritized = None
    if show_merge_picks and not fork_status.is_up_to_date:
        merge_picks_config = app.config.fork.merge_picks
        if merge_picks_config is None:
            merge_picks_config = MergePicksConfig()

        context = MergePickStrategyContext(
            upstream_ref=upstream_ref,
            fork_ref=fork_ref,
        )

        log.info(f"getting prioritized commits for merge picks")
        prioritized = get_prioritized_commits(
            app.repo,
            fork_status.unmerged_commit_shas,
            merge_picks_config,
            context,
        )

    # Build status summary output
    output_lines = build_status_summary(app, fork_status, upstream_ref)

    # Commit listing based on options:
    # -p only: show only merge picks
    # -l only: show all commits
    # -l -p: show all commits with picks marked
    show_commits = (list_commits or show_merge_picks) and not fork_status.is_up_to_date
    if show_commits:
        if show_merge_picks and not list_commits:
            # -p only: show only merge picks
            output_lines.append("Merge picks:")
            output_lines.append("")
            commit_lines = format_commit_list(
                commits=list(reversed(fork_status.unmerged_commits)),
                prioritized_commits=prioritized,
                show_all=False,
            )
        else:
            # -l or -l -p: show all commits (with picks marked if -p)
            output_lines.append("Unmerged commits:")
            output_lines.append("")
            commit_lines = format_commit_list(
                commits=list(reversed(fork_status.unmerged_commits)),
                prioritized_commits=prioritized if show_merge_picks else None,
                show_all=True,
            )
        output_lines.extend(commit_lines)

    # Output via pager if listing commits, otherwise print directly
    output = "\n".join(output_lines)
    print_or_page(output)

@fork.command("merge-pick")
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
@click.option(
    "--list",
    "-l",
    "list_commits",
    is_flag=True,
    default=False,
    help="List all unmerged commits with picks marked (like fork status -lp)",
)
def merge_pick(
    app: AppContext,
    upstream_ref: Optional[str],
    fork_ref: str,
    next_only: bool,
    list_commits: bool,
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
    - conflict: Commits that would cause merge conflicts

    Use --next/-n to get just the hash of the recommended next commit.
    Use --list/-l to show all unmerged commits with picks marked.

    \b
    Examples:
        mergai fork merge-pick                    # List prioritized commits
        mergai fork merge-pick --list             # List all commits with picks marked
        mergai fork merge-pick --next             # Get next commit hash
        mergai fork merge-pick mongodb/master     # Use specific upstream ref
    """
    upstream_ref = resolve_upstream_ref(app, upstream_ref)

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
    context = MergePickStrategyContext(
        upstream_ref=upstream_ref,
        fork_ref=fork_ref,
    )

    # Get prioritized commits
    prioritized = get_prioritized_commits(
        app.repo,
        fork_status.unmerged_commit_shas,
        merge_picks_config,
        context,
        limit=1 if next_only else None,
    )

    if next_only:
        # Output only the hash of the first prioritized commit
        if prioritized:
            click.echo(prioritized[0].commit.hexsha)
        # If no prioritized commits, output nothing (success)
        # TODO: Consider fallback to first unmerged commit if no priority match
        return

    if not fork_status.is_up_to_date:
        output_lines = []

        if list_commits:
            # -l: Show all commits with picks marked (like fork status -lp, but no summary)
            commit_lines = format_commit_list(
                commits=list(reversed(fork_status.unmerged_commits)),
                prioritized_commits=prioritized,
                show_all=True,
                show_prefix=True,
            )
            output_lines.extend(commit_lines)
        else:
            # Default: Show only picks without prefix/numbering
            if prioritized:
                commit_lines = format_commit_list(
                    commits=list(reversed(fork_status.unmerged_commits)),
                    prioritized_commits=prioritized,
                    show_all=False,
                    show_prefix=False,
                )
                output_lines.extend(commit_lines)
            else:
                output_lines.append("(no merge picks found)")

        print_or_page("\n".join(output_lines))
