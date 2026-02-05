"""Branch management commands for merge-related workflows.

This module provides commands for creating and deleting standardized
branch names used in the merge conflict resolution workflow:

- main: Main working branch where all work for merging a given commit will be merged
- conflict: Branch containing a merge commit with committed merge markers
- solution: Branch with solution attempts; PRs are created from solution to conflict
"""

import click
from typing import Optional
from ..app import AppContext
from ..util import BranchNameBuilder, BranchType, ParsedBranchName
from .. import git_utils


# Valid branch types for CLI validation
BRANCH_TYPES = [t.value for t in BranchType]

# Valid branch types for delete command (includes 'all')
DELETE_BRANCH_TYPES = BRANCH_TYPES + ["all"]

# Valid branch types for push command (includes 'all')
PUSH_BRANCH_TYPES = BRANCH_TYPES + ["all"]

# Valid token names for the info command
TOKEN_NAMES = ["target_branch", "merge_commit_short", "type"]


def get_current_branch_parsed(app: AppContext) -> Optional[ParsedBranchName]:
    """Parse the current branch name as a mergai branch.

    Args:
        app: Application context with config and repo.

    Returns:
        ParsedBranchName if current branch matches the format, None otherwise.
    """
    current_branch = git_utils.get_current_branch(app.repo)
    return BranchNameBuilder.parse_branch_name_with_config(
        current_branch, app.config.branch
    )


def get_branch_builder(
    app: AppContext,
    commit: Optional[str],
    target: Optional[str],
) -> BranchNameBuilder:
    """Create a BranchNameBuilder from CLI arguments.

    Uses app.get_merge_context() to resolve commit and target, which checks:
    1. merge_info in note.json
    2. Current mergai branch name
    3. CLI-provided values override auto-detected ones

    Args:
        app: Application context with config and repo.
        commit: Commit SHA or ref to use for branch naming, or None to auto-detect.
        target: Target branch name, or None to auto-detect.

    Returns:
        Configured BranchNameBuilder instance.

    Raises:
        click.ClickException: If commit reference is invalid or cannot be determined.
    """
    ctx = app.get_merge_context(commit=commit, target=target)

    return BranchNameBuilder.from_config(
        app.config.branch,
        target_branch=ctx.target_branch,
        merge_commit_short=ctx.merge_commit_short,
    )


def branch_exists_on_remote(
    app: AppContext, branch_name: str, remote: str = "origin"
) -> bool:
    """Check if a branch exists on a remote.

    Args:
        app: Application context with repo.
        branch_name: Name of the branch to check.
        remote: Name of the remote (default: "origin").

    Returns:
        True if the branch exists on the remote, False otherwise.
    """
    try:
        # List remote refs and check if branch exists
        refs = app.repo.git.ls_remote("--heads", remote, branch_name)
        return bool(refs.strip())
    except Exception:
        return False


def branch_exists_locally(app: AppContext, branch_name: str) -> bool:
    """Check if a branch exists locally.

    Args:
        app: Application context with repo.
        branch_name: Name of the branch to check.

    Returns:
        True if the branch exists locally, False otherwise.
    """
    try:
        app.repo.git.rev_parse("--verify", f"refs/heads/{branch_name}")
        return True
    except Exception:
        return False


@click.group()
def branch():
    """Commands for managing merge-related branches.

    These commands help create and delete standardized branch names
    used in the merge conflict resolution workflow.

    \b
    Branch types:
    - main: Main working branch for merge resolution
    - conflict: Branch with committed merge markers
    - solution: Branch for solution attempts (PRs from here)

    Branch names are generated using the format from .mergai.yaml:
    branch.name_format (default: "mergai/%(target_branch)-%(merge_commit_short)/%(type)")

    When on a mergai branch, COMMIT and --target can often be omitted
    as they will be extracted from the current branch name.
    """
    pass


@branch.command()
@click.pass_obj
@click.argument("type", type=click.Choice(BRANCH_TYPES, case_sensitive=False))
@click.argument("commit", type=str, required=False, default=None)
@click.option(
    "--target",
    "-t",
    type=str,
    default=None,
    help="Target branch name (default: extracted from current branch or current branch name)",
)
def create(app: AppContext, type: str, commit: Optional[str], target: Optional[str]):
    """Create a merge-related branch and check it out.

    Creates a branch using the configured naming format and checks it out.
    Fails if the branch already exists on origin or locally.

    TYPE is one of: main, conflict, solution

    COMMIT is the commit SHA or ref. If omitted and currently on a mergai
    branch, the commit will be extracted from the current branch name.

    \b
    Branch purposes:
    - main: Main working branch where merge resolution work is integrated
    - conflict: Contains the merge commit with committed merge markers
    - solution: Contains solution attempts; PRs go from solution to conflict

    \b
    Examples:
        mergai branch create main abc1234
        mergai branch create solution abc1234 --target v8.0
        mergai branch create conflict HEAD
        mergai branch create solution          # when on a mergai branch
    """
    builder = get_branch_builder(app, commit, target)
    branch_name = builder.get_branch_name(type)

    # Check if branch exists on origin
    if branch_exists_on_remote(app, branch_name):
        raise click.ClickException(
            f"Branch '{branch_name}' already exists on origin. "
            "Delete it first with 'mergai branch delete'."
        )

    # Check if branch exists locally
    if branch_exists_locally(app, branch_name):
        raise click.ClickException(
            f"Branch '{branch_name}' already exists locally. "
            "Delete it first with 'mergai branch delete'."
        )

    # Create and checkout the branch
    try:
        app.repo.git.checkout("-b", branch_name)
        click.echo(f"Created and checked out branch: {branch_name}")
    except Exception as e:
        raise click.ClickException(f"Failed to create branch: {e}")


@branch.command()
@click.pass_obj
@click.argument("type", type=click.Choice(DELETE_BRANCH_TYPES, case_sensitive=False))
@click.argument("commit", type=str, required=False, default=None)
@click.option(
    "--target",
    "-t",
    type=str,
    default=None,
    help="Target branch name (default: extracted from current branch or current branch name)",
)
@click.option(
    "--remote",
    "-r",
    is_flag=True,
    default=False,
    help="Also delete the branch on origin",
)
@click.option(
    "--ignore-missing",
    is_flag=True,
    default=False,
    help="Don't fail if branch doesn't exist",
)
def delete(
    app: AppContext,
    type: str,
    commit: Optional[str],
    target: Optional[str],
    remote: bool,
    ignore_missing: bool,
):
    """Delete a merge-related branch.

    Deletes the specified branch type locally, and optionally on origin.

    TYPE is one of: main, conflict, solution, all

    COMMIT is the commit SHA or ref. If omitted and currently on a mergai
    branch, the commit will be extracted from the current branch name.

    Use 'all' as TYPE to delete all three branch types at once.
    Use --remote to also delete from origin.
    Use --ignore-missing to not fail if branch doesn't exist.

    \b
    Examples:
        mergai branch delete main abc1234
        mergai branch delete solution abc1234 --remote
        mergai branch delete all abc1234 -r
        mergai branch delete main abc1234 --ignore-missing
        mergai branch delete all --remote     # when on a mergai branch
    """
    builder = get_branch_builder(app, commit, target)

    # Determine which branch types to delete
    if type == "all":
        types_to_delete = BRANCH_TYPES
        # When deleting all branches, don't fail on missing - just report
        ignore_missing = True
    else:
        types_to_delete = [type]

    current_branch = git_utils.get_current_branch(app.repo)

    for branch_type in types_to_delete:
        branch_name = builder.get_branch_name(branch_type)

        # Check if we're on the branch we're trying to delete
        if current_branch == branch_name:
            raise click.ClickException(
                f"Cannot delete branch '{branch_name}' - currently checked out. "
                "Switch to another branch first."
            )

        # Delete local branch
        local_exists = branch_exists_locally(app, branch_name)
        if local_exists:
            try:
                app.repo.git.branch("-D", branch_name)
                click.echo(f"Deleted local branch: {branch_name}")
            except Exception as e:
                raise click.ClickException(
                    f"Failed to delete local branch '{branch_name}': {e}"
                )
        elif not ignore_missing:
            raise click.ClickException(f"Local branch '{branch_name}' does not exist")
        else:
            click.echo(f"Local branch '{branch_name}' does not exist (skipped)")

        # Delete remote branch if requested
        if remote:
            remote_exists = branch_exists_on_remote(app, branch_name)
            if remote_exists:
                try:
                    app.repo.git.push("origin", "--delete", branch_name)
                    click.echo(f"Deleted remote branch: origin/{branch_name}")
                except Exception as e:
                    raise click.ClickException(
                        f"Failed to delete remote branch 'origin/{branch_name}': {e}"
                    )
            elif not ignore_missing:
                raise click.ClickException(
                    f"Remote branch 'origin/{branch_name}' does not exist"
                )
            else:
                click.echo(
                    f"Remote branch 'origin/{branch_name}' does not exist (skipped)"
                )


@branch.command()
@click.pass_obj
@click.argument("type", type=click.Choice(PUSH_BRANCH_TYPES, case_sensitive=False))
@click.option(
    "--target",
    "-t",
    type=str,
    default=None,
    help="Target branch name (default: extracted from current branch or current branch name)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force push using --force-with-lease",
)
@click.option(
    "--ignore-missing",
    is_flag=True,
    default=False,
    help="Don't fail if branch doesn't exist locally",
)
def push(
    app: AppContext,
    type: str,
    target: Optional[str],
    force: bool,
    ignore_missing: bool,
):
    """Push a merge-related branch to origin.

    Pushes the specified branch type to origin.

    TYPE is one of: main, conflict, solution, all

    Use 'all' as TYPE to push all three branch types at once (only pushes
    branches that exist locally).
    Use --force to force push using --force-with-lease.
    Use --ignore-missing to not fail if branch doesn't exist locally.

    \b
    Examples:
        mergai branch push main
        mergai branch push solution --force
        mergai branch push all
        mergai branch push all -f
        mergai branch push main --ignore-missing
    """
    builder = get_branch_builder(app, None, target)

    # Determine which branch types to push
    if type == "all":
        types_to_push = BRANCH_TYPES
        # When pushing all branches, don't fail on missing - just report
        ignore_missing = True
    else:
        types_to_push = [type]

    for branch_type in types_to_push:
        branch_name = builder.get_branch_name(branch_type)

        # Check if branch exists locally
        local_exists = branch_exists_locally(app, branch_name)
        if local_exists:
            try:
                push_args = ["origin", branch_name]
                if force:
                    push_args.insert(1, "--force-with-lease")
                app.repo.git.push(*push_args)
                force_msg = " (force)" if force else ""
                click.echo(f"Pushed branch{force_msg}: {branch_name} -> origin/{branch_name}")
            except Exception as e:
                raise click.ClickException(
                    f"Failed to push branch '{branch_name}': {e}"
                )
        elif not ignore_missing:
            raise click.ClickException(f"Local branch '{branch_name}' does not exist")
        else:
            click.echo(f"Local branch '{branch_name}' does not exist (skipped)")


@branch.command()
@click.pass_obj
@click.argument("token", type=click.Choice(TOKEN_NAMES), required=False, default=None)
@click.option(
    "--branch",
    "-b",
    "branch_name",
    type=str,
    default=None,
    help="Branch name to parse (default: current branch)",
)
def info(app: AppContext, token: Optional[str], branch_name: Optional[str]):
    """Print parsed information from a mergai branch name.

    Parses the branch name and prints its components. If TOKEN is specified,
    prints only that token's value. Otherwise, prints all components.

    TOKEN is one of: target_branch, merge_commit_short, type

    \b
    Examples:
        mergai branch info                    # print all info for current branch
        mergai branch info target_branch      # print just the target branch
        mergai branch info type               # print just the branch type
        mergai branch info -b mergai/v8.0-abc123/main  # parse specific branch
    """
    # Determine which branch to parse
    if branch_name is None:
        branch_name = git_utils.get_current_branch(app.repo)

    # Parse the branch name
    parsed = BranchNameBuilder.parse_branch_name_with_config(
        branch_name, app.config.branch
    )

    if parsed is None:
        raise click.ClickException(
            f"Branch '{branch_name}' does not match the mergai branch format. "
            f"Expected format: {app.config.branch.name_format}"
        )

    if token is not None:
        # Print just the requested token value
        value = getattr(parsed, token if token != "type" else "branch_type")
        click.echo(value)
    else:
        # Print all components
        click.echo(f"target_branch: {parsed.target_branch}")
        click.echo(f"merge_commit_short: {parsed.merge_commit_short}")
        click.echo(f"type: {parsed.branch_type}")


@branch.command()
@click.pass_obj
@click.option(
    "--branch",
    "-b",
    "branch_name",
    type=str,
    default=None,
    help="Branch name to check (default: current branch)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Don't print anything, just set exit code",
)
def check(app: AppContext, branch_name: Optional[str], quiet: bool):
    """Check if a branch is a mergai branch.

    Exits with code 0 if the branch matches the mergai format, 1 otherwise.
    Useful for scripting and conditional logic.

    \b
    Examples:
        mergai branch check                   # check current branch
        mergai branch check -b some-branch    # check specific branch
        mergai branch check --quiet           # silent mode for scripts

    \b
    Script usage:
        if mergai branch check --quiet; then
            echo "On a mergai branch"
        fi
    """
    # Determine which branch to check
    if branch_name is None:
        branch_name = git_utils.get_current_branch(app.repo)

    # Parse the branch name
    parsed = BranchNameBuilder.parse_branch_name_with_config(
        branch_name, app.config.branch
    )

    if parsed is not None:
        if not quiet:
            click.echo(f"'{branch_name}' is a mergai branch")
        raise SystemExit(0)
    else:
        if not quiet:
            click.echo(f"'{branch_name}' is not a mergai branch")
        raise SystemExit(1)
