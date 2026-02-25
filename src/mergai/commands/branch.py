"""Branch management commands for merge-related workflows.

This module provides commands for creating and deleting standardized
branch names used in the merge conflict resolution workflow:

- main: Main working branch where all work for merging a given commit will be merged
- conflict: Branch containing a merge commit with committed merge markers
- solution: Branch with solution attempts; PRs are created from solution to conflict
"""

import click
from ..app import AppContext
from ..utils.branch_name_builder import BranchType
from ..utils import git_utils

BRANCH_ALL = "all"

# Excluded types for common operations:
# - target: not valid for create/push/delete commands
EXCLUDED_TYPES = [BranchType.TARGET.value]

# Common valid branch types, includes all except EXCLUDED_TYPES
COMMON_BRANCH_TYPES = [t.value for t in BranchType if t not in EXCLUDED_TYPES]

# Valid branch types for create command ('target' not included)
CREATE_BRANCH_TYPES = COMMON_BRANCH_TYPES

# Valid branch types for delete command (includes 'all'; 'target' not included)
DELETE_BRANCH_TYPES = COMMON_BRANCH_TYPES + [BRANCH_ALL]

# List of branch types to delete when 'all' is specified
DELETE_ALL_BRANCH_TYPES = COMMON_BRANCH_TYPES

# Valid branch types for push command (includes 'all'; 'target' not included)
PUSH_BRANCH_TYPES = COMMON_BRANCH_TYPES + [BRANCH_ALL]

# List of branch types to push when 'all' is specified
PUSH_ALL_BRANCH_TYPES = COMMON_BRANCH_TYPES

# Valid branch types for switch command (includes 'target')
SWITCH_BRANCH_TYPES = COMMON_BRANCH_TYPES + [BranchType.TARGET.value]

# Valid token names for the info command
TOKEN_NAMES = ["target_branch", "target_branch_sha", "merge_commit_sha", "type"]


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
    - target: The branch we're merging into (e.g., 'master' or 'v8.0')

    The following commands support the 'all' type:
    - delete
    - push
    - switch

    The following commands support the 'target' type:
    - switch

    Branch names are generated using the format from .mergai/config.yml:
    branch.name_format (default: "mergai/%(target_branch)-%(merge_commit_short_sha)-%(target_branch_short_sha)/%(type)")
    """
    pass


@branch.command()
@click.pass_obj
@click.argument("type", type=click.Choice(CREATE_BRANCH_TYPES, case_sensitive=False))
def create(app: AppContext, type: str):
    """Create a merge-related branch and check it out.

    Creates a branch using the configured naming format and checks it out.
    Fails if the branch already exists on origin or locally.

    TYPE is one of: main, conflict, solution

    The target branch is not supported.

    \b
    Branch purposes:
    - main: Main working branch where merge resolution work is integrated
    - conflict: Contains the merge commit with committed merge markers
    - solution: Contains solution attempts; PRs go from solution to conflict
    """
    branch_name = app.branches.get_branch_name(type)

    # Check if branch exists on origin
    if git_utils.branch_exists_on_remote(app.repo, branch_name):
        raise click.ClickException(
            f"Branch '{branch_name}' already exists on origin. "
            "Delete it first with 'mergai branch delete -r'."
        )

    # Check if branch exists locally
    if git_utils.branch_exists_locally(app.repo, branch_name):
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
    remote: bool,
    ignore_missing: bool,
):
    """Delete a merge-related branch.

    Deletes the specified branch type locally, and optionally on origin.

    TYPE is one of: main, conflict, solution, all

    The target branch is not supported.

    Use 'all' as TYPE to delete all three branch types at once.
    Use --remote to also delete from origin.
    Use --ignore-missing to not fail if branch doesn't exist.
    """

    # Determine which branch types to delete
    if type == BRANCH_ALL:
        types_to_delete = DELETE_ALL_BRANCH_TYPES
        # When deleting all branches, don't fail on missing - just report
        ignore_missing = True
    else:
        types_to_delete = [type]

    current_branch = git_utils.get_current_branch(app.repo)

    for branch_type in types_to_delete:
        branch_name = app.branches.get_branch_name(branch_type)

        # Check if we're on the branch we're trying to delete
        if current_branch == branch_name:
            raise click.ClickException(
                f"Cannot delete branch '{branch_name}' - currently checked out. "
                "Switch to another branch first."
            )

        # Delete local branch
        local_exists = git_utils.branch_exists_locally(app.repo, branch_name)
        if local_exists:
            try:
                app.repo.git.branch("-D", branch_name)
                click.echo(f"Deleted local branch: {branch_name}")
            except Exception as e:
                raise click.ClickException(
                    f"Failed to delete local branch '{branch_name}': {e}"
                )
        elif not ignore_missing and not remote:
            # Only require local branch when not deleting remote; with -r we allow remote-only delete
            raise click.ClickException(f"Local branch '{branch_name}' does not exist")
        else:
            click.echo(f"Local branch '{branch_name}' does not exist (skipped)")

        # Delete remote branch if requested
        if remote:
            remote_exists = git_utils.branch_exists_on_remote(app.repo, branch_name)
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
    force: bool,
    ignore_missing: bool,
):
    """Push a merge-related branch to origin.

    Pushes the specified branch type to origin.

    TYPE is one of: main, conflict, solution, all

    The target branch is not supported.

    Use 'all' as TYPE to push all three branch types at once (only pushes
    branches that exist locally).
    Use --force to force push using --force-with-lease.
    Use --ignore-missing to not fail if branch doesn't exist locally.

    """

    # Determine which branch types to push
    if type == BRANCH_ALL:
        types_to_push = PUSH_ALL_BRANCH_TYPES
        # When pushing all branches, don't fail on missing - just report
        ignore_missing = True
    else:
        types_to_push = [type]

    for branch_type in types_to_push:
        branch_name = app.branches.get_branch_name(branch_type)

        # Check if branch exists locally
        local_exists = git_utils.branch_exists_locally(app.repo, branch_name)
        if local_exists:
            try:
                push_args = ["origin", branch_name]
                if force:
                    push_args.insert(1, "--force-with-lease")
                app.repo.git.push(*push_args)
                force_msg = " (force)" if force else ""
                click.echo(
                    f"Pushed branch{force_msg}: {branch_name} -> origin/{branch_name}"
                )
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
@click.argument("type", type=click.Choice(SWITCH_BRANCH_TYPES, case_sensitive=False))
def switch(app: AppContext, type: str):
    """Switch to a merge-related branch.

    Switches to the specified branch type. The branch must exist locally.

    TYPE is one of: main, conflict, solution, target

    When TYPE is 'target', switches to the original target branch from merge_info
    (e.g., the branch you're merging into like 'master' or 'v8.0').

    For main/conflict/solution types, the target and commit are auto-detected
    from the current mergai branch or merge_info in note.json.

    \b
    Examples:
        mergai branch switch main          # switch to main working branch
        mergai branch switch conflict      # switch to conflict branch
        mergai branch switch solution      # switch to solution branch
        mergai branch switch target        # switch to target branch (e.g., master)
    """

    branch_name = app.branches.get_branch_name(type)

    # Check if already on this branch
    current_branch = git_utils.get_current_branch(app.repo)
    if current_branch == branch_name:
        click.echo(f"Already on branch: {branch_name}")
        return

    # Check if branch exists locally
    if not git_utils.branch_exists_locally(app.repo, branch_name):
        raise click.ClickException(
            f"Branch '{branch_name}' does not exist locally. "
            f"Create it first with 'mergai branch create {type}' or 'git checkout {branch_name}'."
        )

    # Switch to the branch
    try:
        app.repo.git.checkout(branch_name)
        click.echo(f"Switched to branch: {branch_name}")
    except Exception as e:
        raise click.ClickException(f"Failed to switch to branch: {e}")
