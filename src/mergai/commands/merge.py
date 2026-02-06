"""Merge command for mergai.

This module provides the merge command that performs a git merge
using the SHA from merge_info.
"""

import click
from git.exc import GitCommandError
from ..app import AppContext


# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFLICT = 1
EXIT_ERROR = 2


@click.command()
@click.pass_obj
def merge(app: AppContext):
    """Perform a git merge using the commit from merge_info.

    Executes 'git merge --no-commit --no-ff <sha>' where <sha> is
    the merge_commit from the initialized merge context.

    This command requires merge_info to be initialized first
    (via 'mergai context init').

    \b
    Exit codes:
        0: Merge completed successfully (no conflicts)
        1: Merge resulted in conflicts
        2: Error (missing merge_info, invalid commit, etc.)

    \b
    Examples:
        mergai context init abc1234 --target v8.0
        mergai merge

    \b
    Script usage:
        mergai merge
        case $? in
            0) echo "Merge successful" ;;
            1) echo "Merge has conflicts" ;;
            2) echo "Error occurred" ;;
        esac
    """
    # Load note to get merge_info
    note = app.load_note()

    if note is None or "merge_info" not in note:
        raise SystemExit(EXIT_ERROR)

    merge_info = note["merge_info"]
    merge_commit_sha = merge_info.get("merge_commit")

    if not merge_commit_sha:
        raise SystemExit(EXIT_ERROR)

    # Execute git merge --no-commit --no-ff <sha>
    try:
        app.repo.git.merge("--no-commit", "--no-ff", merge_commit_sha)
        raise SystemExit(EXIT_SUCCESS)
    except GitCommandError as e:
        # GitCommandError is raised when merge has conflicts
        # Git returns exit code 1 for conflicts
        if e.status == 1:
            raise SystemExit(EXIT_CONFLICT)
        else:
            # Other git errors (e.g., invalid ref, not a git repo)
            raise SystemExit(EXIT_ERROR)
    except Exception:
        # Any other unexpected error
        raise SystemExit(EXIT_ERROR)
