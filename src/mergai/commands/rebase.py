"""Rebase command for mergai branches.

This module provides the rebase command which rebases a mergai branch onto
a new base commit while preserving:
- Merge commits as merge commits (using --rebase-merges)
- All mergai notes attached to commits
- The note.json local state

After rebasing, merge_info.target_branch_sha is updated to reflect the new base.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import click

from ..app import AppContext
from ..utils import git_utils
from .notes import NOTES_MARKER_REF, NOTES_REF

# State file for tracking rebase progress
REBASE_STATE_FILE = "rebase_state.json"


@dataclass
class RebaseCommitInfo:
    """Information about a commit being rebased."""

    sha: str
    message: str
    parent_count: int
    has_mergai_note: bool
    has_marker_note: bool
    mergai_note_content: str | None  # Raw JSON string
    marker_note_content: str | None  # Raw text


def _get_rebase_state_path(app: AppContext) -> Path:
    """Get the path to the rebase state file."""
    return app.state.path / REBASE_STATE_FILE


def _rebase_state_exists(app: AppContext) -> bool:
    """Check if a rebase state file exists."""
    return _get_rebase_state_path(app).exists()


def _save_rebase_state(app: AppContext, state: dict) -> None:
    """Save rebase state to file."""
    path = _get_rebase_state_path(app)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _load_rebase_state(app: AppContext) -> dict | None:
    """Load rebase state from file."""
    path = _get_rebase_state_path(app)
    if not path.exists():
        return None
    with open(path) as f:
        result: dict = json.load(f)
        return result


def _remove_rebase_state(app: AppContext) -> None:
    """Remove the rebase state file."""
    path = _get_rebase_state_path(app)
    if path.exists():
        path.unlink()


def _is_git_rebase_in_progress(app: AppContext) -> bool:
    """Check if a git rebase is currently in progress."""
    git_dir = Path(app.repo.git_dir)
    # Git stores rebase state in .git/rebase-merge or .git/rebase-apply
    return (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists()


def _collect_commits_info(app: AppContext, base_sha: str) -> list[RebaseCommitInfo]:
    """Collect information about commits from base_sha to HEAD.

    Args:
        app: AppContext instance.
        base_sha: The base commit SHA (exclusive).

    Returns:
        List of RebaseCommitInfo for each commit, ordered oldest to newest.
    """
    commits_info = []

    # Get commits from base to HEAD (newest first)
    for commit in app.repo.iter_commits(f"{base_sha}..HEAD"):
        # Get mergai note content if exists
        mergai_note_content = None
        has_mergai_note = False
        try:
            mergai_note_content = app.repo.git.notes(
                "--ref", NOTES_REF, "show", commit.hexsha
            )
            has_mergai_note = True
        except Exception:
            # Note doesn't exist for this commit - this is expected for commits
            # without mergai notes
            pass

        # Get marker note content if exists
        marker_note_content = None
        has_marker_note = False
        try:
            marker_note_content = app.repo.git.notes(
                "--ref", NOTES_MARKER_REF, "show", commit.hexsha
            )
            has_marker_note = True
        except Exception:
            # Marker note doesn't exist for this commit - this is expected for
            # commits without mergai marker notes
            pass

        # Get commit message
        message = (
            commit.message
            if isinstance(commit.message, str)
            else commit.message.decode("utf-8", errors="replace")
        )

        commits_info.append(
            RebaseCommitInfo(
                sha=commit.hexsha,
                message=message,
                parent_count=len(commit.parents),
                has_mergai_note=has_mergai_note,
                has_marker_note=has_marker_note,
                mergai_note_content=mergai_note_content,
                marker_note_content=marker_note_content,
            )
        )

    # Reverse to get oldest first
    commits_info.reverse()
    return commits_info


def _build_commit_mapping(
    app: AppContext,
    commits_info: list[RebaseCommitInfo],
    new_base_sha: str,
) -> dict[str, str]:
    """Build mapping from old commit SHAs to new commit SHAs after rebase.

    Matches commits by message and parent count (to detect merge commits).

    Note: This heuristic-based approach relies solely on commit message text and
    parent count. It may produce incorrect mappings when two or more commits in
    the range have identical messages and the same parent count. In such cases,
    notes from the first duplicate could be transferred to the wrong new commit.
    A more robust approach would use git's rebase rewritten-list file or reflog,
    but this implementation trades accuracy for simplicity and portability.

    Args:
        app: AppContext instance.
        commits_info: List of RebaseCommitInfo from before rebase.
        new_base_sha: The new base commit SHA.

    Returns:
        Dict mapping old_sha -> new_sha.
    """
    # Get new commits from new base to HEAD
    new_commits = list(app.repo.iter_commits(f"{new_base_sha}..HEAD"))
    new_commits.reverse()  # oldest first

    mapping: dict[str, str] = {}
    new_idx = 0

    for old_info in commits_info:
        if new_idx >= len(new_commits):
            # No more new commits to match
            break

        new_commit = new_commits[new_idx]
        new_message = (
            new_commit.message
            if isinstance(new_commit.message, str)
            else new_commit.message.decode("utf-8", errors="replace")
        )

        # Match by message and parent count
        if (
            new_message.strip() == old_info.message.strip()
            and len(new_commit.parents) == old_info.parent_count
        ):
            mapping[old_info.sha] = new_commit.hexsha
            new_idx += 1
        else:
            # Try to find a match further in the new commits
            # (rebase might have reordered or skipped some commits)
            for search_idx in range(new_idx, len(new_commits)):
                search_commit = new_commits[search_idx]
                search_message = (
                    search_commit.message
                    if isinstance(search_commit.message, str)
                    else search_commit.message.decode("utf-8", errors="replace")
                )
                if (
                    search_message.strip() == old_info.message.strip()
                    and len(search_commit.parents) == old_info.parent_count
                ):
                    mapping[old_info.sha] = search_commit.hexsha
                    new_idx = search_idx + 1
                    break

    return mapping


def _update_note_target_branch_sha(note_content: str, new_base_sha: str) -> str:
    """Update target_branch_sha in note content JSON.

    Args:
        note_content: Raw JSON string of the note.
        new_base_sha: The new base commit SHA.

    Returns:
        Updated JSON string with new target_branch_sha.
    """
    try:
        note_dict = json.loads(note_content)
        if "merge_info" in note_dict and isinstance(note_dict["merge_info"], dict):
            note_dict["merge_info"]["target_branch_sha"] = new_base_sha
        return json.dumps(note_dict, indent=2)
    except (json.JSONDecodeError, TypeError):
        # If we can't parse/update the JSON, return as-is
        return note_content


def _transfer_notes(
    app: AppContext,
    commits_info: list[RebaseCommitInfo],
    mapping: dict[str, str],
    new_base_sha: str,
) -> tuple[int, int]:
    """Transfer notes from old commits to new commits.

    Updates target_branch_sha in each note's merge_info to reflect the new base.

    Args:
        app: AppContext instance.
        commits_info: List of RebaseCommitInfo with note content.
        mapping: Dict mapping old_sha -> new_sha.
        new_base_sha: The new base commit SHA to set in notes.

    Returns:
        Tuple of (notes_transferred, notes_failed).
    """
    import os
    import tempfile

    notes_transferred = 0
    notes_failed = 0

    for old_info in commits_info:
        old_sha = old_info.sha
        new_sha = mapping.get(old_sha)

        if not new_sha:
            # Old commit wasn't mapped (possibly dropped in rebase)
            if old_info.has_mergai_note or old_info.has_marker_note:
                notes_failed += 1
            continue

        # Transfer mergai note (with updated target_branch_sha)
        if old_info.has_mergai_note and old_info.mergai_note_content:
            try:
                # Update target_branch_sha in the note content
                updated_content = _update_note_target_branch_sha(
                    old_info.mergai_note_content, new_base_sha
                )

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    f.write(updated_content)
                    temp_path = f.name

                try:
                    app.repo.git.notes(
                        "--ref", NOTES_REF, "add", "-f", "-F", temp_path, new_sha
                    )
                    notes_transferred += 1
                finally:
                    os.unlink(temp_path)
            except Exception as e:
                click.echo(
                    f"Warning: Failed to transfer mergai note to {git_utils.short_sha(new_sha)}: {e}",
                    err=True,
                )
                notes_failed += 1

        # Transfer marker note
        if old_info.has_marker_note and old_info.marker_note_content:
            try:
                app.repo.git.notes(
                    "--ref",
                    NOTES_MARKER_REF,
                    "add",
                    "-f",
                    "-m",
                    old_info.marker_note_content,
                    new_sha,
                )
            except Exception as e:
                click.echo(
                    f"Warning: Failed to transfer marker note to {git_utils.short_sha(new_sha)}: {e}",
                    err=True,
                )
                notes_failed += 1

    return notes_transferred, notes_failed


def _update_note_json(
    app: AppContext,
    new_base_sha: str,
    mapping: dict[str, str],
) -> None:
    """Update note.json with new base SHA and updated note_index.

    Args:
        app: AppContext instance.
        new_base_sha: The new base commit SHA.
        mapping: Dict mapping old_sha -> new_sha.
    """
    note = app.note

    # Update target_branch_sha
    note.merge_info.target_branch_sha = new_base_sha

    # Update note_index entries with new SHAs
    if note.has_note_index and note.note_index is not None:
        for entry in note.note_index:
            old_sha = entry.get("sha")
            if old_sha and old_sha in mapping:
                entry["sha"] = mapping[old_sha]

    app.save_note(note)


def _serialize_commits_info(commits_info: list[RebaseCommitInfo]) -> list[dict]:
    """Serialize commits info to dict for JSON storage."""
    return [
        {
            "sha": info.sha,
            "message": info.message,
            "parent_count": info.parent_count,
            "has_mergai_note": info.has_mergai_note,
            "has_marker_note": info.has_marker_note,
            "mergai_note_content": info.mergai_note_content,
            "marker_note_content": info.marker_note_content,
        }
        for info in commits_info
    ]


def _deserialize_commits_info(data: list[dict]) -> list[RebaseCommitInfo]:
    """Deserialize commits info from dict."""
    return [
        RebaseCommitInfo(
            sha=item["sha"],
            message=item["message"],
            parent_count=item["parent_count"],
            has_mergai_note=item["has_mergai_note"],
            has_marker_note=item["has_marker_note"],
            mergai_note_content=item.get("mergai_note_content"),
            marker_note_content=item.get("marker_note_content"),
        )
        for item in data
    ]


@click.command()
@click.pass_obj
@click.argument("onto", required=False)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview what would happen without making changes",
)
@click.option(
    "--continue",
    "do_continue",
    is_flag=True,
    default=False,
    help="Continue after resolving conflicts",
)
@click.option(
    "--abort",
    "do_abort",
    is_flag=True,
    default=False,
    help="Abort an in-progress rebase",
)
def rebase(
    app: AppContext,
    onto: str | None,
    dry_run: bool,
    do_continue: bool,
    do_abort: bool,
):
    """Rebase the current branch onto a new base commit.

    This command rebases the current mergai branch onto the specified
    commit (e.g., upstream/master) while preserving:

    \b
    - Merge commits as merge commits (using --rebase-merges)
    - All mergai notes attached to commits
    - The note.json local state

    After rebasing, merge_info.target_branch_sha is updated to the
    new base commit.

    ONTO is the commit/branch to rebase onto (e.g., "upstream/master").

    \b
    Examples:
        mergai rebase upstream/master           # Rebase onto upstream/master
        mergai rebase upstream/master --dry-run # Preview without changes
        mergai rebase --continue                # Continue after resolving conflicts
        mergai rebase --abort                   # Abort the rebase
    """
    # Validate mutually exclusive options
    if do_continue and do_abort:
        raise click.ClickException("Cannot use both --continue and --abort.")

    if (do_continue or do_abort) and onto:
        raise click.ClickException(
            "Cannot combine --continue or --abort with ONTO argument."
        )

    # Warn if --dry-run is used with --continue or --abort (unsupported)
    if dry_run and (do_continue or do_abort):
        click.echo(
            "Warning: --dry-run is not supported with --continue or --abort and will be ignored.",
            err=True,
        )

    # Handle --abort
    if do_abort:
        _handle_abort(app)
        return

    # Handle --continue
    if do_continue:
        _handle_continue(app)
        return

    # Starting a new rebase - ONTO is required
    if not onto:
        raise click.ClickException(
            "ONTO argument is required when starting a rebase.\n\n"
            "Usage: mergai rebase <onto>\n"
            "Example: mergai rebase upstream/master"
        )

    _handle_rebase(app, onto, dry_run)


def _handle_abort(app: AppContext) -> None:
    """Handle the --abort flag."""
    if not _is_git_rebase_in_progress(app):
        # Check if we have state but git rebase finished
        if _rebase_state_exists(app):
            _remove_rebase_state(app)
            click.echo("Removed stale rebase state (git rebase was not in progress).")
        else:
            click.echo("No rebase in progress.")
        return

    # Abort the git rebase
    try:
        app.repo.git.rebase("--abort")
        click.echo("Git rebase aborted.")
    except Exception as e:
        raise click.ClickException(f"Failed to abort git rebase: {e}") from e

    # Clean up our state
    _remove_rebase_state(app)
    click.echo("Rebase aborted successfully.")


def _handle_continue(app: AppContext) -> None:
    """Handle the --continue flag."""
    if not _rebase_state_exists(app):
        if _is_git_rebase_in_progress(app):
            raise click.ClickException(
                "Git rebase is in progress, but no mergai rebase state found.\n"
                "Use 'git rebase --continue' directly, then manually update notes."
            )
        else:
            click.echo("No rebase in progress.")
            return

    # Load our state
    state = _load_rebase_state(app)
    if state is None:
        raise click.ClickException("Failed to load rebase state.")

    # Check if git rebase is still in progress
    if _is_git_rebase_in_progress(app):
        # Continue the git rebase
        try:
            app.repo.git.rebase("--continue")
        except Exception as e:
            error_msg = str(e)
            if "CONFLICT" in error_msg or "could not apply" in error_msg.lower():
                click.echo("Conflicts remain. Please resolve them and run:")
                click.echo("  mergai rebase --continue")
                click.echo("")
                click.echo("Or abort with:")
                click.echo("  mergai rebase --abort")
                raise SystemExit(1) from e
            else:
                raise click.ClickException(f"Git rebase --continue failed: {e}") from e

    # Git rebase completed - now transfer notes and update note.json
    _finalize_rebase(app, state)


def _handle_rebase(app: AppContext, onto: str, dry_run: bool) -> None:
    """Handle starting a new rebase."""
    # Check for existing rebase
    if _is_git_rebase_in_progress(app):
        raise click.ClickException(
            "A git rebase is already in progress.\n"
            "Use 'mergai rebase --continue' or 'mergai rebase --abort'."
        )

    if _rebase_state_exists(app):
        raise click.ClickException(
            "A mergai rebase state already exists.\n"
            "Use 'mergai rebase --continue' or 'mergai rebase --abort'."
        )

    # Require clean working tree
    if app.repo.is_dirty(untracked_files=False):
        raise click.ClickException(
            "Working directory has uncommitted changes.\n"
            "Please commit or stash them before rebasing."
        )

    # Resolve the "onto" reference
    try:
        onto_sha = git_utils.resolve_ref_sha(app.repo, onto)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    # Get current merge_info
    if not app.has_note:
        raise click.ClickException(
            "No note.json found. This command requires a mergai context.\n"
            "Run 'mergai context init' first."
        )

    current_base_sha = app.note.merge_info.target_branch_sha

    # Check if onto is the same as current base
    if onto_sha == current_base_sha:
        click.echo(f"Already based on {git_utils.short_sha(onto_sha)}. Nothing to do.")
        return

    # Check if onto is an ancestor of current base (would be a backwards rebase)
    try:
        app.repo.git.merge_base("--is-ancestor", onto_sha, current_base_sha)
        # If the above doesn't raise, onto is an ancestor of current base
        click.echo(
            f"Warning: {onto} ({git_utils.short_sha(onto_sha)}) is an ancestor of "
            f"current base ({git_utils.short_sha(current_base_sha)})."
        )
        click.echo("This would move the branch backwards. Proceeding anyway...")
    except Exception:
        # Not an ancestor - this is the normal case (moving forward)
        pass

    # Collect commits info
    click.echo(
        f"Collecting commits from {git_utils.short_sha(current_base_sha)}..HEAD..."
    )
    commits_info = _collect_commits_info(app, current_base_sha)

    if not commits_info:
        click.echo("No commits to rebase.")
        # Update merge_info to reflect new base (we already verified onto_sha != current_base_sha above)
        app.note.merge_info.target_branch_sha = onto_sha
        app.save_note(app.note)
        click.echo(f"Updated target_branch_sha to {git_utils.short_sha(onto_sha)}.")
        return

    # Count merge commits and notes
    merge_commit_count = sum(1 for c in commits_info if c.parent_count > 1)
    notes_count = sum(1 for c in commits_info if c.has_mergai_note)

    click.echo(f"Found {len(commits_info)} commit(s) to rebase:")
    click.echo(f"  - {merge_commit_count} merge commit(s)")
    click.echo(f"  - {notes_count} commit(s) with mergai notes")
    click.echo("")

    # Dry run - show what would happen
    if dry_run:
        click.echo("Dry run - no changes will be made.")
        click.echo("")
        click.echo(f"Would rebase onto: {onto} ({git_utils.short_sha(onto_sha)})")
        click.echo(f"Current base:      {git_utils.short_sha(current_base_sha)}")
        click.echo("")
        click.echo("Commits to rebase:")
        for info in commits_info:
            short_sha = git_utils.short_sha(info.sha)
            first_line = info.message.split("\n")[0].strip()[:50]
            merge_marker = " (merge)" if info.parent_count > 1 else ""
            note_marker = " [note]" if info.has_mergai_note else ""
            click.echo(f"  {short_sha}{merge_marker}{note_marker} {first_line}")
        return

    # Save state before starting rebase
    state = {
        "onto_sha": onto_sha,
        "original_base_sha": current_base_sha,
        "commits": _serialize_commits_info(commits_info),
        "original_note": app.note.to_dict(),
    }
    _save_rebase_state(app, state)

    # Perform the rebase
    click.echo(f"Rebasing onto {onto} ({git_utils.short_sha(onto_sha)})...")
    click.echo("")

    try:
        # Use --rebase-merges to preserve merge commits
        app.repo.git.rebase("--rebase-merges", "--onto", onto_sha, current_base_sha)
    except Exception as e:
        error_msg = str(e)
        if "CONFLICT" in error_msg or "could not apply" in error_msg.lower():
            click.echo("Rebase stopped due to conflicts.")
            click.echo("")
            click.echo(
                "Please resolve the conflicts, stage the resolved files, then run:"
            )
            click.echo("  mergai rebase --continue")
            click.echo("")
            click.echo("Or abort the rebase with:")
            click.echo("  mergai rebase --abort")
            raise SystemExit(1) from e
        else:
            # Unexpected error - clean up state
            _remove_rebase_state(app)
            raise click.ClickException(f"Rebase failed: {e}") from e

    # Rebase completed successfully - finalize
    _finalize_rebase(app, state)


def _finalize_rebase(app: AppContext, state: dict) -> None:
    """Finalize the rebase after git rebase completes.

    Transfers notes to new commits and updates note.json.
    """
    onto_sha = state["onto_sha"]
    commits_info = _deserialize_commits_info(state["commits"])

    click.echo("Rebase completed. Transferring notes...")

    # Build mapping from old SHAs to new SHAs
    mapping = _build_commit_mapping(app, commits_info, onto_sha)

    if len(mapping) != len(commits_info):
        click.echo(
            f"Warning: Only {len(mapping)} of {len(commits_info)} commits were mapped."
        )
        click.echo("Some notes may not have been transferred.")

    # Transfer notes (with updated target_branch_sha)
    notes_transferred, notes_failed = _transfer_notes(
        app, commits_info, mapping, onto_sha
    )

    if notes_transferred > 0:
        click.echo(f"Transferred {notes_transferred} note(s) to new commits.")
    if notes_failed > 0:
        click.echo(f"Failed to transfer {notes_failed} note(s).")

    # Update note.json
    _update_note_json(app, onto_sha, mapping)
    click.echo(f"Updated target_branch_sha to {git_utils.short_sha(onto_sha)}.")

    # Clean up state
    _remove_rebase_state(app)

    click.echo("")
    click.echo(click.style("Rebase completed successfully!", fg="green"))
    click.echo("")
    click.echo("Next steps:")
    click.echo("  - Review the rebased commits with 'git log'")
    click.echo("  - Push with 'mergai branch push -f' (force push required)")
