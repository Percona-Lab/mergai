import os
import json
import click
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from git import Repo
from ..app import AppContext


# Constants for notes refs
NOTES_REF = "mergai"
NOTES_MARKER_REF = "mergai-marker"
NOTES_REMOTE_TMP_REF = "mergai-remote-tmp"
NOTES_MARKER_REMOTE_TMP_REF = "mergai-marker-remote-tmp"


@dataclass
class NoteInfo:
    """Information about a single note."""

    commit_sha: str
    blob_sha: str
    content: Optional[dict] = None


@dataclass
class NotesConflict:
    """Information about a conflicting note."""

    commit_sha: str
    local_content: dict
    remote_content: dict


@dataclass
class NotesMergePreview:
    """Preview of what a notes merge would do."""

    local_only: List[NoteInfo]  # Notes only in local
    remote_only: List[NoteInfo]  # Notes only in remote
    identical: List[str]  # Commit SHAs with identical notes
    conflicts: List[NotesConflict]  # Commit SHAs with conflicting notes


def list_notes_for_ref(repo: Repo, ref: str) -> Dict[str, str]:
    """List all notes in a given ref.

    Args:
        repo: GitPython Repo instance.
        ref: Notes ref name (e.g., "mergai").

    Returns:
        Dict mapping commit SHA to note blob SHA.
    """
    try:
        output = repo.git.notes("--ref", ref, "list")
        if not output.strip():
            return {}

        notes = {}
        for line in output.strip().split("\n"):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    blob_sha, commit_sha = parts[0], parts[1]
                    notes[commit_sha] = blob_sha
        return notes
    except Exception:
        return {}


def get_note_content(repo: Repo, ref: str, commit_sha: str) -> Optional[dict]:
    """Get the JSON content of a note for a specific commit.

    Args:
        repo: GitPython Repo instance.
        ref: Notes ref name.
        commit_sha: The commit SHA to get the note for.

    Returns:
        Parsed JSON dict or None if note doesn't exist or isn't valid JSON.
    """
    try:
        content = repo.git.notes("--ref", ref, "show", commit_sha)
        return json.loads(content)
    except Exception:
        return None


def ref_exists(repo: Repo, ref: str) -> bool:
    """Check if a git ref exists.

    Args:
        repo: GitPython Repo instance.
        ref: Full ref path (e.g., "refs/notes/mergai").

    Returns:
        True if the ref exists, False otherwise.
    """
    try:
        repo.git.rev_parse("--verify", ref)
        return True
    except Exception:
        return False


def notes_merge_in_progress(repo: Repo) -> bool:
    """Check if a notes merge is currently in progress.

    A notes merge is in progress if the NOTES_MERGE_WORKTREE directory exists
    and contains conflict files, OR if NOTES_MERGE_PARTIAL ref exists.

    Args:
        repo: GitPython Repo instance.

    Returns:
        True if there's an in-progress notes merge.
    """
    # Check for NOTES_MERGE_PARTIAL ref (created during merge)
    partial_ref = Path(repo.git_dir) / "NOTES_MERGE_PARTIAL"
    if partial_ref.exists():
        return True

    # Check for non-empty NOTES_MERGE_WORKTREE
    worktree_path = Path(repo.git_dir) / "NOTES_MERGE_WORKTREE"
    if worktree_path.exists():
        # Check if there are any files in the worktree
        return any(worktree_path.iterdir())

    return False


def get_notes_merge_conflicts(repo: Repo) -> List[str]:
    """Get list of commit SHAs with merge conflicts.

    Args:
        repo: GitPython Repo instance.

    Returns:
        List of commit SHAs that have conflicts in the merge worktree.
    """
    worktree_path = Path(repo.git_dir) / "NOTES_MERGE_WORKTREE"
    if not worktree_path.exists():
        return []

    conflicts = []
    for item in worktree_path.iterdir():
        if item.is_file():
            conflicts.append(item.name)
    return conflicts


def preview_notes_merge(repo: Repo, local_ref: str, remote_ref: str) -> NotesMergePreview:
    """Preview what a notes merge would do without actually performing it.

    Compares local and remote notes refs and categorizes each note.

    Args:
        repo: GitPython Repo instance.
        local_ref: Local notes ref (e.g., "mergai").
        remote_ref: Remote notes ref (e.g., "mergai-remote-tmp").

    Returns:
        NotesMergePreview with categorized notes.
    """
    local_notes = list_notes_for_ref(repo, local_ref)
    remote_notes = list_notes_for_ref(repo, remote_ref)

    local_only = []
    remote_only = []
    identical = []
    conflicts = []

    # All commit SHAs from both sides
    all_commits = set(local_notes.keys()) | set(remote_notes.keys())

    for commit_sha in all_commits:
        local_blob = local_notes.get(commit_sha)
        remote_blob = remote_notes.get(commit_sha)

        if local_blob and not remote_blob:
            # Only in local
            local_only.append(
                NoteInfo(
                    commit_sha=commit_sha,
                    blob_sha=local_blob,
                    content=get_note_content(repo, local_ref, commit_sha),
                )
            )
        elif remote_blob and not local_blob:
            # Only in remote
            remote_only.append(
                NoteInfo(
                    commit_sha=commit_sha,
                    blob_sha=remote_blob,
                    content=get_note_content(repo, remote_ref, commit_sha),
                )
            )
        elif local_blob == remote_blob:
            # Identical (same blob SHA)
            identical.append(commit_sha)
        else:
            # Both exist but different - potential conflict
            local_content = get_note_content(repo, local_ref, commit_sha)
            remote_content = get_note_content(repo, remote_ref, commit_sha)

            # Check if content is actually different (blob SHA might differ but content same)
            if local_content == remote_content:
                identical.append(commit_sha)
            else:
                conflicts.append(
                    NotesConflict(
                        commit_sha=commit_sha,
                        local_content=local_content or {},
                        remote_content=remote_content or {},
                    )
                )

    return NotesMergePreview(
        local_only=local_only,
        remote_only=remote_only,
        identical=identical,
        conflicts=conflicts,
    )


def cleanup_temp_refs(repo: Repo):
    """Remove temporary refs created during fetch."""
    for ref in [NOTES_REMOTE_TMP_REF, NOTES_MARKER_REMOTE_TMP_REF]:
        try:
            repo.git.update_ref("-d", f"refs/notes/{ref}")
        except Exception:
            pass


def format_note_summary(note: Optional[dict]) -> str:
    """Format a brief summary of a note's contents."""
    if not note:
        return "(empty or invalid JSON)"

    fields = []
    if "merge_info" in note:
        fields.append("merge_info")
    if "conflict_context" in note:
        fields.append("conflict_context")
    if "merge_context" in note:
        fields.append("merge_context")
    if "solution" in note:
        fields.append("solution")
    if "solutions" in note:
        fields.append(f"solutions[{len(note['solutions'])}]")
    if "merge_description" in note:
        fields.append("merge_description")
    if "pr_comments" in note:
        fields.append("pr_comments")
    if "user_comment" in note:
        fields.append("user_comment")

    return ", ".join(fields) if fields else "(no recognized fields)"


def short_sha(sha: str) -> str:
    """Return shortened SHA."""
    return sha[:11]


@click.group()
@click.pass_obj
def notes(app: AppContext):
    """Manage mergai notes.

    Notes are stored in git notes refs (refs/notes/mergai*) and can be
    synchronized between local and remote repositories.
    """
    pass


@notes.command()
@click.pass_obj
@click.option(
    "-f",
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Force update (overwrite local notes with remote, DESTRUCTIVE)",
)
@click.option(
    "--ours",
    "strategy_ours",
    is_flag=True,
    default=False,
    help="On conflict, keep local notes",
)
@click.option(
    "--theirs",
    "strategy_theirs",
    is_flag=True,
    default=False,
    help="On conflict, keep remote notes",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Preview what would be merged without making changes",
)
@click.argument("remote", default="origin")
def update(
    app: AppContext,
    remote: str,
    force: bool,
    strategy_ours: bool,
    strategy_theirs: bool,
    dry_run: bool,
):
    """Update local notes from remote.

    Fetches notes from the remote and merges them with local notes.
    Notes for different commits are automatically merged (no conflict).

    If the same commit has different notes locally and remotely,
    a conflict is reported with instructions for resolution.

    \b
    Examples:
        mergai notes update              # Merge notes from origin
        mergai notes update upstream     # Merge notes from upstream remote
        mergai notes update --dry-run    # Preview what would be merged
        mergai notes update --ours       # Resolve conflicts keeping local
        mergai notes update --theirs     # Resolve conflicts keeping remote
        mergai notes update -f           # Force overwrite local (DESTRUCTIVE)
    """
    # Validate options
    if sum([force, strategy_ours, strategy_theirs]) > 1:
        raise click.ClickException(
            "Cannot combine --force, --ours, and --theirs. Use only one."
        )

    # Check for in-progress merge
    if notes_merge_in_progress(app.repo):
        conflicts = get_notes_merge_conflicts(app.repo)
        click.echo("Notes merge already in progress.", err=True)
        click.echo("")
        if conflicts:
            click.echo("Conflicting commits:")
            for sha in conflicts:
                click.echo(f"  - {short_sha(sha)}")
            click.echo("")
        click.echo("To resolve:")
        click.echo("  mergai notes merge --commit   # Finalize the merge")
        click.echo("  mergai notes merge --abort    # Abort and start over")
        raise SystemExit(1)

    # Force mode - just overwrite (original behavior with warning)
    if force:
        click.echo(
            click.style("WARNING: ", fg="yellow")
            + "Force mode will overwrite ALL local notes with remote notes."
        )
        click.echo("Any local notes not on the remote will be LOST.")
        click.echo("")

        try:
            refspec = "+refs/notes/mergai*:refs/notes/mergai*"
            app.repo.git.fetch(remote, refspec)
            click.echo(click.style("Done. ", fg="green") + "Local notes overwritten with remote.")
        except Exception as e:
            raise click.ClickException(f"Failed to fetch notes: {e}")
        return

    # Fetch remote notes to temporary refs
    click.echo(f"Fetching notes from {remote}...")

    has_local_notes = ref_exists(app.repo, f"refs/notes/{NOTES_REF}")
    has_local_markers = ref_exists(app.repo, f"refs/notes/{NOTES_MARKER_REF}")

    try:
        # Fetch main notes
        app.repo.git.fetch(
            remote, f"refs/notes/{NOTES_REF}:refs/notes/{NOTES_REMOTE_TMP_REF}"
        )
        has_remote_notes = True
    except Exception:
        has_remote_notes = False

    try:
        # Fetch marker notes
        app.repo.git.fetch(
            remote,
            f"refs/notes/{NOTES_MARKER_REF}:refs/notes/{NOTES_MARKER_REMOTE_TMP_REF}",
        )
        has_remote_markers = True
    except Exception:
        has_remote_markers = False

    if not has_remote_notes and not has_remote_markers:
        click.echo("No remote notes found.")
        cleanup_temp_refs(app.repo)
        return

    # Preview the merge
    if has_remote_notes:
        if has_local_notes:
            preview = preview_notes_merge(app.repo, NOTES_REF, NOTES_REMOTE_TMP_REF)
        else:
            # No local notes, all remote notes are "new"
            remote_notes = list_notes_for_ref(app.repo, NOTES_REMOTE_TMP_REF)
            preview = NotesMergePreview(
                local_only=[],
                remote_only=[
                    NoteInfo(commit_sha=sha, blob_sha=blob)
                    for sha, blob in remote_notes.items()
                ],
                identical=[],
                conflicts=[],
            )

        # Show preview
        click.echo("")
        click.echo("Merge preview:")
        click.echo(f"  - Local-only notes:  {len(preview.local_only)}")
        click.echo(f"  - Remote-only notes: {len(preview.remote_only)}")
        click.echo(f"  - Identical notes:   {len(preview.identical)}")
        click.echo(f"  - Conflicts:         {len(preview.conflicts)}")

        if preview.conflicts:
            click.echo("")
            click.echo(click.style("Conflicts detected:", fg="red"))
            for conflict in preview.conflicts:
                click.echo("")
                click.echo(f"  Commit: {short_sha(conflict.commit_sha)}")
                click.echo(f"    Local:  {format_note_summary(conflict.local_content)}")
                click.echo(f"    Remote: {format_note_summary(conflict.remote_content)}")

        if dry_run:
            click.echo("")
            click.echo("Dry run - no changes made.")
            cleanup_temp_refs(app.repo)
            return

        # If there are conflicts and no strategy specified, fail with instructions
        if preview.conflicts and not strategy_ours and not strategy_theirs:
            click.echo("")
            click.echo(click.style("Cannot auto-merge due to conflicts.", fg="red"))
            click.echo("")
            click.echo("To resolve, choose one of:")
            click.echo("")
            click.echo("  1. Keep YOUR local notes (discard remote changes for conflicts):")
            click.echo("     mergai notes update --ours")
            click.echo("")
            click.echo("  2. Keep REMOTE notes (discard local changes for conflicts):")
            click.echo("     mergai notes update --theirs")
            click.echo("")
            click.echo("  3. Manually inspect and resolve:")
            for conflict in preview.conflicts[:3]:  # Show first 3
                click.echo(f"     git notes --ref=mergai show {short_sha(conflict.commit_sha)}        # Local")
                click.echo(
                    f"     git notes --ref={NOTES_REMOTE_TMP_REF} show {short_sha(conflict.commit_sha)}  # Remote"
                )
                if len(preview.conflicts) > 3:
                    click.echo("     ...")
                break
            click.echo("")
            click.echo("  4. Force overwrite local with remote (DESTRUCTIVE):")
            click.echo("     mergai notes update -f")
            click.echo("")

            # Keep temp refs for manual inspection
            click.echo(
                f"Temporary remote notes kept at refs/notes/{NOTES_REMOTE_TMP_REF} for inspection."
            )
            click.echo(f"Run 'git update-ref -d refs/notes/{NOTES_REMOTE_TMP_REF}' to clean up.")
            raise SystemExit(1)

    # Perform the merge
    click.echo("")
    click.echo("Merging notes...")

    try:
        # Determine strategy
        if strategy_ours:
            strategy = "ours"
            click.echo("Using strategy: ours (keeping local on conflicts)")
        elif strategy_theirs:
            strategy = "theirs"
            click.echo("Using strategy: theirs (keeping remote on conflicts)")
        else:
            strategy = None

        # Merge main notes
        if has_remote_notes:
            if has_local_notes:
                merge_args = ["--ref", NOTES_REF, "merge"]
                if strategy:
                    merge_args.extend(["-s", strategy])
                merge_args.append(f"refs/notes/{NOTES_REMOTE_TMP_REF}")
                app.repo.git.notes(*merge_args)
            else:
                # No local notes - just copy the remote ref
                app.repo.git.update_ref(
                    f"refs/notes/{NOTES_REF}", f"refs/notes/{NOTES_REMOTE_TMP_REF}"
                )

        # Merge marker notes (always use theirs since they're identical content)
        if has_remote_markers:
            click.echo("Merging marker notes (using remote version)...")
            if has_local_markers:
                app.repo.git.notes(
                    "--ref",
                    NOTES_MARKER_REF,
                    "merge",
                    "-s",
                    "theirs",
                    f"refs/notes/{NOTES_MARKER_REMOTE_TMP_REF}",
                )
            else:
                app.repo.git.update_ref(
                    f"refs/notes/{NOTES_MARKER_REF}",
                    f"refs/notes/{NOTES_MARKER_REMOTE_TMP_REF}",
                )

        # Cleanup temp refs
        cleanup_temp_refs(app.repo)

        click.echo("")
        click.echo(click.style("Success! ", fg="green") + f"Notes updated from {remote}.")

        # Show summary
        if has_remote_notes and has_local_notes:
            click.echo(f"  - Kept {len(preview.local_only)} local-only notes")
            click.echo(f"  - Added {len(preview.remote_only)} notes from remote")
            if preview.conflicts:
                if strategy_ours:
                    click.echo(
                        f"  - Resolved {len(preview.conflicts)} conflicts (kept local)"
                    )
                else:
                    click.echo(
                        f"  - Resolved {len(preview.conflicts)} conflicts (kept remote)"
                    )

    except Exception as e:
        error_msg = str(e)
        if "CONFLICT" in error_msg or "Automatic notes merge failed" in error_msg:
            click.echo("")
            click.echo(click.style("Merge conflict occurred.", fg="red"))
            click.echo("")
            click.echo("Git has created a merge worktree for manual resolution.")
            click.echo("")
            click.echo("To resolve:")
            click.echo("  1. Edit files in .git/NOTES_MERGE_WORKTREE/")
            click.echo("  2. Run: mergai notes merge --commit")
            click.echo("")
            click.echo("Or abort: mergai notes merge --abort")
            raise SystemExit(1)
        else:
            cleanup_temp_refs(app.repo)
            raise click.ClickException(f"Failed to merge notes: {e}")


@notes.command()
@click.pass_obj
@click.argument("remote", default="origin")
def push(app: AppContext, remote: str):
    """Push local notes to remote.

    \b
    Examples:
        mergai notes push           # Push to origin
        mergai notes push upstream  # Push to upstream remote
    """
    try:
        app.repo.git.push(remote, "refs/notes/mergai*:refs/notes/mergai*")
        click.echo(f"Notes pushed to {remote}.")
    except Exception as e:
        raise click.ClickException(f"Failed to push notes: {e}")


@notes.command("merge")
@click.pass_obj
@click.option(
    "--commit",
    "do_commit",
    is_flag=True,
    help="Finalize an in-progress notes merge",
)
@click.option(
    "--abort",
    "do_abort",
    is_flag=True,
    help="Abort an in-progress notes merge",
)
def merge_cmd(app: AppContext, do_commit: bool, do_abort: bool):
    """Manage notes merge conflicts.

    After a 'mergai notes update' encounters conflicts, use this command
    to finalize or abort the merge.

    \b
    To resolve conflicts manually:
      1. Edit files in .git/NOTES_MERGE_WORKTREE/
         (each file is named by commit SHA and contains the note content)
      2. Run: mergai notes merge --commit

    \b
    Examples:
        mergai notes merge           # Show merge status
        mergai notes merge --commit  # Finalize the merge
        mergai notes merge --abort   # Abort the merge
    """
    if do_commit and do_abort:
        raise click.ClickException("Cannot use both --commit and --abort.")

    if not notes_merge_in_progress(app.repo):
        if do_commit or do_abort:
            click.echo("No notes merge in progress.")
        else:
            click.echo("No notes merge in progress.")
            click.echo("")
            click.echo("Use 'mergai notes update' to fetch and merge notes from a remote.")
        return

    if do_commit:
        try:
            app.repo.git.notes("--ref", NOTES_REF, "merge", "--commit")
            cleanup_temp_refs(app.repo)
            click.echo(click.style("Success! ", fg="green") + "Notes merge completed.")
        except Exception as e:
            raise click.ClickException(f"Failed to commit notes merge: {e}")

    elif do_abort:
        try:
            app.repo.git.notes("--ref", NOTES_REF, "merge", "--abort")
            cleanup_temp_refs(app.repo)
            click.echo("Notes merge aborted.")
        except Exception as e:
            raise click.ClickException(f"Failed to abort notes merge: {e}")

    else:
        # Show status
        conflicts = get_notes_merge_conflicts(app.repo)
        click.echo("Notes merge in progress.")
        click.echo("")

        if conflicts:
            click.echo(f"Conflicting notes ({len(conflicts)}):")
            for sha in conflicts:
                click.echo(f"  - {short_sha(sha)}")
            click.echo("")
            click.echo("Conflict files are in: .git/NOTES_MERGE_WORKTREE/")
            click.echo("")

        click.echo("To resolve:")
        click.echo("  1. Edit the conflict files (remove conflict markers)")
        click.echo("  2. Run: mergai notes merge --commit")
        click.echo("")
        click.echo("Or abort: mergai notes merge --abort")


@notes.command()
@click.pass_obj
@click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    help="Show detailed information about each note",
)
def status(app: AppContext, verbose: bool):
    """Show status of local notes.

    Displays information about local notes including count,
    and whether a merge is in progress.

    \b
    Examples:
        mergai notes status      # Show summary
        mergai notes status -v   # Show detailed list
    """

    # Check for merge in progress
    if notes_merge_in_progress(app.repo):
        conflicts = get_notes_merge_conflicts(app.repo)
        click.echo(click.style("Notes merge in progress!", fg="yellow"))
        click.echo("")
        if conflicts:
            click.echo(f"Conflicting commits ({len(conflicts)}):")
            for sha in conflicts:
                click.echo(f"  - {short_sha(sha)}")
        click.echo("")
        click.echo("Run 'mergai notes merge' for resolution options.")
        click.echo("")

    # List notes
    has_notes = ref_exists(app.repo, f"refs/notes/{NOTES_REF}")
    has_markers = ref_exists(app.repo, f"refs/notes/{NOTES_MARKER_REF}")

    if not has_notes:
        click.echo("No local notes found.")
        click.echo("")
        click.echo("To fetch notes from remote: mergai notes update")
        return

    notes_map = list_notes_for_ref(app.repo, NOTES_REF)
    markers_map = list_notes_for_ref(app.repo, NOTES_MARKER_REF)

    click.echo(f"Local notes: {len(notes_map)}")
    click.echo(f"Marker notes: {len(markers_map)}")

    if verbose and notes_map:
        click.echo("")
        click.echo("Notes by commit:")
        for commit_sha, blob_sha in sorted(notes_map.items()):
            content = get_note_content(app.repo, NOTES_REF, commit_sha)
            summary = format_note_summary(content)

            # Try to get commit info
            try:
                commit = app.repo.commit(commit_sha)
                commit_msg = commit.summary[:50]
                if len(commit.summary) > 50:
                    commit_msg += "..."
                click.echo(f"  {short_sha(commit_sha)} - {commit_msg}")
            except Exception:
                click.echo(f"  {short_sha(commit_sha)} (commit not found locally)")

            click.echo(f"    Fields: {summary}")

    click.echo("")

    # Check for temp refs (leftover from failed merge)
    has_temp = ref_exists(app.repo, f"refs/notes/{NOTES_REMOTE_TMP_REF}")
    if has_temp:
        click.echo(
            click.style("Note: ", fg="yellow")
            + f"Temporary ref 'refs/notes/{NOTES_REMOTE_TMP_REF}' exists from a previous fetch."
        )
        click.echo(
            f"Run 'git update-ref -d refs/notes/{NOTES_REMOTE_TMP_REF}' to clean up."
        )
