from git import Repo, Commit, Blob
import hashlib
import re
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from enum import StrEnum

log = logging.getLogger(__name__)

BlobsMapType = dict[str, list[Tuple[int, Blob]]]


class ConflictType(StrEnum):
    BOTH_MODIFIED = "both modified"
    BOTH_ADDED = "both added"
    DELETED_BY_THEM = "deleted by them"
    ADDED_BY_US = "added by us"
    DELETED_BY_US = "deleted by us"
    ADDED_BY_THEM = "added by them"
    UNKNOWN = "unknown"


def short_sha(sha: str) -> str:
    return sha[:11]


def resolve_ref_sha(repo: Repo, ref: str, try_remote: bool = True) -> str:
    """Resolve a git reference (branch, tag, SHA) to its full SHA.

    Tries to resolve the reference directly first. If that fails and try_remote
    is True, attempts to resolve with 'origin/' prefix for remote-only branches.

    This is useful in CI environments where branches may exist only on the remote
    (e.g., after 'git fetch' without creating local tracking branches).

    Args:
        repo: GitPython Repo instance.
        ref: Git reference to resolve (branch name, tag, SHA, etc.).
        try_remote: If True, try 'origin/{ref}' when direct resolution fails.

    Returns:
        The full SHA (hexsha) of the resolved reference.

    Raises:
        ValueError: If the reference cannot be resolved locally or remotely.
    """
    # Try direct resolution first
    try:
        return repo.commit(ref).hexsha
    except Exception as direct_error:
        if not try_remote:
            raise ValueError(f"Cannot resolve ref '{ref}': {direct_error}")

        # Try with origin/ prefix for remote-only branches
        try:
            return repo.commit(f"origin/{ref}").hexsha
        except Exception as remote_error:
            raise ValueError(
                f"Cannot resolve ref '{ref}' (tried both local and origin/{ref}): {remote_error}"
            )


def author_to_dict(author):
    """Convert a git Author to a dict.

    Args:
        author: GitPython Actor object.

    Returns:
        Dict with name and email.
    """
    return {
        "name": author.name,
        "email": author.email,
    }


def is_merge_in_progress(repo: Repo) -> bool:
    merge_head = Path(repo.git_dir) / "MERGE_HEAD"
    return merge_head.exists()


def mark_conflict_markers_unresolved(file_path: str) -> None:
    """
    Add (UNRESOLVED) marker to all conflict markers in the given file.

    Transforms:
        <<<<<<< HEAD           ->  <<<<<<< HEAD (UNRESOLVED)
        ||||||| merged common  ->  ||||||| merged common (UNRESOLVED)
        =======                ->  ======= (UNRESOLVED)
        >>>>>>> branch-name    ->  >>>>>>> branch-name (UNRESOLVED)

    Args:
        file_path: Path to the file containing conflict markers.
    """
    path = Path(file_path)
    if not path.exists():
        log.warning(f"File not found, skipping: {file_path}")
        return

    content = path.read_text()

    # Pattern matches conflict markers at the start of a line
    # Group 1: The marker itself (<<<<<<< or ||||||| or ======= or >>>>>>>)
    # Group 2: Optional text after the marker (branch name, etc.)
    pattern = r"^(<{7}|>{7}|\|{7}|={7})(.*)$"

    def add_unresolved_marker(match):
        marker = match.group(1)
        rest = match.group(2)
        # Don't add if already marked
        if "(UNRESOLVED)" in rest:
            return match.group(0)
        return f"{marker}{rest} (UNRESOLVED)"

    modified_content = re.sub(
        pattern, add_unresolved_marker, content, flags=re.MULTILINE
    )
    path.write_text(modified_content)


def get_current_branch(repo: Repo) -> str:
    try:
        return repo.active_branch.name
    except TypeError:
        # Detached HEAD state
        return short_sha(repo.head.commit.hexsha)


def branch_exists_on_remote(
    repo: Repo, branch_name: str, remote: str = "origin"
) -> bool:
    """Check if a branch exists on a remote.

    Args:
        repo: GitPython Repo instance.
        branch_name: Name of the branch to check.
        remote: Name of the remote (default: "origin").

    Returns:
        True if the branch exists on the remote, False otherwise.
    """
    try:
        refs = repo.git.ls_remote("--heads", remote, branch_name)
        return bool(refs.strip())
    except Exception:
        return False


def branch_exists_locally(repo: Repo, branch_name: str) -> bool:
    """Check if a branch exists locally.

    Args:
        repo: GitPython Repo instance.
        branch_name: Name of the branch to check.

    Returns:
        True if the branch exists locally, False otherwise.
    """
    try:
        repo.git.rev_parse("--verify", f"refs/heads/{branch_name}")
        return True
    except Exception:
        return False


def conflict_type_supports_diff(conflict_type: ConflictType) -> bool:
    return conflict_type in {
        ConflictType.BOTH_MODIFIED,
        ConflictType.BOTH_ADDED,
    }


def compress_unified_diff(
    diff_text: str,
    *,
    block_threshold: int = 30,  # minimum block size to start folding
    head: int = 5,  # keep first N lines of big block
    tail: int = 5,  # keep last M lines
) -> str:
    """
    Compress large +/- blocks in a unified diff for display purposes.

    - Only affects contiguous blocks of lines starting with '+' or '-'
      (but not '+++'/'---' file headers).
    - For blocks longer than `block_threshold`, keeps `head` first lines
      and `tail` last lines, and inserts a summary line in the middle.
    """
    lines = diff_text.splitlines()
    out: list[str] = []

    i = 0
    n = len(lines)

    def is_change_line(line: str) -> bool:
        if not line:
            return False
        if line.startswith("+++ ") or line.startswith("--- "):
            return False
        return line[0] in {"+", "-"}

    while i < n:
        line = lines[i]

        if not is_change_line(line):
            out.append(line)
            i += 1
            continue

        # start of a +/- block
        sign = line[0]
        block_start = i
        j = i

        while j < n and is_change_line(lines[j]) and lines[j][0] == sign:
            j += 1

        block = lines[block_start:j]
        block_len = len(block)

        if block_len >= block_threshold and head + tail < block_len:
            # keep head + tail, fold the middle
            kept_head = block[:head]
            kept_tail = block[-tail:]

            middle_count = block_len - head - tail

            if sign == "-":
                summary = f"- (... {middle_count} more deleted lines...)"
            else:
                summary = f"+ (... {middle_count} more added lines...)"

            out.extend(kept_head)
            out.append(summary)
            out.extend(kept_tail)
        else:
            out.extend(block)

        i = j

    return "\n".join(out)


def get_diffs(
    repo: Repo,
    blobs_map: BlobsMapType,
    lines_of_context: int = 0,
    use_compressed_diffs: bool = False,
) -> dict:
    diffs = {}
    args = [
        "--cc",
        f"-U{lines_of_context}",  # lines of context
        "--no-color",
    ]

    for file_path, stages in blobs_map.items():
        conflict_type = get_conflict_type(stages)
        if not conflict_type_supports_diff(conflict_type):
            continue
        diff = repo.git.diff(*args + [file_path])

        if use_compressed_diffs:
            diff = compress_unified_diff(diff)

        diffs[file_path] = diff

    return diffs


def get_their_commits(
    repo: Repo, base: Commit, theirs: Commit, blobs_map: BlobsMapType
) -> dict:
    """Get commits that modified files in their branch since the base.

    Args:
        repo: GitPython Repo instance.
        base: The merge base commit.
        theirs: Their (MERGE_HEAD) commit.
        blobs_map: Map of file paths to blob stages from unmerged_blobs().

    Returns:
        Dict mapping file paths to lists of commit SHAs (hexsha strings).
    """
    their_commits = {}
    for file_path, _ in blobs_map.items():
        file_commits = list(
            repo.iter_commits(f"{base.hexsha}..{theirs.hexsha}", paths=[file_path])
        )
        if file_commits:
            their_commits[file_path] = [commit.hexsha for commit in file_commits]

    return their_commits


def get_conflict_type(stages: list) -> ConflictType:
    stage_numbers = {stage for stage, _ in stages}

    has_base = 1 in stage_numbers
    has_ours = 2 in stage_numbers
    has_theirs = 3 in stage_numbers

    if has_ours and has_theirs:
        if has_base:
            return ConflictType.BOTH_MODIFIED
        else:
            return ConflictType.BOTH_ADDED
    elif has_ours and not has_theirs:
        if has_base:
            return ConflictType.DELETED_BY_THEM
        else:
            return ConflictType.ADDED_BY_US
    elif has_theirs and not has_ours:
        if has_base:
            return ConflictType.DELETED_BY_US
        else:
            return ConflictType.ADDED_BY_THEM
    else:
        return ConflictType.UNKNOWN


def get_conflict_context(
    repo: Repo,
    use_diffs: bool = True,
    lines_of_context: int = 0,
    use_compressed_diffs: bool = False,
    use_their_commits: bool = False,
) -> dict:
    """Get conflict context from the current merge state.

    Returns a dict with only SHA strings for commits (not full commit objects).
    This dict is suitable for storage in note.json.

    Args:
        repo: GitPython Repo instance.
        use_diffs: Include diff hunks for conflicting files.
        lines_of_context: Number of context lines in diffs.
        use_compressed_diffs: Compress large diff blocks.
        use_their_commits: Include list of commits that modified each file.

    Returns:
        Dict with conflict context data, or None if no merge in progress.
        Commits are stored as SHA strings only:
        - ours_commit: str (hexsha)
        - theirs_commit: str (hexsha)
        - base_commit: str (hexsha)
        - their_commits: dict[str, list[str]] (file -> list of hexsha)
    """
    if not is_merge_in_progress(repo):
        return None

    blobs_map = repo.index.unmerged_blobs()

    ours_commit = repo.head.commit
    theirs_commit = repo.commit("MERGE_HEAD")
    base_commit = repo.merge_base(ours_commit, theirs_commit)[0]

    context = {
        "ours_commit": ours_commit.hexsha,
        "theirs_commit": theirs_commit.hexsha,
        "base_commit": base_commit.hexsha,
        "files": list(blobs_map.keys()),
        "conflict_types": {
            file_path: get_conflict_type(stages)
            for file_path, stages in blobs_map.items()
        },
    }

    if use_diffs:
        context["diffs"] = get_diffs(
            repo,
            blobs_map,
            lines_of_context=lines_of_context,
            use_compressed_diffs=use_compressed_diffs,
        )

    if use_their_commits:
        context["their_commits"] = get_their_commits(
            repo, base_commit, theirs_commit, blobs_map
        )

    return context


def merge_has_conflicts(repo: Repo, parent1: Commit, parent2: Commit) -> bool:
    try:
        # Try to perform a dry-run merge to see if there are conflicts
        repo.git.merge_tree(parent1, parent2)
    except Exception as e:
        return True


def read_commit_note(repo: Repo, ref: str, commit: str) -> Optional[str]:
    try:
        note = repo.git.notes("--ref", ref, "show", repo.commit(commit).hexsha)
        return note
    except Exception as e:
        return None


def find_remote_by_url(repo: Repo, url: str) -> Optional[str]:
    """Find a remote by its URL.

    Iterates through all remotes in the repository and returns the name
    of the first remote whose URL matches the provided URL.

    Args:
        repo: GitPython Repo object
        url: The URL to search for (exact match)

    Returns:
        The name of the matching remote, or None if no match is found.
    """
    for remote in repo.remotes:
        # Check all URLs associated with this remote (fetch and push URLs)
        for remote_url in remote.urls:
            if remote_url == url:
                return remote.name
    return None


def is_merge_conflict_style_diff3(repo: Repo) -> bool:
    try:
        merge_conflict_style = repo.git.config("merge.conflictstyle")
        return merge_conflict_style.lower() == "diff3"
    except Exception:
        return False


def get_merge_strategy(repo: Repo) -> str:
    """Get the default merge strategy from git config.

    Checks git config for pull.twohead (the default merge strategy for
    two-head merges). If not configured, returns 'ort' which is the
    default in modern git versions.

    Args:
        repo: GitPython Repo instance.

    Returns:
        The merge strategy name (e.g., 'ort', 'recursive').
    """
    try:
        strategy = repo.git.config("pull.twohead")
        return strategy
    except Exception:
        # Default strategy in modern git is 'ort'
        return "ort"


@dataclass
class GitMergeOutput:
    """Parsed git merge output.

    Attributes:
        auto_merged_files: List of files that were auto-merged by git.
        conflicting_files: Dict mapping file paths to conflict type strings.
        success: True if merge completed without conflicts, False otherwise.
        strategy: The merge strategy used (extracted from output like
                  "Merge made by the 'ort' strategy.").
        raw_output: The original unparsed output.
    """

    auto_merged_files: List[str]
    conflicting_files: dict[str, str]  # file -> conflict type (e.g., "content")
    success: bool
    strategy: Optional[str]
    raw_output: str


def parse_git_merge_output(output: str, repo: Optional[Repo] = None) -> GitMergeOutput:
    """Parse the output of a git merge command.

    Extracts information about:
    - Auto-merged files (lines matching "Auto-merging <file>")
    - Conflicting files (lines matching "CONFLICT (<type>): ...")
    - Whether the merge succeeded or failed
    - The merge strategy used (from "Merge made by the '<strategy>' strategy."
      or from git config if using --no-commit)

    Args:
        output: The raw output from git merge command (stdout + stderr).
        repo: Optional GitPython Repo instance. If provided and strategy
              cannot be parsed from output, will attempt to get default
              strategy from git config.

    Returns:
        GitMergeOutput dataclass with parsed information.

    Example input (success with commit):
        Auto-merging src/file1.cpp
        Auto-merging src/file2.cpp
        Merge made by the 'ort' strategy.
        ...

    Example input (success with --no-commit):
        Auto-merging src/file1.cpp
        Auto-merging src/file2.cpp
        Automatic merge went well; stopped before committing as requested

    Example input (conflict):
        Auto-merging file1.txt
        CONFLICT (content): Merge conflict in file1.txt
        Auto-merging file2.txt
        Automatic merge failed; fix conflicts and then commit the result.
    """
    auto_merged_files = []
    conflicting_files = {}
    strategy = None
    success = True

    for line in output.splitlines():
        # Check for auto-merged files
        match = re.match(r"^Auto-merging (.+)$", line)
        if match:
            auto_merged_files.append(match.group(1))
            continue

        # Check for strategy (indicates success with commit)
        # Example: "Merge made by the 'ort' strategy."
        match = re.match(r"^Merge made by the '([^']+)' strategy\.$", line)
        if match:
            strategy = match.group(1)
            continue

        # Check for conflicts
        # Example: "CONFLICT (content): Merge conflict in file1.txt"
        # Example: "CONFLICT (modify/delete): file.txt deleted in HEAD..."
        match = re.match(
            r"^CONFLICT \(([^)]+)\):\s*(?:Merge conflict in\s+)?(.+?)(?:\s+deleted.*)?$",
            line,
        )
        if match:
            conflict_type = match.group(1)
            file_path = match.group(2).strip()
            conflicting_files[file_path] = conflict_type
            continue

        # Check for failure
        if line.startswith("Automatic merge failed"):
            success = False

    # If strategy not found in output but repo provided, get from git config
    # This happens when using --no-commit flag
    if strategy is None and repo is not None:
        strategy = get_merge_strategy(repo)

    return GitMergeOutput(
        auto_merged_files=auto_merged_files,
        conflicting_files=conflicting_files,
        success=success,
        strategy=strategy,
        raw_output=output,
    )


@dataclass
class ForkStatus:
    """Status information about a fork compared to its upstream base."""

    fork_ref: str
    upstream_ref: str

    # Commit count
    commits_behind: int

    # Key commits
    last_merged_commit: Optional[Commit]
    first_unmerged_commit: Optional[Commit]
    last_unmerged_commit: Optional[Commit]
    merge_base_commit: Optional[Commit]

    # List of unmerged commits (for optional listing)
    unmerged_commits: List[Commit]

    # Divergence stats
    files_affected: int
    total_additions: int
    total_deletions: int

    @property
    def is_up_to_date(self) -> bool:
        """Check if the fork is up to date with upstream."""
        return self.commits_behind == 0

    @property
    def days_behind(self) -> int:
        """Calculate the number of days since the last merged commit."""
        if not self.last_merged_commit:
            return 0

        last_merged_date = datetime.fromtimestamp(
            self.last_merged_commit.authored_date, tz=timezone.utc
        )
        now = datetime.now(tz=timezone.utc)
        return (now - last_merged_date).days

    @property
    def unmerged_date_range(self) -> Optional[Tuple[datetime, datetime]]:
        """Get the date range of unmerged commits (first, last)."""
        if not self.first_unmerged_commit or not self.last_unmerged_commit:
            return None

        first_date = datetime.fromtimestamp(
            self.first_unmerged_commit.authored_date, tz=timezone.utc
        )
        last_date = datetime.fromtimestamp(
            self.last_unmerged_commit.authored_date, tz=timezone.utc
        )
        return (first_date, last_date)

    @property
    def unmerged_days_span(self) -> int:
        """Get the number of days spanned by unmerged commits."""
        date_range = self.unmerged_date_range
        if not date_range:
            return 0
        return (date_range[1] - date_range[0]).days


@dataclass
class CommitStats:
    """Statistics about a commit's changes.

    Attributes:
        files_changed: Number of files changed in this commit.
        lines_added: Number of lines added.
        lines_deleted: Number of lines deleted.
        total_lines: Total lines changed (added + deleted).
        files_modified: List of file paths modified by this commit.
    """

    files_changed: int
    lines_added: int
    lines_deleted: int
    total_lines: int
    files_modified: List[str]


def get_commit_stats(repo: Repo, commit: Commit) -> CommitStats:
    """Get statistics about files and lines changed in a commit.

    Args:
        repo: GitPython Repo object.
        commit: The commit to analyze.

    Returns:
        CommitStats with file and line change information.
    """
    files_modified = []
    lines_added = 0
    lines_deleted = 0

    # Get the parent commit (or empty tree for initial commit)
    if commit.parents:
        parent = commit.parents[0]
        diff_index = parent.diff(commit)
    else:
        # For initial commit, compare against empty tree
        diff_index = commit.diff(None)

    for diff_item in diff_index:
        # Get the path (a_path for deletions, b_path for additions/modifications)
        path = diff_item.b_path if diff_item.b_path else diff_item.a_path
        if path:
            files_modified.append(path)

    # Use git show --numstat for accurate line counts
    try:
        numstat = repo.git.show("--numstat", "--format=", commit.hexsha).strip()
        if numstat:
            for line in numstat.split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    # Handle binary files which show as '-'
                    if parts[0] != "-":
                        lines_added += int(parts[0])
                    if parts[1] != "-":
                        lines_deleted += int(parts[1])
    except Exception as e:
        log.warning(f"Failed to get numstat for commit {commit.hexsha}: {e}")

    return CommitStats(
        files_changed=len(files_modified),
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        total_lines=lines_added + lines_deleted,
        files_modified=files_modified,
    )


def get_commit_modified_files(repo: Repo, commit: Commit) -> List[str]:
    """Get list of files modified by a commit.

    This is a lightweight version that only returns file paths without
    calculating line statistics.

    Args:
        repo: GitPython Repo object.
        commit: The commit to analyze.

    Returns:
        List of file paths modified by the commit.
    """
    files_modified = []

    if commit.parents:
        parent = commit.parents[0]
        diff_index = parent.diff(commit)
    else:
        diff_index = commit.diff(None)

    for diff_item in diff_index:
        path = diff_item.b_path if diff_item.b_path else diff_item.a_path
        if path:
            files_modified.append(path)

    return files_modified


def get_merged_commits(
    repo: Repo,
    target_branch: str,
    merge_commit: str,
) -> List[str]:
    """Get list of commit hashes being merged.

    Finds the merge base between target_branch and merge_commit,
    then returns all commit hashes from base..merge_commit.

    Args:
        repo: GitPython Repo object.
        target_branch: The branch being merged into.
        merge_commit: The commit (or ref) being merged.

    Returns:
        List of commit hexsha strings (full 40-char hashes).
    """
    # Resolve commits
    target = repo.commit(target_branch)
    merge = repo.commit(merge_commit)

    # Find merge base
    bases = repo.merge_base(target, merge)
    if not bases:
        raise ValueError(
            f"No common ancestor found between {target_branch} and {merge_commit}"
        )
    base = bases[0]

    # Get commits from base to merge_commit (excluding base itself)
    commits = list(repo.iter_commits(f"{base.hexsha}..{merge.hexsha}"))

    return [commit.hexsha for commit in commits]


def is_branching_point(
    repo: Repo, commit: Commit, upstream_ref: str
) -> Tuple[bool, int]:
    """Check if a commit is a branching point.

    A commit is considered a branching point if it has multiple children
    within the upstream branch. This typically indicates where branches
    diverged and can be an important merge point.

    Args:
        repo: GitPython Repo object.
        commit: The commit to check.
        upstream_ref: The upstream reference to check children against.

    Returns:
        Tuple of (is_branching_point, child_count).
        is_branching_point is True if the commit has multiple children.
        child_count is the number of children found (0 if none).
    """
    try:
        # Find commits in upstream that have this commit as a parent
        # Use git rev-list with --children to find children
        # Alternative: count commits that have this commit as parent
        children_output = repo.git.rev_list(
            "--children", f"{commit.hexsha}..{upstream_ref}"
        ).strip()

        if not children_output:
            return (False, 0)

        # Parse the output to find children of this specific commit
        # Format: "commit_sha child1 child2 ..."
        child_count = 0
        for line in children_output.split("\n"):
            parts = line.split()
            if len(parts) > 0 and parts[0] == commit.hexsha:
                # This line shows children of our commit
                child_count = len(parts) - 1
                break

        return (child_count > 1, child_count)
    except Exception as e:
        log.warning(f"Failed to check branching point for {commit.hexsha}: {e}")
        return (False, 0)


def get_fork_status(repo: Repo, upstream_ref: str, fork_ref: str) -> ForkStatus:
    """
    Get comprehensive status about how a fork diverges from its upstream base.

    Args:
        repo: GitPython Repo object
        upstream_ref: The upstream/base branch/ref to compare against
        fork_ref: The fork branch/ref (downstream, typically HEAD)

    Returns:
        ForkStatus object with divergence information showing commits
        in upstream that need to be merged into the fork.
    """
    # Get merge base (common ancestor)
    try:
        merge_base = repo.merge_base(fork_ref, upstream_ref)
        merge_base_commit = merge_base[0] if merge_base else None
    except Exception:
        merge_base_commit = None

    # Get commits behind (in upstream but not in fork)
    # These are commits we need to merge from upstream
    # git rev-list fork_ref..upstream_ref = commits in upstream not in fork
    try:
        behind_output = repo.git.rev_list(f"{fork_ref}..{upstream_ref}").strip()
        behind_shas = behind_output.split("\n") if behind_output else []
        unmerged_commits = [repo.commit(sha) for sha in behind_shas if sha]
    except Exception:
        unmerged_commits = []

    # Find last merged commit (most recent commit in upstream that's also in fork)
    # This is essentially the merge base
    last_merged_commit = merge_base_commit

    # First unmerged commit (oldest commit in upstream not in fork)
    # unmerged_commits are in reverse chronological order (newest first)
    first_unmerged_commit = unmerged_commits[-1] if unmerged_commits else None

    # Last unmerged commit (most recent commit in upstream not in fork)
    last_unmerged_commit = unmerged_commits[0] if unmerged_commits else None

    # Get divergence stats using git diff --stat
    files_affected = 0
    total_additions = 0
    total_deletions = 0

    if unmerged_commits:
        numstat = None
        # Try three-dot diff first (uses merge base)
        # Fall back to two-dot diff if no merge base exists
        try:
            numstat = repo.git.diff("--numstat", f"{fork_ref}...{upstream_ref}").strip()
        except Exception:
            try:
                # Fallback: diff between the refs directly
                numstat = repo.git.diff("--numstat", fork_ref, upstream_ref).strip()
            except Exception as e:
                log.warning(f"Failed to get diff stats: {e}")

        if numstat:
            # Use numstat for machine-readable output
            # Format: additions<tab>deletions<tab>filename
            for line in numstat.split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    files_affected += 1
                    # Handle binary files which show as '-'
                    if parts[0] != "-":
                        total_additions += int(parts[0])
                    if parts[1] != "-":
                        total_deletions += int(parts[1])

    return ForkStatus(
        fork_ref=fork_ref,
        upstream_ref=upstream_ref,
        commits_behind=len(unmerged_commits),
        last_merged_commit=last_merged_commit,
        first_unmerged_commit=first_unmerged_commit,
        last_unmerged_commit=last_unmerged_commit,
        merge_base_commit=merge_base_commit,
        unmerged_commits=unmerged_commits,
        files_affected=files_affected,
        total_additions=total_additions,
        total_deletions=total_deletions,
    )
