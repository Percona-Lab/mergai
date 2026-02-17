from git import Repo, Commit, Blob
import re
import json
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

def is_merge_commit(commit: Commit) -> bool:
    return len(commit.parents) > 1

def is_merge_commit_parent(commit: Commit, parent_sha: str) -> bool:
    return is_merge_commit(commit) and commit.parents[1].hexsha == parent_sha

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


def commit_would_conflict(
    repo: Repo, commit: Commit, target_ref: str
) -> Tuple[bool, List[str]]:
    """Check if merging a commit would cause conflicts without modifying working tree.

    Uses `git merge-tree --write-tree` which performs a merge simulation without
    touching the index or working tree. This is a read-only operation.

    Args:
        repo: GitPython Repo instance.
        commit: The commit to check for conflicts.
        target_ref: The target reference to merge into (e.g., "HEAD", branch name).

    Returns:
        Tuple of (has_conflicts, conflicting_files).
        has_conflicts is True if the merge would have conflicts.
        conflicting_files is a list of file paths that would conflict.
    """
    from git.exc import GitCommandError

    try:
        # git merge-tree --write-tree --name-only target_ref commit
        # Exit code 0 = clean merge, exit code 1 = conflicts
        # --name-only makes conflicting file info easier to parse
        repo.git.merge_tree("--write-tree", "--name-only", target_ref, commit.hexsha)
        # If we get here, no conflicts (exit code 0)
        return (False, [])
    except GitCommandError as e:
        # Exit code 1 means conflicts - parse the output from stdout
        # The output format with --name-only and conflicts is:
        # <tree-oid>
        # <conflicting-file1>
        # <conflicting-file2>
        # <empty line>
        # Auto-merging <file>
        # CONFLICT (content): Merge conflict in <file>
        # ...

        # Get stdout from the exception - it contains the merge-tree output
        # GitCommandError.stdout contains the formatted output string
        output = e.stdout if hasattr(e, "stdout") and e.stdout else str(e)

        conflicting_files = []
        lines = output.split("\n")

        # Parse: skip tree OID, collect file names until empty line or info messages
        in_file_section = False
        for line in lines:
            # Don't strip - preserve original but remove leading "  stdout: '" if present
            line = line.strip()

            # Skip empty lines - they mark end of file section
            if not line:
                if in_file_section:
                    break  # Empty line after files means end of file list
                continue

            # Skip GitPython error prefix lines
            if line.startswith("Cmd(") or line.startswith("cmdline:") or line.startswith("stdout:"):
                # Handle "  stdout: '<tree-oid>" format
                if "stdout:" in line and "'" in line:
                    # Extract the actual content after stdout: '
                    idx = line.find("'")
                    if idx != -1:
                        line = line[idx + 1 :].rstrip("'")
                    else:
                        continue
                else:
                    continue

            # Tree OID is 40 hex chars - marks start of file section
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                in_file_section = True
                continue

            # Stop at informational messages
            if line.startswith("Auto-merging") or line.startswith("CONFLICT"):
                break

            # Collect file paths
            if in_file_section and line:
                conflicting_files.append(line)

        return (True, conflicting_files)
    except Exception:
        # Other errors (not exit code 1) - assume no conflict info available
        return (True, [])


def get_note_from_commit(repo: Repo, ref: str, commit: str) -> Optional[str]:
    try:
        note = repo.git.notes("--ref", ref, "show", repo.commit(commit).hexsha)
        return note
    except Exception as e:
        return None

def get_note_from_commit_as_dict(repo: Repo, ref: str, commit: str) -> Optional[dict]:
    note_str = get_note_from_commit(repo, ref, commit)
    if not note_str:
        return None

    return json.loads(note_str)


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


class ForkStatus:
    """Status information about a fork compared to its upstream base.
    
    This class uses lazy loading for commits - commit SHAs are stored
    and full Commit objects are only created when needed. This significantly
    improves performance when there are many unmerged commits.
    """

    def __init__(
        self,
        repo: Repo,
        fork_ref: str,
        upstream_ref: str,
        commits_behind: int,
        last_merged_commit: Optional[Commit],
        first_unmerged_commit: Optional[Commit],
        last_unmerged_commit: Optional[Commit],
        merge_base_commit: Optional[Commit],
        unmerged_commit_shas: List[str],
        files_affected: int,
        total_additions: int,
        total_deletions: int,
    ):
        self._repo = repo
        self.fork_ref = fork_ref
        self.upstream_ref = upstream_ref
        self.commits_behind = commits_behind
        self.last_merged_commit = last_merged_commit
        self.first_unmerged_commit = first_unmerged_commit
        self.last_unmerged_commit = last_unmerged_commit
        self.merge_base_commit = merge_base_commit
        self._unmerged_commit_shas = unmerged_commit_shas
        self._unmerged_commits_cache: Optional[List[Commit]] = None
        self.files_affected = files_affected
        self.total_additions = total_additions
        self.total_deletions = total_deletions

    @property
    def unmerged_commit_shas(self) -> List[str]:
        """Get the list of unmerged commit SHAs (newest first).
        
        This is efficient as it doesn't require loading full Commit objects.
        """
        return self._unmerged_commit_shas

    @property
    def unmerged_commits(self) -> List[Commit]:
        """Get the list of unmerged commits (newest first).
        
        Note: This loads all Commit objects on first access. For better
        performance when only needing SHAs or a subset of commits, use
        unmerged_commit_shas or get_commit() instead.
        """
        if self._unmerged_commits_cache is None:
            self._unmerged_commits_cache = [
                self._repo.commit(sha) for sha in self._unmerged_commit_shas
            ]
        return self._unmerged_commits_cache

    def get_commit(self, sha: str) -> Commit:
        """Get a single commit by SHA.
        
        Args:
            sha: The commit SHA to retrieve.
            
        Returns:
            The Commit object.
        """
        return self._repo.commit(sha)

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
        num_of_dirs: Number of unique directories containing modified files.
    """

    files_changed: int
    lines_added: int
    lines_deleted: int
    total_lines: int
    files_modified: List[str]
    num_of_dirs: int


def get_batch_commit_stats(repo: Repo, commit_shas: List[str]) -> dict[str, CommitStats]:
    """Batch calculate stats for multiple commits in a single git call.

    This is much more efficient than calling get_commit_stats() for each commit
    individually, as it uses a single `git log --numstat` command.

    Args:
        repo: GitPython Repo object.
        commit_shas: List of commit SHA strings to analyze.

    Returns:
        Dictionary mapping commit SHA to CommitStats.
    """
    if not commit_shas:
        return {}

    result = {}

    try:
        # Use git log with custom format to get all stats in one call
        # Format: SHA, then numstat output, separated by a marker
        # We use --stdin to pass the list of commits
        output = repo.git.log(
            "--numstat",
            "--format=COMMIT_START %H",
            "--no-walk",
            *commit_shas
        )

        current_sha = None
        current_files = []
        current_additions = 0
        current_deletions = 0
        current_dirs = set()

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("COMMIT_START "):
                # Save previous commit if exists
                if current_sha is not None:
                    result[current_sha] = CommitStats(
                        files_changed=len(current_files),
                        lines_added=current_additions,
                        lines_deleted=current_deletions,
                        total_lines=current_additions + current_deletions,
                        files_modified=current_files,
                        num_of_dirs=len(current_dirs),
                    )
                # Start new commit
                current_sha = line.split()[1]
                current_files = []
                current_additions = 0
                current_deletions = 0
                current_dirs = set()
            else:
                # Parse numstat line: additions<tab>deletions<tab>filename
                parts = line.split("\t")
                if len(parts) >= 3:
                    file_path = parts[2]
                    current_files.append(file_path)
                    # Handle binary files which show as '-'
                    if parts[0] != "-":
                        current_additions += int(parts[0])
                    if parts[1] != "-":
                        current_deletions += int(parts[1])
                    # Calculate directory
                    parent = str(Path(file_path).parent)
                    if parent != ".":
                        current_dirs.add(parent)

        # Save last commit
        if current_sha is not None:
            result[current_sha] = CommitStats(
                files_changed=len(current_files),
                lines_added=current_additions,
                lines_deleted=current_deletions,
                total_lines=current_additions + current_deletions,
                files_modified=current_files,
                num_of_dirs=len(current_dirs),
            )

    except Exception as e:
        log.warning(f"Failed to batch get commit stats: {e}")
        # Return empty dict, caller can fall back to individual calls

    return result


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

    # Calculate unique directories from file paths
    unique_dirs = set()
    for file_path in files_modified:
        # Get parent directory (or '.' for root-level files)
        parent = str(Path(file_path).parent)
        if parent != ".":
            unique_dirs.add(parent)

    return CommitStats(
        files_changed=len(files_modified),
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        total_lines=lines_added + lines_deleted,
        files_modified=files_modified,
        num_of_dirs=len(unique_dirs),
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


def get_batch_branching_points(
    repo: Repo, base_sha: str, upstream_ref: str
) -> dict[str, int]:
    """Batch detect all branching points in a commit range.

    A branching point is a commit with multiple children. This function
    efficiently detects all branching points in a single git command.

    Args:
        repo: GitPython Repo object.
        base_sha: The base commit SHA (oldest commit in range, exclusive).
        upstream_ref: The upstream reference (newest commit in range).

    Returns:
        Dictionary mapping commit SHA to child count for commits with >1 child.
        Only commits with multiple children are included in the result.
    """
    result = {}

    try:
        # git rev-list --children base..upstream gives:
        # <commit_sha> <child1> <child2> ...
        # for each commit in the range
        children_output = repo.git.rev_list(
            "--children", f"{base_sha}..{upstream_ref}"
        ).strip()

        if not children_output:
            return result

        for line in children_output.split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) > 2:  # More than 1 child
                commit_sha = parts[0]
                child_count = len(parts) - 1
                result[commit_sha] = child_count

    except Exception as e:
        log.warning(f"Failed to batch detect branching points: {e}")

    return result


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

    Uses lazy loading for commits - only commit SHAs are stored initially,
    and full Commit objects are created only when needed. This significantly
    improves performance when there are many unmerged commits.

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
    # NOTE: We only store SHAs here for lazy loading - full Commit objects
    # are loaded on-demand via ForkStatus.unmerged_commits property
    try:
        behind_output = repo.git.rev_list(f"{fork_ref}..{upstream_ref}").strip()
        unmerged_commit_shas = [sha for sha in behind_output.split("\n") if sha] if behind_output else []
    except Exception:
        unmerged_commit_shas = []

    # Find last merged commit (most recent commit in upstream that's also in fork)
    # This is essentially the merge base
    last_merged_commit = merge_base_commit

    # First and last unmerged commits - these we load eagerly since they're
    # needed for the status summary display
    # unmerged_commit_shas are in reverse chronological order (newest first)
    first_unmerged_commit = repo.commit(unmerged_commit_shas[-1]) if unmerged_commit_shas else None
    last_unmerged_commit = repo.commit(unmerged_commit_shas[0]) if unmerged_commit_shas else None

    # Get divergence stats using git diff --stat
    files_affected = 0
    total_additions = 0
    total_deletions = 0

    if unmerged_commit_shas:
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
        repo=repo,
        fork_ref=fork_ref,
        upstream_ref=upstream_ref,
        commits_behind=len(unmerged_commit_shas),
        last_merged_commit=last_merged_commit,
        first_unmerged_commit=first_unmerged_commit,
        last_unmerged_commit=last_unmerged_commit,
        merge_base_commit=merge_base_commit,
        unmerged_commit_shas=unmerged_commit_shas,
        files_affected=files_affected,
        total_additions=total_additions,
        total_deletions=total_deletions,
    )
