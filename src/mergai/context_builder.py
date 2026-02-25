"""Builder for creating MergeContext and ConflictContext."""

import logging

import git

from .models import ConflictContext, MergeContext, MergeInfo
from .utils import git_utils

log = logging.getLogger(__name__)


class ContextBuilder:
    """Builder for creating MergeContext and ConflictContext.

    This class is responsible for creating context objects from the current
    git repository state. It does not handle persistence - callers are
    responsible for saving the created contexts to the note.

    Attributes:
        repo: GitPython Repo instance.
        merge_info: MergeInfo with target branch and merge commit information.
        important_files: List of important files to track modifications for.
    """

    def __init__(
        self,
        repo: git.Repo,
        merge_info: MergeInfo,
        important_files: list[str] | None = None,
    ):
        """Initialize the ContextBuilder.

        Args:
            repo: GitPython Repo instance.
            merge_info: MergeInfo with target branch and merge commit information.
            important_files: Optional list of important files to track.
        """
        self.repo = repo
        self.merge_info = merge_info
        self.important_files = important_files or []

    def create_conflict_context(
        self,
        use_diffs: bool,
        diff_lines_of_context: int,
        use_compressed_diffs: bool,
        use_their_commits: bool,
    ) -> ConflictContext:
        """Create conflict context from current merge state.

        Args:
            use_diffs: Include diffs in the context.
            diff_lines_of_context: Number of context lines in diffs.
            use_compressed_diffs: Use compressed diffs.
            use_their_commits: Include their commits in context.

        Returns:
            The created ConflictContext.

        Raises:
            Exception: If no merge in progress.
        """
        context_dict = git_utils.get_conflict_context(
            self.repo,
            use_diffs=use_diffs,
            lines_of_context=diff_lines_of_context,
            use_compressed_diffs=use_compressed_diffs,
            use_their_commits=use_their_commits,
        )
        if context_dict is None:
            raise Exception("No merge in progress")

        return ConflictContext.from_dict(context_dict, self.repo)

    def create_merge_context(
        self,
        auto_merged_files: list[str] | None = None,
        merge_strategy: str | None = None,
    ) -> MergeContext:
        """Create merge context from merge_info.

        Calculates the list of commits being merged by finding the
        merge base between target_branch and merge_commit, then
        listing all commits from base..merge_commit. Also identifies
        which important files (from config) were modified.

        Args:
            auto_merged_files: List of files that were auto-merged by git.
            merge_strategy: The merge strategy used (e.g., 'ort', 'recursive').

        Returns:
            The created MergeContext.
        """
        target_branch = self.merge_info.target_branch
        merge_commit_sha = self.merge_info.merge_commit_sha

        # Resolve merge commit
        merge_commit = self.repo.commit(merge_commit_sha)
        merge_commit_hexsha = merge_commit.hexsha

        log.info(
            f"getting merged commits for merge context: target_branch={target_branch}, merge_commit={merge_commit_sha}"
        )
        # Get the list of merged commits
        merged_commits = git_utils.get_merged_commits(
            self.repo,
            target_branch,
            merge_commit_sha,
        )
        log.info(f"found {len(merged_commits)} merged commits for merge context")

        # Find which important files were modified by any of the merged commits
        important_files_modified = []
        if self.important_files:
            all_modified_files = set()
            for commit_sha in merged_commits:
                commit = self.repo.commit(commit_sha)
                modified = git_utils.get_commit_modified_files(self.repo, commit)
                all_modified_files.update(modified)

            important_files_modified = sorted(
                set(self.important_files) & all_modified_files
            )

        context_dict: dict = {
            "merge_commit": merge_commit_hexsha,
            "merged_commits": merged_commits,
            "important_files_modified": important_files_modified,
        }

        # Add auto_merged info if provided
        if auto_merged_files is not None or merge_strategy is not None:
            context_dict["auto_merged"] = {
                "strategy": merge_strategy,
                "files": auto_merged_files or [],
            }

        return MergeContext.from_dict(context_dict, self.repo)
