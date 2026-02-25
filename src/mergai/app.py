import contextlib
import json
import logging
import tempfile
import textwrap

import click
import git
import github
from github import Repository as GithubRepository

from .agent_executor import AgentExecutionError, AgentExecutor
from .agents.base import Agent
from .agents.factory import create_agent
from .config import MergaiConfig
from .context_builder import ContextBuilder
from .models import (
    MergaiNote,
)
from .prompt_builder import PromptBuilder
from .utils import git_utils, util
from .utils.branch_name_builder import BranchNameBuilder
from .utils.pr_title_builder import PRTitleBuilder
from .utils.state_store import StateStore

log = logging.getLogger(__name__)


class AppContext:
    def __init__(self, config: MergaiConfig | None = None):
        self.config: MergaiConfig = config if config is not None else MergaiConfig()
        self.repo: git.Repo = git.Repo(".")
        working_dir = self.repo.working_tree_dir
        self.state: StateStore = StateStore(str(working_dir) if working_dir else ".")
        self.gh_repo_str: str | None = None
        gh_token = util.gh_auth_token()
        self.gh = github.Github(gh_token) if gh_token else None
        self._note: MergaiNote | None = None

    def _require_note(self) -> MergaiNote:
        """Load note and raise ClickException if missing.

        Returns:
            MergaiNote instance with repo bound.

        Raises:
            click.ClickException: If no note found.
        """
        if self._note is not None:
            return self._note

        if self.state.note_exists():
            note_dict = self.state.load_note()
            self._note = MergaiNote.from_dict(note_dict, self.repo)

        if not self._note:
            raise click.ClickException(
                "No note found. Run 'mergai context init' first."
            )

        return self._note

    @property
    def has_note(self) -> bool:
        """Check if a note file exists."""
        return self.state.note_exists()

    @property
    def note(self) -> MergaiNote:
        """Get the current note, loading it if necessary.

        Returns:
            MergaiNote instance with repo bound.

        Raises:
            click.ClickException: If no note found.
        """
        return self._require_note()

    def load_note(self) -> MergaiNote | None:
        """Load note from storage without caching.

        Returns:
            MergaiNote instance or None if not found.
        """
        if not self.state.note_exists():
            return None
        note_dict = self.state.load_note()
        return MergaiNote.from_dict(note_dict, self.repo)

    def save_note(self, note: MergaiNote):
        """Save a MergaiNote to storage.

        Args:
            note: MergaiNote instance to save.
        """
        self.state.save_note(note.to_dict())
        # Update cached note
        self._note = note

    @property
    def gh_repo(self) -> GithubRepository.Repository:
        if self.gh is None:
            raise Exception(
                "GitHub token not found. Please set GITHUB_TOKEN or GH_TOKEN."
            )
        if self.gh_repo_str is None:
            raise Exception("GitHub repository not set. Please provide --repo option.")
        return self.gh.get_repo(self.gh_repo_str)

    @property
    def branches(self) -> BranchNameBuilder:
        try:
            return BranchNameBuilder.from_config(
                self.config.branch,
                self.note.merge_info,
            )
        except ValueError as e:
            raise click.ClickException(
                f"Invalid branch name format in config: {e}\n\n"
                f"The format string must contain:\n"
                f"  - %(target_branch)\n"
                f"  - Either %(merge_commit_sha) or %(merge_commit_short_sha)\n"
                f"  - Either %(target_branch_sha) or %(target_branch_short_sha)\n\n"
                f"Current format: {self.config.branch.name_format}"
            ) from e

    @property
    def commit_footer(self) -> str:
        """Get the commit footer from config.

        Returns:
            The footer text to append to MergAI-generated commits.
        """
        return self.config.commit.footer

    @property
    def pr_titles(self) -> PRTitleBuilder:
        """Get a PRTitleBuilder instance for generating PR titles.

        Returns:
            PRTitleBuilder configured with current config and merge_info.

        Raises:
            click.ClickException: If no note found.
        """
        return PRTitleBuilder.from_config(self.config.pr, self.note.merge_info)

    def get_note_from_commit(self, commit: str) -> dict | None:
        try:
            return git_utils.get_note_from_commit_as_dict(self.repo, "mergai", commit)
        except Exception as e:
            raise click.ClickException(
                f"Failed to get note for commit {commit}: {e}"
            ) from e

    def try_get_note_from_commit(self, commit: str) -> MergaiNote | None:
        try:
            note_dict = git_utils.get_note_from_commit_as_dict(
                self.repo, "mergai", commit
            )
            if note_dict is None:
                return None
            return MergaiNote.from_dict(note_dict, self.repo)
        except Exception:
            return None

    @property
    def prompt_builder(self) -> PromptBuilder:
        """Get a PromptBuilder instance for the current note.

        Returns:
            PromptBuilder configured with current note and prompt config.

        Raises:
            click.ClickException: If no note found.
        """
        return PromptBuilder(self.note, self.config.prompt)

    @property
    def context_builder(self) -> ContextBuilder:
        """Get a ContextBuilder instance for creating contexts.

        Returns:
            ContextBuilder configured with repo, note's merge_info, and important_files.

        Raises:
            click.ClickException: If no note found.
        """
        return ContextBuilder(
            repo=self.repo,
            merge_info=self.note.merge_info,
            important_files=self.config.important_files,
        )

    def drop_all(self):
        """Drop the entire note."""
        self.state.remove_note()
        self._note = None

    def get_agent(self, agent_desc: str | None = None, yolo: bool = False) -> "Agent":
        """Get an agent instance for conflict resolution.

        Args:
            agent_desc: Agent descriptor (e.g., "gemini-cli", "opencode:model").
                       If None, uses the value from config.resolve.agent.
            yolo: Enable YOLO mode.

        Returns:
            An Agent instance configured with the specified settings.
        """
        # Use config default if not explicitly provided
        if agent_desc is None:
            agent_desc = self.config.resolve.agent

        agent_type = agent_desc.split(":")[0]
        model = agent_desc.split(":")[1] if ":" in agent_desc else ""

        return create_agent(agent_type, model, yolo=yolo)

    def resolve(self, force: bool, yolo: bool):
        """Run the AI agent to resolve conflicts.

        Args:
            force: If True, overwrite existing uncommitted solution.
            yolo: Enable YOLO mode for the agent.

        Raises:
            Exception: If no note found or agent fails.
        """
        # Check if there's an uncommitted solution
        uncommitted = self.note.get_uncommitted_solution()
        if uncommitted is not None and not force:
            raise Exception(
                "An uncommitted solution already exists in the note. Use -f/--force to overwrite."
            )

        # If force and there's an uncommitted solution, we'll replace it
        # Otherwise, we'll append a new solution

        prompt = self.prompt_builder.build_resolve_prompt()
        agent = self.get_agent(yolo=yolo)

        executor = AgentExecutor(
            agent=agent,
            state_dir=self.state.path,
            max_attempts=self.config.resolve.max_attempts,
            repo=self.repo,
        )

        try:
            solution = executor.run_with_retry(
                prompt=prompt,
                validator=executor.validate_solution_files,
            )
        except AgentExecutionError as e:
            raise Exception(str(e)) from e

        # Add solution to solutions array
        if uncommitted is not None and force:
            # Replace the uncommitted solution
            uncommitted_idx, _ = uncommitted
            self.note.set_solution_at(uncommitted_idx, solution)
        else:
            # Append new solution
            self.note.add_solution(solution)

        self.save_note(self.note)

    def describe(self, force: bool, max_attempts: int | None = None):
        """Generate a description of the merge using an AI agent.

        This method runs an agent to analyze the merge context and generate
        a description without modifying any files. The description is stored
        in the note as 'merge_description'.

        Args:
            force: If True, overwrite existing merge_description.
            max_attempts: Maximum number of retry attempts on validation failure.

        Raises:
            Exception: If no note found, merge_description exists (without force),
                      or agent fails to produce a valid description.
        """
        if max_attempts is None:
            max_attempts = self.config.resolve.max_attempts

        if self.note.has_merge_description and not force:
            raise Exception(
                "Merge description already exists in the note. Use -f/--force to overwrite."
            )

        if self.note.has_merge_description:
            self.note.drop_merge_description()

        prompt = self.prompt_builder.build_describe_prompt()

        # No YOLO mode for describe - we don't want file modifications
        agent = self.get_agent(yolo=False)

        # Check repo state before running agent
        was_dirty_before = self.repo.is_dirty(untracked_files=True)

        executor = AgentExecutor(
            agent=agent,
            state_dir=self.state.path,
            max_attempts=max_attempts,
            repo=self.repo,
        )

        try:
            description = executor.run_with_retry(
                prompt=prompt,
                validator=executor.create_describe_validator(was_dirty_before),
            )
        except AgentExecutionError as e:
            raise Exception(str(e)) from e

        self.note.set_merge_description(description)
        self.save_note(self.note)

    def add_note(self, commit: str):
        self.repo.git.notes(
            "--ref", "mergai", "add", "-f", "-F", self.state.note_path(), commit
        )
        self.repo.git.notes(
            "--ref",
            "mergai-marker",
            "add",
            "-f",
            "-m",
            self.config.config.git.notes.marker_text,
            commit,
        )

    def add_selective_note(self, commit: str, fields: list[str]):
        """Add a git note containing only merge_info and specified fields.

        Creates a git note attached to the specified commit containing merge_info
        (if available) plus the specified fields from the current note. Also updates
        note_index in note.json to track which commits have which fields attached.

        For solutions, use field format "solutions[N]" where N is the solution index.
        The git note will contain only that single solution (not the entire array).

        Args:
            commit: The commit SHA to attach the note to
            fields: List of field names to include in the note (in addition to merge_info).
                    For solutions, use "solutions[N]" format.
        """
        import os
        import re

        note_dict = self.note.to_dict()

        # Build selective note content - always include merge_info and mergai_version
        selective_note = {
            "merge_info": note_dict["merge_info"],
            "mergai_version": note_dict["mergai_version"],
        }

        for field in fields:
            # Handle solutions[N] format
            match = re.match(r"solutions\[(\d+)\]", field)
            if match:
                idx = int(match.group(1))
                if (
                    self.note.has_solutions
                    and self.note.solutions is not None
                    and idx < len(self.note.solutions)
                ):
                    # Store as "solutions" array with single element
                    selective_note["solutions"] = [self.note.solutions[idx]]
            elif field == "conflict_context" and self.note.has_conflict_context:
                selective_note["conflict_context"] = note_dict["conflict_context"]
            elif field == "merge_context" and self.note.has_merge_context:
                selective_note["merge_context"] = note_dict["merge_context"]
            elif field == "pr_comments" and self.note.has_pr_comments:
                selective_note["pr_comments"] = self.note.pr_comments
            elif field == "user_comment" and self.note.has_user_comment:
                selective_note["user_comment"] = self.note.user_comment
            elif field == "merge_description" and self.note.has_merge_description:
                selective_note["merge_description"] = self.note.merge_description

        # Write selective note to temp file and attach as git note
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(selective_note, f, indent=2)
            temp_path = f.name

        try:
            self.repo.git.notes("--ref", "mergai", "add", "-f", "-F", temp_path, commit)
            self.repo.git.notes(
                "--ref",
                "mergai-marker",
                "add",
                "-f",
                "-m",
                self.config.config.git.notes.marker_text,
                commit,
            )
        finally:
            os.unlink(temp_path)

        # Update note_index in note
        self.note.add_note_index_entry(commit, fields)
        self.save_note(self.note)

    def commit_solution(self):
        """Commit the current solution to the repository."""
        # Find the uncommitted solution
        uncommitted = self.note.get_uncommitted_solution()
        if uncommitted is None:
            if not self.note.has_solutions:
                raise Exception("No solution found in the note.")
            else:
                raise Exception("All solutions have already been committed.")

        uncommitted_idx, solution = uncommitted

        if not self.repo.is_dirty():
            raise Exception("No changes to commit in the repository.")

        # Track modified files (files changed but not in the solution's resolved list)
        modified_files = []
        if self.repo.is_dirty():
            for item in self.repo.index.diff(None):
                if item.a_path not in solution["response"]["resolved"]:
                    message = f"Unstaged changes found in file {item.a_path}, which is not in the solution."
                    message += "\n\n"
                    message += "This use case is not supported yet."
                    message += (
                        "Please stage only the files in the solution and try again."
                    )
                    raise Exception(message)

                self.repo.index.add([item.a_path])

        # Build commit message following Git's merge commit style
        target_branch = self.branches.target_branch
        if self.note.has_conflict_context:
            theirs_sha = git_utils.short_sha(
                self.note.conflict_context.theirs_commit_sha
            )
            message = f"Resolve conflicts for merge commit '{theirs_sha}' into {target_branch}\n\n"
        else:
            message = "Resolve conflicts for merge\n\n"

        # Add summary from AI response
        summary = solution["response"].get("summary", "")
        if summary:
            wrapped_summary = textwrap.fill(summary, width=72)
            message += f"{wrapped_summary}\n\n"

        # Add resolved files section
        resolved_files = solution["response"].get("resolved", {})
        if resolved_files:
            message += "Resolved:\n"
            for file_path in resolved_files:
                message += f"\t{file_path}\n"
            message += "\n"

        # Add unresolved files section (with conflict markers warning)
        unresolved_files = solution["response"].get("unresolved", {})
        if unresolved_files:
            # Mark conflict markers as unresolved and stage the files
            for file_path in unresolved_files:
                git_utils.mark_conflict_markers_unresolved(file_path)
                self.repo.index.add([file_path])

            message += "Unresolved (contains conflict markers):\n"
            for file_path in unresolved_files:
                message += f"\t{file_path}\n"
            message += "\n"

        # Add modified files section (files not in the solution)
        if modified_files:
            message += "Modified:\n"
            for file_path in modified_files:
                message += f"\t{file_path}\n"
            message += "\n"

        # Add MergAI footer
        message += self.commit_footer

        self.repo.index.commit(message)

        # Add note with the specific solution index
        self.add_selective_note(
            self.repo.head.commit.hexsha, [f"solutions[{uncommitted_idx}]"]
        )

    def commit_conflict(self):
        """Commit conflict context to the repository."""
        if not self.note.has_conflict_context:
            raise Exception(
                "No conflict context found in the note.\n\n"
                "Please prepare the conflict context by running:\nmergai context create conflict"
            )

        conflict_ctx = self.note.conflict_context
        theirs_sha = git_utils.short_sha(conflict_ctx.theirs_commit_sha)
        files = conflict_ctx.files
        target_branch = self.branches.target_branch

        # Build commit message following Git's merge commit style
        message = f"Merge commit '{theirs_sha}' into {target_branch}\n\n"

        # Add conflicts section with tab-prefixed file list
        message += "Conflicts:\n"
        for file_path in files:
            message += f"\t{file_path}\n"
        message += "\n"

        # Add note about conflict markers
        message += "Note: This commit contains unresolved conflict markers.\n\n"

        # Add MergAI footer
        message += self.commit_footer

        for path in files:
            self.repo.git.add(path)

        self.repo.git.commit(
            "-m",
            message,
        )

        fields = ["conflict_context", "merge_context"]
        if self.note.has_merge_description:
            fields.append("merge_description")
        self.add_selective_note(self.repo.head.commit.hexsha, fields)

    def commit_merge(self):
        """Commit the current staged changes as a merge commit.

        Creates a commit with the message "Merge commit '<short sha>'" where
        the short sha comes from merge_info.

        Raises:
            Exception: If no note found.
        """
        # Build commit message (use short SHA for display)
        message = f"Merge commit '{git_utils.short_sha(self.note.merge_info.merge_commit_sha)}'\n\n"

        # Add MergAI footer
        message += self.commit_footer

        self.repo.git.commit("-m", message)

        fields = ["merge_context"]
        if self.note.has_merge_description:
            fields.append("merge_description")
        self.add_selective_note(self.repo.head.commit.hexsha, fields)

    def _collect_commits_for_squash(
        self, target_sha: str
    ) -> list[tuple[git.Commit, dict | None]]:
        """Collect commits from HEAD to target_sha with their notes.

        Iterates from HEAD backwards until reaching target_sha (exclusive),
        collecting each commit and its mergai git note (if any).

        Args:
            target_sha: The SHA to stop at (not included in results).

        Returns:
            List of (commit, note) tuples, ordered from oldest to newest.
            The note is None if the commit has no mergai note.

        Raises:
            click.ClickException: If target_sha is not an ancestor of HEAD.
        """
        commits_with_notes = []
        target_sha_full = self.repo.commit(target_sha).hexsha

        for commit in self.repo.iter_commits():
            if commit.hexsha == target_sha_full:
                break
            git_note = self.get_note_from_commit(commit.hexsha)
            commits_with_notes.append((commit, git_note))
        else:
            # We didn't find target_sha in the history
            raise click.ClickException(
                f"Target commit {target_sha[:11]} is not an ancestor of HEAD. "
                "Cannot determine commits to squash."
            )

        # Reverse to get oldest-first order
        commits_with_notes.reverse()
        return commits_with_notes

    def _build_squash_commit_message(
        self,
        merge_info: dict,
        commits_with_notes: list[tuple[git.Commit, dict | None]],
        combined_note: dict,
    ) -> str:
        """Build the commit message for the squashed merge commit.

        Creates a message in the format:
            Merge commit '<short sha>' into <target_branch>

            Conflicts:
                <conflicted_file1>
                <conflicted_file2>

            Modified:
                <modified_file1>
                <modified_file2>

            Squashed commits:
                <sha1> <message1>
                <sha2> <message2>

            Note: commit created by mergai

        Args:
            merge_info: The merge_info dict containing target_branch and merge_commit.
            commits_with_notes: List of (commit, note) tuples for reference.
            combined_note: The combined note containing solutions and conflict_context.

        Returns:
            The formatted commit message string.
        """
        target_branch = merge_info["target_branch"]
        merge_commit = merge_info["merge_commit"]

        # Header
        message = f"Merge commit '{git_utils.short_sha(merge_commit)}' into {target_branch}\n\n"

        # Get conflicted files from conflict_context
        conflict_context = combined_note.get("conflict_context", {})
        conflict_files = set(conflict_context.get("files", []))

        # Collect modified files ONLY from solution commits (not from merge-related commits)
        # Solution commits are identified by having "solutions" in their note
        # or by NOT having "conflict_context" or "merge_context"
        # (merge-related commits have conflict_context and/or merge_context)
        solution_modified_files = set()
        for commit, note in commits_with_notes:
            # Skip if this commit has conflict_context or merge_context
            # (these are mergai-managed commits, not solution commits)
            if note and ("conflict_context" in note or "merge_context" in note):
                continue
            # Include files from commits that have a solution or no note at all
            # (PR merge commits may not have mergai notes)
            modified = git_utils.get_commit_modified_files(self.repo, commit)
            solution_modified_files.update(modified)

        # Modified files are those changed by solution commits but not in the conflict list
        modified_files = solution_modified_files - conflict_files

        # Conflicts section
        if conflict_files:
            message += "Conflicts:\n"
            for file_path in sorted(conflict_files):
                message += f"\t{file_path}\n"
            message += "\n"

        # Modified section (files changed by solutions that weren't in conflict)
        if modified_files:
            message += "Modified:\n"
            for file_path in sorted(modified_files):
                message += f"\t{file_path}\n"
            message += "\n"

        # Include original commit messages as reference
        message += "Squashed commits:\n"
        for commit, _ in commits_with_notes:
            short_sha = git_utils.short_sha(commit.hexsha)
            # Get first line of commit message
            commit_message = (
                commit.message
                if isinstance(commit.message, str)
                else commit.message.decode("utf-8", errors="replace")
            )
            first_line = commit_message.split("\n")[0].strip()
            message += f"\t{short_sha} {first_line}\n"
        message += "\n"

        # Add MergAI footer
        message += self.commit_footer

        return message

    def squash_to_merge(self):
        """Squash all commits from HEAD to merge commit into a single merge commit.

        This method:
        1. Validates merge_info exists and we have commits to squash
        2. Collects all commits and their notes from HEAD to target_branch_sha
        3. Builds a combined note merging all individual notes
        4. Creates a new merge commit with two parents (target_branch_sha, merge_commit)
        5. Attaches the combined note to the new commit

        Raises:
            click.ClickException: If prerequisites are not met or operation fails.
        """
        # Check for uncommitted changes
        if self.repo.is_dirty(untracked_files=False):
            raise click.ClickException(
                "Working directory has uncommitted changes. "
                "Please commit or stash them before squashing."
            )

        # Load and validate merge_info
        target_branch_sha = self.note.merge_info.target_branch_sha
        merge_commit_sha = self.note.merge_info.merge_commit_sha

        if not target_branch_sha or not merge_commit_sha:
            raise click.ClickException(
                "merge_info is incomplete (missing target_branch_sha or merge_commit). "
                "Please reinitialize with 'mergai context init <commit>'."
            )

        # Check if HEAD is already at target_branch_sha (nothing to squash)
        head_sha = self.repo.head.commit.hexsha
        target_sha_full = self.repo.commit(target_branch_sha).hexsha

        if head_sha == target_sha_full:
            raise click.ClickException(
                "HEAD is already at target_branch_sha. No commits to squash."
            )

        # Collect commits to squash
        click.echo(
            f"Collecting commits from HEAD to {git_utils.short_sha(target_branch_sha)}..."
        )
        commits_with_notes = self._collect_commits_for_squash(target_branch_sha)

        if not commits_with_notes:
            raise click.ClickException("No commits found to squash.")

        click.echo(f"Found {len(commits_with_notes)} commit(s) to squash.")

        # Check if there's only 1 commit and it's already a proper merge commit
        # with the expected parents (target_branch_sha, merge_commit_sha).
        # In this case, squashing is a no-op - the commit is already in the desired state.
        if len(commits_with_notes) == 1:
            head_commit = self.repo.head.commit
            if (
                len(head_commit.parents) == 2
                and head_commit.parents[0].hexsha == target_sha_full
                and head_commit.parents[1].hexsha == merge_commit_sha
            ):
                click.echo(
                    "HEAD is already a merge commit with the expected structure. "
                    "Nothing to squash."
                )
                return

        # Soft reset to target_branch_sha to stage all changes
        click.echo(f"Resetting to {git_utils.short_sha(target_branch_sha)}...")
        self.repo.git.reset("--soft", target_branch_sha)

        # Create the merge commit with two parents using git commit-tree
        # First, write the current index as a tree
        tree_sha = self.repo.git.write_tree()

        # Build combined note from all commits
        combined_note = MergaiNote.combine_from_dicts(commits_with_notes, self.repo)

        # Ensure merge_info is present (in case git notes didn't have it)
        if combined_note.merge_info is None:
            combined_note.merge_info = self.note.merge_info

        # Build commit message
        merge_info_dict = combined_note.merge_info.to_dict()
        message = self._build_squash_commit_message(
            merge_info_dict, commits_with_notes, combined_note.to_dict()
        )

        # Create commit with two parents: target_branch_sha (first parent) and merge_commit_sha (second parent)
        click.echo("Creating squashed merge commit...")
        new_commit_sha = self.repo.git.commit_tree(
            tree_sha, "-p", target_sha_full, "-p", merge_commit_sha, "-m", message
        )

        # Update HEAD to point to the new commit
        self.repo.git.reset("--hard", new_commit_sha)

        # Set note_index to reference all fields to the new squashed commit
        combined_note.set_note_index_for_all_fields(new_commit_sha)

        # Save combined note to local state
        self.save_note(combined_note)

        # Attach note to the new commit
        click.echo("Attaching combined note to squashed commit...")
        self.add_note(new_commit_sha)

        click.echo(
            f"Successfully squashed {len(commits_with_notes)} commit(s) into {git_utils.short_sha(new_commit_sha)}"
        )

    def finalize(self, mode: str | None = None):
        """Finalize the solution PR based on configured mode.

        This method is typically called after a solution PR is merged into the
        conflict branch. It handles post-merge processing according to the
        configured finalize mode.

        Args:
            mode: Override mode from config. Options: 'squash', 'keep', 'fast-forward'.
                  If None, uses config.finalize.mode.

        Raises:
            click.ClickException: If mode is invalid or operation fails.
        """
        effective_mode = mode or self.config.finalize.mode

        if effective_mode == "squash":
            self.squash_to_merge()
        elif effective_mode == "keep":
            self._finalize_keep()
        elif effective_mode == "fast-forward":
            self._finalize_fast_forward()
        else:
            raise click.ClickException(
                f"Invalid finalize mode: '{effective_mode}'. "
                "Valid options are 'squash', 'keep', or 'fast-forward'."
            )

    def _finalize_keep(self):
        """Finalize in 'keep' mode - validate state and print summary.

        This mode validates the repository state and prints a summary of
        commits without modifying any commits. Useful when you want to
        preserve the individual commit history from the solution PR.

        Raises:
            click.ClickException: If merge_info is incomplete or invalid.
        """
        # Validate merge_info exists
        target_branch_sha = self.note.merge_info.target_branch_sha
        merge_commit_sha = self.note.merge_info.merge_commit_sha

        if not target_branch_sha or not merge_commit_sha:
            raise click.ClickException(
                "merge_info is incomplete (missing target_branch_sha or merge_commit). "
                "Please reinitialize with 'mergai context init <commit>'."
            )

        # Check if HEAD is already at target_branch_sha (nothing to finalize)
        head_sha = self.repo.head.commit.hexsha
        target_sha_full = self.repo.commit(target_branch_sha).hexsha

        if head_sha == target_sha_full:
            raise click.ClickException(
                "HEAD is already at target_branch_sha. No commits to finalize."
            )

        # Collect commits for summary
        commits_with_notes = self._collect_commits_for_squash(target_branch_sha)

        if not commits_with_notes:
            raise click.ClickException("No commits found to finalize.")

        # Print validation summary
        click.echo("Finalize mode: keep (commits preserved)")
        click.echo(f"Commits from HEAD to {git_utils.short_sha(target_branch_sha)}:\n")
        for commit, note in commits_with_notes:
            short_sha = git_utils.short_sha(commit.hexsha)
            first_line = commit.message.split("\n")[0].strip()
            click.echo(f"  {short_sha} {first_line}")
            if note:
                fields = self._get_note_fields_summary(note)
                click.echo(f"              note: {fields}")
            else:
                click.echo("              note: (none)")

        click.echo(f"\nTotal: {len(commits_with_notes)} commit(s)")
        click.echo("No changes made (keep mode).")

    def _get_note_fields_summary(self, note: dict) -> str:
        """Get a summary of fields present in a note dict.

        Args:
            note: The note dictionary from a git note.

        Returns:
            A comma-separated string of field names present in the note.
            For solutions, shows count if multiple (e.g., "solutions(2)").
        """
        fields = []

        # Check for each possible field
        if "merge_info" in note:
            fields.append("merge_info")
        if "conflict_context" in note:
            fields.append("conflict_context")
        if "merge_context" in note:
            fields.append("merge_context")
        if "solutions" in note:
            count = len(note["solutions"])
            if count == 1:
                fields.append("solution")
            else:
                fields.append(f"solutions({count})")
        if "merge_description" in note:
            fields.append("merge_description")
        if "pr_comments" in note:
            fields.append("pr_comments")
        if "user_comment" in note:
            fields.append("user_comment")

        return ", ".join(fields) if fields else "(empty)"

    def _finalize_fast_forward(self):
        """Finalize in 'fast-forward' mode - remove PR merge commit.

        This mode removes the GitHub PR merge commit to simulate a fast-forward
        merge. It keeps the original solution commits with their mergai notes
        intact.

        GitHub PR merge commits have two parents:
        - First parent: The base branch (conflict branch) HEAD before merge
        - Second parent: The head branch (solution branch) with the actual commits

        We want to reset to the second parent (solution branch), which contains
        the linear history of conflict commit -> solution commit(s).

        The operation only proceeds if:
        1. Working directory is clean (no uncommitted changes)
        2. HEAD is a merge commit (has 2 parents)
        3. HEAD has no mergai note attached
        4. The second parent of HEAD has a mergai note attached

        If HEAD already has a note or is not a merge commit, the command
        assumes the PR was merged with fast-forward and does nothing.

        Raises:
            click.ClickException: If working directory is dirty or validation fails.
        """
        # Check for uncommitted changes
        if self.repo.is_dirty(untracked_files=False):
            raise click.ClickException(
                "Working directory has uncommitted changes. "
                "Please commit or stash them before finalizing."
            )

        head_commit = self.repo.head.commit
        head_sha = head_commit.hexsha
        short_head_sha = git_utils.short_sha(head_sha)

        # Check if HEAD has a mergai note
        head_note = self.try_get_note_from_commit(head_sha)
        if head_note is not None:
            click.echo(f"HEAD ({short_head_sha}) already has a mergai note attached.")
            click.echo("Assuming PR was merged with fast-forward. Nothing to do.")
            return

        # Check if HEAD is a merge commit (has 2 parents)
        if len(head_commit.parents) < 2:
            click.echo(f"HEAD ({short_head_sha}) is not a merge commit.")
            click.echo("Assuming PR was merged with fast-forward. Nothing to do.")
            return

        # HEAD is a merge commit without a note - this is likely the GitHub PR merge commit
        # Second parent is the solution branch (the one being merged in)
        second_parent = head_commit.parents[1]
        second_parent_sha = second_parent.hexsha
        short_second_parent_sha = git_utils.short_sha(second_parent_sha)

        # Check if the second parent has a mergai note
        second_parent_note = self.try_get_note_from_commit(second_parent_sha)
        if second_parent_note is None:
            raise click.ClickException(
                f"Second parent ({short_second_parent_sha}) has no mergai note attached.\n"
                "Cannot safely remove the merge commit. Expected the solution branch "
                "tip to have a note.\n\n"
                "This might indicate an unexpected repository state. Please verify manually."
            )

        # All checks passed - reset to second parent (solution branch tip)
        first_line = head_commit.message.split("\n")[0].strip()
        click.echo(f"Removing PR merge commit: {short_head_sha} {first_line}")
        click.echo(f"Resetting HEAD to: {short_second_parent_sha}")

        self.repo.git.reset("--hard", second_parent_sha)

        click.echo("\nSuccessfully removed merge commit.")
        click.echo(f"HEAD is now at {short_second_parent_sha}")

    def rebuild_note_from_commits(self) -> MergaiNote:
        """Rebuild note.json from git commit notes.

        Scans commits from HEAD backwards to find and collect all mergai notes,
        reconstructing the local note.json with:
        - merge_info (from the first commit that has it)
        - conflict_context (at most one)
        - merge_context (at most one)
        - solutions array (collected from all solution commits)
        - note_index (rebuilt based on which commit has what)

        The scan range is determined by:
        1. If on a mergai branch, use parsed target_branch_sha as the stop point
        2. Otherwise, scan until finding a commit with merge_info and use its
           target_branch_sha as the stop point

        Returns:
            The rebuilt MergaiNote.

        Raises:
            click.ClickException: If merge_info is inconsistent across commits
                                  or cannot determine scan range.
        """
        current_branch = git_utils.get_current_branch(self.repo)
        parsed = BranchNameBuilder.parse_branch_name_with_config(
            current_branch, self.config.branch
        )

        target_sha = None
        if parsed:
            # Mergai branch - we know the target SHA
            with contextlib.suppress(Exception):
                target_sha = self.repo.commit(parsed.target_branch_sha).hexsha

        # Collect commits and their notes
        commits_with_notes = []
        for commit in self.repo.iter_commits():
            # Stop BEFORE processing the target commit - its note belongs to a
            # previous merge operation and should not be included
            if target_sha and commit.hexsha == target_sha:
                break

            git_note = self.get_note_from_commit(commit.hexsha)
            if git_note:
                commits_with_notes.append((commit.hexsha, git_note))

                # If we don't have target_sha yet, try to get it from merge_info
                if target_sha is None and "merge_info" in git_note:
                    target_branch_sha = git_note["merge_info"].get("target_branch_sha")
                    if target_branch_sha:
                        target_sha = target_branch_sha

        if not commits_with_notes:
            raise click.ClickException(
                "No commits with mergai notes found. Cannot rebuild note.json."
            )

        # Reverse to process oldest first (for consistent indexing)
        commits_with_notes.reverse()

        # Build the new note as a dict first
        note_dict: dict = {
            "solutions": [],
            "note_index": [],
        }

        reference_merge_info = None

        for commit_sha, git_note in commits_with_notes:
            fields_for_this_commit = []

            # Handle mergai_version - take from first note that has it
            if "mergai_version" in git_note and "mergai_version" not in note_dict:
                note_dict["mergai_version"] = git_note["mergai_version"]

            # Handle merge_info
            if "merge_info" in git_note:
                if reference_merge_info is None:
                    reference_merge_info = git_note["merge_info"]
                    note_dict["merge_info"] = git_note["merge_info"]
                else:
                    # Check consistency
                    current_mi = git_note["merge_info"]
                    if current_mi.get("target_branch") != reference_merge_info.get(
                        "target_branch"
                    ) or current_mi.get("merge_commit") != reference_merge_info.get(
                        "merge_commit"
                    ):
                        raise click.ClickException(
                            f"Inconsistent merge_info found across commits.\n"
                            f"Reference (from earlier commit): target_branch={reference_merge_info.get('target_branch')}, "
                            f"merge_commit={reference_merge_info.get('merge_commit')}\n"
                            f"Current (commit {commit_sha[:11]}): target_branch={current_mi.get('target_branch')}, "
                            f"merge_commit={current_mi.get('merge_commit')}\n\n"
                            f"Please use 'mergai context init <commit> --target <branch>' to explicitly set merge_info."
                        )

            # Handle conflict_context (at most one)
            if "conflict_context" in git_note and "conflict_context" not in note_dict:
                note_dict["conflict_context"] = git_note["conflict_context"]
                fields_for_this_commit.append("conflict_context")

            # Handle merge_context (at most one)
            if "merge_context" in git_note and "merge_context" not in note_dict:
                note_dict["merge_context"] = git_note["merge_context"]
                fields_for_this_commit.append("merge_context")

            # Handle solution (singular in git note -> add to solutions array)
            # Handle solutions array
            if "solutions" in git_note:
                for solution in git_note["solutions"]:
                    idx = len(note_dict["solutions"])
                    note_dict["solutions"].append(solution)
                    fields_for_this_commit.append(f"solutions[{idx}]")

            # Handle merge_description
            if "merge_description" in git_note and "merge_description" not in note_dict:
                note_dict["merge_description"] = git_note["merge_description"]
                fields_for_this_commit.append("merge_description")

            # Handle pr_comments
            if "pr_comments" in git_note and "pr_comments" not in note_dict:
                note_dict["pr_comments"] = git_note["pr_comments"]
                fields_for_this_commit.append("pr_comments")

            # Handle user_comment
            if "user_comment" in git_note and "user_comment" not in note_dict:
                note_dict["user_comment"] = git_note["user_comment"]
                fields_for_this_commit.append("user_comment")

            # Add to note_index if we collected any fields
            if fields_for_this_commit:
                note_dict["note_index"].append(
                    {
                        "sha": commit_sha,
                        "fields": fields_for_this_commit,
                    }
                )

        # Clean up empty collections
        if not note_dict["solutions"]:
            del note_dict["solutions"]
        if not note_dict["note_index"]:
            del note_dict["note_index"]

        # Set mergai_version to current version if not found in any git note
        if "mergai_version" not in note_dict:
            from .version import __version__

            note_dict["mergai_version"] = __version__

        # Convert to MergaiNote
        return MergaiNote.from_dict(note_dict, self.repo)

    def _is_pr_merge_commit(self, commit: git.Commit) -> bool:
        """Check if a commit is a GitHub PR merge commit.

        A PR merge commit is identified by:
        1. It's a merge commit (2 parents)
        2. The first parent is on the conflict branch lineage (has conflict_context)
        3. The second parent is on the solution branch (the branch being merged in)

        This detects commits like "Merge pull request #123 from user/solution-branch"
        which should not be treated as human solution commits.

        Args:
            commit: The commit to check.

        Returns:
            True if this is a PR merge commit, False otherwise.
        """
        if len(commit.parents) != 2:
            return False

        # First parent should be on the conflict branch - check if it has conflict_context
        # or if it's an ancestor of the conflict branch tip
        first_parent = commit.parents[0]
        first_parent_note = self.get_note_from_commit(first_parent.hexsha)

        # If first parent has conflict_context, this is likely a PR merge into conflict branch
        if first_parent_note is not None and "conflict_context" in first_parent_note:
            return True

        # Also check if first parent is a previous PR merge commit (chain of PR merges)
        # by checking if it's a merge commit whose first parent has conflict_context
        if len(first_parent.parents) == 2:
            grandparent_note = self.get_note_from_commit(first_parent.parents[0].hexsha)
            if grandparent_note is not None and "conflict_context" in grandparent_note:
                return True

        return False

    def get_unsynced_commits(
        self, force: bool = False
    ) -> list[tuple[git.Commit, bool]]:
        """Get commits that need syncing (don't have solutions attached).

        Iterates from HEAD backwards to target_branch_sha and identifies
        commits that:
        1. Don't have a mergai git note with a solution, OR
        2. Have a note but force=True (for re-syncing)

        Skips:
        - Commits that already have solutions (unless force=True)
        - Commits with conflict_context or merge_context (mergai-managed commits)
        - PR merge commits (merge commits that merge solution branch into conflict branch)

        Args:
            force: If True, include commits that already have notes.

        Returns:
            List of (commit, has_existing_note) tuples, ordered oldest to newest.
        """
        target_branch_sha = self.note.merge_info.target_branch_sha
        target_sha_full = self.repo.commit(target_branch_sha).hexsha

        commits_to_sync = []

        for commit in self.repo.iter_commits():
            if commit.hexsha == target_sha_full:
                break

            # Check if this commit has a mergai note
            git_note = self.get_note_from_commit(commit.hexsha)

            # Check if commit already has a solution
            has_solution = git_note is not None and "solutions" in git_note

            # Check if commit has conflict_context or merge_context - these are
            # mergai-managed commits (from 'mergai commit conflict' or 'mergai commit merge'),
            # not human fixes
            has_context = git_note is not None and (
                "conflict_context" in git_note or "merge_context" in git_note
            )

            # Skip PR merge commits (merge commits that merge solution branch into conflict branch)
            # These are GitHub-generated commits like "Merge pull request #123..."
            if self._is_pr_merge_commit(commit):
                continue

            # Skip mergai-managed commits (conflict/merge context) - these should never be synced
            if has_context:
                continue

            if has_solution and not force:
                # Skip - already has a solution
                continue

            commits_to_sync.append((commit, git_note is not None))

        # Reverse to get oldest-first order
        commits_to_sync.reverse()
        return commits_to_sync

    def create_human_solution_from_commit(self, commit: git.Commit) -> dict:
        """Build a solution dict from a human commit.

        Creates a solution structure similar to AI solutions but with
        author information instead of agent_info.

        Args:
            commit: The git commit to create a solution from.

        Returns:
            Solution dict with response, author, and commit_sha fields.
        """
        # Get files modified by this commit
        modified_files = git_utils.get_commit_modified_files(self.repo, commit)

        # Determine which files were previously unresolved
        previously_unresolved: set[str] = set()
        if self.note.has_solutions and self.note.solutions is not None:
            for solution in self.note.solutions:
                unresolved = solution.get("response", {}).get("unresolved", {})
                previously_unresolved.update(unresolved.keys())

        # Also check conflict_context for original conflicting files
        conflict_files: set[str] = set()
        if self.note.has_conflict_context and self.note.conflict_context is not None:
            conflict_files = set(self.note.conflict_context.files)

        # Categorize files as resolved, unresolved, or modified
        resolved = {}
        unresolved = {}
        modified = {}

        for file_path in modified_files:
            # Check if file still has conflict markers
            has_markers = git_utils.file_has_conflict_markers(
                self.repo, commit.hexsha, file_path
            )

            if has_markers:
                unresolved[file_path] = "Contains conflict markers"
            else:
                # Determine appropriate description
                if file_path in previously_unresolved:
                    resolved[file_path] = (
                        "Resolved by human (was previously unresolved)"
                    )
                elif file_path in conflict_files:
                    resolved[file_path] = "Resolved by human"
                else:
                    # Files not in conflict list go to modified
                    modified[file_path] = "Modified by human"

        # Parse commit message
        commit_message = (
            commit.message
            if isinstance(commit.message, str)
            else commit.message.decode("utf-8", errors="replace")
        )
        message_lines = commit_message.strip().split("\n")
        summary = message_lines[0] if message_lines else "Human changes"
        review_notes = (
            "\n".join(message_lines[1:]).strip() if len(message_lines) > 1 else ""
        )

        return {
            "response": {
                "summary": summary,
                "resolved": resolved,
                "unresolved": unresolved,
                "modified": modified,
                "review_notes": review_notes if review_notes else None,
            },
            "author": {
                "name": commit.author.name,
                "email": commit.author.email,
                "type": "human",
            },
            "commit_sha": commit.hexsha,
        }

    def sync_human_commits(self, force: bool = False, dry_run: bool = False) -> int:
        """Scan commits and create solutions for human-made changes.

        Iterates through commits from HEAD to target_branch_sha and creates
        solution entries for commits that don't have solutions attached.

        Args:
            force: If True, re-sync commits that already have notes.
            dry_run: If True, only show what would be synced without making changes.

        Returns:
            Number of commits that were synced (or would be synced in dry_run mode).

        Raises:
            click.ClickException: If no note found or operation fails.
        """
        commits_to_sync = self.get_unsynced_commits(force=force)

        if not commits_to_sync:
            click.echo("No commits to sync.")
            return 0

        click.echo(f"Found {len(commits_to_sync)} commit(s) to sync.")

        if dry_run:
            click.echo("\nDry run - showing what would be synced:\n")
            for commit, has_existing_note in commits_to_sync:
                short_sha = git_utils.short_sha(commit.hexsha)
                commit_message = (
                    commit.message
                    if isinstance(commit.message, str)
                    else commit.message.decode("utf-8", errors="replace")
                )
                first_line = commit_message.split("\n")[0].strip()
                status = " (has note, will overwrite)" if has_existing_note else ""
                click.echo(f"  {short_sha} {first_line}{status}")

                # Show what files would be in the solution
                modified_files = git_utils.get_commit_modified_files(self.repo, commit)
                for file_path in modified_files:
                    has_markers = git_utils.file_has_conflict_markers(
                        self.repo, commit.hexsha, file_path
                    )
                    status = (
                        " (UNRESOLVED - has conflict markers)" if has_markers else ""
                    )
                    click.echo(f"    - {file_path}{status}")
            return len(commits_to_sync)

        # Perform the sync
        synced_count = 0
        for commit, _has_existing_note in commits_to_sync:
            short_sha = git_utils.short_sha(commit.hexsha)
            commit_message = (
                commit.message
                if isinstance(commit.message, str)
                else commit.message.decode("utf-8", errors="replace")
            )
            first_line = commit_message.split("\n")[0].strip()

            click.echo(f"Syncing {short_sha} {first_line}...")

            # Create human solution from commit
            solution = self.create_human_solution_from_commit(commit)

            # Add to solutions array
            solution_idx = self.note.add_solution(solution)

            # Save note first (so note_index gets updated properly)
            self.save_note(self.note)

            # Attach git note to the commit
            self.add_selective_note(commit.hexsha, [f"solutions[{solution_idx}]"])

            synced_count += 1

            # Show summary
            resolved_count = len(solution["response"]["resolved"])
            unresolved_count = len(solution["response"]["unresolved"])
            modified_count = len(solution["response"].get("modified", {}))
            summary_parts = [f"{resolved_count} resolved"]
            if unresolved_count > 0:
                summary_parts.append(f"{unresolved_count} unresolved")
            if modified_count > 0:
                summary_parts.append(f"{modified_count} modified")
            click.echo(
                f"  -> Added solution [{solution_idx}]: " f"{', '.join(summary_parts)}"
            )

        click.echo(f"\nSuccessfully synced {synced_count} commit(s).")
        return synced_count
