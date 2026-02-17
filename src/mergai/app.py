import git
import click
import logging
import textwrap
from .utils import git_utils
from .utils import util
from .agents.factory import create_agent
from .utils.state_store import StateStore
from .config import MergaiConfig
from .models import (
    ConflictContext,
    MergeContext,
    MergaiNote,
)
import github
from github import Repository as GithubRepository
import json
from typing import Optional, Tuple, List
import tempfile
from .agents.base import Agent
from .utils.branch_name_builder import BranchNameBuilder
from .utils.pr_title_builder import PRTitleBuilder
from .prompt_builder import PromptBuilder
from .context_builder import ContextBuilder
from .agent_executor import AgentExecutor, AgentExecutionError

log = logging.getLogger(__name__)


class AppContext:
    def __init__(self, config: MergaiConfig = None):
        self.config: MergaiConfig = config if config is not None else MergaiConfig()
        self.repo: git.Repo = git.Repo(".")
        self.state: StateStore = StateStore(self.repo.working_tree_dir)
        self.gh_repo_str: Optional[str] = None
        gh_token = util.gh_auth_token()
        self.gh = github.Github(gh_token) if gh_token else None
        self._note: Optional[MergaiNote] = None

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

    def load_note(self) -> Optional[MergaiNote]:
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
            )

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

    def get_note_from_commit(self, commit: str) -> Optional[dict]:
        try:
            return git_utils.get_note_from_commit_as_dict(self.repo, "mergai", commit)
        except Exception as e:
            raise click.ClickException(f"Failed to get note for commit {commit}: {e}")

    def try_get_note_from_commit(self, commit: str) -> Optional[MergaiNote]:
        try:
            note_dict = git_utils.get_note_from_commit_as_dict(
                self.repo, "mergai", commit
            )
            if note_dict is None:
                return None
            return MergaiNote.from_dict(note_dict, self.repo)
        except Exception as e:
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

    def get_agent(self, agent_desc: str = None, yolo: bool = False) -> "Agent":
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
        model = agent_desc.split(":")[1] if ":" in agent_desc else None

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
            raise Exception(str(e))

        # Add solution to solutions array
        if uncommitted is not None and force:
            # Replace the uncommitted solution
            uncommitted_idx, _ = uncommitted
            self.note.set_solution_at(uncommitted_idx, solution)
        else:
            # Append new solution
            self.note.add_solution(solution)

        self.save_note(self.note)

    def describe(self, force: bool, max_attempts: int = None):
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
            raise Exception(str(e))

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
            "MergaAI note available, use `mergai show <commit>` to view it.",
            commit,
        )

    def add_selective_note(self, commit: str, fields: List[str]):
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

        # Build selective note content - always include merge_info
        selective_note = {"merge_info": note_dict["merge_info"]}

        for field in fields:
            # Handle solutions[N] format
            match = re.match(r"solutions\[(\d+)\]", field)
            if match:
                idx = int(match.group(1))
                if self.note.has_solutions and idx < len(self.note.solutions):
                    # In the git note, store as "solution" (singular) for the specific solution
                    selective_note["solution"] = self.note.solutions[idx]
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
                "MergaAI note available, use `mergai show <commit>` to view it.",
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
            for file_path in resolved_files.keys():
                message += f"\t{file_path}\n"
            message += "\n"

        # Add unresolved files section (with conflict markers warning)
        unresolved_files = solution["response"].get("unresolved", {})
        if unresolved_files:
            # Mark conflict markers as unresolved and stage the files
            for file_path in unresolved_files.keys():
                git_utils.mark_conflict_markers_unresolved(file_path)
                self.repo.index.add([file_path])

            message += "Unresolved (contains conflict markers):\n"
            for file_path in unresolved_files.keys():
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

        self.add_selective_note(
            self.repo.head.commit.hexsha, ["conflict_context", "merge_context"]
        )

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

        self.add_selective_note(self.repo.head.commit.hexsha, ["merge_context"])

    def _collect_commits_for_squash(
        self, target_sha: str
    ) -> List[Tuple[git.Commit, Optional[dict]]]:
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
        commits_with_notes: List[Tuple[git.Commit, Optional[dict]]],
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

        # Collect modified files ONLY from solution commits (not from the merge commit)
        # Solution commits are identified by having a "solution" in their note
        # or by NOT having "conflict_context" (the merge commit has conflict_context)
        solution_modified_files = set()
        for commit, note in commits_with_notes:
            # Skip if this commit has conflict_context (it's the merge commit)
            if note and "conflict_context" in note:
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
            first_line = commit.message.split("\n")[0].strip()
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
            try:
                target_sha = self.repo.commit(parsed.target_branch_sha).hexsha
            except Exception:
                pass

        # Collect commits and their notes
        commits_with_notes = []
        for commit in self.repo.iter_commits():
            git_note = self.get_note_from_commit(commit.hexsha)
            if git_note:
                commits_with_notes.append((commit.hexsha, git_note))

                # If we don't have target_sha yet, try to get it from merge_info
                if target_sha is None and "merge_info" in git_note:
                    target_branch_sha = git_note["merge_info"].get("target_branch_sha")
                    if target_branch_sha:
                        target_sha = target_branch_sha

            # Stop if we've reached the target
            if target_sha and commit.hexsha == target_sha:
                break

        if not commits_with_notes:
            raise click.ClickException(
                "No commits with mergai notes found. Cannot rebuild note.json."
            )

        # Reverse to process oldest first (for consistent indexing)
        commits_with_notes.reverse()

        # Build the new note as a dict first
        note_dict = {
            "solutions": [],
            "note_index": [],
        }

        reference_merge_info = None

        for commit_sha, git_note in commits_with_notes:
            fields_for_this_commit = []

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
            if "conflict_context" in git_note:
                if "conflict_context" not in note_dict:
                    note_dict["conflict_context"] = git_note["conflict_context"]
                    fields_for_this_commit.append("conflict_context")

            # Handle merge_context (at most one)
            if "merge_context" in git_note:
                if "merge_context" not in note_dict:
                    note_dict["merge_context"] = git_note["merge_context"]
                    fields_for_this_commit.append("merge_context")

            # Handle solution (singular in git note -> add to solutions array)
            if "solution" in git_note:
                idx = len(note_dict["solutions"])
                note_dict["solutions"].append(git_note["solution"])
                fields_for_this_commit.append(f"solutions[{idx}]")

            # Handle solutions (array in git note -> add all to solutions array)
            if "solutions" in git_note:
                for solution in git_note["solutions"]:
                    idx = len(note_dict["solutions"])
                    note_dict["solutions"].append(solution)
                    fields_for_this_commit.append(f"solutions[{idx}]")

            # Handle merge_description
            if "merge_description" in git_note:
                if "merge_description" not in note_dict:
                    note_dict["merge_description"] = git_note["merge_description"]
                    fields_for_this_commit.append("merge_description")

            # Handle pr_comments
            if "pr_comments" in git_note:
                if "pr_comments" not in note_dict:
                    note_dict["pr_comments"] = git_note["pr_comments"]
                    fields_for_this_commit.append("pr_comments")

            # Handle user_comment
            if "user_comment" in git_note:
                if "user_comment" not in note_dict:
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

        # Convert to MergaiNote
        return MergaiNote.from_dict(note_dict, self.repo)
