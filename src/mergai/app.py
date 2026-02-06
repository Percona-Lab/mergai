import git
import subprocess
import click
import logging
import textwrap
from . import git_utils
from . import util
from . import prompts
from .agents.factory import create_agent
from .state_store import StateStore
from .config import MergaiConfig
import github
from github import Repository as GithubRepository
import json
from typing import Optional, Tuple, List
from datetime import datetime, timezone
from dataclasses import dataclass
import tempfile
from pathlib import Path
from .agents.base import Agent

log = logging.getLogger(__name__)

# TODO: Make this configurable via settings/config file
MERGAI_COMMIT_FOOTER = "Note: commit created by mergai"


@dataclass
class MergeContext:
    """Context information for a merge operation.

    Attributes:
        target_branch: The branch being merged into (e.g., "v8.0", "master").
        target_branch_sha: Full SHA of the target branch HEAD (40 chars).
        merge_commit_sha: Full SHA of the commit being merged (40 chars).
    """

    target_branch: str
    target_branch_sha: str
    merge_commit_sha: str


def gh_auth_token() -> str:
    import os

    token = os.getenv("GITHUB_TOKEN")
    if token is not None:
        return token
    token = os.getenv("GH_TOKEN")
    if token is not None:
        return token

    try:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except:
        token = None

    return token


def convert_note(
    note: dict,
    format: str,
    pretty: bool = False,
    show_context: bool = True,
    show_solution: bool = True,
    show_pr_comments: bool = True,
    show_user_comment: bool = True,
    show_summary: bool = True,
    show_merge_info: bool = True,
    show_merge_context: bool = True,
    show_merge_description: bool = True,
) -> str:
    if format == "json":
        return json.dumps(note, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        output_str = ""
        if show_merge_info and "merge_info" in note:
            output_str += util.merge_info_to_markdown(note["merge_info"]) + "\n"
        if show_merge_context and "merge_context" in note:
            output_str += util.merge_context_to_markdown(note["merge_context"]) + "\n"
        if show_context and "conflict_context" in note:
            output_str += (
                util.conflict_context_to_markdown(note["conflict_context"]) + "\n"
            )
        if show_pr_comments and "pr_comments" in note:
            output_str += util.pr_comments_to_markdown(note["pr_comments"]) + "\n"
        if show_user_comment and "user_comment" in note:
            output_str += util.user_comment_to_markdown(note["user_comment"]) + "\n"
        # Handle both legacy "solution" and new "solutions" array
        if show_solution:
            if "solutions" in note:
                for idx, solution in enumerate(note["solutions"]):
                    if len(note["solutions"]) > 1:
                        output_str += f"## Solution {idx + 1}\n\n"
                    output_str += util.conflict_solution_to_markdown(solution) + "\n"
            elif "solution" in note:
                output_str += util.conflict_solution_to_markdown(note["solution"]) + "\n"
        if show_merge_description and "merge_description" in note:
            output_str += (
                util.merge_description_to_markdown(note["merge_description"]) + "\n"
            )

        return output_str + "\n"
    return str(note)


# TODO: refactor
class AppContext:
    def __init__(self, config: MergaiConfig = None):
        self.config: MergaiConfig = config if config is not None else MergaiConfig()
        self.repo: git.Repo = git.Repo(".")
        self.state: StateStore = StateStore(self.repo.working_tree_dir)
        gh_token = gh_auth_token()
        self.gh_repo_str: Optional[str] = None
        self.gh = github.Github(gh_auth_token()) if gh_token else None

    def get_repo(self) -> git.Repo:
        return self.repo

    def get_gh_repo(self) -> GithubRepository.Repository:
        if self.gh is None:
            raise Exception(
                "GitHub token not found. Please set GITHUB_TOKEN or GH_TOKEN."
            )
        if self.gh_repo_str is None:
            raise Exception("GitHub repository not set. Please provide --repo option.")
        return self.gh.get_repo(self.gh_repo_str)

    def get_merge_context(
        self,
        commit: Optional[str] = None,
        target: Optional[str] = None,
    ) -> MergeContext:
        """Get merge context from note.json or current branch.

        Resolves target_branch, target_branch_sha, and merge_commit_sha using the following
        priority:
        1. If merge_info exists in note.json, use those values
        2. Otherwise, try to parse from current mergai branch name
        3. If neither exists, raise an exception
        4. If both exist and values differ, raise an exception

        CLI-provided values (commit, target) override the auto-detected values.

        Args:
            commit: Optional commit SHA or ref to override auto-detection.
            target: Optional target branch to override auto-detection.

        Returns:
            MergeContext with target_branch, target_branch_sha, and merge_commit_sha.

        Raises:
            click.ClickException: If context cannot be determined or conflicts.
        """
        # Get values from note.json if merge_info exists
        note = self.load_note()
        note_target = None
        note_target_sha = None
        note_commit = None
        if note and "merge_info" in note:
            merge_info = note["merge_info"]
            note_target = merge_info.get("target_branch")
            note_target_sha = merge_info.get("target_branch_sha")
            note_commit = merge_info.get("merge_commit")

        # Get values from current branch name
        current_branch = git_utils.get_current_branch(self.repo)
        parsed = util.BranchNameBuilder.parse_branch_name_with_config(
            current_branch, self.config.branch
        )
        branch_target = parsed.target_branch if parsed else None
        branch_target_sha = parsed.target_branch_sha if parsed else None
        branch_commit = parsed.merge_commit_sha if parsed else None

        # Check for conflicts between note and branch (when both exist)
        # Compare by resolving both to full SHA for accurate comparison
        if note_target and branch_target and note_target != branch_target:
            raise click.ClickException(
                f"Conflict: target_branch in note.json ({note_target}) differs from "
                f"current branch ({branch_target}). Use --target to specify explicitly."
            )
        if note_commit and branch_commit:
            # Resolve branch_commit (possibly short) to full SHA for comparison
            try:
                branch_commit_full = self.repo.commit(branch_commit).hexsha
                if note_commit != branch_commit_full:
                    raise click.ClickException(
                        f"Conflict: merge_commit in note.json ({note_commit}) differs from "
                        f"current branch ({branch_commit}). Use COMMIT argument to specify explicitly."
                    )
            except Exception:
                # If we can't resolve, compare as-is
                if note_commit != branch_commit:
                    raise click.ClickException(
                        f"Conflict: merge_commit in note.json ({note_commit}) differs from "
                        f"current branch ({branch_commit}). Use COMMIT argument to specify explicitly."
                    )

        # Resolve final values: CLI args > note.json > branch > error
        # Resolve merge_commit to full SHA
        if commit is not None:
            # CLI-provided commit - resolve to full SHA
            try:
                resolved = self.repo.commit(commit)
                final_commit = resolved.hexsha
            except Exception as e:
                raise click.ClickException(f"Invalid commit reference '{commit}': {e}")
        elif note_commit:
            final_commit = note_commit
        elif branch_commit:
            # Resolve to full SHA
            try:
                final_commit = self.repo.commit(branch_commit).hexsha
            except Exception as e:
                raise click.ClickException(f"Cannot resolve commit from branch name '{branch_commit}': {e}")
        else:
            raise click.ClickException(
                "Cannot determine merge commit. Either:\n"
                "  - Run 'mergai context init' first, or\n"
                "  - Switch to a mergai branch, or\n"
                "  - Provide COMMIT argument explicitly."
            )

        # Resolve target_branch
        if target is not None:
            final_target = target
        elif note_target:
            final_target = note_target
        elif branch_target:
            final_target = branch_target
        else:
            # Fall back to current branch name (matches old behavior)
            final_target = current_branch

        # Resolve target_branch_sha
        if note_target_sha:
            final_target_sha = note_target_sha
        elif branch_target_sha:
            # Resolve to full SHA
            try:
                final_target_sha = self.repo.commit(branch_target_sha).hexsha
            except Exception:
                # If can't resolve, use as-is
                final_target_sha = branch_target_sha
        else:
            # Resolve from target branch
            try:
                final_target_sha = self.repo.commit(final_target).hexsha
            except Exception as e:
                raise click.ClickException(f"Cannot resolve target branch '{final_target}': {e}")

        return MergeContext(
            target_branch=final_target,
            target_branch_sha=final_target_sha,
            merge_commit_sha=final_commit,
        )

    def _get_target_branch_name(self) -> str:
        """Get the target branch name for commit messages.

        Returns the target branch name by:
        1. Checking merge_info in note.json
        2. Parsing the current branch if it's a mergai branch
        3. Otherwise, returning the current branch name (Git's default behavior)

        Returns:
            The target branch name to use in commit messages.
        """
        # Try to get from merge context (note.json or branch)
        note = self.load_note()
        if note and "merge_info" in note:
            target = note["merge_info"].get("target_branch")
            if target:
                return target

        current_branch = git_utils.get_current_branch(self.repo)
        parsed = util.BranchNameBuilder.parse_branch_name_with_config(
            current_branch, self.config.branch
        )
        if parsed:
            return parsed.target_branch
        return current_branch

    def read_note(self, commit: str) -> Optional[dict]:
        note_str = git_utils.read_commit_note(self.repo, "mergai", commit)
        if not note_str:
            return None

        try:
            note = json.loads(note_str)
            return note
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse note for commit {commit} as JSON: {e}")

    def load_note(self) -> Optional[dict]:
        if self.state.note_exists():
            return self.state.load_note()

        return None

    def get_notes(self):
        def get_notes_from_commits():
            notes = []
            for commit in self.repo.iter_commits():
                note_str = git_utils.read_commit_note(
                    self.repo, "mergai", commit.hexsha
                )
                if note_str is None:
                    # TODO: handle commits without notes ?
                    return notes

                note = json.loads(note_str)
                notes.append((commit, note))

            notes.reverse()
            return notes

        notes = get_notes_from_commits()
        notes.reverse()
        return notes

    def load_or_create_note(self) -> dict:
        if self.state.note_exists():
            return self.state.load_note()

        return {}

    def save_note(self, note: dict):
        self.state.save_note(note)

    def build_resolve_prompt(self, current_note: dict, use_history: bool) -> str:
        if use_history:
            raise Exception("use_history in build_resolve_prompt is not supported yet.")

        system_prompt_resolve = prompts.load_system_prompt_resolve()
        project_invariants = util.load_if_exists(".mergai/invariants.md")

        prompt = system_prompt_resolve + "\n\n"
        if project_invariants:
            prompt += project_invariants + "\n\n"

        if "conflict_context" in current_note:
            prompt += prompts.load_conflict_context_prompt() + "\n\n"

        if "pr_comments" in current_note:
            prompt += prompts.load_pr_comments_prompt() + "\n\n"

        if "user_comment" in current_note:
            prompt += prompts.load_user_comment_prompt() + "\n\n"

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(current_note, indent=2)
        prompt += "\n```\n"

        return prompt

    def build_describe_prompt(self, current_note: dict) -> str:
        system_prompt_describe = prompts.load_system_prompt_describe()
        prompt = system_prompt_describe + "\n\n"

        project_invariants = util.load_if_exists(".mergai/invariants.md")
        if project_invariants:
            prompt += project_invariants + "\n\n"

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(current_note, indent=2)
        prompt += "\n```\n"

        return prompt

    def drop_conflict_prompt(self):
        self.state.drop_conflict_prompt()

    def drop_note_field(self, field: str):
        note = self.load_or_create_note()
        if field in note:
            del note[field]
            if len(note) == 0:
                self.state.remove_note()
            else:
                self.save_note(note)

    def drop_all(self):
        self.state.remove_note()

    def drop_solution(self, all: bool = False):
        """Drop solution(s) from the note.

        Args:
            all: If True, drop all solutions. If False (default), only drop
                 uncommitted solutions (those not in note_index).
        """
        note = self.load_or_create_note()

        # Handle legacy "solution" field - migrate to "solutions"
        if "solution" in note:
            if "solutions" not in note:
                note["solutions"] = [note["solution"]]
            del note["solution"]

        if "solutions" not in note:
            return

        if all:
            # Drop all solutions
            del note["solutions"]
            # Also remove solutions entries from note_index
            if "note_index" in note:
                note["note_index"] = [
                    entry for entry in note["note_index"]
                    if not any(f.startswith("solutions[") for f in entry.get("fields", []))
                ]
        else:
            # Only drop uncommitted solutions
            committed_indices = self._get_committed_solution_indices(note)
            if committed_indices:
                # Keep only committed solutions
                note["solutions"] = [
                    note["solutions"][i] for i in sorted(committed_indices)
                    if i < len(note["solutions"])
                ]
            else:
                # No committed solutions, drop all
                del note["solutions"]

        if len(note) == 0:
            self.state.remove_note()
        else:
            self.save_note(note)

    def _get_committed_solution_indices(self, note: dict) -> set:
        """Get indices of solutions that have been committed.

        Args:
            note: The note dict containing note_index.

        Returns:
            Set of solution indices that are in the note_index.
        """
        committed = set()
        if "note_index" not in note:
            return committed

        import re
        for entry in note["note_index"]:
            for field in entry.get("fields", []):
                # Match "solutions[N]" pattern
                match = re.match(r"solutions\[(\d+)\]", field)
                if match:
                    committed.add(int(match.group(1)))
                # Also handle legacy "solution" field
                if field == "solution":
                    committed.add(0)

        return committed

    def _get_uncommitted_solution_index(self, note: dict) -> Optional[int]:
        """Get the index of the last uncommitted solution.

        Args:
            note: The note dict.

        Returns:
            Index of the last uncommitted solution, or None if all are committed.
        """
        solutions = note.get("solutions", [])
        if not solutions:
            return None

        committed = self._get_committed_solution_indices(note)
        # Find the last index that is not committed
        for i in range(len(solutions) - 1, -1, -1):
            if i not in committed:
                return i
        return None

    def _migrate_solution_to_solutions(self, note: dict) -> dict:
        """Migrate legacy 'solution' field to 'solutions' array.

        Args:
            note: The note dict to migrate.

        Returns:
            The migrated note dict.
        """
        if "solution" in note and "solutions" not in note:
            note["solutions"] = [note["solution"]]
            del note["solution"]
            # Also migrate note_index entries
            if "note_index" in note:
                for entry in note["note_index"]:
                    entry["fields"] = [
                        "solutions[0]" if f == "solution" else f
                        for f in entry.get("fields", [])
                    ]
        return note

    def drop_pr_comments(self):
        self.drop_note_field("pr_comments")

    def drop_conflict_context(self):
        self.drop_note_field("conflict_context")

    def drop_user_comment(self):
        self.drop_note_field("user_comment")

    def drop_merge_info(self):
        self.drop_note_field("merge_info")

    def drop_merge_description(self):
        self.drop_note_field("merge_description")

    def create_conflict_context(
        self,
        use_diffs: bool,
        diff_lines_of_context: int,
        use_compressed_diffs: bool,
        use_their_commits: bool,
        force: bool,
    ) -> dict:
        note = self.load_or_create_note()

        context = git_utils.get_conflict_context(
            self.repo,
            use_diffs=use_diffs,
            lines_of_context=diff_lines_of_context,
            use_compressed_diffs=use_compressed_diffs,
            use_their_commits=use_their_commits,
        )
        if context is None:
            raise Exception("No merge in progress")

        if "conflict_context" in note and not force:
            raise Exception(
                "Conflict context already exists in the note. Use -f/--force to overwrite."
            )

        note["conflict_context"] = context

        self.save_note(note)

        return context

    def create_merge_context(self, force: bool = False) -> dict:
        """Create merge context from merge_info.

        Calculates the list of commits being merged by finding the
        merge base between target_branch and merge_commit, then
        listing all commits from base..merge_commit. Also identifies
        which important files (from config) were modified.

        Args:
            force: If True, overwrite existing merge_context.

        Returns:
            The created merge context dict.

        Raises:
            Exception: If merge_info is not initialized or merge_context
                       already exists (without force).
        """
        note = self.load_or_create_note()

        # Require merge_info to be initialized
        if "merge_info" not in note:
            raise Exception(
                "merge_info not found. Run 'mergai context init' first."
            )

        # Check for existing merge_context
        if "merge_context" in note and not force:
            raise Exception(
                "merge_context already exists. Use -f/--force to overwrite."
            )

        merge_info = note["merge_info"]
        target_branch = merge_info["target_branch"]
        merge_commit_sha = merge_info["merge_commit"]

        # Resolve merge commit
        merge_commit = self.repo.commit(merge_commit_sha)
        merge_commit_hexsha = merge_commit.hexsha

        log.info(f"getting merged commits for merge context: target_branch={target_branch}, merge_commit={merge_commit_sha}")
        # Get the list of merged commits
        merged_commits = git_utils.get_merged_commits(
            self.repo,
            target_branch,
            merge_commit_sha,
        )
        log.info(f"found {len(merged_commits)} merged commits for merge context")

        # Get important files from config
        important_files = self._get_important_files_from_config()

        # Find which important files were modified by any of the merged commits
        important_files_modified = []
        if important_files:
            all_modified_files = set()
            for commit_sha in merged_commits:
                commit = self.repo.commit(commit_sha)
                modified = git_utils.get_commit_modified_files(self.repo, commit)
                all_modified_files.update(modified)

            important_files_modified = sorted(
                set(important_files) & all_modified_files
            )

        context = {
            "merge_commit": merge_commit_hexsha,
            "merged_commits": merged_commits,
            "important_files_modified": important_files_modified,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        note["merge_context"] = context
        self.save_note(note)

        return context

    def _get_important_files_from_config(self) -> List[str]:
        """Extract important files list from merge_picks config.

        Looks for the important_files strategy in fork.merge_picks
        and returns its file list.

        Returns:
            List of file paths, or empty list if not configured.
        """
        if not self.config.fork.merge_picks:
            return []

        for strategy in self.config.fork.merge_picks.strategies:
            if strategy.name == "important_files":
                return strategy.config.files

        return []

    def drop_merge_context(self):
        """Drop the merge_context from the note."""
        self.drop_note_field("merge_context")

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

    def check_solution_files_dirty(self, solution: dict) -> Optional[str]:
        not_dirty_files = []
        for path in solution["response"]["resolved"].keys():
            click.echo(
                f"Checking file '{path}': {'dirty' if self.repo.is_dirty(path) else 'not dirty'}"
            )
            if not self.repo.is_dirty(path):
                not_dirty_files.append(path)
        if len(not_dirty_files):
            message = "The following files in the solution have no unstaged changes: "
            message += ", ".join(not_dirty_files)

            return message

        return None

    def error_to_prompt(self, error: str) -> str:
        return f"An error occurred while trying to process the output: {error}"

    def resolve(
        self, force: bool, use_history: bool, yolo: bool, max_attempts: int = None
    ):
        if max_attempts is None:
            max_attempts = self.config.resolve.max_attempts
        if use_history:
            raise Exception("use_history is not supported yet.")

        note = self.load_note()
        if note is None:
            raise Exception("No note found. Please prepare the context first.")

        # Migrate legacy solution field
        note = self._migrate_solution_to_solutions(note)

        # Check if there's an uncommitted solution
        uncommitted_idx = self._get_uncommitted_solution_index(note)
        if uncommitted_idx is not None and not force:
            raise Exception(
                "An uncommitted solution already exists in the note. Use -f/--force to overwrite."
            )

        # If force and there's an uncommitted solution, we'll replace it
        # Otherwise, we'll append a new solution

        # TODO: implement use_history=True
        prompt = self.build_resolve_prompt(note, use_history=False)

        agent = self.get_agent(yolo=yolo)

        tmp = tempfile.NamedTemporaryFile(dir=Path.cwd(), mode="w", delete=False)
        tmp.write(prompt)
        tmp.flush()
        tmp.close()
        prompt_path = Path(tmp.name)

        prompt = f"See @{prompt_path} make sure the output is in specified format"

        error = None
        solution = None
        # TODO: refactor retry logic
        for attempt in range(max_attempts):
            if error is not None:
                click.echo(
                    f"Attempt {attempt + 1} failed with error: {error}. Retrying..."
                )

            if attempt == max_attempts - 1:
                click.echo("Max attempts reached. Failed to obtain a valid solution.")
                solution = None
                break

            result = agent.run(prompt)
            if not result.success():
                click.echo(f"Agent execution failed: {result.error()}")
                prompt = self.error_to_prompt(str(result.error()))
                continue

            click.echo("Agent execution succeeded. Checking result...")
            solution = result.result()

            failed_files = self.check_solution_files_dirty(solution)
            if failed_files is not None:
                click.echo(
                    f"Checking resolved files from solution failed: {failed_files}"
                )
                prompt = self.error_to_prompt(failed_files)
                continue

            click.echo("Solution verified.")
            break

        if tmp is not None:
            prompt_path.unlink()

        if solution is None:
            raise Exception("Failed to obtain a valid solution from the agent.")

        # Add solution to solutions array
        if "solutions" not in note:
            note["solutions"] = []

        if uncommitted_idx is not None and force:
            # Replace the uncommitted solution
            note["solutions"][uncommitted_idx] = solution
        else:
            # Append new solution
            note["solutions"].append(solution)

        self.save_note(note)

    def check_describe_response_format(self, response: dict) -> Optional[str]:
        """Check if the describe response has the correct format.

        Args:
            response: The response dict from the agent.

        Returns:
            None if valid, or an error message string if invalid.
        """
        required_fields = ["summary", "auto_merged", "review_notes"]
        missing_fields = [f for f in required_fields if f not in response]
        if missing_fields:
            return f"Missing required fields: {', '.join(missing_fields)}"

        if not isinstance(response.get("auto_merged"), dict):
            return "'auto_merged' field must be a dictionary"

        return None

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

        note = self.load_note()
        if note is None:
            raise Exception("No note found. Please prepare the context first.")

        if "merge_description" in note and not force:
            raise Exception(
                "Merge description already exists in the note. Use -f/--force to overwrite."
            )

        if "merge_description" in note:
            del note["merge_description"]

        prompt = self.build_describe_prompt(note)

        # No YOLO mode for describe - we don't want file modifications
        agent = self.get_agent(yolo=False)

        # Check repo state before running agent
        was_dirty_before = self.repo.is_dirty(untracked_files=True)

        tmp = tempfile.NamedTemporaryFile(dir=Path.cwd(), mode="w", delete=False)
        tmp.write(prompt)
        tmp.flush()
        tmp.close()
        prompt_path = Path(tmp.name)

        prompt = f"See @{prompt_path} make sure the output is in specified format"

        error = None
        description = None
        for attempt in range(max_attempts):
            if error is not None:
                click.echo(
                    f"Attempt {attempt + 1} failed with error: {error}. Retrying..."
                )

            if attempt == max_attempts - 1:
                click.echo("Max attempts reached. Failed to obtain a valid description.")
                description = None
                break

            result = agent.run(prompt)
            if not result.success():
                click.echo(f"Agent execution failed: {result.error()}")
                prompt = self.error_to_prompt(str(result.error()))
                continue

            click.echo("Agent execution succeeded. Checking result...")
            description = result.result()

            # Validate response format
            format_error = self.check_describe_response_format(description["response"])
            if format_error is not None:
                click.echo(f"Response format validation failed: {format_error}")
                prompt = self.error_to_prompt(format_error)
                error = format_error
                continue

            # Check that no files were modified
            is_dirty_after = self.repo.is_dirty(untracked_files=True)
            if is_dirty_after and not was_dirty_before:
                error_msg = "Files were modified during describe operation. No file modifications are allowed."
                click.echo(f"Validation failed: {error_msg}")
                prompt = self.error_to_prompt(error_msg)
                error = error_msg
                continue
            elif is_dirty_after and was_dirty_before:
                # Check if new files were modified (compare dirty files)
                # For simplicity, we'll just warn but allow it if repo was already dirty
                click.echo(
                    "Warning: Repository was already dirty before describe. "
                    "Cannot verify if new modifications were made."
                )

            click.echo("Description verified.")
            error = None
            break

        if tmp is not None:
            prompt_path.unlink()

        if description is None:
            raise Exception("Failed to obtain a valid description from the agent.")

        note["merge_description"] = description
        self.save_note(note)

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

        note = self.state.load_note()

        # Migrate legacy solution field
        note = self._migrate_solution_to_solutions(note)

        # Build selective note content - always include merge_info
        selective_note = {}
        if "merge_info" in note:
            selective_note["merge_info"] = note["merge_info"]

        for field in fields:
            # Handle solutions[N] format
            match = re.match(r"solutions\[(\d+)\]", field)
            if match:
                idx = int(match.group(1))
                if "solutions" in note and idx < len(note["solutions"]):
                    # In the git note, store as "solution" (singular) for the specific solution
                    selective_note["solution"] = note["solutions"][idx]
            elif field in note:
                selective_note[field] = note[field]

        # Write selective note to temp file and attach as git note
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
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

        # Update note_index in note.json
        note_index_entry = {"sha": commit, "fields": fields}
        if "note_index" not in note:
            note["note_index"] = []
        note["note_index"].append(note_index_entry)
        self.state.save_note(note)

    def commit_solution(self):
        if not self.state.note_exists():
            raise Exception("No note found.")

        note = self.state.load_note()

        # Migrate legacy solution field
        note = self._migrate_solution_to_solutions(note)
        self.save_note(note)

        # Find the uncommitted solution
        uncommitted_idx = self._get_uncommitted_solution_index(note)
        if uncommitted_idx is None:
            if "solutions" not in note or len(note["solutions"]) == 0:
                raise Exception("No solution found in the note.")
            else:
                raise Exception("All solutions have already been committed.")

        solution = note["solutions"][uncommitted_idx]
        conflict_context = note.get("conflict_context")

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
        target_branch = self._get_target_branch_name()
        if conflict_context:
            theirs_sha = conflict_context["theirs_commit"]["short_sha"]
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
        message += MERGAI_COMMIT_FOOTER

        self.repo.index.commit(message)

        # Add note with the specific solution index
        self.add_selective_note(self.repo.head.commit.hexsha, [f"solutions[{uncommitted_idx}]"])

    def commit_conflict(self):
        hint_msg = "Please prepare the conflict context by running:\nmergai create-conflict-context"
        note = self.load_note()
        if note is None:
            raise Exception(f"No note found.\n\n{hint_msg}")

        context = note.get("conflict_context")
        if context is None:
            raise Exception(f"No conflict context found in the note.\n\n{hint_msg}")

        theirs_sha = context["theirs_commit"]["short_sha"]
        files = context["files"]
        target_branch = self._get_target_branch_name()

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
        message += MERGAI_COMMIT_FOOTER

        for path in files:
            self.repo.git.add(path)

        self.repo.git.commit(
            "-m",
            message,
        )

        self.add_selective_note(self.repo.head.commit.hexsha, ["conflict_context"])

    def commit_merge(self):
        """Commit the current staged changes as a merge commit.

        Creates a commit with the message "Merge commit '<short sha>'" where
        the short sha comes from merge_info. Requires merge_info to be
        initialized in the note.

        Raises:
            Exception: If no note found or merge_info is missing.
        """
        hint_msg = "Please initialize merge context by running:\nmergai context init <commit>"
        note = self.load_note()
        if note is None:
            raise Exception(f"No note found.\n\n{hint_msg}")

        merge_info = note.get("merge_info")
        if merge_info is None:
            raise Exception(f"No merge_info found in the note.\n\n{hint_msg}")

        merge_commit = merge_info["merge_commit"]

        # Build commit message (use short SHA for display)
        message = f"Merge commit '{git_utils.short_sha(merge_commit)}'\n\n"

        # Add MergAI footer
        message += MERGAI_COMMIT_FOOTER

        self.repo.git.commit("-m", message)

        self.add_selective_note(self.repo.head.commit.hexsha, ["merge_context"])

    def get_merge_conflict(
        self, ref: str
    ) -> Tuple[Optional[git.Commit], Optional[dict]]:
        note = None
        conflict_context = None
        for commit in self.repo.iter_commits(ref):
            note = self.read_note(commit.hexsha)

            if not conflict_context and note and "conflict_context" in note:
                conflict_context = note["conflict_context"]

            if git_utils.is_merge_commit(commit) and conflict_context:
                parent1_sha = commit.parents[0].hexsha
                parent2_sha = commit.parents[1].hexsha

                ours_sha = conflict_context["ours_commit"]["hexsha"]
                theirs_sha = conflict_context["theirs_commit"]["hexsha"]

                if ours_sha == parent1_sha and theirs_sha == parent2_sha:
                    return (commit, conflict_context)

        return (None, conflict_context)

    def rebuild_note_from_commits(self) -> dict:
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
            The rebuilt note dict.

        Raises:
            click.ClickException: If merge_info is inconsistent across commits
                                  or cannot determine scan range.
        """
        current_branch = git_utils.get_current_branch(self.repo)
        parsed = util.BranchNameBuilder.parse_branch_name_with_config(
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
            git_note = self.read_note(commit.hexsha)
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

        # Build the new note
        note = {
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
                    note["merge_info"] = git_note["merge_info"]
                else:
                    # Check consistency
                    current_mi = git_note["merge_info"]
                    if (current_mi.get("target_branch") != reference_merge_info.get("target_branch") or
                        current_mi.get("merge_commit") != reference_merge_info.get("merge_commit")):
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
                if "conflict_context" not in note:
                    note["conflict_context"] = git_note["conflict_context"]
                    fields_for_this_commit.append("conflict_context")

            # Handle merge_context (at most one)
            if "merge_context" in git_note:
                if "merge_context" not in note:
                    note["merge_context"] = git_note["merge_context"]
                    fields_for_this_commit.append("merge_context")

            # Handle solution (singular in git note -> add to solutions array)
            if "solution" in git_note:
                idx = len(note["solutions"])
                note["solutions"].append(git_note["solution"])
                fields_for_this_commit.append(f"solutions[{idx}]")

            # Handle merge_description
            if "merge_description" in git_note:
                if "merge_description" not in note:
                    note["merge_description"] = git_note["merge_description"]
                    fields_for_this_commit.append("merge_description")

            # Handle pr_comments
            if "pr_comments" in git_note:
                if "pr_comments" not in note:
                    note["pr_comments"] = git_note["pr_comments"]
                    fields_for_this_commit.append("pr_comments")

            # Handle user_comment
            if "user_comment" in git_note:
                if "user_comment" not in note:
                    note["user_comment"] = git_note["user_comment"]
                    fields_for_this_commit.append("user_comment")

            # Add to note_index if we collected any fields
            if fields_for_this_commit:
                note["note_index"].append({
                    "sha": commit_sha,
                    "fields": fields_for_this_commit,
                })

        # Clean up empty collections
        if not note["solutions"]:
            del note["solutions"]
        if not note["note_index"]:
            del note["note_index"]

        return note
