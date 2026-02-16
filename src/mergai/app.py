import git
import click
import logging
import textwrap
from . import git_utils
from . import util
from . import prompts
from .agents.factory import create_agent
from .state_store import StateStore
from .config import MergaiConfig
from .models import ConflictContext, MergeContext, MergeInfo, MergaiNote
import github
from github import Repository as GithubRepository
import json
from typing import Optional, Tuple, List
import tempfile
from pathlib import Path
from .agents.base import Agent
from .utils.branch_name_builder import BranchNameBuilder

log = logging.getLogger(__name__)

# TODO: Make this configurable via settings/config file
MERGAI_COMMIT_FOOTER = "Note: commit created by mergai"


def convert_note(
    note: dict,
    format: str,
    repo: git.Repo = None,
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
    """Convert a note to the specified format.

    Args:
        note: The note dict from note.json.
        format: Output format ('json' or 'markdown').
        repo: Optional GitPython Repo for hydrating contexts in markdown format.
        pretty: If True, format JSON with indentation.
        show_context: Include conflict_context in output.
        show_solution: Include solutions in output.
        show_pr_comments: Include PR comments in output.
        show_user_comment: Include user comment in output.
        show_summary: Include summary in output.
        show_merge_info: Include merge_info in output.
        show_merge_context: Include merge_context in output.
        show_merge_description: Include merge_description in output.

    Returns:
        Formatted string representation of the note.
    """
    if format == "json":
        return json.dumps(note, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        output_str = ""
        if show_merge_info and "merge_info" in note:
            merge_info = MergeInfo.from_dict(note["merge_info"], repo)
            output_str += util.merge_info_to_markdown(merge_info) + "\n"
        if show_merge_context and "merge_context" in note:
            merge_ctx = MergeContext.from_dict(note["merge_context"], repo)
            output_str += util.merge_context_to_markdown(merge_ctx) + "\n"
        if show_context and "conflict_context" in note:
            conflict_ctx = ConflictContext.from_dict(note["conflict_context"], repo)
            output_str += util.conflict_context_to_markdown(conflict_ctx) + "\n"
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
                output_str += (
                    util.conflict_solution_to_markdown(note["solution"]) + "\n"
                )
        if show_merge_description and "merge_description" in note:
            output_str += (
                util.merge_description_to_markdown(note["merge_description"]) + "\n"
            )

        return output_str + "\n"
    return str(note)


class AppContext:
    def __init__(self, config: MergaiConfig = None):
        self.config: MergaiConfig = config if config is not None else MergaiConfig()
        self.repo: git.Repo = git.Repo(".")
        self.state: StateStore = StateStore(self.repo.working_tree_dir)
        self.gh_repo_str: Optional[str] = None
        gh_token = util.gh_auth_token()
        self.gh = github.Github(gh_token) if gh_token else None
        self._note: Optional[dict] = None

    def _require_note(self) -> dict:
        """Load note and raise ClickException if missing."""

        if self._note is not None:
            return self._note

        if self.state.note_exists():
            self._note = self.state.load_note()

        if not self._note:
            raise click.ClickException(
                "No note found. Run 'mergai context init' first."
            )

        return self._note

    @property
    def has_note(self) -> bool:
        return self.state.note_exists()

    @property
    def note(self) -> dict:
        if not self.has_note:
            raise click.ClickException(
                "No note found. Run 'mergai context init' first."
            )
        return self._require_note()

    def _require_merge_info(self) -> MergeInfo:
        """Return merge info from note or raise ClickException if missing."""

        self._require_note()

        if "merge_info" not in self._note:
            raise click.ClickException(
                "No merge info found in note. Run 'mergai context init' first."
            )

        return MergeInfo.from_dict(self._note["merge_info"], self.repo)

    @property
    def has_merge_info(self) -> bool:
        self._require_note()
        return "merge_info" in self.note

    @property
    def merge_info(self) -> MergeInfo:
        return self._require_merge_info()

    @property
    def conflict_context(self) -> ConflictContext:
        """Get the conflict context from note, bound to the repo.

        Returns:
            ConflictContext object with repo bound for commit resolution.

        Raises:
            click.ClickException: If no conflict context found in note.
        """
        if not self.has_conflict_context:
            raise click.ClickException(
                "No conflict context found in note. Run 'mergai context create conflict' first."
            )
        return ConflictContext.from_dict(self._note["conflict_context"], self.repo)

    @property
    def has_conflict_context(self) -> bool:
        self._require_note()
        return "conflict_context" in self._note

    @property
    def has_merge_context(self) -> bool:
        self._require_note()
        return "merge_context" in self._note

    @property
    def merge_context(self) -> MergeContext:
        """Get the merge context from note, bound to the repo.

        Returns:
            MergeContext object with repo bound for commit resolution.

        Raises:
            click.ClickException: If no merge context found in note.
        """
        if not self.has_merge_context:
            raise click.ClickException(
                "No merge context found in note. Run 'mergai context create merge' first."
            )
        return MergeContext.from_dict(self._note["merge_context"], self.repo)

    def check_all_solutions_committed(self) -> bool:
        solutions = self.solutions
        committed = self._get_committed_solution_indices(self._note)
        return all(i in committed for i in range(len(solutions)))

    @property
    def solutions(self) -> List[dict]:
        self._require_note()
        if "solutions" not in self._note:
            raise click.ClickException(
                "No solutions found in note. Run 'mergai resolve' first."
            )
        return self._note["solutions"]

    @property
    def has_solutions(self) -> bool:
        self._require_note()
        return "solutions" in self._note

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
        merge_info = self.merge_info
        try:
            return BranchNameBuilder.from_config(
                self.config.branch,
                merge_info,
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


    def get_note_from_commit(self, commit: str) -> Optional[dict]:
        try:
            return git_utils.get_note_from_commit_as_dict(self.repo, "mergai", commit)
        except Exception as e:
            raise click.ClickException(f"Failed to get note for commit {commit}: {e}")

    def try_get_note_from_commit(self, commit: str) -> Optional[MergaiNote]:
        try:
            note_dict = git_utils.get_note_from_commit_as_dict(self.repo, "mergai", commit)
            if note_dict is None:
                return None
            return MergaiNote.from_dict(note_dict, self.repo)
        except Exception as e:
            return None

    # TODO: refactor

    def load_or_create_note(self) -> dict:
        if self.state.note_exists():
            return self.state.load_note()

        return {}

    def save_note(self, note: dict):
        self.state.save_note(note)

    def build_resolve_prompt(self, current_note: dict) -> str:
        if self.has_solutions:
            raise Exception(
                "Current note already has solutions. Cannot build resolve prompt for an existing solution. Please drop existing solutions first."
            )

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

        # Prepare note data for prompt serialization
        # Hydrate conflict_context with configurable commit fields
        note_for_prompt = self._prepare_note_for_prompt(current_note)

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(note_for_prompt, indent=2)
        prompt += "\n```\n"

        return prompt

    def _prepare_note_for_prompt(self, note: dict) -> dict:
        """Prepare a note dict for prompt serialization.

        Hydrates context fields (conflict_context, merge_context) using the
        configurable prompt serialization settings from config.

        Args:
            note: The note dict from note.json (storage format).

        Returns:
            A copy of the note with context fields hydrated for prompt use.
        """
        from .models import ConflictContext, MergeContext, MergeInfo

        note_copy = note.copy()
        prompt_config = self.config.prompt.to_prompt_serialization_config()

        if self.has_conflict_context:
            note_copy["conflict_context"] = self.conflict_context.to_dict(prompt_config)

        if self.has_merge_context:
            note_copy["merge_context"] = self.merge_context.to_dict(prompt_config)

        return note_copy

    def build_describe_prompt(self, current_note: dict) -> str:
        if self.has_solutions:
            raise Exception(
                "Current note already has solutions. Cannot build describe prompt for an existing solution. Please drop existing solutions first."
            )

        system_prompt_describe = prompts.load_system_prompt_describe()
        prompt = system_prompt_describe + "\n\n"

        project_invariants = util.load_if_exists(".mergai/invariants.md")
        if project_invariants:
            prompt += project_invariants + "\n\n"

        # Prepare note data for prompt serialization
        note_for_prompt = self._prepare_note_for_prompt(current_note)

        prompt += "## Note Data\n\n"
        prompt += "```json\n"
        prompt += json.dumps(note_for_prompt, indent=2)
        prompt += "\n```\n"

        return prompt

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
                    entry
                    for entry in note["note_index"]
                    if not any(
                        f.startswith("solutions[") for f in entry.get("fields", [])
                    )
                ]
        else:
            # Only drop uncommitted solutions
            committed_indices = self._get_committed_solution_indices(note)
            if committed_indices:
                # Keep only committed solutions
                note["solutions"] = [
                    note["solutions"][i]
                    for i in sorted(committed_indices)
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

    def drop_merge_context(self):
        self.drop_note_field("merge_context")

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

    def create_merge_context(
        self,
        force: bool = False,
        auto_merged_files: Optional[List[str]] = None,
        merge_strategy: Optional[str] = None,
    ) -> dict:
        """Create merge context from merge_info.

        Calculates the list of commits being merged by finding the
        merge base between target_branch and merge_commit, then
        listing all commits from base..merge_commit. Also identifies
        which important files (from config) were modified.

        Args:
            force: If True, overwrite existing merge_context.
            auto_merged_files: List of files that were auto-merged by git.
            merge_strategy: The merge strategy used (e.g., 'ort', 'recursive').

        Returns:
            The created merge context dict.

        Raises:
            Exception: If merge_info is not initialized or merge_context
                       already exists (without force).
        """
        note = self.load_or_create_note()

        # Require merge_info to be initialized
        if "merge_info" not in note:
            raise Exception("merge_info not found. Run 'mergai context init' first.")

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

            important_files_modified = sorted(set(important_files) & all_modified_files)

        context = {
            "merge_commit": merge_commit_hexsha,
            "merged_commits": merged_commits,
            "important_files_modified": important_files_modified,
        }

        # Add auto_merged info if provided
        if auto_merged_files is not None or merge_strategy is not None:
            context["auto_merged"] = {
                "strategy": merge_strategy,
                "files": auto_merged_files or [],
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
        modified_files = [item.a_path for item in self.repo.index.diff(None)]
        for path in solution["response"]["resolved"].keys():
            click.echo(
                f"Checking file '{path}': {'dirty' if path in modified_files else 'not dirty'}"
            )
            if path not in modified_files:
                not_dirty_files.append(path)
        if len(not_dirty_files):
            message = "The following files in the solution have no unstaged changes: "
            message += ", ".join(not_dirty_files)

            return message

        return None

    def error_to_prompt(self, error: str) -> str:
        return f"An error occurred while trying to process the output: {error}"

    def resolve(self, force: bool, yolo: bool):
        note = self.load_note()
        if note is None:
            raise Exception("No note found. Please prepare the context first.")

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
        max_attempts = self.config.resolve.max_attempts
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
                click.echo(
                    "Max attempts reached. Failed to obtain a valid description."
                )
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

        # Find the uncommitted solution
        uncommitted_idx = self._get_uncommitted_solution_index(note)
        if uncommitted_idx is None:
            if "solutions" not in note or len(note["solutions"]) == 0:
                raise Exception("No solution found in the note.")
            else:
                raise Exception("All solutions have already been committed.")

        solution = note["solutions"][uncommitted_idx]

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
        if self.has_conflict_context:
            theirs_sha = git_utils.short_sha(self.conflict_context.theirs_commit_sha)
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
        self.add_selective_note(
            self.repo.head.commit.hexsha, [f"solutions[{uncommitted_idx}]"]
        )

    def commit_conflict(self):
        hint_msg = "Please prepare the conflict context by running:\nmergai create-conflict-context"
        note = self.load_note()
        if note is None:
            raise Exception(f"No note found.\n\n{hint_msg}")

        context_dict = note.get("conflict_context")
        if context_dict is None:
            raise Exception(f"No conflict context found in the note.\n\n{hint_msg}")

        conflict_ctx = ConflictContext.from_dict(context_dict, self.repo)
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
        message += MERGAI_COMMIT_FOOTER

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
        the short sha comes from merge_info. Requires merge_info to be
        initialized in the note.

        Raises:
            Exception: If no note found or merge_info is missing.
        """
        hint_msg = (
            "Please initialize merge context by running:\nmergai context init <commit>"
        )
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

    def _build_combined_note(
        self,
        commits_with_notes: List[Tuple[git.Commit, Optional[dict]]],
        new_commit_sha: str,
    ) -> dict:
        """Build a combined note from multiple commit notes.

        Merges all note data from the provided commits into a single note:
        - merge_info: taken from the first commit that has it
        - conflict_context: taken from the first commit that has it
        - merge_context: taken from the first commit that has it
        - solutions: all solutions combined into a single array
        - merge_description: taken from the first commit that has it
        - pr_comments: taken from the first commit that has it
        - user_comment: taken from the first commit that has it
        - note_index: rebuilt to reference the new squashed commit

        Args:
            commits_with_notes: List of (commit, note) tuples from _collect_commits_for_squash.
            new_commit_sha: The SHA of the new squashed commit for note_index.

        Returns:
            The combined note dict.
        """
        combined = {}
        all_fields = []

        for commit, git_note in commits_with_notes:
            if git_note is None:
                continue

            # merge_info - take from first commit that has it
            if "merge_info" in git_note and "merge_info" not in combined:
                combined["merge_info"] = git_note["merge_info"]

            # conflict_context - take from first commit that has it
            if "conflict_context" in git_note and "conflict_context" not in combined:
                combined["conflict_context"] = git_note["conflict_context"]
                all_fields.append("conflict_context")

            # merge_context - take from first commit that has it
            if "merge_context" in git_note and "merge_context" not in combined:
                combined["merge_context"] = git_note["merge_context"]
                all_fields.append("merge_context")

            # solutions - combine all into array
            # Handle both "solution" (singular in git note) and "solutions" (array)
            if "solution" in git_note:
                if "solutions" not in combined:
                    combined["solutions"] = []
                idx = len(combined["solutions"])
                combined["solutions"].append(git_note["solution"])
                all_fields.append(f"solutions[{idx}]")

            if "solutions" in git_note:
                if "solutions" not in combined:
                    combined["solutions"] = []
                for solution in git_note["solutions"]:
                    idx = len(combined["solutions"])
                    combined["solutions"].append(solution)
                    all_fields.append(f"solutions[{idx}]")

            # merge_description - take from first commit that has it
            if "merge_description" in git_note and "merge_description" not in combined:
                combined["merge_description"] = git_note["merge_description"]
                all_fields.append("merge_description")

            # pr_comments - take from first commit that has it
            if "pr_comments" in git_note and "pr_comments" not in combined:
                combined["pr_comments"] = git_note["pr_comments"]
                all_fields.append("pr_comments")

            # user_comment - take from first commit that has it
            if "user_comment" in git_note and "user_comment" not in combined:
                combined["user_comment"] = git_note["user_comment"]
                all_fields.append("user_comment")

        # Build note_index pointing to the new squashed commit
        if all_fields:
            combined["note_index"] = [{"sha": new_commit_sha, "fields": all_fields}]

        return combined

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
        message += MERGAI_COMMIT_FOOTER

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
        note = self.load_note()
        if note is None:
            raise click.ClickException(
                "No note found. Please initialize merge context by running:\n"
                "mergai context init <commit>"
            )

        merge_info = note.get("merge_info")
        if merge_info is None:
            raise click.ClickException(
                "No merge_info found in the note. Please initialize merge context by running:\n"
                "mergai context init <commit>"
            )

        target_branch_sha = merge_info.get("target_branch_sha")
        merge_commit_sha = merge_info.get("merge_commit")

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

        # Build combined note (we'll use a placeholder SHA for now, update after commit)
        combined_note = self._build_combined_note(commits_with_notes, "PLACEHOLDER")

        # Build commit message
        message = self._build_squash_commit_message(
            merge_info, commits_with_notes, combined_note
        )

        # Create commit with two parents: target_branch_sha (first parent) and merge_commit_sha (second parent)
        click.echo("Creating squashed merge commit...")
        new_commit_sha = self.repo.git.commit_tree(
            tree_sha, "-p", target_sha_full, "-p", merge_commit_sha, "-m", message
        )

        # Update HEAD to point to the new commit
        self.repo.git.reset("--hard", new_commit_sha)

        # Rebuild combined note with correct SHA
        combined_note = self._build_combined_note(commits_with_notes, new_commit_sha)

        # Also include merge_info from our local note (in case git notes didn't have it)
        if "merge_info" not in combined_note:
            combined_note["merge_info"] = merge_info

        # Save combined note to local state
        self.save_note(combined_note)

        # Attach note to the new commit
        click.echo("Attaching combined note to squashed commit...")
        self.add_note(new_commit_sha)

        click.echo(
            f"Successfully squashed {len(commits_with_notes)} commit(s) into {git_utils.short_sha(new_commit_sha)}"
        )

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

            # Handle solutions (array in git note -> add all to solutions array)
            if "solutions" in git_note:
                for solution in git_note["solutions"]:
                    idx = len(note["solutions"])
                    note["solutions"].append(solution)
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
                note["note_index"].append(
                    {
                        "sha": commit_sha,
                        "fields": fields_for_this_commit,
                    }
                )

        # Clean up empty collections
        if not note["solutions"]:
            del note["solutions"]
        if not note["note_index"]:
            del note["note_index"]

        return note
