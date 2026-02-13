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
from .models import ConflictContext, MergeContext, MergeInfo
import github
from github import Repository as GithubRepository
import json
from typing import Optional, Tuple, List
import tempfile
from pathlib import Path
from .agents.base import Agent
from .util import BranchNameBuilder

log = logging.getLogger(__name__)

# TODO: Make this configurable via settings/config file
MERGAI_COMMIT_FOOTER = "Note: commit created by mergai"


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
        gh_token = gh_auth_token()
        self.gh_repo_str: Optional[str] = None
        self.gh = github.Github(gh_auth_token()) if gh_token else None
        self._note: Optional[dict] = None

    def _require_note(self) -> dict:
        """Load note and raise ClickException if missing."""

        if self._note is not None:
            return self._note

        self._note = self.load_note()

        if not self._note:
            raise click.ClickException(
                "No note found. Run 'mergai context init' first."
            )

        return self._note

    def _require_merge_info(self) -> MergeInfo:
        """Return merge info from note or raise ClickException if missing."""

        self._require_note()

        if "merge_info" not in self._note:
            raise click.ClickException(
                "No merge info found in note. Run 'mergai context init' first."
            )

        return MergeInfo.from_dict(self._note["merge_info"], self.repo)

    @property
    def branches(self) -> BranchNameBuilder:
        merge_info = self.merge_info
        try:
            return util.BranchNameBuilder.from_config(
                self.config.branch,
                merge_info.target_branch,
                merge_info.merge_commit_sha,
                merge_info.target_branch_sha,
            )
        except ValueError as e:
            raise click.ClickException(f"Invalid branch name format in config: {e}")

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
        return "solutions" in self._note

    # TODO: refactor

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

    def get_merge_info(
        self,
        commit: Optional[str] = None,
        target: Optional[str] = None,
    ) -> MergeInfo:
        """Get merge info from note.json or current branch.

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
            MergeInfo with target_branch, target_branch_sha, and merge_commit_sha.

        Raises:
            click.ClickException: If info cannot be determined or conflicts.
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
                raise click.ClickException(
                    f"Cannot resolve commit from branch name '{branch_commit}': {e}"
                )
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
            # Uses fallback to origin/ for remote-only branches (common in CI)
            try:
                final_target_sha = git_utils.resolve_ref_sha(self.repo, final_target)
            except ValueError as e:
                raise click.ClickException(str(e))

        return MergeInfo(
            target_branch=final_target,
            target_branch_sha=final_target_sha,
            merge_commit_sha=final_commit,
            _repo=self.repo,
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
        """Return all commits with notes from HEAD until the merge commit (from merge_info).

        Each item is (commit, note) where note is the parsed note dict or None if the
        commit has no mergai note. Stops when the merge commit (merge_info["merge_commit"])
        is reached, inclusive.
        """
        merge_commit_sha = None
        note = self.load_note()
        if note and "merge_info" in note:
            try:
                merge_commit_sha = self.repo.commit(
                    note["merge_info"]["merge_commit"]
                ).hexsha
            except Exception:
                pass

        notes = []
        for commit in self.repo.iter_commits():
            note_str = git_utils.read_commit_note(self.repo, "mergai", commit.hexsha)
            if note_str is not None:
                try:
                    note = json.loads(note_str)
                except json.JSONDecodeError:
                    note = None
            else:
                note = None

            if note and "merge_info" in note and merge_commit_sha is None:
                try:
                    merge_commit_sha = self.repo.commit(
                        note["merge_info"]["merge_commit"]
                    ).hexsha
                except Exception:
                    pass

            notes.append((commit, note))

            if merge_commit_sha and commit.hexsha == merge_commit_sha:
                break

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

        result = note.copy()
        prompt_config = self.config.context.to_prompt_serialization_config()

        # Hydrate conflict_context if present
        if "conflict_context" in result:
            ctx = ConflictContext.from_dict(result["conflict_context"], self.repo)
            result["conflict_context"] = ctx.to_dict(prompt_config)

        # Hydrate merge_context if present
        if "merge_context" in result:
            ctx = MergeContext.from_dict(result["merge_context"], self.repo)
            result["merge_context"] = ctx.to_dict(prompt_config)

        return result

    def build_describe_prompt(self, current_note: dict) -> str:
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

    def resolve(self, force: bool, yolo: bool, max_attempts: int = None):
        if max_attempts is None:
            max_attempts = self.config.resolve.max_attempts

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
            git_note = self.read_note(commit.hexsha)
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
