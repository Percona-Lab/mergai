import git
import subprocess
import click
from . import git_utils
from . import util
from . import prompts
from .agents.factory import create_agent
from .state_store import StateStore
from .config import MergaiConfig
import github
from github import Repository as GithubRepository
import json
from typing import Optional, Tuple
from dataclasses import dataclass
import tempfile
from pathlib import Path
from .agents.base import Agent


# TODO: Make this configurable via settings/config file
MERGAI_COMMIT_FOOTER = "Note: commit created by mergai"


@dataclass
class MergeContext:
    """Context information for a merge operation.

    Attributes:
        target_branch: The branch being merged into (e.g., "v8.0", "master").
        merge_commit_short: Short SHA of the commit being merged (11 chars).
    """

    target_branch: str
    merge_commit_short: str


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
) -> str:
    if format == "json":
        return json.dumps(note, indent=2 if pretty else None) + "\n"
    elif format == "markdown":
        output_str = ""
        if show_context and "conflict_context" in note:
            output_str += (
                util.conflict_context_to_markdown(note["conflict_context"]) + "\n"
            )
        if show_pr_comments and "pr_comments" in note:
            output_str += util.pr_comments_to_markdown(note["pr_comments"]) + "\n"
        if show_user_comment and "user_comment" in note:
            output_str += util.user_comment_to_markdown(note["user_comment"]) + "\n"
        if show_solution and "solution" in note:
            output_str += util.conflict_solution_to_markdown(note["solution"]) + "\n"

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

        Resolves target_branch and merge_commit_short using the following
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
            MergeContext with target_branch and merge_commit_short.

        Raises:
            click.ClickException: If context cannot be determined or conflicts.
        """
        # Get values from note.json if merge_info exists
        note = self.load_note()
        note_target = None
        note_commit = None
        if note and "merge_info" in note:
            merge_info = note["merge_info"]
            note_target = merge_info.get("target_branch")
            note_commit = merge_info.get("merge_commit_short")

        # Get values from current branch name
        current_branch = git_utils.get_current_branch(self.repo)
        parsed = util.BranchNameBuilder.parse_branch_name_with_config(
            current_branch, self.config.branch
        )
        branch_target = parsed.target_branch if parsed else None
        branch_commit = parsed.merge_commit_short if parsed else None

        # Check for conflicts between note and branch (when both exist)
        if note_target and branch_target and note_target != branch_target:
            raise click.ClickException(
                f"Conflict: target_branch in note.json ({note_target}) differs from "
                f"current branch ({branch_target}). Use --target to specify explicitly."
            )
        if note_commit and branch_commit and note_commit != branch_commit:
            raise click.ClickException(
                f"Conflict: merge_commit_short in note.json ({note_commit}) differs from "
                f"current branch ({branch_commit}). Use COMMIT argument to specify explicitly."
            )

        # Resolve final values: CLI args > note.json > branch > error
        # Resolve merge_commit_short
        if commit is not None:
            # CLI-provided commit - resolve to short SHA
            try:
                resolved = self.repo.commit(commit)
                final_commit = git_utils.short_sha(resolved.hexsha)
            except Exception as e:
                raise click.ClickException(f"Invalid commit reference '{commit}': {e}")
        elif note_commit:
            final_commit = note_commit
        elif branch_commit:
            final_commit = branch_commit
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

        return MergeContext(
            target_branch=final_target,
            merge_commit_short=final_commit,
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

    def build_prompt(self, current_note: dict, use_history: bool) -> str:
        if use_history:
            raise Exception("use_history in build_prompt is not supported yet.")

        system_prompt = prompts.load_system_prompt()
        project_invariants = util.load_if_exists(".mergai/invariants.md")

        prompt = system_prompt + "\n\n"
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

    def drop_solution(self):
        self.drop_note_field("solution")

    def drop_pr_comments(self):
        self.drop_note_field("pr_comments")

    def drop_conflict_context(self):
        self.drop_note_field("conflict_context")

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

        if "solution" in note and not force:
            raise Exception(
                "Solution already exists in the note. Use -f/--force to overwrite."
            )

        if "solution" in note:
            del note["solution"]

        # TODO: implement use_history=True
        prompt = self.build_prompt(note, use_history=False)

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

        note["solution"] = solution
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

    def commit_solution(self):
        if not self.state.note_exists():
            raise Exception("No note found.")

        note = self.state.load_note()
        if "solution" not in note:
            raise Exception("No solution found in the note.")

        if not self.repo.is_dirty():
            raise Exception("No changes to commit in the repository.")

        solution = note["solution"]
        conflict_context = note.get("conflict_context")

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
            message = f"Merge commit '{theirs_sha}' into {target_branch}\n\n"
        else:
            message = "Merge conflict solution\n\n"

        # Add summary from AI response
        summary = solution["response"].get("summary", "")
        if summary:
            message += f"{summary}\n\n"

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

        self.add_note(self.repo.head.commit.hexsha)
        self.drop_all()

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
