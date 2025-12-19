import git
import subprocess
import click
from . import git_utils
from . import util
from .prompts import load_system_prompt
from .agents.factory import create_agent
from .state_store import StateStore
import github
from github import Repository as GithubRepository
import json
from typing import Optional, Tuple
import tempfile
from pathlib import Path
from .agents.base import Agent


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
        if show_solution and "solution" in note:
            output_str += util.conflict_solution_to_markdown(note["solution"]) + "\n"

        return output_str + "\n"
    return str(note)


# TODO: refactor
class AppContext:
    def __init__(self):
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
        system_prompt = load_system_prompt()
        project_invariants = util.load_if_exists(".mergai/invariants.md")

        prompt = system_prompt + "\n\n"
        if project_invariants:
            prompt += project_invariants + "\n\n"

        prompt += f"# Current Note\n"
        prompt += convert_note(current_note, "markdown")

        if use_history:
            notes = self.get_notes()
            for idx, (_, note) in enumerate(notes):
                prompt += f"# History Note {idx + 1}\n"
                prompt += (
                    convert_note(
                        note,
                        format="markdown",
                    )
                    + "\n\n"
                )

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

    def get_agent(self, agent_desc: str = "gemini-cli", yolo: bool = False) -> "Agent":
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
        self, force: bool, use_history: bool, yolo: bool, max_attempts: int = 3
    ):
        note = self.load_note()
        if note is None:
            raise Exception("No note found. Please prepare the context first.")

        if "solution" in note and not force:
            raise Exception(
                "Solution already exists in the note. Use -f/--force to overwrite."
            )

        if "solution" in note:
            del note["solution"]

        prompt = self.build_prompt(note, use_history=use_history)

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

    def commit(self):
        if not self.state.note_exists():
            raise Exception("No note found.")

        note = self.state.load_note()
        if "solution" not in note:
            raise Exception("No solution found in the note.")

        if not self.repo.is_dirty():
            raise Exception("No changes to commit in the repository.")

        solution = note["solution"]
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

        self.repo.index.commit(
            "MergaAI: merge conflict solution\n\n" + solution["response"]["summary"]
        )

        commit_sha = self.repo.head.commit.hexsha
        self.repo.git.notes(
            "--ref", "mergai", "add", "-f", "-F", self.state.note_path(), commit_sha
        )
        self.repo.git.notes(
            "--ref",
            "mergai-marker",
            "add",
            "-f",
            "-m",
            "MergaAI note available, use `mergai show <commit>` to view it.",
            commit_sha,
        )

        self.drop_all()

    def commit_conflict(self):
        hint_msg = "Please prepare the conflict context by running:\nmergai create-conflict-context"
        note = self.load_note()
        if note is None:
            raise Exception(f"No note found.\n\n{hint_msg}")

        context = note.get("conflict_context")
        if context is None:
            raise Exception(f"No conflict context found in the note.\n\n{hint_msg}")

        ours_sha = context["ours_commit"]["short_sha"]
        theirs_sha = context["theirs_commit"]["short_sha"]
        message = f"MergeAI: baseline for merging {theirs_sha} to {ours_sha}"

        for path in context["files"]:
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
