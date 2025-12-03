import click
import git
from pathlib import Path

from .. import git_utils
from ..app import AppContext
from .conflict_context import conflict_context_flags
import subprocess
from typing import List
import yaml
from .. import util


class ReplayApp:
    BASE_DIR = Path(".mergai_state") / "replay"

    @classmethod
    def scan(cls) -> List[str]:
        if not cls.BASE_DIR.exists():
            return []

        commits = []
        for entry in cls.BASE_DIR.iterdir():
            if entry.is_dir():
                try:
                    commits.append(entry.name)
                except git.exc.BadName:
                    pass
        return commits

    def __init__(self, app: AppContext, commit: str):
        self.app = app
        self.commit = self.app.repo.commit(commit)
        self.replay_dir = self.BASE_DIR / self.commit.hexsha

    def dir_exists(self) -> bool:
        return self.replay_dir.exists()

    def check_or_create_replay_dir(self, force: bool) -> Path:
        if self.replay_dir.exists() and not force:
            click.echo(
                f"Replay directory '{self.replay_dir}' already exists. Use -f/--force to overwrite."
            )
            exit(1)

        self.replay_dir.mkdir(parents=True, exist_ok=True)

    def write_to_dir(self, filename: str, content: str):
        file_path = self.replay_dir / filename
        with open(file_path, "w") as f:
            f.write(content)

    def exec_to_dir(self, filename: str, *args) -> str:
        subprocess.run(
            args,
            stdout=open(self.replay_dir / filename, "w"),
            stderr=subprocess.PIPE,
            check=True,
        )

    def get_dir(self, filename) -> Path:
        return self.replay_dir / filename

    def read_from_dir(self, filename: str) -> str:
        file_path = self.replay_dir / filename
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            return f.read().strip()

    def remove_from_dir(self, filename: str):
        file_path = self.replay_dir / filename
        if file_path.exists():
            file_path.unlink()

    def remove_dir(self):
        if self.replay_dir.exists():
            self.replay_dir.rmdir()

    def replay_merge(self, commit: git, branch):
        commit = self.app.repo.commit(commit)
        parent1 = commit.parents[0]
        parent2 = commit.parents[1]

        short_parent1_sha = git_utils.short_sha(parent1.hexsha)
        short_parent2_sha = git_utils.short_sha(parent2.hexsha)

        try:
            if branch is False:
                self.app.repo.git.checkout(parent1)
            else:
                if branch is True:
                    branch = f"mergai/merging-{short_parent2_sha}-to-{short_parent1_sha}/baseline"
                self.app.repo.git.checkout("-b", branch, parent1.hexsha)
        except git.GitCommandError as e:
            click.echo(e)
            exit(1)

        try:
            self.app.repo.git.merge(parent2.hexsha)
        except git.GitCommandError as e:
            pass


def commit_and_create_solution_branch(app: AppContext):
    note = app.load_note()
    context = note.get("conflict_context", None)
    if context is None:
        raise Exception("No conflict context found in the note.")

    short_parent1_sha = context["ours_commit"]["short_sha"]
    short_parent2_sha = context["theirs_commit"]["short_sha"]
    message = (
        f"MergeAI: baseline for merging {short_parent2_sha} to {short_parent1_sha}"
    )

    for path in context["files"]:
        click.echo(f"Adding file '{path}' to index")
        app.repo.git.add(path)

    click.echo(f"Creating baseline commit with message: '{message}'")
    app.repo.git.commit(
        "-m",
        message,
    )

    branch = f"mergai/merging-{short_parent2_sha}-to-{short_parent1_sha}/solution"

    click.echo(f"Creating solution branch '{branch}'")
    app.repo.git.checkout("-b", branch)


def check_commit_is_merge_with_conflicts(repo: git.Repo, commit_sha: str):
    if not git_utils.is_valid_commit(repo, commit_sha):
        click.echo(f"Ref '{commit_sha}' is not valid.")
        exit(1)

    commit = repo.commit(commit_sha)
    if not git_utils.is_merge_commit(commit):
        click.echo(f"Commit {commit.hexsha} is not a merge commit.")
        exit(1)

    if not git_utils.commit_has_conflicts(repo, commit):
        click.echo(f"Commit {commit.hexsha} does not have merge conflicts.")
        exit(1)


@click.group()
def replay():
    pass


@replay.command()
@click.pass_obj
@click.argument("revision", default="HEAD")
@click.option(
    "--max-count",
    default=0,
    help="Maximum number of merge commits to retrieve. 0 means no limit.",
)
def log(app: AppContext, revision, max_count):
    for commit in git_utils.get_merge_conflicts(app.repo, revision, max_count):
        replay = ReplayApp(app, commit.hexsha)
        commit_line = ""
        if replay.dir_exists():
            baseline_branch = replay.read_from_dir("branch_base.txt")
            solution_branch = replay.read_from_dir("branch_solution.txt")
            original_branch = replay.read_from_dir("branch_original.txt")
            commit_line = f"(stored: base='{baseline_branch}', solution='{solution_branch}', original='{original_branch}')"
        click.echo(f"{commit.hexsha} {commit_line}")


@replay.command()
@click.pass_obj
@click.argument("commit")
@click.option(
    "-b/--branch",
    "branch",
    is_flag=True,
    flag_value=True,
    default=False,
    help="Create a new branch at the merge replay point.",
)
def merge(app: AppContext, commit, branch):
    check_commit_is_merge_with_conflicts(app.repo, commit)
    replay = ReplayApp(app, commit)
    replay.replay_merge(commit, branch)


@replay.command()
@click.pass_obj
@click.argument("commit")
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing results.",
)
@conflict_context_flags
def resolve(
    app: AppContext,
    commit: str,
    use_diffs: bool,
    diff_lines_of_context: int,
    use_compressed_diffs: bool,
    use_their_commits: bool,
    force: bool,
):
    click.echo(f"Checking if commit '{commit}' is a merge with conflicts...")
    check_commit_is_merge_with_conflicts(app.repo, commit)

    commit = app.repo.commit(commit)

    replay = ReplayApp(app, commit)

    replay.check_or_create_replay_dir(force)

    click.echo("Dropping all existing mergai state...")
    app.drop_all()

    replay.write_to_dir("branch_prev.txt", git_utils.get_current_branch(app.repo))

    replay.replay_merge(commit, branch=True)
    base_branch = git_utils.get_current_branch(app.repo)
    replay.write_to_dir("branch_base.txt", base_branch)

    click.echo("Creating conflict context...")
    context = app.create_conflict_context(
        use_diffs,
        diff_lines_of_context,
        use_compressed_diffs,
        use_their_commits,
        force,
    )

    short_ours_sha = context["ours_commit"]["short_sha"]
    short_theirs_sha = context["theirs_commit"]["short_sha"]

    click.echo("Creating solution branch...")
    commit_and_create_solution_branch(app)
    solution_branch = git_utils.get_current_branch(app.repo)
    replay.write_to_dir("branch_solution.txt", solution_branch)

    click.echo("Creating original solution branch...")
    original_branch = f"mergai/merging-{short_theirs_sha}-to-{short_ours_sha}/original"
    replay.write_to_dir("branch_original.txt", original_branch)
    app.repo.git.checkout("-b", original_branch, base_branch)
    app.repo.git.checkout(commit.hexsha, "--", ".")
    app.repo.git.commit(
        "-m",
        f"MergeAI: original solution for merging {short_theirs_sha} to {short_ours_sha}",
    )
    app.repo.git.switch(solution_branch)

    click.echo("Storing prompt...")
    replay.exec_to_dir("prompt.md", "mergai", "prompt")

    click.echo("Resolving merge conflict...")
    app.resolve(force=True, use_history=False, yolo=True)

    app.commit()

    replay.exec_to_dir(
        "solution.md", "mergai", "show", "--solution", "--format", "markdown"
    )
    replay.exec_to_dir(
        "solution.json", "mergai", "show", "--solution", "--format", "json", "--pretty"
    )
    replay.exec_to_dir(
        "context.json", "mergai", "show", "--context", "--format", "json", "--pretty"
    )


@replay.command()
@click.pass_obj
@click.argument("commit")
def drop(
    app: AppContext,
    commit: str,
):
    if git_utils.is_merge_in_progress(app.repo):
        click.echo("Cannot drop replay results while a merge is in progress.")
        exit(1)

    check_commit_is_merge_with_conflicts(app.repo, commit)
    replay = ReplayApp(app, commit)

    if replay.dir_exists() is False:
        click.echo(f"Replay directory '{replay.replay_dir}' does not exist.")
        exit(0)

    branch_prev = replay.read_from_dir("branch_prev.txt")
    if branch_prev:
        click.echo(f"Checking out previous branch '{branch_prev}'")
        app.repo.git.checkout(branch_prev)
        replay.remove_from_dir("branch_prev.txt")

    branch_base = replay.read_from_dir("branch_base.txt")
    branch_solution = replay.read_from_dir("branch_solution.txt")
    current_branch = git_utils.get_current_branch(app.repo)
    branch_original = replay.read_from_dir("branch_original.txt")

    if current_branch == branch_base:
        click.echo(
            f"Checked out on base branch '{branch_base}'. Please switch to another branch to drop the replay results"
        )
        exit(1)

    if current_branch == branch_solution:
        click.echo(
            f"Checked out on solution branch '{branch_solution}'. Please switch to another branch to drop the replay results"
        )
        exit(1)

    if current_branch == branch_original:
        click.echo(
            f"Checked out on original solution branch '{branch_original}'. Please switch to another branch to drop the replay results"
        )
        exit(1)

    click.echo(f"Removing base branch '{branch_base}'")
    git_utils.remove_branch_if_exists(app.repo, branch_base)
    replay.remove_from_dir("branch_base.txt")

    click.echo(f"Removing solution branch '{branch_solution}'")
    git_utils.remove_branch_if_exists(app.repo, branch_solution)
    replay.remove_from_dir("branch_solution.txt")

    click.echo(f"Removing original solution branch '{branch_original}'")
    git_utils.remove_branch_if_exists(app.repo, branch_original)
    replay.remove_from_dir("branch_original.txt")

    app.drop_all()

    click.echo(f"Removing replay directory '{replay.replay_dir}'")

    replay.remove_from_dir("solution.md")
    replay.remove_from_dir("solution.json")
    replay.remove_from_dir("prompt.md")
    replay.remove_from_dir("context.json")
    replay.remove_dir()


def replay_status_to_markdown_table(replay_status: dict) -> str:
    use_keys = ["conflict", "build"]
    output_str = "\n"
    output_str += "| Commit | " + " | ".join(k.capitalize() for k in use_keys) + " |\n"
    output_str += "|--------|" + "|".join(["--------" for _ in use_keys]) + "|\n"

    for commit, status in replay_status.items():
        output_str += (
            f"| {commit} | "
            + " | ".join(status.get(k, "N/A") for k in use_keys)
            + " |\n"
        )
    return output_str


def get_replay_stats(replay_status: dict) -> dict:
    stats = {
        "total": len(replay_status),
        "conflict": {
            "success": 0,
            "failed": 0,
            "unknown": 0,
        },
        "build": {
            "success": 0,
            "failed": 0,
            "unknown": 0,
        },
    }

    for commit, status in replay_status.items():
        conflict_status = status.get("conflict", "N/A")
        if conflict_status == "success":
            stats["conflict"]["success"] += 1
        elif conflict_status == "unknown":
            stats["conflict"]["unknown"] += 1
        else:
            stats["conflict"]["failed"] += 1

        build_status = status.get("build", "N/A")
        if build_status == "success":
            stats["build"]["success"] += 1
        elif build_status == "unknown":
            stats["build"]["unknown"] += 1
        else:
            stats["build"]["failed"] += 1

    return stats


def replay_stats_to_markdown_table(replay_stats: dict, total: int) -> str:
    def get_number(count: int, total: int):
        if total == 0:
            return "0"
        percent = (count / total) * 100
        return f"{count} ({percent:.1f}%)"

    output_str = "\n"
    output_str += "| Metric | Success | Failed | Unknown |\n"
    output_str += "|--------|---------|--------|---------|\n"

    output_str += f"| Conflict Resolution | {get_number(replay_stats['conflict']['success'], total)} | {get_number(replay_stats['conflict']['failed'], total)} | {get_number(replay_stats['conflict']['unknown'], total)} |\n"
    output_str += f"| Build | {get_number(replay_stats['build']['success'], total)} | {get_number(replay_stats['build']['failed'], total)} | {get_number(replay_stats['build']['unknown'], total)} |\n"

    return output_str


def commits_to_replay_status_dict(app: AppContext, commits: List[str]) -> dict:
    replay_status = {}
    for commit in commits:
        replay = ReplayApp(app, commit)
        status_str = replay.read_from_dir("status.yml")
        if not status_str:
            click.echo(f"WARNING: No status found for commit '{commit}', skipping...")
            continue

        try:
            status = yaml.safe_load(status_str)
        except yaml.YAMLError as e:
            click.echo(
                f"ERROR: parsing status YAML for commit '{commit}' in '{replay.get_dir('status.yml')}': {e}"
            )
            continue

        replay_status[commit] = status
    return replay_status


@replay.command()
@click.pass_obj
def summary(
    app: AppContext,
):

    commits = ReplayApp.scan()
    if not commits:
        click.echo("No replayed merge commits found.")
        exit(0)

    replay_status = commits_to_replay_status_dict(app, commits)
    replay_stats = get_replay_stats(replay_status)

    output_str = "# Replay Summary\n\n"

    output_str += "## Stats\n\n"
    output_str += replay_stats_to_markdown_table(replay_stats, len(replay_status))
    output_str += f"Total replayed merge commits: {len(replay_status)}\n\n"
    output_str += replay_status_to_markdown_table(replay_status)

    output_str += "\n"

    output_str += "## Details\n"
    for commit, status in replay_status.items():
        replay = ReplayApp(app, commit)
        output_str += f"### Commit {commit}\n\n"
        output_str += f"- Conflict Resolution: {status.get('conflict', 'N/A')}\n"
        output_str += f"- Build: {status.get('build', 'N/A')}\n"
        output_str += f"- Baseline Branch: {replay.read_from_dir('branch_base.txt')}\n"
        output_str += (
            f"- Solution Branch: {replay.read_from_dir('branch_solution.txt')}\n"
        )
        output_str += (
            f"- Original Branch: {replay.read_from_dir('branch_original.txt')}\n"
        )
        output_str += f"- Notes: \n"
        for note in status.get("notes", []):
            output_str += f"  - {note}\n"

        output_str += "\n"

    util.print_or_page(output_str, format="markdown")
