import click
from ..app import AppContext
from github import Github
from github import PullRequest as GithubPullRequest
from github import IssueComment as GithubIssueComment
from github import PullRequestComment as GithubPullRequestComment
from .. import git_utils
from .. import util
from typing import Optional, List, Iterable


@click.group()
@click.pass_obj
@click.option(
    "--repo",
    "repo",
    type=str,
    required=False,
    envvar="GH_REPO",
    help="The repository where the PR is located.",
)
def pr(app: AppContext, repo: Optional[str]):
    if repo is None:
        raise click.ClickException(
            "GitHub repository not set. Use --repo or set GH_REPO environment variable."
        )

    app.gh_repo_str = repo
    pass


@pr.command()
@click.pass_obj
@click.option(
    "--base", "base", type=str, required=True, help="The base branch for the PR."
)
def create(app: AppContext, base: str):
    try:
        note = app.read_note("HEAD")
        if not note:
            raise Exception("No note found for HEAD.")

        context = note.get("conflict_context")
        if not context:
            raise Exception("No conflict context found in the note.")

        solution = note.get("solution")
        if not solution:
            raise Exception("No solution found in the note.")

        body = util.conflict_solution_to_str(solution, format="markdown")
        head = git_utils.get_current_branch(app.repo)
        gh_repo = app.get_gh_repo()
        title = f"MergAI: Automated Merge Conflict Resolution [{context["theirs_commit"]["short_sha"]} into {context["ours_commit"]["short_sha"]}]"
        click.echo(
            f"Creating PR:\nrepo:  {gh_repo.full_name}\nfrom:  {head}\nto:    {base}\ntitle: {title}"
        )
        pr = gh_repo.create_pull(
            title=title,
            body=body,
            head=head,
            base=base,
        )
        print(f"PR created: {pr.html_url}")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


def get_prs_for_current_branch(app: AppContext) -> List[GithubPullRequest.PullRequest]:
    gh_repo = app.get_gh_repo()
    pulls = gh_repo.get_pulls(
        sort="created", head=git_utils.get_current_branch(app.repo)
    )
    return list(
        filter(lambda pr: pr.head.ref == git_utils.get_current_branch(app.repo), pulls)
    )


def get_pr_by_number(app: AppContext, pr_number: int) -> GithubPullRequest.PullRequest:
    gh_repo = app.get_gh_repo()
    pr = gh_repo.get_pull(pr_number)
    return pr


def show_prs(prs):
    for pr in prs:
        click.echo(f"#{pr.number}: ({pr.html_url})")
        click.echo(f"  Title      : {pr.title}")
        click.echo(f"  Created at : {pr.created_at}")
        click.echo(f"  Author     : {pr.user.login}")
        click.echo(f"  Head       : {pr.head.ref}")
        click.echo(f"  Base       : {pr.base.ref}")
        click.echo(f"  State      : {pr.state}")


@pr.command()
@click.pass_obj
def show(app: AppContext):
    try:
        prs = get_prs_for_current_branch(app)
        if len(prs) == 0:
            click.echo("No open pull requests found for the current branch.")
            exit(0)

        if len(prs) > 1:
            click.echo("Multiple open pull requests found for the current branch:")

        else:
            show_prs(prs)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


def handle_pr_number(app: AppContext, pr_number: Optional[int]):
    if pr_number is None:
        prs = get_prs_for_current_branch(app)
        if len(prs) == 0:
            click.echo("No open pull requests found for the current branch.")
            exit(0)
        if len(prs) > 1:
            click.echo(
                "Multiple open pull requests found for the current branch. Please specify a PR number using --pr-number."
            )
            show_prs(prs)
            exit(1)
        pr_number = prs[0].number
    return pr_number


def filter_comments_by_feedback(
    comments: Iterable[
        GithubIssueComment.IssueComment | GithubPullRequestComment.PullRequestComment
    ],
    all: bool,
) -> List[
    GithubIssueComment.IssueComment | GithubPullRequestComment.PullRequestComment
]:
    filtered_comments = []
    for comment in comments:
        feedback = comment.reactions.get("+1", 0) - comment.reactions.get("-1", 0)
        if all or feedback > 0:
            filtered_comments.append(comment)
    return filtered_comments


@pr.command()
@click.pass_obj
@click.option(
    "--pr-number",
    "pr_number",
    type=int,
    required=False,
    help="The pull request number to show comments for.",
)
@click.option(
    "--all",
    "all",
    is_flag=True,
    default=False,
    help="Show comments for all PRs on the current branch. Ignore the reactions.",
)
def show_comments(app: AppContext, pr_number: Optional[int], all: bool):
    try:
        pr_number = handle_pr_number(app, pr_number)
        pr = get_pr_by_number(app, pr_number)
        issue_comments = filter_comments_by_feedback(pr.get_issue_comments(), all)
        for comment in issue_comments:
            click.echo("------------------------------")
            click.echo(f"Comment by {comment.user.login} at {comment.created_at}:")
            click.echo(f"             ID: {comment.id}")
            click.echo(
                f"+1/-1 Reactions: {comment.reactions['+1']}/{comment.reactions['-1']}"
            )
            click.echo(f"      Body     :\n{comment.body}")
            click.echo("------------------------------\n")

        comments = filter_comments_by_feedback(pr.get_comments(), all)
        # TODO: there is no "resolved" state for comments in GitHub API
        for comment in comments:
            line_str = (
                comment.line
                if comment.start_line is None
                else f"{comment.start_line}-{comment.line}"
            )
            click.echo("------------------------------")
            click.echo(f"Comment by {comment.user.login} at {comment.created_at}:")
            click.echo(f"             ID: {comment.id}")
            click.echo(f"      Commit ID: {comment.commit_id}")
            click.echo(f"      Review ID: {comment.pull_request_review_id}")
            click.echo(
                f"+1/-1 Reactions: {comment.reactions['+1']}/{comment.reactions['-1']}"
            )
            click.echo(f"      Path     : {comment.path}:{line_str}")
            click.echo(f"      Body     :\n{comment.body}")
            click.echo("------------------------------\n")
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)


@pr.command()
@click.pass_obj
@click.option(
    "--pr-number",
    "pr_number",
    type=int,
    required=False,
    help="The pull request number to use comments from.",
)
@click.option(
    "-f/--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing saved conflict context.",
)
def add_comments_to_context(app: AppContext, pr_number: Optional[int], force: bool):
    pr_number = handle_pr_number(app, pr_number)
    pr = get_pr_by_number(app, pr_number)
    comments = filter_comments_by_feedback(pr.get_comments(), all=False)
    issue_comments = filter_comments_by_feedback(pr.get_issue_comments(), all=False)

    note = app.load_or_create_note()

    pr_comments = {}
    added_count = 0
    for comment in issue_comments:
        pr_comments[comment.id] = {
            "user": comment.user.login,
            "created_at": comment.created_at.isoformat(),
            "body": comment.body,
        }
        added_count += 1

    for comment in comments:
        pr_comments[comment.id] = {
            "commit_id": comment.commit_id,
            "user": comment.user.login,
            "created_at": comment.created_at.isoformat(),
            "body": comment.body,
            "path": comment.path,
            "line": comment.line,
            "start_line": comment.start_line,
            "line_str": (
                comment.line
                if comment.start_line is None
                else f"{comment.start_line}-{comment.line}"
            ),
            "body": comment.body,
        }

        added_count += 1

    if added_count == 0:
        click.echo("No comments with positive feedback found to add to the context.")
        exit(0)

    note = app.load_or_create_note()
    if "pr_comments" in note and not force:
        click.echo(
            "PR comments already exist in the note. Use -f/--force to overwrite."
        )
        exit(1)

    note["pr_comments"] = pr_comments

    app.state.save_note(note)


@pr.command()
@click.pass_obj
@click.option(
    "--pr-number",
    "pr_number",
    type=int,
    required=False,
    help="The pull request number to use comments from.",
)
@click.argument(
    "commit",
    type=str,
    required=False,
    default="HEAD",
)
def add_comment_with_solution(app: AppContext, pr_number: Optional[int], commit: str):
    pr_number = handle_pr_number(app, pr_number)
    pr = get_pr_by_number(app, pr_number)

    note = app.read_note(commit)
    if not note:
        raise Exception(f"No note found for commit {commit}.")

    solution = note.get("solution")
    if not solution:
        click.echo(
            f"No solution found in the note for commit {commit} to add as a comment."
        )
        exit(1)

    body = util.conflict_solution_to_str(solution, format="markdown")

    pr.create_issue_comment(body=body)


@pr.command()
@click.pass_obj
def list(app: AppContext):
    try:
        gh_repo = app.get_gh_repo()

        click.echo(
            f"WARNING: Filtering PRs by branch name. This may be unreliable (but there is TODO)."
        )

        # TODO: this should check if a branch has commits with mergai notes
        def filter_pr(pr):
            return pr.head.ref.startswith("mergai/")

        pulls = filter(filter_pr, gh_repo.get_pulls())

        for pr in pulls:
            click.echo(f"#{pr.number}: [{pr.created_at}] title: {pr.title}")
        # show_prs(pulls)
    except Exception as e:
        click.echo(f"Error: {e}")
        exit(1)
