"""Tests for the ``mergai pr comment`` command, focused on ``--pr-number``.

``--pr-number`` targets a PR by number regardless of state (open/closed/merged),
so a fast-forward that already merged the PR can still be commented on -- the
default open-PR lookup can no longer resolve it. Also covers the guard against
passing both PR_TYPE and ``--pr-number`` and ``--allow-missing`` on a 404. No
network: ``gh_repo`` is stubbed.
"""

from types import SimpleNamespace

from click.testing import CliRunner
from github import GithubException

from mergai.commands.pr import pr

BRANCH = "mergai/master-deadbeef000/main"


class _FakePR:
    def __init__(self, number, *, head_ref=BRANCH, state="open"):
        self.number = number
        self.state = state
        self.html_url = f"https://github.com/o/r/pull/{number}"
        self.head = SimpleNamespace(ref=head_ref, sha=f"sha{number}")
        self.comments = []

    def create_issue_comment(self, body):
        self.comments.append(body)
        return SimpleNamespace(html_url=f"{self.html_url}#issuecomment-1")


class _FakeRepo:
    full_name = "o/r"

    def __init__(self, *, pull=None, pulls=None, raise_status=None):
        self._pull = pull
        self._pulls = pulls or []
        self._raise_status = raise_status
        self.got_pull = None

    def get_pull(self, number):
        self.got_pull = number
        if self._raise_status is not None:
            raise GithubException(self._raise_status, {"message": "Not Found"}, None)
        return self._pull

    def get_pulls(self, state="open", sort=None, head=None):  # noqa: A002
        return [p for p in self._pulls if p.state == state]


def _app(repo, *, run_link=False):
    return SimpleNamespace(
        gh_repo=repo,
        config=SimpleNamespace(run_link=SimpleNamespace(enabled=run_link)),
        branches=SimpleNamespace(
            get_branch_name=lambda t: f"mergai/master-deadbeef000/{t}"
        ),
    )


def _run(app, args):
    return CliRunner().invoke(pr, ["--repo", "o/r", "comment", *args], obj=app)


def test_pr_number_comments_on_closed_pr():
    merged = _FakePR(42, state="closed")
    repo = _FakeRepo(pull=merged)
    res = _run(_app(repo), ["--pr-number", "42", "--body", "Merged."])
    assert res.exit_code == 0, res.output
    assert repo.got_pull == 42
    assert merged.comments == ["Merged."]
    assert "Commented on PR #42" in res.output


def test_pr_number_and_type_are_mutually_exclusive():
    repo = _FakeRepo(pull=_FakePR(1))
    res = _run(_app(repo), ["main", "--pr-number", "1", "--body", "x"])
    assert res.exit_code != 0
    # Click writes UsageError to stderr; assert there (repo convention).
    assert "not both" in res.stderr
    # Nothing was posted.
    assert repo.got_pull is None


def test_pr_number_missing_without_allow_missing_fails():
    repo = _FakeRepo(raise_status=404)
    res = _run(_app(repo), ["--pr-number", "99", "--body", "x"])
    assert res.exit_code != 0
    assert "Failed to fetch PR #99" in res.stderr


def test_pr_number_missing_with_allow_missing_is_noop():
    repo = _FakeRepo(raise_status=404)
    res = _run(_app(repo), ["--pr-number", "99", "--body", "x", "--allow-missing"])
    assert res.exit_code == 0, res.output
    # The warning is written to stderr.
    assert "Skipping comment" in res.stderr


def test_open_pr_path_still_resolves_by_type():
    open_pr = _FakePR(7, state="open")
    repo = _FakeRepo(pulls=[open_pr])
    res = _run(_app(repo), ["main", "--body", "hi"])
    assert res.exit_code == 0, res.output
    assert open_pr.comments == ["hi"]
    # Resolved via the open-PR lookup, not get_pull.
    assert repo.got_pull is None
