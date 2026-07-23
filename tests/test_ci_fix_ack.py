"""Tests for `mergai ci fix --ack` (A3).

The ack is a one-line trigger acknowledgement posted to the PR, summarising
how many failing checks were found / fixed - posted even when zero, so a
comment-triggered run always gives feedback.
"""

from types import SimpleNamespace

import mergai.commands.ci as ci
from mergai.config import MergaiConfig


class _FakePull:
    def __init__(self, recorder):
        self._recorder = recorder

    def create_issue_comment(self, body):
        self._recorder.append(body)
        return SimpleNamespace(html_url="https://example/c/1")


class _FakeRepo:
    def __init__(self):
        self.posted: list[str] = []
        self.pull_numbers: list[int] = []

    def get_pull(self, number):
        self.pull_numbers.append(number)
        return _FakePull(self.posted)


class _FakeApp:
    def __init__(self):
        self.gh_repo = _FakeRepo()
        self.config = MergaiConfig()


# --- _post_ci_ack ---------------------------------------------------------


def test_ack_zero_found_posts_nothing_to_address():
    app = _FakeApp()
    ci._post_ci_ack(app, pr_override=7, found=0, fixed=0)
    assert app.gh_repo.posted == ["mergai ci fix: no failing checks to address."]
    assert app.gh_repo.pull_numbers == [7]


def test_ack_reports_fixed_of_found():
    app = _FakeApp()
    ci._post_ci_ack(app, pr_override=7, found=2, fixed=1)
    assert app.gh_repo.posted == ["mergai ci fix: fixed 1 of 2 failing check(s)."]


def test_ack_resolves_pr_from_branch_when_no_override(monkeypatch):
    app = _FakeApp()
    monkeypatch.setattr(
        ci, "get_prs_for_current_branch", lambda a: [SimpleNamespace(number=42)]
    )
    ci._post_ci_ack(app, pr_override=None, found=0, fixed=0)
    assert app.gh_repo.pull_numbers == [42]


def test_ack_skips_when_branch_pr_ambiguous(monkeypatch):
    app = _FakeApp()
    monkeypatch.setattr(
        ci,
        "get_prs_for_current_branch",
        lambda a: [SimpleNamespace(number=1), SimpleNamespace(number=2)],
    )
    ci._post_ci_ack(app, pr_override=None, found=1, fixed=1)
    assert app.gh_repo.posted == []


# --- fix --ack with zero actionable runs ----------------------------------


def test_fix_all_ack_zero_runs_prints_count_and_posts(monkeypatch, capsys):
    app = _FakeApp()
    monkeypatch.setattr(ci, "_resolve_target_runs", lambda a, t, force=False: [])
    monkeypatch.setattr(
        ci, "get_prs_for_current_branch", lambda a: [SimpleNamespace(number=9)]
    )

    ci.fix.callback.__wrapped__(
        app,
        target="all",
        workflow=None,
        pr=None,
        artifacts_dir=None,
        force=False,
        ack=True,
    )

    out = capsys.readouterr().out
    assert "0 unprocessed actionable run(s)" in out
    assert app.gh_repo.posted == ["mergai ci fix: no failing checks to address."]
