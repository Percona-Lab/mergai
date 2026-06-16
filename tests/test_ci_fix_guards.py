"""Tests for `_fix_one_run` guards.

Two behaviours, both exercised by driving `_fix_one_run` with a stubbed
workflow-context builder and handler:

* an explicit run id that already has a recorded ci_comment (judged
  unfixable / already-resolved) is skipped without invoking the agent, and
* a commit failure rolls the just-added solution back out of the note so the
  run isn't wrongly treated as processed on a retry.
"""

import contextlib
from types import SimpleNamespace

import click
import pytest

import mergai.commands.ci as ci


class _FakeNote:
    def __init__(self, *, solution=None, comment=None):
        self._solution = solution
        self._comment = comment
        self.solutions: list[dict] = []
        self.dropped: list[set] = []

    def get_ci_solution_for_run(self, run_id):
        return self._solution

    def get_ci_comment_for_run(self, run_id):
        return self._comment

    def get_ci_solutions(self, workflow_name):
        return []

    def add_solution(self, solution):
        self.solutions.append(solution)
        return len(self.solutions) - 1

    def drop_solutions_at_indices(self, indices):
        self.dropped.append(set(indices))
        self.solutions = [s for i, s in enumerate(self.solutions) if i not in indices]
        return self


class _FakeApp:
    def __init__(self, note):
        self.note = note
        self.saved = 0
        self.repo = SimpleNamespace(
            head=SimpleNamespace(commit=SimpleNamespace(hexsha="abc1234def5"))
        )

    def save_note(self, note):
        self.saved += 1

    def commit_ci_fix_solution(self, idx):  # overridden per-test
        pass


def _patch_context(monkeypatch, *, max_attempts=3):
    context = SimpleNamespace(workflow_name="format", pr_number=7, summary="boom")
    config = SimpleNamespace(max_attempts=max_attempts)

    @contextlib.contextmanager
    def fake_builder(app, run_id, **kw):
        yield (context, config)

    monkeypatch.setattr(ci, "build_workflow_context_for_run", fake_builder)


def _run(app):
    return ci._fix_one_run(
        app,
        "99",
        workflow_override=None,
        pr_override=None,
        artifacts_dir_override=None,
        check_staleness=False,
    )


def test_skips_run_with_existing_ci_comment_without_running_agent(monkeypatch):
    note = _FakeNote(comment={"outcome": "unfixable", "run_id": "99"})
    app = _FakeApp(note)
    _patch_context(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("agent must not run for an already-decided run")

    monkeypatch.setattr(ci, "get_handler", boom)

    assert _run(app) == "skip"


def test_rolls_back_phantom_solution_when_commit_fails(monkeypatch):
    note = _FakeNote()  # no prior solution / comment
    app = _FakeApp(note)
    _patch_context(monkeypatch)

    agent_solution = {"response": {"resolved": {"a.py": "fix"}, "status": "fixed"}}
    monkeypatch.setattr(
        ci,
        "get_handler",
        lambda app, config: SimpleNamespace(execute=lambda ctx: agent_solution),
    )

    def fail_commit(idx):
        raise RuntimeError("git hook rejected the commit")

    app.commit_ci_fix_solution = fail_commit

    with pytest.raises(click.ClickException):
        _run(app)

    # The solution was added then rolled back, so a retry isn't blocked.
    assert note.solutions == []
    assert note.dropped == [{0}]
