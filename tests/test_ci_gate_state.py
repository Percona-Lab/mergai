"""Tests for ``_actionable_state`` — the ``mergai ci status --state`` gate.

The gate's ``failure`` token must mean *mergai has actionable work* (the fixer
would run), not merely "a watched run didn't return success". These tests pin
the cases that used to spin the privileged CI-fix handler up for nothing:
a plain cancellation, an already-handled run, and a passing code-scanning run
whose findings can't be queried (no ``security-events`` read).
"""

from types import SimpleNamespace

import github

from mergai.ci.gate import _actionable_state

HEAD_SHA = "a" * 40


class _Run:
    def __init__(
        self,
        *,
        name,
        conclusion,
        status="completed",
        steps_conclusions=(),
    ):
        self.id = abs(hash(name)) % 100000
        self.name = name
        self.conclusion = conclusion
        self.status = status
        self.head_branch = "mergai/x"
        self.head_sha = HEAD_SHA
        self.url = "https://api.github.com/repos/o/r/actions/runs/123"
        self.pull_requests = [SimpleNamespace(number=7)]
        self._steps_conclusions = steps_conclusions

    def jobs(self):
        steps = [SimpleNamespace(conclusion=c) for c in self._steps_conclusions]
        return [SimpleNamespace(steps=steps)]


def _app(
    *,
    code_scanning=False,
    solution_for_run=None,
    comment_for_run=None,
    requester=None,
):
    """Build a minimal AppContext double.

    * ``code_scanning`` toggles ``code_scanning_check`` on every workflow.
    * ``solution_for_run`` / ``comment_for_run`` are callables ``run_id -> …``
      standing in for the note lookups (default: nothing recorded).
    * ``requester`` overrides ``requestJsonAndCheck`` (the code-scanning /
      approvals side-calls); default returns an empty analyses list.
    """
    workflow_config = SimpleNamespace(
        enabled=True,
        context=SimpleNamespace(
            type="bazel",
            code_scanning_check=code_scanning,
            code_scanning_tool_name=None,
        ),
    )
    note = SimpleNamespace(
        get_ci_comment_for_run=comment_for_run or (lambda run_id: None),
        get_ci_solution_for_run=solution_for_run or (lambda run_id: None),
        solutions=[],
    )
    if requester is None:
        requester = lambda method, url, **kw: ({}, [])  # noqa: E731
    return SimpleNamespace(
        has_note=True,
        note=note,
        gh_repo=SimpleNamespace(
            url="https://api.github.com/repos/o/r",
            _requester=SimpleNamespace(requestJsonAndCheck=requester),
        ),
        repo=SimpleNamespace(
            head=SimpleNamespace(commit=SimpleNamespace(hexsha=HEAD_SHA))
        ),
        config=SimpleNamespace(
            branch=SimpleNamespace(working_prefix="mergai/"),
            workflows=SimpleNamespace(
                get=lambda name: workflow_config,
                workflows={"format", "clang-tidy", "build-and-test"},
            ),
        ),
    )


def _by_workflow(*runs):
    return {run.name: run for run in runs}


def test_none_when_no_runs():
    assert _actionable_state(_app(), {}) == "none"


def test_in_progress_wins_over_actionable_failure():
    # One run still running, another already failed: hold at in-progress so the
    # handler fires only once the whole set for HEAD is done.
    app = _app()
    runs = _by_workflow(
        _Run(name="format", conclusion=None, status="in_progress"),
        _Run(name="clang-tidy", conclusion="failure", steps_conclusions=("failure",)),
    )
    assert _actionable_state(app, runs) == "in-progress"


def test_real_failure_is_failure():
    app = _app()
    runs = _by_workflow(
        _Run(
            name="build-and-test", conclusion="failure", steps_conclusions=("failure",)
        )
    )
    assert _actionable_state(app, runs) == "failure"


def test_fail_fast_masked_cancellation_is_failure():
    # cancelled roll-up masking a real failing step -> actionable.
    app = _app()
    runs = _by_workflow(
        _Run(
            name="build-and-test",
            conclusion="cancelled",
            steps_conclusions=("success", "failure"),
        )
    )
    assert _actionable_state(app, runs) == "failure"


def test_plain_cancellation_is_not_failure():
    # The headline fix: a user/timeout cancellation is nothing to fix, so the
    # gate must NOT report failure (which would spin up the privileged handler).
    app = _app()
    runs = _by_workflow(
        _Run(name="format", conclusion="success"),
        _Run(
            name="build-and-test",
            conclusion="cancelled",
            steps_conclusions=("success",),
        ),
    )
    assert _actionable_state(app, runs) == "success"


def test_all_success_is_success():
    app = _app()
    runs = _by_workflow(
        _Run(name="format", conclusion="success"),
        _Run(name="clang-tidy", conclusion="success"),
        _Run(name="build-and-test", conclusion="success"),
    )
    assert _actionable_state(app, runs) == "success"


def test_already_handled_failure_is_not_re_triggered():
    # A run mergai already fixed (a solution is recorded for its run_id) is
    # `applied`, not `pending` -> the gate stays green (idempotent).
    solution = {"request": {"attempt_number": 1}}
    app = _app(solution_for_run=lambda run_id: solution)
    app.note.solutions = [solution]
    runs = _by_workflow(
        _Run(
            name="build-and-test", conclusion="failure", steps_conclusions=("failure",)
        )
    )
    assert _actionable_state(app, runs) == "success"


def test_passing_code_scanning_run_with_findings_is_failure():
    def requester(method, url, **kw):
        if url.endswith("/code-scanning/analyses"):
            return ({}, [{"commit_sha": HEAD_SHA, "results_count": 3}])
        return ({}, [])

    app = _app(code_scanning=True, requester=requester)
    runs = _by_workflow(_Run(name="clang-tidy", conclusion="success"))
    assert _actionable_state(app, runs) == "failure"


def test_passing_code_scanning_run_without_findings_is_success():
    app = _app(code_scanning=True)  # analyses lookup returns []
    runs = _by_workflow(_Run(name="clang-tidy", conclusion="success"))
    assert _actionable_state(app, runs) == "success"


def test_code_scanning_query_denied_degrades_to_non_actionable(capsys):
    # No security-events read: the analyses lookup 403s. The gate must not crash
    # and must not report failure on a signal it couldn't confirm.
    def requester(method, url, **kw):
        if url.endswith("/code-scanning/analyses"):
            raise github.GithubException(
                403, {"message": "Resource not accessible"}, None
            )
        return ({}, [])

    app = _app(code_scanning=True, requester=requester)
    runs = _by_workflow(_Run(name="clang-tidy", conclusion="success"))
    assert _actionable_state(app, runs) == "success"
    assert "could not classify run" in capsys.readouterr().err


def test_non_github_side_call_error_degrades_to_non_actionable(capsys):
    # A raw requester/network error (not a GithubException) underneath the
    # code-scanning lookup must also degrade rather than crash the gate.
    def requester(method, url, **kw):
        if url.endswith("/code-scanning/analyses"):
            raise ConnectionError("connection reset")
        return ({}, [])

    app = _app(code_scanning=True, requester=requester)
    runs = _by_workflow(_Run(name="clang-tidy", conclusion="success"))
    assert _actionable_state(app, runs) == "success"
    assert "could not classify run" in capsys.readouterr().err
