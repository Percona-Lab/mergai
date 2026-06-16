"""Tests for ``classify_run`` non-code-failure skips and ``_skip_message``.

Pure logic against hand-built run / job / approvals stubs - no network. The
focus is A1: a ``failure`` conclusion is only treated as actionable when there
is positive evidence it is a real code failure (a failing step, no rejected
deployment approval).
"""

from types import SimpleNamespace

from mergai.ci.dispatch import RunDispatchDecision, _skip_message, classify_run

HEAD_SHA = "a" * 40


class _Requester:
    """Stub for ``gh_repo._requester`` returning canned approvals data."""

    def __init__(self, approvals, *, raises=False):
        self._approvals = approvals
        self._raises = raises
        self.calls = 0

    def requestJsonAndCheck(self, method, url):  # noqa: N802 - mirror PyGithub
        self.calls += 1
        if self._raises:
            raise RuntimeError("approvals endpoint blew up")
        return {}, self._approvals


class _Run:
    def __init__(
        self,
        *,
        conclusion="failure",
        steps_conclusions=("failure",),
        head_branch="mergai/x",
        jobs_raise=False,
    ):
        self.conclusion = conclusion
        self.status = "completed"
        self.head_branch = head_branch
        self.head_sha = HEAD_SHA
        self.url = "https://api.github.com/repos/o/r/actions/runs/1"
        self.pull_requests = []
        self._steps_conclusions = steps_conclusions
        self._jobs_raise = jobs_raise
        self.jobs_calls = 0

    def jobs(self):
        self.jobs_calls += 1
        if self._jobs_raise:
            raise RuntimeError("jobs() blew up")
        steps = [SimpleNamespace(conclusion=c) for c in self._steps_conclusions]
        return [SimpleNamespace(steps=steps)]


def _app(*, approvals=None, approvals_raise=False):
    requester = _Requester(approvals or [], raises=approvals_raise)
    workflow_config = SimpleNamespace(
        enabled=True,
        context=SimpleNamespace(
            code_scanning_check=False, code_scanning_tool_name=None
        ),
    )
    return SimpleNamespace(
        gh_repo=SimpleNamespace(_requester=requester),
        repo=SimpleNamespace(
            head=SimpleNamespace(commit=SimpleNamespace(hexsha=HEAD_SHA))
        ),
        config=SimpleNamespace(
            branch=SimpleNamespace(working_prefix="mergai/"),
            workflows=SimpleNamespace(get=lambda name: workflow_config),
        ),
    )


def _classify(app, run, **kw):
    return classify_run(app, run, workflow_name="format", pr_number=7, **kw)


# --- A1: non-code-failure skips ------------------------------------------


def test_failure_with_failing_step_is_actionable():
    app = _app(approvals=[])
    decision = _classify(app, _Run(steps_conclusions=("success", "failure")))
    assert decision.actionable
    assert decision.kind == "failure"


def test_rejected_approval_is_skipped():
    app = _app(approvals=[{"state": "rejected", "environments": [{"name": "rbe"}]}])
    decision = _classify(app, _Run(steps_conclusions=()))
    assert decision.skip_reason == "approval_rejected"


def test_failure_with_no_failing_step_is_skipped():
    app = _app(approvals=[])
    decision = _classify(app, _Run(steps_conclusions=("success",)))
    assert decision.skip_reason == "no_failing_step"


def test_cancelled_run_is_skipped():
    app = _app(approvals=[])
    decision = _classify(app, _Run(conclusion="cancelled"))
    assert decision.skip_reason == "cancelled"


def test_approval_checked_before_failing_step():
    # A rejected approval reports first even when (hypothetically) a step also
    # failed - the approval signal is the precise one.
    app = _app(approvals=[{"state": "rejected"}])
    decision = _classify(app, _Run(steps_conclusions=("failure",)))
    assert decision.skip_reason == "approval_rejected"


def test_approvals_side_call_failure_fails_open():
    # A flaky approvals endpoint must never block a real fix.
    app = _app(approvals_raise=True)
    decision = _classify(app, _Run(steps_conclusions=("failure",)))
    assert decision.actionable


def test_jobs_side_call_failure_fails_open():
    app = _app(approvals=[])
    decision = _classify(app, _Run(jobs_raise=True))
    assert decision.actionable


def test_check_failure_kind_false_skips_side_calls():
    # `ci list` path: a failure stays actionable without the approvals / jobs
    # side-calls, and neither stub is touched.
    app = _app(approvals=[{"state": "rejected"}])
    run = _Run(steps_conclusions=(), jobs_raise=True)
    decision = _classify(app, run, check_failure_kind=False)
    assert decision.actionable
    assert app.gh_repo._requester.calls == 0
    assert run.jobs_calls == 0


def test_not_mergai_branch_skipped_before_side_calls():
    app = _app(approvals=[{"state": "rejected"}])
    run = _Run(head_branch="feature/x", steps_conclusions=())
    decision = _classify(app, run)
    assert decision.skip_reason == "not_mergai_branch"
    assert app.gh_repo._requester.calls == 0


# --- A1: _skip_message wording -------------------------------------------


def _decision(reason):
    return RunDispatchDecision(
        kind=None,
        head_status="current",
        pr_number=7,
        skip_reason=reason,
        findings_queried=False,
    )


def test_skip_message_approval_rejected():
    run = SimpleNamespace(
        conclusion="failure", head_sha=HEAD_SHA, head_branch="mergai/x"
    )
    msg = _skip_message(_decision("approval_rejected"), run, "format")
    assert "deployment approval was rejected" in msg


def test_skip_message_cancelled():
    run = SimpleNamespace(
        conclusion="cancelled", head_sha=HEAD_SHA, head_branch="mergai/x"
    )
    msg = _skip_message(_decision("cancelled"), run, "format")
    assert "cancelled" in msg.lower()


def test_skip_message_no_failing_step():
    run = SimpleNamespace(
        conclusion="failure", head_sha=HEAD_SHA, head_branch="mergai/x"
    )
    msg = _skip_message(_decision("no_failing_step"), run, "format")
    assert "no step reported a failure" in msg
