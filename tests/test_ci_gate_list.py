"""Tests for ``_list_run_status`` notes — the ``ci list`` Status/Notes columns.

Focus: a fail-fast-cancelled run that masks a real job failure shows up as
``pending`` with a note that explains the otherwise-contradictory pairing of a
``cancelled`` conclusion and a failure note.
"""

from types import SimpleNamespace

from mergai.ci.gate import _list_run_status

HEAD_SHA = "a" * 40


class _Run:
    def __init__(self, *, conclusion, steps_conclusions, name="build-and-test"):
        self.id = 123
        self.name = name
        self.conclusion = conclusion
        self.status = "completed"
        self.head_branch = "mergai/x"
        self.head_sha = HEAD_SHA
        self.url = "https://api.github.com/repos/o/r/actions/runs/123"
        self.pull_requests = [SimpleNamespace(number=7)]
        self._steps_conclusions = steps_conclusions

    def jobs(self):
        steps = [SimpleNamespace(conclusion=c) for c in self._steps_conclusions]
        return [SimpleNamespace(steps=steps)]


def _app():
    workflow_config = SimpleNamespace(
        enabled=True,
        context=SimpleNamespace(
            type="bazel",
            code_scanning_check=False,
            code_scanning_tool_name=None,
        ),
    )
    note = SimpleNamespace(
        get_ci_comment_for_run=lambda run_id: None,
        get_ci_solution_for_run=lambda run_id: None,
        solutions=[],
    )
    return SimpleNamespace(
        has_note=True,
        note=note,
        gh_repo=SimpleNamespace(
            _requester=SimpleNamespace(requestJsonAndCheck=lambda method, url: ({}, []))
        ),
        repo=SimpleNamespace(
            head=SimpleNamespace(commit=SimpleNamespace(hexsha=HEAD_SHA))
        ),
        config=SimpleNamespace(
            branch=SimpleNamespace(working_prefix="mergai/"),
            workflows=SimpleNamespace(get=lambda name: workflow_config),
        ),
    )


def test_cancelled_masking_failure_note_explains_pending():
    app = _app()
    run = _Run(conclusion="cancelled", steps_conclusions=("success", "failure"))
    status, note = _list_run_status(app, run, check_findings=False)
    assert status == "pending"
    assert "fail-fast" in note
    assert "bazel artifact" in note


def test_plain_cancellation_still_skips():
    app = _app()
    run = _Run(conclusion="cancelled", steps_conclusions=("success",))
    status, note = _list_run_status(app, run, check_findings=False)
    assert status == "skip"
    assert "cancelled" in note.lower()


def test_plain_failure_note_unchanged():
    app = _app()
    run = _Run(conclusion="failure", steps_conclusions=("failure",))
    status, note = _list_run_status(app, run, check_findings=False)
    assert status == "pending"
    assert "fail-fast" not in note
    assert "bazel artifact" in note
