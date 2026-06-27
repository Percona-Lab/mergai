"""Tests for ``mergai ci list`` helpers: per-job sub-rows, relative age, and
the effective run conclusion.

The list is laid out like GitHub's PR "checks" section: a run header row
followed by one sub-row per job. These tests pin the sub-row shape (blank
run-level columns, ``workflow / job`` name in the Workflow column, job
conclusion in the Conclusion column), the jobs()-error fallback, and the
fail-fast ``cancelled`` -> ``failure (cancelled)`` relabel.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mergai.commands.ci import (
    _display_conclusion,
    _iter_workflow_runs,
    _job_rows,
    _relative_age,
)


def _job(name, conclusion=None, status="completed", job_id=999, steps=None):
    return SimpleNamespace(
        id=job_id, name=name, conclusion=conclusion, status=status, steps=steps
    )


def _run(conclusion):
    return SimpleNamespace(conclusion=conclusion, status="completed")


def test_job_rows_shape():
    # 8-column shape: blank Head SHA / Run ID, Job ID set, `workflow / job`
    # name, conclusion, then blank age / status / notes (those belong to the
    # run header).
    jobs = [
        _job("gate", conclusion="success", job_id=101),
        _job("build", conclusion="failure", job_id=102),
    ]
    rows = _job_rows("build-and-test", jobs)
    assert rows == [
        ("", "", "101", "build-and-test / gate", "success", "", "", ""),
        ("", "", "102", "build-and-test / build", "failure", "", "", ""),
    ]


def test_job_rows_falls_back_to_status_when_not_concluded():
    # An in-progress job has conclusion=None; show its status instead.
    jobs = [_job("lint", conclusion=None, status="in_progress", job_id=103)]
    rows = _job_rows("build-and-test", jobs)
    assert rows == [("", "", "103", "build-and-test / lint", "in_progress", "", "", "")]


def test_job_rows_handles_unavailable_jobs():
    # The caller passes None when the jobs() call failed.
    rows = _job_rows("build-and-test", None)
    assert rows == [("", "", "", "(jobs unavailable)", "", "", "", "")]


# --- effective conclusion --------------------------------------------------


def _step(conclusion):
    return SimpleNamespace(conclusion=conclusion)


def test_display_conclusion_passes_through_non_cancelled():
    assert _display_conclusion(_run("success"), None) == "success"
    assert _display_conclusion(_run("failure"), None) == "failure"


def test_display_conclusion_relabels_fail_fast_cancelled():
    # One job has a failing step, siblings cancelled -> the run rolled up to
    # `cancelled`, but the real outcome is a failure.
    jobs = [
        _job("gate", conclusion="success", steps=[_step("success")]),
        _job("dbtests", conclusion="failure", steps=[_step("failure")]),
        _job("jstests", conclusion="cancelled", steps=[]),
    ]
    assert _display_conclusion(_run("cancelled"), jobs) == "failure (cancelled)"


def test_display_conclusion_keeps_plain_cancellation():
    # No failing step (all cancelled) -> a genuine cancellation, left as-is.
    jobs = [
        _job("gate", conclusion="cancelled", steps=[]),
        _job("build", conclusion="cancelled", steps=[_step("cancelled")]),
    ]
    assert _display_conclusion(_run("cancelled"), jobs) == "cancelled"


def test_display_conclusion_cancelled_without_jobs_stays_cancelled():
    # jobs() failed (None) -> can't tell, so don't relabel.
    assert _display_conclusion(_run("cancelled"), None) == "cancelled"


def _ago(**kw):
    return datetime.now(timezone.utc) - timedelta(**kw)


def test_relative_age_tiers():
    assert _relative_age(None) == "-"
    assert _relative_age(_ago(seconds=5)) == "just now"
    assert _relative_age(_ago(minutes=1, seconds=2)) == "1 minute ago"
    assert _relative_age(_ago(minutes=15, seconds=2)) == "15 minutes ago"
    assert _relative_age(_ago(hours=1, seconds=2)) == "1 hour ago"
    assert _relative_age(_ago(hours=2, seconds=2)) == "2 hours ago"
    assert _relative_age(_ago(days=3, seconds=2)) == "3 days ago"
    assert _relative_age(_ago(days=14, seconds=2)) == "2 weeks ago"


def test_relative_age_naive_datetime_treated_as_utc():
    # PyGithub can hand back a naive (tz-less) UTC datetime; it must not crash.
    naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
    assert _relative_age(naive) == "10 minutes ago"


def test_relative_age_old_run_falls_back_to_date():
    # Capture once: two _ago() calls could straddle UTC midnight and format to
    # different dates.
    old = _ago(days=40)
    assert _relative_age(old) == old.strftime("%Y-%m-%d")


# --- lazy run iteration ----------------------------------------------------


def test_iter_workflow_runs_caps_total():
    assert list(_iter_workflow_runs(iter(range(100)), 3)) == [0, 1, 2]


def test_iter_workflow_runs_stops_pulling_when_caller_breaks():
    # The whole point: a caller that stops early must not pull further pages.
    pulled = []

    def gen():
        for i in range(100):
            pulled.append(i)
            yield i

    it = _iter_workflow_runs(gen(), 50)
    taken = [next(it), next(it)]
    assert taken == [0, 1]
    assert pulled == [0, 1]  # only what was consumed, despite the cap of 50


def test_iter_workflow_runs_swallows_pagination_indexerror():
    def gen():
        yield 1
        yield 2
        raise IndexError("pagination Link promised more than the page returned")

    assert list(_iter_workflow_runs(gen(), 10)) == [1, 2]
