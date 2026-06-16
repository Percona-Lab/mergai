"""``mergai ci status`` gate — aggregate the watched workflows' state for HEAD.

The gate the CI auto-fix loop reads to decide between squash / fix / wait.
"""

from typing import Literal

import click
import github

from ..app import AppContext
from .dispatch import (
    _failure_note,
    _resolve_pr_number,
    _run_head_status,
    _take_workflow_runs,
    classify_run,
)


def _list_run_status(
    app: AppContext,
    run: "github.WorkflowRun.WorkflowRun",
    *,
    check_findings: bool,
) -> tuple[str, str]:
    """Return ``(status, notes)`` describing what mergai sees for this run.

    Status is one of:
    - ``applied``: a ``type: ci_fix`` solution exists with a matching ``run_id``.
    - ``pending``: not processed and mergai *would* act (failure, or
      success with code_scanning_check + findings present).
    - ``skip``: not processed and mergai would not act (passing run with
      no opt-in, conclusion not actionable, workflow disabled, etc.).
    """
    run_id = str(run.id)
    comment = app.note.get_ci_comment_for_run(run_id) if app.has_note else None
    existing = app.note.get_ci_solution_for_run(run_id) if app.has_note else None
    if existing is not None:
        solutions = app.note.solutions or []
        idx = next(
            (i for i, s in enumerate(solutions) if s is existing),
            -1,
        )
        attempt = existing.get("request", {}).get("attempt_number", "?")
        note = f"solutions[{idx}], attempt {attempt}"
        if comment is not None:
            note += (
                f"; comment posted {comment['posted_at']}"
                if comment.get("posted_at")
                else "; comment pending"
            )
        return "applied", note

    if comment is not None:
        if comment.get("posted_at"):
            return "commented", f"unable to fix; comment posted {comment['posted_at']}"
        return "diagnosed", "agent unable to fix; comment pending"

    decision = classify_run(
        app,
        run,
        workflow_name=run.name,
        pr_number=_resolve_pr_number(run),
        check_findings=check_findings,
        # Listing should stay cheap and still show a failed run as a failure
        # entry; the non-code-failure side-calls (approvals / job steps) are
        # only worth making on the real dispatch path.
        check_failure_kind=False,
    )

    if decision.actionable:
        config = app.config.workflows.get(run.name)
        assert config is not None  # actionable ⇒ configured + enabled
        if decision.kind == "failure":
            return "pending", _failure_note(config)
        # code_scanning: findings confirmed, or deferred when not queried.
        if not decision.findings_queried:
            return (
                "pending",
                "passed; Code Scanning check enabled (findings not queried)",
            )
        return "pending", "passed, but Code Scanning has findings"

    reason = decision.skip_reason
    if reason in ("no_config", "disabled"):
        return "skip", "workflow not enabled in config"
    if reason == "not_mergai_branch":
        return "skip", f"head_branch '{run.head_branch}' is not mergai/*"
    if reason == "superseded":
        return "skip", "superseded by newer commits on the branch"
    if reason == "obsolete":
        return "skip", "head_sha not reachable from HEAD (force-pushed?)"
    if reason == "incomplete":
        # Not completed yet — neither actionable now nor a reason to give up.
        # `wait` differentiates from `skip` so the table reads as still moving.
        return "wait", f"still {run.status}"
    if reason == "passed":
        return "skip", "passed"
    if reason == "no_pr":
        if run.conclusion == "failure":
            return "skip", "failed, but no associated PR"
        return "skip", "passed; Code Scanning check enabled, but no associated PR"
    if reason == "no_findings":
        return "skip", "passed; Code Scanning check enabled, no findings"
    if reason == "cancelled":
        return "skip", "cancelled; nothing to fix"
    if reason == "approval_rejected":
        return "skip", "failed: deployment approval rejected; nothing to fix"
    if reason == "no_failing_step":
        return "skip", "failed: no failing step (blocked approval / infra)"
    # unusual_conclusion: completed with timed_out / neutral / …
    return "skip", f"conclusion '{run.conclusion}'"


def _watched_runs_for_head(
    app: AppContext,
) -> dict[str, "github.WorkflowRun.WorkflowRun"]:
    """Latest run per watched workflow whose ``head_sha`` is the branch HEAD.

    "Watched" means configured in ``.mergai/config.yml`` (``format``,
    ``clang-tidy``, ``build-and-test``). Runs on a superseded / obsolete
    commit are ignored — the gate only reasons about the current HEAD.
    ``get_workflow_runs`` returns newest-first, so the first run seen per
    workflow name is the latest one.
    """
    try:
        branch = app.repo.active_branch.name
    except TypeError as e:
        raise click.ClickException(
            "HEAD is detached; cannot determine the branch to gate on."
        ) from e

    runs = app.gh_repo.get_workflow_runs(branch=branch)  # type: ignore[arg-type]
    runs_list = _take_workflow_runs(runs, 50)

    latest: dict[str, github.WorkflowRun.WorkflowRun] = {}
    for run in runs_list:
        if run.name not in app.config.workflows.workflows:
            continue
        if _run_head_status(app, run) != "current":
            continue
        latest.setdefault(run.name, run)
    return latest


def _aggregate_state(
    runs_by_workflow: dict[str, "github.WorkflowRun.WorkflowRun"],
) -> Literal["in-progress", "success", "failure", "none"]:
    """Reduce the per-workflow latest runs to a single gate token.

    * ``none``        — no watched runs for HEAD (e.g. all skipped).
    * ``in-progress`` — at least one watched run hasn't completed.
    * ``success``     — every watched run completed with ``success``.
    * ``failure``     — all completed, but at least one did not succeed
                        (``failure`` / ``cancelled`` / ``timed_out`` / …).
    """
    if not runs_by_workflow:
        return "none"
    runs = list(runs_by_workflow.values())
    if any(run.status != "completed" for run in runs):
        return "in-progress"
    if all(run.conclusion == "success" for run in runs):
        return "success"
    return "failure"
