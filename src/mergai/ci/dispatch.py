"""Workflow-run dispatch decisions and context building for ``mergai ci``."""

import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
import github

from ..app import AppContext
from ..config import WorkflowConfig
from ..utils import git_utils
from ..utils.artifact_downloader import download_workflow_run_artifacts
from .context_builders import WorkflowContext, get_context_builder
from .context_builders.sarif import SARIFContextBuilder


def _take_workflow_runs(
    runs: Iterable["github.WorkflowRun.WorkflowRun"], limit: int
) -> list["github.WorkflowRun.WorkflowRun"]:
    """Collect up to ``limit`` workflow runs, tolerating pagination races.

    Iterates the ``PaginatedList`` directly and stops at ``limit`` rather
    than slicing it (``runs[:limit]``). The slice path indexes by position
    and trusts the pagination ``Link`` header, so it raises ``IndexError``
    when GitHub promises more runs than a page actually returns — a race
    seen in the seconds after a push, while new runs are still being
    created. Plain iteration only yields what was fetched, so it degrades
    to a short list instead of crashing ``ci list`` / ``ci fix``.
    """
    collected: list[github.WorkflowRun.WorkflowRun] = []
    try:
        for run in runs:
            collected.append(run)
            if len(collected) >= limit:
                break
    except IndexError:
        # Defensive: a page can still come back short mid-iteration.
        pass
    return collected


def _resolve_target_runs(
    app: AppContext, target: str, *, force: bool = False
) -> list[str]:
    """Resolve ``"all"`` / workflow-name to a list of run IDs to process.

    Lists recent workflow_runs on the current branch, filters out runs
    that are not configured / not actionable / already processed, and
    returns IDs in newest-first order.

    A workflow-name target is just the ``"all"`` filter narrowed to one
    workflow.

    Args:
        force: When True, do not filter out runs whose head_sha is not
            current HEAD. Other actionability filters (workflow enabled,
            mergai/* branch, conclusion type) still apply.
    """
    workflow_filter = None if target == "all" else target
    if (
        workflow_filter is not None
        and workflow_filter not in app.config.workflows.workflows
    ):
        raise click.ClickException(
            f"Unknown workflow '{workflow_filter}'. Configured workflows: "
            + ", ".join(sorted(app.config.workflows.workflows))
            + "."
        )

    try:
        branch = app.repo.active_branch.name
    except TypeError as e:
        raise click.ClickException(
            "HEAD is detached; pass an explicit run ID instead of "
            "'all' / workflow name."
        ) from e

    runs = app.gh_repo.get_workflow_runs(branch=branch)  # type: ignore[arg-type]
    runs_list = _take_workflow_runs(runs, 50)

    # Only the newest run per workflow matters: while a slow check re-runs,
    # GitHub may list several runs of the same workflow (re-runs / repeated
    # triggers). Processing more than one in a single `ci fix all` is what
    # produced duplicate / contradictory comments (e.g. fixing the latest
    # clang-tidy run, then reporting an older clang-tidy run as "already
    # resolved"). Iterating newest-first, take the first run per workflow and
    # skip the rest entirely — don't fall back to an older run if the newest
    # is already processed or not actionable.
    selected: list[str] = []
    seen_workflows: set[str] = set()
    for run in runs_list:
        if workflow_filter is not None and run.name != workflow_filter:
            continue
        if run.name not in app.config.workflows.workflows:
            continue
        if run.name in seen_workflows:
            continue
        seen_workflows.add(run.name)
        run_id = str(run.id)
        if app.has_note and app.note.get_ci_solution_for_run(run_id) is not None:
            continue
        if app.has_note and app.note.get_ci_comment_for_run(run_id) is not None:
            continue
        if not _run_is_actionable(app, run, force=force):
            continue
        selected.append(run_id)

    return selected


SkipReason = Literal[
    "not_mergai_branch",
    "no_config",
    "disabled",
    "obsolete",
    "superseded",
    "incomplete",
    "passed",
    "no_pr",
    "no_findings",
    "unusual_conclusion",
]


@dataclass(frozen=True)
class RunDispatchDecision:
    """Single source of truth for "would mergai act on this run, and how?".

    Computed once by :func:`classify_run` and consumed by the three paths
    that used to each re-derive it: :func:`build_workflow_context_for_run`
    (the real dispatch), :func:`_run_is_actionable` (the ``fix all`` /
    ``prompt ci all`` filter), and :func:`_list_run_status` (the ``ci list``
    display). Keeping the decision in one place stops those paths drifting —
    e.g. ``ci list`` reporting ``pending`` while ``ci fix`` says "nothing to
    do."

    Attributes:
        kind: Dispatch path when actionable — ``"failure"`` (build context
            from the run's artifacts) or ``"code_scanning"`` (build from
            Code Scanning findings on a passing run); ``None`` when not
            actionable.
        head_status: ``run.head_sha`` relative to the local branch HEAD
            (see :func:`_run_head_status`).
        pr_number: The PR the run belongs to (non-``None`` when actionable;
            both dispatch paths need it).
        skip_reason: ``None`` iff actionable; otherwise a stable code naming
            why mergai would not act. Callers map it to their own wording.
        findings_queried: Whether the Code Scanning findings lookup actually
            ran. ``ci list`` can defer that network call and still mark a run
            ``pending``.
    """

    kind: Literal["failure", "code_scanning"] | None
    head_status: Literal["current", "superseded", "obsolete"]
    pr_number: int | None
    skip_reason: SkipReason | None
    findings_queried: bool

    @property
    def actionable(self) -> bool:
        """True when mergai would build a context and act on this run."""
        return self.skip_reason is None

    @property
    def needs_artifacts(self) -> bool:
        """True when the dispatch path downloads the run's artifacts."""
        return self.kind == "failure"


def classify_run(
    app: AppContext,
    run: "github.WorkflowRun.WorkflowRun",
    *,
    workflow_name: str,
    pr_number: int | None,
    check_findings: bool = True,
    check_staleness: bool = True,
    force: bool = False,
) -> RunDispatchDecision:
    """Decide whether mergai would act on ``run``, and how.

    The one place the actionability rules live — branch namespace, workflow
    enabled, head-commit staleness, conclusion, and (for passing runs that
    opt in) Code Scanning findings. The three former copies now delegate
    here.

    Args:
        workflow_name: Resolved workflow name (callers pass an override when
            they have one, else ``run.name``); config is looked up from it.
        pr_number: The run's PR, already resolved by the caller. Required by
            both dispatch paths; ``None`` yields a ``no_pr`` skip.
        check_findings: When False, a passing code-scanning workflow is
            reported actionable without running the findings lookup
            (``findings_queried=False``). ``ci list`` uses this to skip the
            network call.
        check_staleness: When False (``build_workflow_context_for_run``,
            whose callers vet staleness separately), ``superseded`` /
            ``obsolete`` runs are not skipped here.
        force: When True, accept ``superseded`` runs (``ci fix --force``).
            ``obsolete`` runs are skipped regardless — their work was
            force-pushed away.
    """
    head_status = _run_head_status(app, run)
    config = app.config.workflows.get(workflow_name)

    def skip(reason: SkipReason) -> RunDispatchDecision:
        return RunDispatchDecision(
            kind=None,
            head_status=head_status,
            pr_number=pr_number,
            skip_reason=reason,
            findings_queried=False,
        )

    def act(
        kind: Literal["failure", "code_scanning"], *, findings_queried: bool
    ) -> RunDispatchDecision:
        return RunDispatchDecision(
            kind=kind,
            head_status=head_status,
            pr_number=pr_number,
            skip_reason=None,
            findings_queried=findings_queried,
        )

    if not (run.head_branch or "").startswith(app.config.branch.working_prefix):
        return skip("not_mergai_branch")
    if config is None:
        return skip("no_config")
    if not config.enabled:
        return skip("disabled")
    if check_staleness:
        if head_status == "obsolete":
            return skip("obsolete")
        if head_status == "superseded" and not force:
            return skip("superseded")
    if run.status != "completed":
        return skip("incomplete")

    if run.conclusion == "failure":
        if pr_number is None:
            return skip("no_pr")
        return act("failure", findings_queried=False)

    if run.conclusion == "success":
        if not config.context.code_scanning_check:
            return skip("passed")
        if pr_number is None:
            return skip("no_pr")
        if not check_findings:
            return act("code_scanning", findings_queried=False)
        if _code_scanning_has_findings(
            app,
            tool_name=config.context.code_scanning_tool_name or workflow_name,
            head_sha=run.head_sha,
            pr_number=pr_number,
        ):
            return act("code_scanning", findings_queried=True)
        return skip("no_findings")

    return skip("unusual_conclusion")


def _run_is_actionable(
    app: AppContext,
    run: "github.WorkflowRun.WorkflowRun",
    *,
    force: bool = False,
) -> bool:
    """Whether ``ci fix all`` / ``prompt ci all`` would handle this run.

    Thin wrapper over :func:`classify_run` (see there for the rules). Used
    to filter the run list before iterating. ``obsolete`` runs are always
    skipped; ``superseded`` runs are skipped unless ``force`` — which keeps
    ``ci fix --force`` aligned with ``ci list``, which also hides obsolete
    runs.
    """
    decision = classify_run(
        app,
        run,
        workflow_name=run.name,
        pr_number=_resolve_pr_number(run),
        force=force,
    )
    return decision.actionable


def _failure_note(config: WorkflowConfig) -> str:
    """Describe how `ci fix` would source context for a failed run.

    The note text is shown by ``ci list`` so the user can tell at a
    glance which path mergai would take. The phrasing depends on the
    configured context builder — only the SARIF builder has a log
    fallback, the diff builder doesn't.
    """
    ctx_type = config.context.type
    if ctx_type == "sarif":
        return "failure -> SARIF artifact (log fallback if missing)"
    if ctx_type == "diff":
        return "failure -> diff artifact"
    return f"failure -> {ctx_type} artifact"


def _run_head_status(
    app: AppContext, run: "github.WorkflowRun.WorkflowRun"
) -> Literal["current", "superseded", "obsolete"]:
    """Classify a workflow run's head commit relative to the local branch.

    * ``current``    — ``run.head_sha`` equals the working tree HEAD.
                       Findings still describe the same source the user
                       has checked out, so the run is safe to act on.
    * ``superseded`` — ``run.head_sha`` is an ancestor of HEAD but not
                       equal. Newer commits exist on the branch since
                       the run; its findings may be stale.
    * ``obsolete``   — ``run.head_sha`` isn't reachable from HEAD at
                       all. Typically means the branch was force-pushed
                       (or the SHA was never fetched locally).
    """
    head_sha = app.repo.head.commit.hexsha
    if run.head_sha == head_sha:
        return "current"
    # Ancestor of HEAD → superseded (newer commits since the run); not
    # reachable at all (status 1, or an unknown SHA) → obsolete: we can't act
    # on a SHA we don't have or can't reach.
    if git_utils.is_ancestor(app.repo, run.head_sha):
        return "superseded"
    return "obsolete"


def _skip_message(
    decision: RunDispatchDecision,
    run: "github.WorkflowRun.WorkflowRun",
    workflow_name: str,
) -> str:
    """Human-readable "nothing to do" line for a non-actionable run.

    Maps the classifier's :attr:`RunDispatchDecision.skip_reason` back to
    the message ``build_workflow_context_for_run`` echoes. Staleness reasons
    (``obsolete`` / ``superseded``) can't occur here — that path runs the
    classifier with ``check_staleness=False``.
    """
    reason = decision.skip_reason
    if reason == "no_pr":
        return "No PR associated with this workflow run; nothing to do."
    if reason == "not_mergai_branch":
        return (
            f"Head branch '{run.head_branch}' is not a mergai/* branch; "
            f"nothing to do."
        )
    if reason == "no_config":
        return f"No configuration for workflow '{workflow_name}'; nothing to do."
    if reason == "disabled":
        return f"Workflow '{workflow_name}' handling is disabled; nothing to do."
    if reason == "no_findings":
        return (
            f"Workflow '{workflow_name}' passed and Code Scanning has "
            f"no findings for {run.head_sha[:7]}; nothing to do."
        )
    # passed / incomplete / unusual_conclusion
    return (
        f"Run conclusion '{run.conclusion}' is not actionable for "
        f"'{workflow_name}'; nothing to do."
    )


@contextmanager
def build_workflow_context_for_run(
    app: AppContext,
    run_id: str,
    *,
    workflow_override: str | None = None,
    pr_override: int | None = None,
    artifacts_dir_override: str | None = None,
) -> Iterator[tuple[WorkflowContext, WorkflowConfig] | None]:
    """Resolve a workflow run and build its :class:`WorkflowContext`.

    Yields ``(context, config)`` when the run is actionable, or ``None``
    after echoing a human-readable skip reason. Cleans up any temporary
    artifact directory on exit.

    Shared by ``mergai ci fix`` (which then runs the handler) and
    ``mergai prompt ci`` (which renders the prompt without invoking the
    agent), so both consume the same dispatch decision and the same
    context shape.
    """
    repo = app.gh_repo
    try:
        run = repo.get_workflow_run(int(run_id))
    except github.GithubException as e:
        raise click.ClickException(f"Could not fetch workflow run {run_id}: {e}") from e

    workflow_name = workflow_override or run.name
    pr_number = pr_override if pr_override is not None else _resolve_pr_number(run)
    head_sha = run.head_sha

    # Staleness is vetted by the caller (`_fix_one_run` for an explicit run
    # ID, `_resolve_target_runs` for `all`), so don't re-skip on it here:
    # within `fix all`, applying one fix moves HEAD past the not-yet-handled
    # runs that were already deemed valid. `check_staleness=False` preserves
    # that while routing the branch/config/conclusion decision through the
    # shared classifier.
    decision = classify_run(
        app,
        run,
        workflow_name=workflow_name,
        pr_number=pr_number,
        check_staleness=False,
    )
    if not decision.actionable:
        click.echo(_skip_message(decision, run, workflow_name))
        yield None
        return

    config = app.config.workflows.get(workflow_name)
    assert config is not None  # actionable ⇒ workflow is configured + enabled
    assert decision.pr_number is not None  # actionable ⇒ PR resolved

    builder_artifacts_dir: str | None = None
    builder_head_sha: str | None = None
    tmp_dir: tempfile.TemporaryDirectory | None = None

    try:
        if decision.kind == "failure":
            if artifacts_dir_override:
                builder_artifacts_dir = artifacts_dir_override
            else:
                tmp_dir = tempfile.TemporaryDirectory(prefix="mergai-ci-")
                download_workflow_run_artifacts(run, Path(tmp_dir.name))
                builder_artifacts_dir = tmp_dir.name
        else:  # "code_scanning" — findings already confirmed by the classifier
            builder_head_sha = head_sha

        builder = get_context_builder(app, config.context.type)
        context = builder.build_context(
            config.context,
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=decision.pr_number,
            artifacts_dir=builder_artifacts_dir,
            head_sha=builder_head_sha,
        )
        yield context, config
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _resolve_pr_number(run: "github.WorkflowRun.WorkflowRun") -> int | None:
    """Return the first PR number associated with a workflow run, or None."""
    prs = run.pull_requests or []
    if not prs:
        return None
    return prs[0].number


def _code_scanning_has_findings(
    app: AppContext, *, tool_name: str, head_sha: str, pr_number: int
) -> bool:
    """Return True if Code Scanning has results for this commit + tool.

    ``tool_name`` is the Code Scanning tool/driver name to query, resolved by
    the caller from ``context.code_scanning_tool_name`` (falling back to the
    workflow name when unset).
    """
    builder = get_context_builder(app, "sarif")
    if not isinstance(builder, SARIFContextBuilder):  # defensive
        return False
    analysis = builder.find_code_scanning_analysis(
        tool_name=tool_name, head_sha=head_sha, pr_number=pr_number
    )
    return analysis is not None and analysis.get("results_count", 0) > 0
