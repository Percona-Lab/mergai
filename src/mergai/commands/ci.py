"""``mergai ci`` click command group.

Entry point invoked by the ``ci-fix`` job in ``.github/workflows/mergai.yml``
on every ``workflow_run.completed`` for a watched CI workflow on a
``mergai/*`` branch. Given a run ID (or ``"all"`` / a workflow name for
manual catch-up), this command:

* Fetches the workflow run (resolves workflow name, head SHA, PR number,
  conclusion) so callers don't have to extract them from the event payload.
* For ``conclusion == "failure"``: downloads the run's artifacts to a
  temp directory and dispatches to the configured context builder
  (which falls back to job logs when the SARIF is missing).
* For ``conclusion == "success"`` and a workflow whose context has
  ``code_scanning_check: true``: queries Code Scanning for findings on
  the run's commit. If any exist, dispatches to the SARIF builder via
  the Code Scanning path; otherwise exits no-op.
* Otherwise exits no-op.

The handler edits the working tree; this command commits the result as
a ``type: ci_fix`` solution on the note. The outer workflow pushes the
new commit so CI re-runs.

Every attempt that yields a usable verdict also records one entry in
``note.ci_comments`` — ``outcome="fixed"`` (with the commit) or
``outcome="unfixable"`` (no commit) — so ``mergai ci comment post`` can
publish an explanation to the PR. Those records live only in the cache
note: ``ci fix`` and the post step share a CI job.

``mergai ci fix <run-id>`` is what GitHub Actions calls. Manually
you can also pass ``"all"`` or a workflow name to process every
unprocessed actionable run on the current branch.
"""

import logging
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import click
import git
import github

from ..app import AppContext
from ..ci.context_builders import WorkflowContext, get_context_builder
from ..ci.context_builders.sarif import SARIFContextBuilder
from ..ci.handlers import get_handler
from ..config import WorkflowConfig
from ..solution_types import CI_FIX
from ..utils.artifact_downloader import download_workflow_run_artifacts

log = logging.getLogger(__name__)


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
def ci(app: AppContext, repo: str | None):
    """CI workflow integration commands."""
    if repo is None:
        raise click.ClickException(
            "GitHub repository not set. Use --repo or set GH_REPO environment variable."
        )
    app.gh_repo_str = repo


@ci.command()
@click.pass_obj
@click.argument("target", required=True)
@click.option(
    "--workflow",
    required=False,
    help="Override workflow name (default: from the run).",
)
@click.option(
    "--pr",
    required=False,
    type=int,
    help="Override PR number (default: first PR associated with the run).",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help=(
        "Pre-downloaded artifacts directory (default: download to a temp "
        "directory). Each artifact must already be extracted into a "
        "subdirectory named after it. Useful for manual / offline runs."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Act on runs whose head_sha has been superseded by newer commits "
        "on the branch (the same runs ``ci list`` shows as 'skip — "
        "superseded'). Use when the intervening commits are known to be "
        "unrelated to the failure. Runs whose head_sha is no longer "
        "reachable from HEAD at all (force-pushed away) are still "
        "skipped — those describe code that doesn't exist on this branch."
    ),
)
def fix(
    app: AppContext,
    target: str,
    workflow: str | None,
    pr: int | None,
    artifacts_dir: str | None,
    force: bool,
) -> None:
    """Apply a fix for one or more workflow runs on the current branch.

    \b
    TARGET can be:
      * a numeric run ID — process that specific run
      * "all"            — process every unprocessed actionable run on
                           the current branch (newest first), respecting
                           per-workflow `max_attempts`
      * a workflow name  — like "all" but filtered to that workflow

    For each selected run, builds a `WorkflowContext` via
    `build_workflow_context_for_run`, enforces `max_attempts`,
    dispatches to the configured handler, and commits the result as a
    `type: ci_fix` solution.
    """
    # `check_staleness` only applies at the entry — once a run has been
    # vetted (either by the user passing its run-id explicitly or by
    # `_resolve_target_runs` filtering the listing) the per-iteration
    # head_sha check just gets in our own way: applying a fix for run #1
    # commits something on top of HEAD, which would then disqualify
    # run #2 even though both runs were valid against the same commit at
    # the start of the loop.
    if target.isdigit():
        run_ids = [target]
        check_staleness = not force
    else:
        run_ids = _resolve_target_runs(app, target, force=force)
        if not run_ids:
            click.echo(f"No unprocessed actionable runs found for target '{target}'.")
            return
        click.echo(
            f"Found {len(run_ids)} unprocessed actionable run(s) for target '{target}'."
        )
        check_staleness = False

    for run_id in run_ids:
        _fix_one_run(
            app,
            run_id,
            workflow_override=workflow,
            pr_override=pr,
            artifacts_dir_override=artifacts_dir,
            check_staleness=check_staleness,
        )


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
    try:
        app.repo.git.merge_base("--is-ancestor", run.head_sha, "HEAD")
        return "superseded"
    except git.GitCommandError:
        # status 1 → not an ancestor; anything else (e.g. unknown SHA) →
        # also obsolete from our perspective. We can't act on a SHA we
        # don't have or can't reach.
        return "obsolete"


def _fix_one_run(
    app: AppContext,
    run_id: str,
    *,
    workflow_override: str | None,
    pr_override: int | None,
    artifacts_dir_override: str | None,
    check_staleness: bool = True,
) -> None:
    """Apply a fix for a single workflow run.

    Body of the per-run flow, hoisted out so that the top-level ``fix``
    command can iterate over multiple runs for the ``all`` /
    workflow-name targets.

    Args:
        check_staleness: If True (default; for explicit run-id targets),
            reject the run when its head_sha isn't the current branch
            HEAD. ``fix all`` / ``fix <workflow>`` pass False because
            ``_resolve_target_runs`` has already vetted the runs at the
            top of the loop, and applying a fix for run #1 will move
            HEAD past run #2 even though both were originally valid.
    """
    if check_staleness:
        try:
            run = app.gh_repo.get_workflow_run(int(run_id))
        except github.GithubException as e:
            raise click.ClickException(
                f"Could not fetch workflow run {run_id}: {e}"
            ) from e
        head_status = _run_head_status(app, run)
        if head_status != "current":
            reason = (
                "superseded by newer commits"
                if head_status == "superseded"
                else "head_sha not reachable from HEAD (force-pushed?)"
            )
            click.echo(
                f"Run {run_id} ({run.head_sha[:7]}) is {head_status}: {reason}; "
                f"skipping."
            )
            return

    with build_workflow_context_for_run(
        app,
        run_id,
        workflow_override=workflow_override,
        pr_override=pr_override,
        artifacts_dir_override=artifacts_dir_override,
    ) as built:
        if built is None:
            return
        context, config = built

        existing = app.note.get_ci_solution_for_run(run_id)
        if existing is not None:
            click.echo(
                f"Run {run_id} was already processed "
                f"(solution recorded as attempt "
                f"{existing.get('request', {}).get('attempt_number', '?')}); "
                f"nothing to do."
            )
            return

        prior_solutions = app.note.get_ci_solutions(context.workflow_name)
        attempt_number = len(prior_solutions) + 1
        if attempt_number > config.max_attempts:
            click.echo(
                f"Max attempts ({config.max_attempts}) reached for "
                f"'{context.workflow_name}'; giving up."
            )
            _post_max_attempts_comment(
                app, context.pr_number, context.workflow_name, config
            )
            return

        click.echo(
            f"Handling '{context.workflow_name}' run {run_id} "
            f"(attempt {attempt_number} of {config.max_attempts}). "
            f"Context: {context.summary}"
        )

        handler = get_handler(app, config)
        agent_solution = handler.execute(context)

        if agent_solution is None:
            click.echo(
                f"Handler did not produce a solution for "
                f"'{context.workflow_name}'. Will retry on the next workflow run."
            )
            return

        # Agent investigated but produced no code change (empty resolved +
        # modified). The agent itself classifies why via response.status:
        #   * "already_resolved" -> the failure no longer applies to the
        #     current code (e.g. an earlier fix in this run addressed the same
        #     root cause). Not a problem; recorded calmly.
        #   * anything else ("unfixable" / missing) -> needs manual attention.
        # Whether mergai knows it was addressed is the agent's call, not a
        # heuristic. Either way record a ci_comment so the run isn't
        # re-investigated, but don't commit or burn a `max_attempts` slot.
        response = agent_solution.get("response", {}) or {}
        if not response.get("resolved") and not response.get("modified"):
            outcome: Literal["unfixable", "already_resolved"] = (
                "already_resolved"
                if response.get("status") == "already_resolved"
                else "unfixable"
            )
            comment_idx = _record_ci_comment(
                app,
                outcome=outcome,
                context=context,
                run_id=run_id,
                attempt_number=attempt_number,
                response=response,
                commit_sha=None,
            )
            click.echo(
                f"Agent produced no code change for '{context.workflow_name}' "
                f"run {run_id} (outcome={outcome}). "
                f"Recorded comment (ci_comments[{comment_idx}]); "
                f"no commit, no attempt slot consumed. "
                f"Run `mergai ci comment post {run_id}` to publish a PR comment."
            )
            return

        ci_solution = {
            "type": CI_FIX,
            "request": {
                "workflow": context.workflow_name,
                "run_id": run_id,
                "pr_number": context.pr_number,
                "attempt_number": attempt_number,
                "context_summary": context.summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            **agent_solution,
        }
        solution_idx = app.note.add_solution(ci_solution)
        app.save_note(app.note)

        try:
            app.commit_ci_fix_solution(solution_idx)
        except Exception as e:
            raise click.ClickException(f"Failed to commit CI fix: {e}") from e

        # Record the fix as a postable ci_comment too, so every attempt —
        # success or not — leaves an explanation for `ci comment post`.
        # The commit exists now; anchor the comment to it for the footer.
        comment_idx = _record_ci_comment(
            app,
            outcome="fixed",
            context=context,
            run_id=run_id,
            attempt_number=attempt_number,
            response=response,
            commit_sha=app.repo.head.commit.hexsha,
        )
        click.echo(
            f"Applied fix for '{context.workflow_name}' run {run_id} "
            f"(attempt {attempt_number} of {config.max_attempts}). "
            f"Recorded comment (ci_comments[{comment_idx}]); "
            f"run `mergai ci comment post {run_id}` to publish a PR comment."
        )


def _record_ci_comment(
    app: AppContext,
    *,
    outcome: Literal["fixed", "unfixable", "already_resolved"],
    context: WorkflowContext,
    run_id: str,
    attempt_number: int,
    response: dict,
    commit_sha: str | None,
) -> int:
    """Record a postable ci_comment for a fix attempt and save the note.

    Both terminal outcomes of ``ci fix`` that carry agent text record one
    entry here so `mergai ci comment post` can publish an explanation:

    * ``outcome="fixed"`` — the agent changed files; ``commit_sha`` is the
      commit `commit_ci_fix_solution` just created.
    * ``outcome="unfixable"`` — the agent investigated but produced no
      code change; ``commit_sha`` is ``None`` (no commit was made).

    The entry is a self-contained comment payload (it embeds the agent
    ``response`` so the renderer needs no lookup) and lives only in the
    cache note — `ci fix` and the post step share a CI job. Returns the
    index of the appended entry.
    """
    entry = {
        "outcome": outcome,
        "workflow": context.workflow_name,
        "run_id": run_id,
        "pr_number": context.pr_number,
        "attempt_number": attempt_number,
        "context_summary": context.summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "response": response,
        "posted_at": None,
        "posted_comment_url": None,
    }
    idx = app.note.add_ci_comment(entry)
    app.save_note(app.note)
    return idx


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


def _post_max_attempts_comment(
    app: AppContext, pr_number: int, workflow: str, config: WorkflowConfig
) -> None:
    """Post a PR comment when the per-workflow max-attempts cap is hit."""
    if app.gh is None:
        log.warning("GitHub auth not available; skipping max-attempts PR comment.")
        return
    body = (
        f"mergai gave up auto-fixing the **{workflow}** workflow after "
        f"{config.max_attempts} attempts. Manual intervention required."
    )
    try:
        app.gh_repo.get_pull(pr_number).create_issue_comment(body)
    except Exception as e:  # noqa: BLE001 — best-effort notification
        log.warning("Failed to post PR comment on #%s: %s", pr_number, e)


@ci.command(name="list")
@click.pass_obj
@click.option(
    "--branch",
    "-b",
    "branch_override",
    required=False,
    help="Branch to list runs for (default: current branch).",
)
@click.option(
    "--limit",
    "-n",
    default=20,
    type=int,
    show_default=True,
    help="Maximum number of recent runs to show.",
)
@click.option(
    "--check-findings/--no-check-findings",
    default=True,
    show_default=True,
    help=(
        "For passing runs whose workflow has 'code_scanning_check', query "
        "Code Scanning to determine the actual finding count. Disable to "
        "skip the extra API calls."
    ),
)
def list_runs(
    app: AppContext,
    branch_override: str | None,
    limit: int,
    check_findings: bool,
) -> None:
    """List recent workflow runs and their mergai status.

    For each configured workflow run on the branch, shows whether
    mergai has already applied a fix (matching ``ci_fix`` solution) or,
    if not, what action mergai would take if the run was handled now.
    """
    repo = app.gh_repo

    branch = branch_override
    if branch is None:
        try:
            branch = app.repo.active_branch.name
        except TypeError as e:
            raise click.ClickException(
                "HEAD is detached; pass --branch explicitly."
            ) from e

    # PyGithub's type stub asks for a Branch object, but the API
    # accepts a branch name string. Pass the string directly.
    runs = repo.get_workflow_runs(branch=branch)  # type: ignore[arg-type]

    rows: list[tuple[str, ...]] = []
    runs_list = _take_workflow_runs(runs, limit)
    for run in runs_list:
        if run.name not in app.config.workflows.workflows:
            continue
        # Drop runs whose head commit isn't reachable from HEAD — they
        # belong to a prior incarnation of this branch (force-pushed
        # away). They're still in GitHub's history for the branch name
        # but they're noise here.
        if _run_head_status(app, run) == "obsolete":
            continue
        status, notes = _list_run_status(app, run, check_findings=check_findings)
        rows.append(
            (
                str(run.id),
                run.name,
                run.conclusion or run.status or "-",
                (run.head_sha or "")[:8],
                status,
                notes,
            )
        )

    if not rows:
        click.echo(
            f"No configured workflow runs found for branch '{branch}' "
            f"(showing first {limit})."
        )
        return

    headers = ("Run ID", "Workflow", "Conclusion", "Head SHA", "Status", "Notes")
    click.echo(_format_ascii_table(headers, rows))


def _format_ascii_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a plain-ASCII table with `|` columns and `-` rules.

    No Unicode box-drawing characters — works in any terminal /
    log-aggregator without font surprises. Columns are padded to the
    widest cell; the last column is left as-is so long notes don't
    force wide gutters on everything else.
    """
    cols = list(zip(headers, *rows, strict=False))
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def render_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            str(cell).ljust(width) for cell, width in zip(row, widths, strict=False)
        ).rstrip()

    rule = "  ".join("-" * w for w in widths)
    lines = [render_row(headers), rule]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


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
    # unusual_conclusion: completed with cancelled / timed_out / neutral / …
    return "skip", f"conclusion '{run.conclusion}'"


# ---------------------------------------------------------------------------
# `mergai ci status` — aggregate the watched workflows' state for HEAD. The
# gate the CI auto-fix loop reads to decide between squash / fix / wait.
# ---------------------------------------------------------------------------


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


@ci.command(name="status")
@click.pass_obj
@click.option(
    "--state",
    "state_only",
    is_flag=True,
    default=False,
    help=(
        "Print only the aggregate state token "
        "(in-progress|success|failure|none) on stdout and nothing else, so "
        "a workflow can do STATE=$(mergai ci status --state)."
    ),
)
def status(app: AppContext, state_only: bool) -> None:
    """Aggregate the watched workflows' state for the current branch HEAD.

    Looks at the latest run of each watched workflow (``format``,
    ``clang-tidy``, ``build-and-test`` from ``.mergai/config.yml``) whose
    ``head_sha`` matches HEAD and reduces them to one token:

    \b
    * in-progress — at least one watched run for HEAD hasn't completed.
    * success     — every watched run for HEAD completed successfully.
    * failure     — all completed, but at least one didn't succeed.
    * none        — no watched runs for HEAD (e.g. all skipped).

    Exits 0 in every case; the state is communicated on stdout. With
    ``--state`` only the bare token is printed (for shell capture).
    """
    runs_by_workflow = _watched_runs_for_head(app)
    state = _aggregate_state(runs_by_workflow)

    if state_only:
        click.echo(state)
        return

    if runs_by_workflow:
        headers = ("Workflow", "Run ID", "Status", "Conclusion")
        rows: list[tuple[str, ...]] = [
            (
                name,
                str(run.id),
                run.status or "-",
                run.conclusion or "-",
            )
            for name, run in sorted(runs_by_workflow.items())
        ]
        click.echo(_format_ascii_table(headers, rows))
    else:
        click.echo("No watched workflow runs for the current HEAD.")
    click.echo(f"\nState: {state}")


# ---------------------------------------------------------------------------
# `mergai ci comment` — view and post the explanation `ci fix` records for
# every attempt (a fix it applied, or a failure it couldn't fix).
# ---------------------------------------------------------------------------


@ci.group(name="comment")
def comment() -> None:
    """View and publish the PR comment `ci fix` records for each attempt.

    \b
    Every time `mergai ci fix` invokes the agent and gets a usable
    verdict, it records one comment on the local note:
      * outcome `fixed`     — what mergai changed (a commit was made).
      * outcome `unfixable` — why it couldn't fix the failure (no commit,
                              no `max_attempts` slot consumed).
    `mergai ci comment list` shows recorded comments;
    `mergai ci comment post` publishes one (or all pending) to the PR.
    """


@comment.command(name="list")
@click.pass_obj
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Also print the rendered comment body for each entry.",
)
@click.option(
    "--pending",
    "pending_only",
    is_flag=True,
    default=False,
    help="Show only comments that haven't been posted yet.",
)
def comment_list(app: AppContext, verbose: bool, pending_only: bool) -> None:
    """List recorded CI comments.

    By default lists both pending and posted entries. Use ``--pending``
    to see only what ``mergai ci comment post`` would publish.
    """
    if not app.has_note or not app.note.ci_comments:
        click.echo("No CI comments recorded.")
        return

    entries = app.note.pending_ci_comments() if pending_only else app.note.ci_comments
    if not entries:
        click.echo("No pending CI comments.")
        return

    rows: list[tuple[str, ...]] = []
    for c in entries:
        rows.append(
            (
                str(c.get("run_id", "?")),
                str(c.get("workflow", "?")),
                str(c.get("outcome", "?")),
                str(c.get("pr_number", "?")),
                str(c.get("created_at", "?")),
                str(c.get("posted_at") or "pending"),
            )
        )

    headers = ("Run ID", "Workflow", "Outcome", "PR", "Recorded at", "Posted at")
    click.echo(_format_ascii_table(headers, rows))

    if verbose:
        for c in entries:
            click.echo("")
            click.echo(f"--- run {c.get('run_id')} ({c.get('workflow')}) ---")
            click.echo(_render_ci_comment(c))


@comment.command(name="post")
@click.pass_obj
@click.argument("target", required=False, default="all")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the comment that would be posted, but don't call GitHub.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-post even if the comment was already posted.",
)
@click.option(
    "--review-pr",
    type=int,
    default=None,
    help=(
        "Post the notification on this PR instead of the failing run's own "
        "PR. Use when the fix was relocated to a review PR (e.g. a main-branch "
        "failure whose fix lives on a semantic PR)."
    ),
)
def comment_post(
    app: AppContext,
    target: str,
    dry_run: bool,
    force: bool,
    review_pr: int | None,
) -> None:
    """Post a short CI-fix notification to the PR.

    For each recorded attempt, posts a one-line notice: which check was fixed
    in which commit (outcome ``fixed``), or that mergai could not fix it
    (``unfixable``). The full per-solution detail belongs in the PR body,
    maintained by ``mergai pr update``; this command only pings the PR so
    reviewers notice the change (body edits are silent on GitHub).

    \b
    TARGET (optional, default "all"):
      * "all" / omitted   — post every pending notification. No-op when
                            nothing is pending, so this is safe to run
                            unconditionally from CI / a mergai workflow.
      * a numeric run ID  — post the notification for that run.

    ``--review-pr`` redirects the notice to a review PR when the fix was
    relocated off the failing run's own PR.
    """
    comments = _resolve_comments_for_post(app, target, include_posted=force)
    if not comments:
        if target == "all":
            click.echo("No pending CI comments to post.")
        else:
            click.echo(f"No comment found for run '{target}'.")
        return

    if app.gh is None and not dry_run:
        raise click.ClickException("GitHub auth not available; cannot post PR comment.")

    now = datetime.now(timezone.utc).isoformat()
    posted_any = False

    for c in comments:
        run_id = str(c.get("run_id"))
        if c.get("posted_at") is not None and not force:
            click.echo(
                f"Comment for run {run_id} was already posted at "
                f"{c['posted_at']}; skipping (use --force to re-post)."
            )
            continue

        pr_number = c.get("pr_number")
        target_pr = int(review_pr) if review_pr is not None else pr_number
        if target_pr is None:
            click.echo(f"Run {run_id}: no PR number recorded; cannot post.")
            continue

        body = _render_ci_notification(c)
        if dry_run:
            click.echo(f"--- would post for run {run_id} on #{target_pr} ---")
            click.echo(body)
            click.echo("--- end ---")
            continue

        posted = _create_pr_comment(app, int(target_pr), body, run_id)
        app.note.mark_ci_comment_posted(
            run_id, posted_at=now, comment_url=getattr(posted, "html_url", None)
        )
        posted_any = True
        click.echo(f"Posted CI notification for run {run_id} on #{target_pr}")

    if posted_any and not dry_run:
        app.save_note(app.note)


def _create_pr_comment(app: AppContext, pr_number: int, body: str, run_id: str) -> Any:
    """Create an issue comment on a PR, wrapping API errors."""
    try:
        return app.gh_repo.get_pull(int(pr_number)).create_issue_comment(body)
    except Exception as e:  # noqa: BLE001 — wrap external API errors
        raise click.ClickException(
            f"Failed to post PR comment for run {run_id} on #{pr_number}: {e}"
        ) from e


def _resolve_comments_for_post(
    app: AppContext, target: str, *, include_posted: bool
) -> list[dict]:
    """Return CI comments matching ``target``.

    ``target == "all"`` returns pending comments (or all if
    ``include_posted=True``); a specific run id returns that single
    comment (regardless of posted state — the caller's
    skip-unless-force check applies per entry).
    """
    if not app.has_note or not app.note.ci_comments:
        return []
    if target == "all":
        if include_posted:
            return list(app.note.ci_comments)
        return app.note.pending_ci_comments()
    comment = app.note.get_ci_comment_for_run(target)
    return [comment] if comment is not None else []


def _render_ci_comment(entry: dict) -> str:
    """Format a recorded CI fix attempt as Markdown for a PR comment.

    Branches on ``outcome``: a ``fixed`` entry explains what mergai
    changed; an ``unfixable`` entry explains why it couldn't and what
    needs manual attention. Both render from the agent ``response`` shape
    (``summary`` / ``resolved`` / ``unresolved`` / ``modified`` /
    ``review_notes``).
    """
    outcome = entry.get("outcome", "unfixable")
    workflow = entry.get("workflow", "?")
    run_id = entry.get("run_id", "?")
    attempt = entry.get("attempt_number", "?")
    created_at = entry.get("created_at", "?")
    commit_sha = entry.get("commit_sha")
    response = entry.get("response") or {}
    summary = (response.get("summary") or "").strip()
    review_notes = (response.get("review_notes") or "").strip()

    if outcome == "fixed":
        lines: list[str] = [
            f"### mergai auto-fixed `{workflow}` failure",
            "",
        ]
        if summary:
            lines += [summary, ""]
        changed = {
            **(response.get("resolved") or {}),
            **(response.get("modified") or {}),
        }
        if changed:
            lines += ["**Changed files:**", ""]
            for path, note in changed.items():
                note_str = note.strip() if isinstance(note, str) else str(note)
                lines.append(f"- `{path}`: {note_str}" if note_str else f"- `{path}`")
            lines.append("")
        if review_notes:
            lines += ["**Review notes:**", "", review_notes, ""]
        footer = f"_Workflow: `{workflow}` run {run_id} · attempt {attempt}_"
        if commit_sha:
            footer += f"\n_Commit: `{commit_sha[:12]}`_"
        lines.append(footer)
        return "\n".join(lines)

    # outcome == "unfixable"
    unresolved = response.get("unresolved") or {}
    lines = [
        f"### mergai: unable to auto-fix `{workflow}` failure",
        "",
    ]
    if summary:
        lines += [summary, ""]
    if unresolved:
        lines += ["**Unresolved:**", ""]
        for key, note in unresolved.items():
            note_str = note.strip() if isinstance(note, str) else str(note)
            lines.append(f"- `{key}`: {note_str}")
        lines.append("")
    if review_notes:
        lines += ["**Review notes:**", "", review_notes, ""]
    lines += [
        f"_Workflow: `{workflow}` run {run_id} · attempt {attempt}_",
        f"_Recorded: {created_at}_",
    ]
    return "\n".join(lines)


def _render_ci_notification(entry: dict) -> str:
    """Render the short PR notification for a CI-fix attempt.

    A terse, one-line notice — which check was fixed in which commit, or that
    it couldn't be fixed. The full per-solution detail lives in the PR body
    (maintained by ``mergai pr update``); this just pings the PR.
    """
    workflow = entry.get("workflow", "?")
    commit_sha = entry.get("commit_sha")
    outcome = entry.get("outcome")
    response = entry.get("response") or {}
    summary = (response.get("summary") or "").strip()
    review_notes = (response.get("review_notes") or "").strip()

    if outcome == "fixed":
        where = f" in commit `{commit_sha[:12]}`" if commit_sha else ""
        return f"The `{workflow}` check fixed{where}. See the PR comment for details."

    if outcome == "already_resolved":
        lines = [
            f"No fix needed for the `{workflow}` check — the agent found the "
            "failure already resolved in the current code."
        ]
        if summary:
            lines += ["", summary]
        return "\n".join(lines)

    # unfixable — include the agent's reasoning so reviewers know *why* it
    # could not be fixed, not just that it wasn't.
    lines = [
        f"The `{workflow}` check could not be auto-fixed; it needs manual attention."
    ]
    if summary:
        lines += ["", summary]
    unresolved = response.get("unresolved") or {}
    if unresolved:
        lines += ["", "**Unresolved:**"]
        for path, reason in unresolved.items():
            reason_str = reason.strip() if isinstance(reason, str) else str(reason)
            lines.append(f"- `{path}`: {reason_str}" if reason_str else f"- `{path}`")
    if review_notes:
        lines += ["", review_notes]
    return "\n".join(lines)
