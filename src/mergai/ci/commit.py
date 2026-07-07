"""Committing a CI-fix solution to the working tree.

The CI-specific counterpart to :meth:`AppContext.commit_solution` (the
conflict-resolution path). Kept out of :class:`AppContext` so the CI commit
logic lives next to the rest of the ``mergai.ci`` package; ``AppContext``
keeps a thin delegator.
"""

import os
from typing import Any

from ..solution_types import CI_FIX
from ..utils import git_utils
from ..utils.formatters import _render_solution_body


# ``app`` is an :class:`~mergai.app.AppContext`, typed ``Any`` so this module
# does not import ``AppContext`` — that would create an
# ``app`` ↔ ``mergai.ci.commit`` import cycle. It must provide ``note``,
# ``repo``, ``config``, ``commit_footer``, and ``add_selective_note``.
def commit_ci_fix_solution(app: Any, solution_idx: int) -> None:
    """Commit the CI-fix solution at ``solution_idx`` and attach the note.

    Mirrors :meth:`AppContext.commit_solution`'s structure and voice for the
    CI-fix path: title `Fix <workflow> failure for merge commit
    '<sha>' into <target_branch>` (echoes "Resolve conflicts for
    merge commit ..." / "Merge commit ... into ..."), body with
    the agent's summary + Resolved / Unresolved / Modified
    sections + a ``CI:`` trailer carrying the run id and the
    attempt index for traceability. Stages every file the agent
    touched, commits, and attaches the solution as a selective
    git note via ``add_selective_note(sha, ["solutions[N]"])``.

    The solution at ``solution_idx`` must have ``type == "ci_fix"``
    and a populated ``request`` dict (workflow / run_id /
    attempt_number). Raises if the working tree is clean — that
    means the agent claimed a fix but didn't touch any files (which
    ``validate_solution`` should have caught earlier).
    """
    if not app.note.has_solutions or app.note.solutions is None:
        raise Exception("No solutions in note.")
    if solution_idx >= len(app.note.solutions):
        raise Exception(
            f"Solution index {solution_idx} out of range "
            f"(have {len(app.note.solutions)})."
        )

    solution = app.note.solutions[solution_idx]
    if solution.get("type") != CI_FIX:
        raise Exception(
            f"Solution at {solution_idx} is "
            f"type={solution.get('type')!r}, expected 'ci_fix'."
        )
    if not app.repo.is_dirty(untracked_files=True):
        raise Exception(
            "No changes to commit — agent reported a fix but the "
            "working tree is clean."
        )

    request = solution.get("request") or {}
    response = solution.get("response") or {}
    workflow = request.get("workflow", "ci")
    attempt_number = request.get("attempt_number", "?")
    run_id = request.get("run_id", "?")
    workflow_config = app.config.workflows.get(workflow)
    max_attempts = workflow_config.max_attempts if workflow_config is not None else "?"

    # Stage every file the agent touched (resolved + modified).
    # Untracked files won't show up in index.diff(None), so add them
    # explicitly via git index.add — same approach commit_solution uses.
    # A file the agent deleted to resolve a conflict no longer exists on
    # disk, so it must be staged via index.remove instead — index.add
    # would try to read it and raise FileNotFoundError.
    files_to_stage = list(response.get("resolved", {}).keys())
    files_to_stage += list(response.get("modified", {}).keys())
    working_dir = app.repo.working_tree_dir or ""
    files_to_add = []
    files_to_remove = []
    for f in files_to_stage:
        if os.path.exists(os.path.join(working_dir, f)):
            files_to_add.append(f)
        else:
            files_to_remove.append(f)
    if files_to_add:
        app.repo.index.add(files_to_add)
    if files_to_remove:
        app.repo.index.remove(files_to_remove)

    # Build commit message matching mergai's voice. The title is
    # configurable (commit.ci_fix_title_format) so it can mirror the PR
    # title style; tokens are substituted below.
    target_branch = app.note.merge_info.target_branch
    merge_sha = git_utils.short_sha(app.note.merge_info.merge_commit_sha)
    title = app.config.commit.ci_fix_title_format
    for token, value in {
        "%(workflow)": workflow,
        "%(merge_commit_sha)": app.note.merge_info.merge_commit_sha,
        "%(merge_commit_short_sha)": merge_sha,
        "%(target_branch)": target_branch,
    }.items():
        title = title.replace(token, str(value))
    message = f"{title}\n\n"
    message += _render_solution_body(
        response.get("summary", ""),
        response.get("resolved", {}),
        response.get("unresolved", {}),
        response.get("modified", {}),
    )
    message += (
        f"CI: {workflow} run {run_id} (attempt {attempt_number} of {max_attempts})\n\n"
    )
    message += app.commit_footer

    app.repo.index.commit(message)
    app.add_selective_note(app.repo.head.commit.hexsha, [f"solutions[{solution_idx}]"])
