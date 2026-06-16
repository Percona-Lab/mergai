"""Committing a review-fix solution to the working tree.

The review-fix counterpart to :func:`mergai.ci.commit.commit_ci_fix_solution`.
Stages every file the agent touched, builds a commit message in mergai's
voice, commits, and attaches the solution as a selective git note.
"""

from typing import Any

from ..solution_types import REVIEW_FIX
from ..utils import git_utils
from ..utils.formatters import _render_solution_body


# ``app`` is an :class:`~mergai.app.AppContext`, typed ``Any`` to avoid an
# ``app`` ↔ ``mergai.review.commit`` import concern. It must provide ``note``,
# ``repo``, ``config``, ``commit_footer``, and ``add_selective_note``.
def commit_review_fix_solution(app: Any, solution_idx: int) -> None:
    """Commit the review-fix solution at ``solution_idx`` and attach the note.

    Mirrors :func:`mergai.ci.commit.commit_ci_fix_solution`: title from
    ``commit.review_fix_title_format``, body with the agent's summary +
    Resolved / Unresolved / Modified sections + a ``Review:`` trailer with the
    addressed/total comment counts (no PR number - it can be ambiguous in
    history). Raises if the working tree is clean (the agent claimed a fix but
    touched nothing).
    """
    if not app.note.has_solutions or app.note.solutions is None:
        raise Exception("No solutions in note.")
    if solution_idx >= len(app.note.solutions):
        raise Exception(
            f"Solution index {solution_idx} out of range "
            f"(have {len(app.note.solutions)})."
        )

    solution = app.note.solutions[solution_idx]
    if solution.get("type") != REVIEW_FIX:
        raise Exception(
            f"Solution at {solution_idx} is "
            f"type={solution.get('type')!r}, expected 'review_fix'."
        )
    if not app.repo.is_dirty(untracked_files=True):
        raise Exception(
            "No changes to commit - agent reported a fix but the "
            "working tree is clean."
        )

    request = solution.get("request") or {}
    response = solution.get("response") or {}
    pr_number = request.get("pr_number", "?")
    addressed = response.get("addressed", {}) or {}
    unaddressed = response.get("unaddressed", {}) or {}

    # Stage every file the agent touched (resolved + modified). Untracked
    # files won't show up in index.diff(None), so add them explicitly.
    files_to_stage = list(response.get("resolved", {}).keys())
    files_to_stage += list(response.get("modified", {}).keys())
    if files_to_stage:
        app.repo.index.add(files_to_stage)

    target_branch = app.note.merge_info.target_branch
    merge_sha = git_utils.short_sha(app.note.merge_info.merge_commit_sha)
    title = app.config.commit.review_fix_title_format
    for token, value in {
        "%(pr_number)": pr_number,
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
    total = len(addressed) + len(unaddressed)
    message += f"Review: addressed {len(addressed)} of {total} comment(s)\n\n"
    message += app.commit_footer

    app.repo.index.commit(message)
    app.add_selective_note(app.repo.head.commit.hexsha, [f"solutions[{solution_idx}]"])
