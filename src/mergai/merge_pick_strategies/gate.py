"""Deterministic merge gate.

The gate decides *whether* to merge now; it does not decide *which* commit to
merge to (that is the pick, deterministic or AI). It is a pure function over
already-computed data (fork status + prioritized commits + gate config), so it
needs no AI tokens and is safe to run in the unprivileged periodic phase. It is
also mode-agnostic: the same decision drives both deterministic and AI picks.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import MergeGateConfig
    from ..utils.git_utils import ForkStatus
    from .base import MergePickCommit


@dataclass
class GateDecision:
    """The gate's go/no-go verdict.

    Attributes:
        open: True if a merge should happen now.
        reason: Human-readable explanation (e.g. ``"force:conflict"``,
            ``"min_commits (63 >= 50)"``, ``"wait (12 < 50 commits; ...)"``).
    """

    open: bool
    reason: str


def evaluate_merge_gate(
    fork_status: "ForkStatus",
    prioritized: "list[MergePickCommit] | None",
    cfg: "MergeGateConfig",
) -> GateDecision:
    """Decide whether to merge now.

    Open if (in priority order):
      - any prioritized match's strategy is in ``force_strategies``
        (reason ``"force:<name>"``), or
      - ``commits_behind >= min_commits`` (reason ``"min_commits"``), or
      - the oldest unmerged commit is at least ``max_age_days`` old
        (reason ``"max_age"``).
    Otherwise wait.

    Args:
        fork_status: Fork divergence info (provides ``commits_behind`` and
            ``unmerged_oldest_age_days``).
        prioritized: Prioritized commits within the candidate window (used only
            for the force-strategy check). May be None/empty.
        cfg: The merge-gate configuration.

    Returns:
        A :class:`GateDecision`.
    """
    force = set(cfg.force_strategies or [])
    if force and prioritized:
        for pick in prioritized:
            if pick.strategy_name in force:
                return GateDecision(open=True, reason=f"force:{pick.strategy_name}")

    commits_behind = fork_status.commits_behind
    if commits_behind >= cfg.min_commits:
        return GateDecision(
            open=True,
            reason=f"min_commits ({commits_behind} >= {cfg.min_commits})",
        )

    age = fork_status.unmerged_oldest_age_days
    if age is not None and age >= cfg.max_age_days:
        return GateDecision(
            open=True,
            reason=f"max_age ({age:.1f}d >= {cfg.max_age_days}d)",
        )

    age_str = f"{age:.1f}d" if age is not None else "n/a"
    return GateDecision(
        open=False,
        reason=(
            f"wait ({commits_behind} < {cfg.min_commits} commits; "
            f"oldest {age_str} < {cfg.max_age_days}d)"
        ),
    )
