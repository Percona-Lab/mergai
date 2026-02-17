"""Merge-pick strategies for commit prioritization.

This package provides a pluggable strategy system for prioritizing commits
during fork synchronization. Each strategy checks commits against specific
criteria and returns detailed results.

Example usage:
    from mergai.merge_pick_strategies import (
        MergePickStrategyContext,
        create_strategy,
    )

    # Create a strategy from config (expression string)
    strategy = create_strategy("huge_commit", "num_of_files > 100 or num_of_lines > 1000")

    # Check a commit
    context = MergePickStrategyContext(upstream_ref="origin/master")
    result = strategy.check(repo, commit, context)
    if result:
        print(f"Matched: {result.format_short()}")

To add a new strategy:
    1. Create a new module in this package
    2. Define a MergePickStrategyConfig dataclass with from_dict() class method
    3. Define a MergePickStrategyResult dataclass with format_short() method
    4. Define a Strategy class inheriting from MergePickStrategy
    5. Register the strategy in registry.py
"""

from .base import MergePickStrategy, MergePickStrategyResult, MergePickStrategyContext, MergePickCommit
from .huge_commit import HugeCommitStrategy, HugeCommitStrategyConfig, HugeCommitResult
from .important_files import (
    ImportantFilesStrategy,
    ImportantFilesStrategyConfig,
    ImportantFilesResult,
)
from .branching_point import (
    BranchingPointStrategy,
    BranchingPointStrategyConfig,
    BranchingPointResult,
)
from .conflict import ConflictStrategy, ConflictStrategyConfig, ConflictResult
from .registry import STRATEGY_REGISTRY, create_strategy

__all__ = [
    # Base classes
    "MergePickStrategy",
    "MergePickStrategyResult",
    "MergePickStrategyContext",
    "MergePickCommit",
    # Huge commit strategy
    "HugeCommitStrategy",
    "HugeCommitStrategyConfig",
    "HugeCommitResult",
    # Important files strategy
    "ImportantFilesStrategy",
    "ImportantFilesStrategyConfig",
    "ImportantFilesResult",
    # Branching point strategy
    "BranchingPointStrategy",
    "BranchingPointStrategyConfig",
    "BranchingPointResult",
    # Conflict strategy
    "ConflictStrategy",
    "ConflictStrategyConfig",
    "ConflictResult",
    # Registry
    "STRATEGY_REGISTRY",
    "create_strategy",
]
