"""Pick strategies for commit prioritization.

This package provides a pluggable strategy system for prioritizing commits
during fork synchronization. Each strategy checks commits against specific
criteria and returns detailed results.

Example usage:
    from mergai.strategies import (
        StrategyContext,
        create_strategy,
    )

    # Create a strategy from config
    strategy = create_strategy("huge_commit", {"min_changed_files": 50})

    # Check a commit
    context = StrategyContext(upstream_ref="origin/master")
    result = strategy.check(repo, commit, context)
    if result:
        print(f"Matched: {result.format_short()}")

To add a new strategy:
    1. Create a new module in this package
    2. Define a StrategyConfig dataclass with from_dict() class method
    3. Define a StrategyResult dataclass with format_short() method
    4. Define a Strategy class inheriting from PickStrategy
    5. Register the strategy in registry.py
"""

from .base import PickStrategy, StrategyResult, StrategyContext
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
    "PickStrategy",
    "StrategyResult",
    "StrategyContext",
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
