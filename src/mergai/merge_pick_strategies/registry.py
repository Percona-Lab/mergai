"""Strategy registry and factory.

This module provides a central registry for all merge-pick strategies and
a factory function to create strategy instances from configuration data.
"""

import logging

from .base import MergePickStrategy
from .branching_point import BranchingPointStrategy, BranchingPointStrategyConfig
from .conflict import ConflictStrategy, ConflictStrategyConfig
from .huge_commit import HugeCommitStrategy, HugeCommitStrategyConfig
from .important_files import ImportantFilesStrategy, ImportantFilesStrategyConfig
from .most_recent import MostRecentStrategy, MostRecentStrategyConfig

log = logging.getLogger(__name__)

# Registry mapping config keys to (ConfigClass, StrategyClass)
# To add a new strategy:
# 1. Create the strategy module with Config, Result, and Strategy classes
# 2. Import them here
# 3. Add an entry to STRATEGY_REGISTRY
STRATEGY_REGISTRY: dict[str, tuple[type, type[MergePickStrategy]]] = {
    "huge_commit": (HugeCommitStrategyConfig, HugeCommitStrategy),
    "important_files": (ImportantFilesStrategyConfig, ImportantFilesStrategy),
    "branching_point": (BranchingPointStrategyConfig, BranchingPointStrategy),
    "conflict": (ConflictStrategyConfig, ConflictStrategy),
    "most_recent": (MostRecentStrategyConfig, MostRecentStrategy),
}


def create_strategy(strategy_type: str, data) -> MergePickStrategy | None:
    """Create a strategy instance from config data.

    Args:
        strategy_type: The strategy type name (e.g., "huge_commit").
        data: The configuration data for this strategy.

    Returns:
        MergePickStrategy instance, or None if unknown type.
    """
    if strategy_type not in STRATEGY_REGISTRY:
        log.warning(f"Unknown strategy type: {strategy_type}")
        return None
    config_cls, strategy_cls = STRATEGY_REGISTRY[strategy_type]
    config = config_cls.from_dict(data)
    return strategy_cls(config)
