"""Strategy registry and factory.

This module provides a central registry for all merge-pick strategies and
a factory function to create strategy instances from configuration data.
"""

from typing import Optional, Dict, Tuple, Type
import logging

from .base import MergePickStrategy
from .huge_commit import HugeCommitStrategyConfig, HugeCommitStrategy
from .important_files import ImportantFilesStrategyConfig, ImportantFilesStrategy
from .branching_point import BranchingPointStrategyConfig, BranchingPointStrategy
from .conflict import ConflictStrategyConfig, ConflictStrategy
from .most_recent import MostRecentStrategyConfig, MostRecentStrategy

log = logging.getLogger(__name__)

# Registry mapping config keys to (ConfigClass, StrategyClass)
# To add a new strategy:
# 1. Create the strategy module with Config, Result, and Strategy classes
# 2. Import them here
# 3. Add an entry to STRATEGY_REGISTRY
STRATEGY_REGISTRY: Dict[str, Tuple[Type, Type[MergePickStrategy]]] = {
    "huge_commit": (HugeCommitStrategyConfig, HugeCommitStrategy),
    "important_files": (ImportantFilesStrategyConfig, ImportantFilesStrategy),
    "branching_point": (BranchingPointStrategyConfig, BranchingPointStrategy),
    "conflict": (ConflictStrategyConfig, ConflictStrategy),
    "most_recent": (MostRecentStrategyConfig, MostRecentStrategy),
}


def create_strategy(strategy_type: str, data) -> Optional[MergePickStrategy]:
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
