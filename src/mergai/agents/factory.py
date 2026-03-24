from typing import TYPE_CHECKING

from .base import CliAgent
from .claude_cli import ClaudeCLIAgent
from .gemini_cli import GeminiCLIAgent

if TYPE_CHECKING:
    from typing import TypeAlias

    # Type alias for agent classes that accept model and yolo parameters
    CliAgentClass: TypeAlias = type[ClaudeCLIAgent] | type[GeminiCLIAgent]

REGISTRY: dict[str, "CliAgentClass"] = {
    "claude-cli": ClaudeCLIAgent,
    "gemini-cli": GeminiCLIAgent,
}


def create_agent(agent_type: str, model: str, yolo: bool) -> CliAgent:
    agent_class = REGISTRY.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent_class(model=model, yolo=yolo)
