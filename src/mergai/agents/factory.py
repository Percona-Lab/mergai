from .base import Agent
from .claude_cli import ClaudeCLIAgent
from .gemini_cli import GeminiCLIAgent
from .opencode_cli import OpenCodeCLIAgent

REGISTRY: dict[
    str, type[ClaudeCLIAgent] | type[GeminiCLIAgent] | type[OpenCodeCLIAgent]
] = {
    "claude-cli": ClaudeCLIAgent,
    "gemini-cli": GeminiCLIAgent,
    "opencode-cli": OpenCodeCLIAgent,
}


def create_agent(agent_type: str, model: str, yolo: bool) -> Agent:
    agent_class = REGISTRY.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent_class(model=model, yolo=yolo)
