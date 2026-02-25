from .base import Agent
from .gemini_cli import GeminiCLIAgent

REGISTRY: dict[str, type[GeminiCLIAgent]] = {
    "gemini-cli": GeminiCLIAgent,
}


def create_agent(agent_type: str, model: str, yolo: bool) -> Agent:
    agent_class = REGISTRY.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent_class(model=model, yolo=yolo)
