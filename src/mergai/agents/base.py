class Agent:
    def __init__(self, model: str):
        self._model = model

    def get_model(self) -> str:
        return self._model

    def is_cli(self) -> bool:
        return isinstance(self, CliAgent)

    def to_cli(self) -> "CliAgent":
        if self.is_cli():
            return self  # type: ignore
        raise TypeError("Agent is not a CLI agent")


class CliAgent(Agent):
    pass
