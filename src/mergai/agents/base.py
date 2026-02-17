from typing import Optional


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

    def get_session_data(self) -> Optional[dict]:
        """Get session data from the last agent run.

        Override this method in subclasses to provide session data
        that can be saved for debugging failed runs.

        Returns:
            Dict containing session data, or None if not available.
        """
        return None

    def get_session_id(self) -> Optional[str]:
        """Get session ID from the last agent run.

        Returns:
            Session ID string, or None if not available.
        """
        return None


class CliAgent(Agent):
    pass
