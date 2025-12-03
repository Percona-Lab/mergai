from enum import Enum
from typing import Optional


class AgentErrorType(str, Enum):
    PARSING_RESULT = "Error parsing agent result"
    AGENT_EXECUTION = "Error running agent"


class AgentError(Exception):
    def __init__(self, error_type: AgentErrorType, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(f"{error_type.value}: {message}")


class AgentResult:
    def __init__(
        self, result: Optional[dict] = None, error: Optional[AgentError] = None
    ):
        self._result = result
        self._error = error

    def success(self) -> bool:
        return self._error is None

    def error(self) -> AgentError:
        return self._error

    def result(self) -> Optional[dict]:
        return self._result
