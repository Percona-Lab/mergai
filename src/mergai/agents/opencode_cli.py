import json
import os
import subprocess
from pathlib import Path

import click

from .base import CliAgent
from .error import AgentError, AgentErrorType, AgentResult
from .response_utils import parse_response_json


class OpenCodeCLIAgent(CliAgent):
    """OpenCode CLI Agent for running prompts via opencode CLI.

    This agent integrates with the OpenCode CLI (https://opencode.ai)
    to run AI-powered coding tasks. It uses JSON streaming output
    for structured communication.
    """

    def __init__(self, model: str, yolo: bool = False, debug: bool = False):
        """Initialize the OpenCode CLI agent.

        Args:
            model: The model to use in provider/model format (e.g., "anthropic/claude-4-sonnet").
            yolo: Reserved for future use (auto-approve permissions).
            debug: Enable debug logging.
        """
        super().__init__(model)
        self.session_id: str | None = None
        self.yolo = yolo  # Reserved for future use
        self.debug = debug
        self._session_data: dict | None = None

    def get_version(self) -> str:
        """Get the OpenCode CLI version.

        Returns:
            Version string from the CLI.

        Raises:
            AgentError: If the version command fails.
        """
        result = subprocess.run(
            ["opencode", "--version"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"OpenCode CLI error: {result.stderr.strip()}",
            )

        return result.stdout.strip()

    def _get_storage_dir(self) -> Path:
        """Get the OpenCode storage directory.

        Returns:
            Path to the storage directory (~/.local/share/opencode/storage/).
        """
        return Path.home() / ".local" / "share" / "opencode" / "storage"

    def read_session(self, session_id: str) -> dict | None:
        """Read session data from OpenCode's storage directory.

        Args:
            session_id: The session ID to look up.

        Returns:
            Dict containing session data, or None if not found.
        """
        storage_dir = self._get_storage_dir()

        # Sessions are stored in per-project directories
        session_base_dir = storage_dir / "session"
        if not session_base_dir.exists():
            return None

        # Search through all project directories for the session
        for project_dir in session_base_dir.iterdir():
            if not project_dir.is_dir():
                continue

            session_file = project_dir / f"{session_id}.json"
            if session_file.exists():
                try:
                    with open(session_file) as f:
                        data: dict = json.load(f)
                        return data
                except (json.JSONDecodeError, OSError):
                    continue

        return None

    def read_messages(self, session_id: str) -> list[dict]:
        """Read all messages for a session.

        Args:
            session_id: The session ID to read messages for.

        Returns:
            List of message dictionaries.
        """
        storage_dir = self._get_storage_dir()
        message_dir = storage_dir / "message" / session_id

        if not message_dir.exists():
            return []

        messages = []
        for msg_file in sorted(message_dir.glob("*.json")):
            try:
                with open(msg_file) as f:
                    messages.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

        return messages

    def parse_stats(self, session_data: dict) -> dict:
        """Parse token usage statistics from session messages.

        Args:
            session_data: Session data dictionary (not used directly,
                         but kept for interface consistency).

        Returns:
            Dict containing token usage statistics per model.
        """
        if not self.session_id:
            return {}

        messages = self.read_messages(self.session_id)
        models: dict[str, dict] = {}

        for msg in messages:
            if "tokens" not in msg:
                continue

            tokens = msg["tokens"]
            provider_id = msg.get("providerID", "unknown")
            model_id = msg.get("modelID", "unknown")
            model_key = f"{provider_id}/{model_id}"

            if model_key not in models:
                models[model_key] = {"tokens": {}}

            for key, value in tokens.items():
                if isinstance(value, dict):
                    # Handle nested token counts (e.g., cache.read, cache.write)
                    for nested_key, nested_value in value.items():
                        full_key = f"{key}.{nested_key}"
                        if full_key not in models[model_key]["tokens"]:
                            models[model_key]["tokens"][full_key] = 0
                        models[model_key]["tokens"][full_key] += nested_value
                else:
                    if key not in models[model_key]["tokens"]:
                        models[model_key]["tokens"][key] = 0
                    models[model_key]["tokens"][key] += value

        return {"models": models}

    def read_session_content(self, session_id: str) -> dict:
        """Read session messages and thoughts, excluding tool outputs.

        Reads the part files for each message and extracts:
        - Text content (thoughts/responses from the assistant)
        - Tool call names (but NOT their input/output data)

        Args:
            session_id: The session ID to read content for.

        Returns:
            Dict with 'messages' list containing message content.
        """
        storage_dir = self._get_storage_dir()
        part_dir = storage_dir / "part"
        messages = self.read_messages(session_id)

        result_messages = []

        for msg in messages:
            msg_id = msg.get("id")
            role = msg.get("role")

            if not msg_id or role != "assistant":
                continue

            # Read parts for this message
            msg_part_dir = part_dir / msg_id
            if not msg_part_dir.exists():
                continue

            msg_content: dict = {
                "role": role,
                "model": msg.get("modelID"),
                "thoughts": [],
                "tools_used": [],
            }

            for part_file in sorted(msg_part_dir.glob("*.json")):
                try:
                    with open(part_file) as f:
                        part = json.load(f)

                    part_type = part.get("type")

                    if part_type == "text":
                        # Extract text content (thoughts/responses)
                        text = part.get("text", "").strip()
                        if text:
                            msg_content["thoughts"].append(text)

                    elif part_type == "tool":
                        # Only record tool name, NOT input/output
                        tool_name = part.get("tool", "unknown")
                        msg_content["tools_used"].append(tool_name)

                except (json.JSONDecodeError, OSError):
                    continue

            # Only include messages that have content
            if msg_content["thoughts"] or msg_content["tools_used"]:
                result_messages.append(msg_content)

        return {"messages": result_messages}

    def build_args(self, prompt: str) -> list:
        """Build command line arguments for opencode run.

        Args:
            prompt: The prompt to send to the agent.

        Returns:
            List of command line arguments.
        """
        args = [
            "opencode",
            "run",
            "--format",
            "json",  # JSON event streaming
        ]

        if self.debug:
            args.extend(["--log-level", "DEBUG"])

        if self.session_id:
            args.extend(["--session", self.session_id])

        if self.get_model():
            args.extend(["--model", self.get_model()])

        # Add the prompt as the final argument
        args.append(prompt)

        return args

    def run_prompt(self, prompt: str) -> dict:
        """Execute the prompt using opencode CLI.

        Args:
            prompt: The prompt to send to the agent.

        Returns:
            Dict containing the response and metadata.

        Raises:
            AgentError: If the command execution fails.
        """
        args = self.build_args(prompt)

        click.echo(f"Running command: '{' '.join(args)}'")
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )

        result: dict = {}
        response_parts: list[str] = []
        tokens: dict = {}

        if proc.stdout is None:
            raise AgentError(AgentErrorType.AGENT_EXECUTION, "stdout is None")

        for line in proc.stdout:
            click.echo(f"opencode: {line}", nl=False)
            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                # Extract session ID from events
                if "sessionID" in event and not self.session_id:
                    self.session_id = event["sessionID"]
                    click.echo(f"opencode: received session_id: {self.session_id}")

                part = event.get("part", {})

                if event_type == "text":
                    # Text content from the assistant
                    text = part.get("text", "")
                    response_parts.append(text)

                elif event_type == "step_finish":
                    # Step completion with token usage
                    result["status"] = part.get("reason", "stop")
                    result["cost"] = part.get("cost", 0)

                    # Extract token usage
                    if "tokens" in part:
                        tokens = part["tokens"]

                elif event_type == "tool_call":
                    # Tool calls (file operations, bash, etc.)
                    tool_name = part.get("tool", "")
                    click.echo(f"opencode: tool call: {tool_name}")

                elif event_type == "error":
                    # Error event
                    error_msg = event.get("error", "Unknown error")
                    raise AgentError(AgentErrorType.AGENT_EXECUTION, error_msg)

            except json.JSONDecodeError:
                # Non-JSON line, might be status message
                continue

        result["response"] = "".join(response_parts)
        result["tokens"] = tokens

        # Read session data after completion
        if self.session_id:
            session = self.read_session(self.session_id)
            if session is None:
                session = {}
            # Include session content (messages and thoughts, but not tool outputs)
            session_content = self.read_session_content(self.session_id)
            session["messages"] = session_content.get("messages", [])
            result["session"] = session
            self._session_data = session
            result["stats"] = self.parse_stats(session)

        rc = proc.wait()
        if rc != 0:
            stderr_output = proc.stderr.read() if proc.stderr else ""
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"opencode: exited with code {rc}, stderr: {stderr_output}",
            )

        return result

    def run(self, prompt: str, response_file: Path | None = None) -> AgentResult:
        """Run the agent with the given prompt.

        Args:
            prompt: The prompt to send to the agent.
            response_file: Optional path to file where agent should write JSON response.
                          If provided, response is read from this file instead of stdout.

        Returns:
            AgentResult with the result or error.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")

        click.echo(f"Running OpenCode CLI agent with prompt:\n{prompt}")

        try:
            result = self.run_prompt(prompt)
        except AgentError as e:
            click.echo(f"Agent execution error: {e}")
            return AgentResult(error=e)

        # If response_file is provided, try reading from file first
        if response_file is not None:
            if response_file.exists():
                # Preferred: read from file
                try:
                    response_text = response_file.read_text()
                    response_json = json.loads(response_text)
                    result["response"] = response_json
                except json.JSONDecodeError as e:
                    return AgentResult(
                        error=AgentError(
                            AgentErrorType.PARSING_RESULT,
                            f"Invalid JSON in response file: {e}",
                        )
                    )
            else:
                # Fallback: parse JSON from stdout response
                click.echo(
                    "Warning: Response file not found, falling back to stdout parsing..."
                )
                try:
                    result = parse_response_json(result)
                except AgentError as e:
                    click.echo("Error parsing OpenCode CLI response")
                    click.echo("--- Start of OpenCode CLI response ---")
                    click.echo(f"{result.get('response', '')}")
                    click.echo("--- End of OpenCode CLI response ---")
                    return AgentResult(error=e)

        try:
            version = self.get_version()
        except AgentError as e:
            click.echo(f"Error getting version: {e}")
            return AgentResult(error=e)

        result["agent_info"] = {
            "agent_type": "opencode_cli",
            "version": version,
        }

        return AgentResult(result=result)

    def get_session_data(self) -> dict | None:
        """Get session data from the last OpenCode CLI run.

        Returns:
            Dict containing session data, or None if no session available.
        """
        if self.session_id is None:
            return None
        return self._session_data or self.read_session(self.session_id)

    def get_session_id(self) -> str | None:
        """Get session ID from the last OpenCode CLI run.

        Returns:
            Session ID string, or None if no session available.
        """
        return self.session_id
