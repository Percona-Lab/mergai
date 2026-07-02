import json
import os
import subprocess
from pathlib import Path

from ..utils.output import echo_err as _echo
from .base import CliAgent
from .env import agent_subprocess_env
from .error import AgentError, AgentErrorType, AgentResult
from .response_utils import parse_response_json


class ClaudeCLIAgent(CliAgent):
    """Claude CLI Agent for running prompts via Claude Code CLI.

    This agent integrates with the Claude Code CLI (claude command)
    to run AI-powered coding tasks. It uses JSON streaming output
    for structured communication.
    """

    def __init__(self, model: str, yolo: bool = False, debug: bool = False):
        """Initialize the Claude CLI agent.

        Args:
            model: The model to use (e.g., "claude-sonnet-4-20250514" or "sonnet").
            yolo: Enable bypass of all permission checks (--dangerously-skip-permissions).
            debug: Enable debug logging.
        """
        super().__init__(model)
        self.session_id: str | None = None
        self.yolo = yolo
        self.debug = debug
        self._session_data: dict | None = None

    def get_version(self) -> str:
        """Get the Claude CLI version.

        Returns:
            Version string from the CLI.

        Raises:
            AgentError: If the version command fails.
        """
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"Claude CLI error: {result.stderr.strip()}",
            )

        return result.stdout.strip()

    def _get_storage_dir(self) -> Path:
        """Get the Claude storage directory.

        Returns:
            Path to the storage directory (~/.claude/projects/).
        """
        return Path.home() / ".claude" / "projects"

    def _encode_project_path(self, cwd: str) -> str:
        """Encode a project path for use in Claude's storage directory.

        Claude encodes paths by replacing '/' with '-'.
        e.g., '/home/user/project' -> '-home-user-project'

        This method normalizes the path to POSIX format for cross-platform
        compatibility, handling Windows backslashes and drive letters.

        Args:
            cwd: The current working directory path.

        Returns:
            Encoded path string suitable for directory name.
        """
        # Normalize to POSIX format for cross-platform compatibility
        normalized = Path(cwd).resolve().as_posix()
        return normalized.replace("/", "-")

    def _get_project_dir(self, cwd: str | None = None) -> Path | None:
        """Get the project-specific storage directory.

        Args:
            cwd: Optional working directory. If not provided, uses current directory.

        Returns:
            Path to the project directory, or None if it doesn't exist.
        """
        if cwd is None:
            cwd = os.getcwd()

        storage_dir = self._get_storage_dir()
        project_name = self._encode_project_path(cwd)
        project_dir = storage_dir / project_name

        if project_dir.exists():
            return project_dir
        return None

    def read_session(self, session_id: str) -> dict | None:
        """Read session data from Claude's storage directory.

        Claude stores sessions as JSONL files in project-specific directories.

        Args:
            session_id: The session ID (UUID) to look up.

        Returns:
            Dict containing parsed session data, or None if not found.
        """
        storage_dir = self._get_storage_dir()

        if not storage_dir.exists():
            return None

        # First, check the current project's directory (most likely location)
        current_project_dir = self._get_project_dir()
        if current_project_dir is not None:
            session_file = current_project_dir / f"{session_id}.jsonl"
            if session_file.exists():
                try:
                    return self._parse_session_file(session_file)
                except (json.JSONDecodeError, OSError):
                    pass  # Fall through to full scan

        # Fallback: Search through all project directories for the session
        for project_dir in storage_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Skip current project dir (already checked)
            if current_project_dir is not None and project_dir == current_project_dir:
                continue

            session_file = project_dir / f"{session_id}.jsonl"
            if session_file.exists():
                try:
                    return self._parse_session_file(session_file)
                except (json.JSONDecodeError, OSError):
                    continue

        return None

    def _parse_session_file(self, session_file: Path) -> dict:
        """Parse a Claude session JSONL file.

        Args:
            session_file: Path to the session JSONL file.

        Returns:
            Dict containing parsed session data with messages list.
        """
        messages = []
        metadata: dict = {}

        with open(session_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry_type = entry.get("type")

                    if entry_type in ("user", "assistant"):
                        messages.append(entry)
                    elif entry_type == "summary":
                        metadata["summary"] = entry.get("summary")
                    elif entry_type == "last-prompt":
                        metadata["lastPrompt"] = entry.get("lastPrompt")

                except json.JSONDecodeError:
                    continue

        return {"messages": messages, "metadata": metadata}

    def read_messages(self, session_id: str) -> list[dict]:
        """Read all messages for a session.

        Args:
            session_id: The session ID to read messages for.

        Returns:
            List of message dictionaries.
        """
        session_data = self.read_session(session_id)
        if session_data is None:
            return []
        messages: list[dict] = session_data.get("messages", [])
        return messages

    def parse_stats(self, result_data: dict) -> dict:
        """Parse token usage statistics from result data.

        Args:
            result_data: The result dict containing modelUsage from Claude CLI output.

        Returns:
            Dict containing token usage statistics per model.
        """
        model_usage = result_data.get("modelUsage", {})
        models: dict[str, dict] = {}

        for model_name, usage in model_usage.items():
            tokens: dict[str, int] = {}

            # Map Claude CLI token fields to our format
            if "inputTokens" in usage:
                tokens["input"] = usage["inputTokens"]
            if "outputTokens" in usage:
                tokens["output"] = usage["outputTokens"]
            if "cacheReadInputTokens" in usage:
                tokens["cache_read"] = usage["cacheReadInputTokens"]
            if "cacheCreationInputTokens" in usage:
                tokens["cache_creation"] = usage["cacheCreationInputTokens"]

            if tokens:
                models[model_name] = {"tokens": tokens}

            # Include cost if available (initialize model dict if needed)
            if "costUSD" in usage:
                if model_name not in models:
                    models[model_name] = {}
                models[model_name]["cost_usd"] = usage["costUSD"]

        return {"models": models}

    def read_session_content(self, session_id: str) -> dict:
        """Read session messages and thoughts, excluding tool outputs.

        Extracts assistant messages content without tool input/output data.

        Args:
            session_id: The session ID to read content for.

        Returns:
            Dict with 'messages' list containing message content.
        """
        messages = self.read_messages(session_id)
        result_messages = []

        for msg in messages:
            if msg.get("type") != "assistant":
                continue

            msg_data = msg.get("message", {})
            content_parts = msg_data.get("content", [])

            msg_content: dict = {
                "role": "assistant",
                "model": msg_data.get("model"),
                "thoughts": [],
                "tools_used": [],
            }

            for part in content_parts:
                part_type = part.get("type")

                if part_type == "text":
                    text = part.get("text", "").strip()
                    if text:
                        msg_content["thoughts"].append(text)

                elif part_type == "tool_use":
                    tool_name = part.get("name", "unknown")
                    msg_content["tools_used"].append(tool_name)

            # Only include messages that have content
            if msg_content["thoughts"] or msg_content["tools_used"]:
                result_messages.append(msg_content)

        return {"messages": result_messages}

    def build_args(
        self, prompt: str, allowed_write_paths: list[Path] | None = None
    ) -> list:
        """Build command line arguments for claude CLI.

        Args:
            prompt: The prompt to send to the agent.
            allowed_write_paths: Optional list of paths the agent is allowed to write to.

        Returns:
            List of command line arguments.
        """
        args = [
            "claude",
            "--print",  # Non-interactive mode
            "--output-format",
            "stream-json",  # JSON event streaming
            "--verbose",  # Required for stream-json
        ]

        if self.yolo:
            args.append("--dangerously-skip-permissions")
        elif allowed_write_paths:
            # Use acceptEdits mode to auto-approve write operations.
            # NOTE: Claude CLI's acceptEdits mode does not restrict edits to specific paths;
            # it broadly auto-approves all file edits. The allowed_write_paths parameter is
            # accepted for interface compatibility but cannot enforce path-level restrictions
            # in print mode. For strict path isolation, consider running the agent in a
            # sandboxed working directory.
            args.extend(["--permission-mode", "acceptEdits"])

        if self.debug:
            args.append("--debug")

        if self.session_id:
            args.extend(["--resume", self.session_id])

        if self.get_model():
            args.extend(["--model", self.get_model()])

        # Add the prompt as the final argument
        args.append(prompt)

        return args

    def run_prompt(
        self, prompt: str, allowed_write_paths: list[Path] | None = None
    ) -> dict:
        """Execute the prompt using Claude CLI.

        Args:
            prompt: The prompt to send to the agent.
            allowed_write_paths: Optional list of paths the agent is allowed to write to.

        Returns:
            Dict containing the response and metadata.

        Raises:
            AgentError: If the command execution fails.
        """
        args = self.build_args(prompt, allowed_write_paths)

        _echo(f"Running command: '{' '.join(args)}'")
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout to avoid deadlock
            text=True,
            # Strip GitHub write credentials so the agent cannot push or issue
            # GitHub write APIs; local file/git operations are unaffected.
            env=agent_subprocess_env(),
        )

        result: dict = {}
        response_parts: list[str] = []

        if proc.stdout is None:
            raise AgentError(AgentErrorType.AGENT_EXECUTION, "stdout is None")

        for line in proc.stdout:
            _echo(f"claude: {line}", nl=False)
            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                # Extract session ID from init event
                if event_type == "system" and event.get("subtype") == "init":
                    if "session_id" in event and not self.session_id:
                        self.session_id = event["session_id"]
                        _echo(f"claude: received session_id: {self.session_id}")

                elif event_type == "assistant":
                    # Assistant message with content
                    message = event.get("message", {})
                    content_parts = message.get("content", [])
                    for part in content_parts:
                        if part.get("type") == "text":
                            text = part.get("text", "")
                            response_parts.append(text)

                elif event_type == "result":
                    # Final result event with stats
                    result["status"] = event.get("subtype", "unknown")
                    result["is_error"] = event.get("is_error", False)
                    result["duration_ms"] = event.get("duration_ms", 0)
                    result["num_turns"] = event.get("num_turns", 0)
                    result["stop_reason"] = event.get("stop_reason")
                    result["total_cost_usd"] = event.get("total_cost_usd", 0)

                    # Extract token usage from modelUsage
                    if "modelUsage" in event:
                        result["modelUsage"] = event["modelUsage"]
                        result["stats"] = self.parse_stats(event)

                    # Store the final text result
                    if "result" in event:
                        result["text_result"] = event["result"]

                    _echo(f"claude: received result: {result.get('status')}")

                elif event_type == "rate_limit_event":
                    # Log rate limit info
                    rate_info = event.get("rate_limit_info", {})
                    if rate_info.get("status") != "allowed":
                        _echo(f"claude: rate limit status: {rate_info}")

            except json.JSONDecodeError:
                # Non-JSON line, might be status message
                continue

        result["response"] = "".join(response_parts)

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

        rc = proc.wait()
        if rc != 0:
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"claude: exited with code {rc}",
            )

        # Check if result indicates an error
        if result.get("is_error"):
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"claude: execution failed with status {result.get('status')}",
            )

        return result

    def run(
        self,
        prompt: str,
        response_file: Path | None = None,
        allowed_write_paths: list[Path] | None = None,
    ) -> AgentResult:
        """Run the agent with the given prompt.

        Args:
            prompt: The prompt to send to the agent.
            response_file: Optional path to file where agent should write JSON response.
                          If provided, response is read from this file instead of stdout.
            allowed_write_paths: Optional list of paths the agent is allowed to write to.
                Used to grant specific write permissions without enabling full yolo mode.

        Returns:
            AgentResult with the result or error.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")

        _echo(f"Running Claude CLI agent with prompt:\n{prompt}")

        try:
            result = self.run_prompt(prompt, allowed_write_paths)
        except AgentError as e:
            _echo(f"Agent execution error: {e}")
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
                _echo(
                    "Warning: Response file not found, falling back to stdout parsing..."
                )
                try:
                    result = parse_response_json(result)
                except AgentError as e:
                    _echo("Error parsing Claude CLI response")
                    _echo("--- Start of Claude CLI response ---")
                    _echo(f"{result.get('response', '')}")
                    _echo("--- End of Claude CLI response ---")
                    return AgentResult(error=e)
        else:
            # No response file provided: parse JSON from stdout response
            try:
                result = parse_response_json(result)
            except AgentError as e:
                _echo("Error parsing Claude CLI response")
                _echo("--- Start of Claude CLI response ---")
                _echo(f"{result.get('response', '')}")
                _echo("--- End of Claude CLI response ---")
                return AgentResult(error=e)

        try:
            version = self.get_version()
        except AgentError as e:
            _echo(f"Error getting version: {e}")
            return AgentResult(error=e)

        result["agent_info"] = {
            "agent_type": "claude_cli",
            "version": version,
        }

        return AgentResult(result=result)

    def get_session_data(self) -> dict | None:
        """Get session data from the last Claude CLI run.

        Returns:
            Dict containing session data, or None if no session available.
        """
        if self.session_id is None:
            return None
        return self._session_data or self.read_session(self.session_id)

    def get_session_id(self) -> str | None:
        """Get session ID from the last Claude CLI run.

        Returns:
            Session ID string, or None if no session available.
        """
        return self.session_id
