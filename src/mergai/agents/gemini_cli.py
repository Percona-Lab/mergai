import json
import os
import subprocess
from pathlib import Path

from ..utils.output import echo_err as _echo
from .base import CliAgent
from .env import agent_subprocess_env
from .error import AgentError, AgentErrorType, AgentResult
from .response_utils import parse_response_json


class GeminiCLIAgent(CliAgent):
    def __init__(self, model: str, yolo: bool, debug: bool = False):
        super().__init__(model)
        self.session_id: str | None = None
        self.yolo = yolo
        self.debug = debug

    def get_version(self) -> str:
        result = subprocess.run(
            ["gemini", "--version"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"Gemini CLI error: {result.stderr.strip()}",
            )

        return result.stdout.strip()

    def read_session(self, session_id: str) -> dict | None:
        gemini_tmp_dir = Path.home() / ".gemini" / "tmp"

        for file in gemini_tmp_dir.iterdir():
            chats_dir = file / "chats"
            if chats_dir.exists() and chats_dir.is_dir():
                for chat_file in chats_dir.iterdir():
                    try:
                        with open(chat_file) as f:
                            chat_data: dict = json.load(f)
                            if chat_data.get("sessionId") == session_id:
                                return chat_data
                    except Exception:
                        continue
        return None

    def parse_stats(self, session_data: dict) -> dict:
        models: dict[str, dict] = {}
        for msg in session_data.get("messages", []):
            if "tokens" not in msg:
                continue
            tokens = msg["tokens"]
            model = msg.get("model", "unknown")

            for key, value in tokens.items():
                if model not in models:
                    models[model] = {"tokens": {}}
                if key not in models[model]["tokens"]:
                    models[model]["tokens"][key] = 0
                models[model]["tokens"][key] += value

        return {"models": models}

    def build_args(self, prompt: str) -> list:
        # Always use auto_edit: auto-approve file edits in the working repo, but
        # NOT other tools. yolo deliberately does NOT map to Gemini's "yolo"
        # approval mode (which auto-approves every tool, incl. shell commands
        # that could push or reach the remote). This mirrors the Claude adapter,
        # where yolo maps to acceptEdits rather than --dangerously-skip-permissions.
        args = [
            "gemini",
            "--approval-mode",
            "auto_edit",
            "-o",
            "stream-json",
        ]

        if self.debug:
            args.append("-d")

        if self.session_id:
            args.extend(["-r", self.session_id])

        if self.get_model():
            args.extend(["--model", self.get_model()])

        args.append("-p")
        args.append(prompt)

        return args

    def run_prompt(self, prompt: str):
        args = self.build_args(prompt)

        _echo(f"Running command: '{' '.join(args)}'")
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Strip GitHub write credentials so the agent cannot push or issue
            # GitHub write APIs; local file/git operations are unaffected.
            env=agent_subprocess_env(),
        )

        result: dict = {}
        response = ""
        if proc.stdout is None:
            raise AgentError(AgentErrorType.AGENT_EXECUTION, "stdout is None")
        for line in proc.stdout:
            _echo(f"gemini-cli: {line}", nl=False)
            event = json.loads(line)

            if event["type"] == "message" and event["role"] == "assistant":
                response += event["content"]

            if event["type"] == "init":
                self.session_id = event["session_id"]
                _echo(f"gemini-cli: received session_id: {self.session_id}")

            if event["type"] == "result":
                result["timestamp"] = event["timestamp"]
                result["status"] = event["status"]
                _echo(f"gemini-cli: received result: {result}")
                break

        result["response"] = response

        if self.session_id is not None:
            session = self.read_session(self.session_id)
            result["session"] = session
            if session is not None:
                result["stats"] = self.parse_stats(session)

        rc = proc.wait()
        if rc != 0:
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"gemini-cli: exited with code {rc}, stderr: {proc.stderr}",
            )

        return result

    def run(
        self,
        prompt: str,
        response_file: Path | None = None,
        allowed_write_paths: list[Path] | None = None,  # noqa: ARG002
    ) -> AgentResult:
        # Note: allowed_write_paths is accepted for interface compatibility but not used.
        # Gemini CLI's auto_edit mode already auto-approves file edits.
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")

        _echo(f"Running Gemini CLI agent with prompt:\n{prompt}")

        try:
            result = self.run_prompt(prompt)
        except AgentError as e:
            _echo(f"Agent execution error: {e}")
            return AgentResult(error=e)

        # If response_file is provided, try reading from file first
        if response_file is not None:
            if response_file.exists():
                try:
                    response_text = response_file.read_text()
                    response_json = json.loads(response_text)
                    result["response"] = response_json
                except json.JSONDecodeError as e:
                    # Fallback to parsing from stdout if JSON in file is invalid
                    _echo(
                        f"Warning: Invalid JSON in response file ({e}), "
                        "falling back to stdout parsing..."
                    )
                    try:
                        result = parse_response_json(result)
                    except AgentError:
                        # If legacy parsing also fails, return the original JSON error
                        return AgentResult(
                            error=AgentError(
                                AgentErrorType.PARSING_RESULT,
                                f"Invalid JSON in response file: {e}",
                            )
                        )
            else:
                # Fallback to parsing from stdout if file is missing
                _echo(
                    "Warning: Response file not found, falling back to stdout parsing..."
                )
                try:
                    result = parse_response_json(result)
                except AgentError as e:
                    # If legacy parsing also fails, surface the parsing error
                    _echo("Error parsing Gemini CLI response after file was not found")
                    _echo("--- Start of Gemini CLI response ---")
                    _echo(f"{result}")
                    _echo("--- End of Gemini CLI response ---")
                    return AgentResult(error=e)
        else:
            # No response file provided - parse from stdout (legacy behavior)
            try:
                result = parse_response_json(result)
            except AgentError as e:
                _echo("Error parsing Gemini CLI response")
                _echo("--- Start of Gemini CLI response ---")
                _echo(f"{result}")
                _echo("--- End of Gemini CLI response ---")
                return AgentResult(error=e)

        try:
            version = self.get_version()
        except AgentError as e:
            _echo(f"Error getting version: {e}")
            return AgentResult(error=e)

        result["agent_info"] = {
            "agent_type": "gemini_cli",
            "version": version,
        }

        return AgentResult(result=result)

    def get_session_data(self) -> dict | None:
        """Get session data from the last Gemini CLI run.

        Returns:
            Dict containing session data, or None if no session available.
        """
        if self.session_id is None:
            return None
        return self.read_session(self.session_id)

    def get_session_id(self) -> str | None:
        """Get session ID from the last Gemini CLI run.

        Returns:
            Session ID string, or None if no session available.
        """
        return self.session_id
