from .base import CliAgent
from .error import AgentError, AgentErrorType, AgentResult
import click
import subprocess
import os
import json
import re
from pathlib import Path
from typing import Optional


JSON_BLOCK_RE = re.compile(
    r"```json\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


# Extracts the content of the first ```json ... ``` block.
# Returns the JSON string or None if not found.
def extract_json_block(text: str) -> str | None:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def fix_response_json(result_json: dict) -> str:
    if "response" not in result_json:
        raise AgentError(
            AgentErrorType.PARSING_RESULT, "invalid response: 'response' field missing"
        )

    response = result_json["response"]

    extracted = extract_json_block(response)
    if extracted is not None:
        response = extracted

    try:
        response_json = json.loads(response)
        result_json["response"] = response_json
    except json.JSONDecodeError as e:
        raise AgentError(AgentErrorType.PARSING_RESULT, f"JSON decode error: {e}")

    return result_json


class GeminiCLIAgent(CliAgent):

    def __init__(self, model: str, yolo: bool, debug: bool = False):
        super().__init__(model)
        self.session_id: Optional[str] = None
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

    def read_session(self, session_id: str) -> dict:
        gemini_tmp_dir = Path.home() / ".gemini" / "tmp"

        for file in gemini_tmp_dir.iterdir():
            chats_dir = file / "chats"
            if chats_dir.exists() and chats_dir.is_dir():
                for chat_file in chats_dir.iterdir():
                    try:
                        with open(chat_file, "r") as f:
                            chat_data = json.load(f)
                            if chat_data.get("sessionId") == session_id:
                                return chat_data
                    except Exception:
                        continue
        return None

    def parse_stats(self, session_data: dict) -> dict:
        models = {}
        for msg in session_data.get("messages", []):
            if not "tokens" in msg:
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
        args = [
            "gemini",
            "--approval-mode", "yolo" if self.yolo else "auto_edit",
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

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )

        result = {}
        response = ""
        for line in proc.stdout:
            click.echo(f"gemini-cli: {line}", nl=False)
            event = json.loads(line)

            if event["type"] == "message" and event["role"] == "assistant":
                response += event["content"]

            if event["type"] == "init":
                self.session_id = event["session_id"]
                click.echo(f"gemini-cli: received session_id: {self.session_id}")

            if event["type"] == "result":
                result["timestamp"] = event["timestamp"]
                result["status"] = event["status"]
                click.echo(f"gemini-cli: received result: {result}")
                break

        result["response"] = response

        if self.session_id is not None:
            result["session"] = self.read_session(self.session_id)
            result["stats"] = self.parse_stats(result["session"])

        rc = proc.wait()
        if rc != 0:
            raise AgentError(
                AgentErrorType.AGENT_EXECUTION,
                f"gemini-cli: exited with code {rc}, stderr: {proc.stderr}",
            )

        return result

    def run(self, prompt: str) -> AgentResult:
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")

        click.echo(f"Running Gemini CLI agent with prompt:\n{prompt}")
        response = self.run_prompt(prompt)

        try:
            result = fix_response_json(response)
            version = self.get_version()
        except AgentError as e:
            click.echo(f"Error parsing Gemini CLI response")
            click.echo(f"--- Start of Gemini CLI response ---")
            click.echo(f"{response}")
            click.echo(f"--- End of Gemini CLI response ---")
            return AgentResult(error=e)

        result["agent_info"] = {
            "agent_type": "gemini_cli",
            "version": version,
        }

        return AgentResult(result=result)
