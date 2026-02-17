"""Agent execution with retry logic and result validation.

This module provides the AgentExecutor class which encapsulates the logic
for running AI agents with retry capabilities and result validation.
"""

import click
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import git

from .agents.base import Agent
from .prompt_builder import PromptBuilder


class AgentExecutionError(Exception):
    """Raised when agent fails to produce a valid result after max attempts."""

    pass


class AgentExecutor:
    """Executes AI agents with retry logic and result validation.

    This class encapsulates the common pattern of:
    1. Writing a prompt to a file in the state directory
    2. Running an agent with the prompt
    3. Validating the result
    4. Retrying on failure up to max_attempts
    5. Cleaning up prompt file on success, keeping it on failure

    Attributes:
        agent: The AI agent to execute.
        max_attempts: Maximum number of retry attempts.
        repo: Optional git repo for validation that requires repo state.
        state_dir: Directory to store prompt and session files.

    Example usage:
        executor = AgentExecutor(
            agent, max_attempts=3, repo=repo, state_dir=state.path
        )
        result = executor.run_with_retry(
            prompt=prompt_text,
            validator=executor.validate_solution_files
        )
    """

    def __init__(
        self,
        agent: Agent,
        state_dir: Path,
        max_attempts: int = 3,
        repo: git.Repo = None,
    ):
        """Initialize AgentExecutor.

        Args:
            agent: The AI agent instance to execute.
            state_dir: Directory to store prompt and session files.
            max_attempts: Maximum number of retry attempts (default: 3).
            repo: Optional GitPython Repo for validations requiring repo state.
        """
        self.agent = agent
        self.state_dir = Path(state_dir)
        self.max_attempts = max_attempts
        self.repo = repo

    def _generate_prompt_filename(self) -> str:
        """Generate unique prompt filename with timestamp.

        Returns:
            Filename string in format: prompt_YYYYMMDD_HHMMSS_ffffff.md
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"prompt_{timestamp}.md"

    def _generate_session_filename(self, session_id: str) -> str:
        """Generate session filename.

        Args:
            session_id: The session ID from the agent.

        Returns:
            Filename string in format: session_{id}_{timestamp}.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{session_id}_{timestamp}.json"

    def _save_session_on_failure(self) -> Optional[Path]:
        """Save agent session data to file on failure.

        Retrieves session data from the agent and saves it to the state
        directory for debugging purposes.

        Returns:
            Path to the saved session file, or None if no session data available.
        """
        session_data = self.agent.get_session_data()
        if session_data is None:
            return None

        session_id = self.agent.get_session_id() or "unknown"
        filename = self._generate_session_filename(session_id)
        session_path = self.state_dir / filename

        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=2)

        return session_path

    def run_with_retry(
        self,
        prompt: str,
        validator: Callable[[dict], Optional[str]] = None,
    ) -> dict:
        """Run agent with retry logic and optional result validation.

        Creates a file with the prompt in the current working directory (so it's
        accessible to the agent), runs the agent, validates the result, and
        retries on failure.

        On success: prompt file is removed.
        On failure: prompt file is moved to state_dir, session file is saved,
                   and paths are echoed.

        Args:
            prompt: The initial prompt text to send to the agent.
            validator: Optional callback that validates the result dict.
                      Should return None if valid, or an error string if invalid.

        Returns:
            The validated result dict from the agent.

        Raises:
            AgentExecutionError: If max attempts reached without valid result.
        """
        # Generate unique filename and write prompt to cwd (not state_dir)
        # so the agent can access it (state_dir may be ignored by agent)
        prompt_filename = self._generate_prompt_filename()
        prompt_path = Path.cwd() / prompt_filename
        prompt_path.write_text(prompt)

        success = False
        try:
            result = self._execute_with_retry(prompt_path, validator)
            success = True
            return result
        except AgentExecutionError:
            # Move prompt file to state_dir for preservation
            final_prompt_path = self.state_dir / prompt_filename
            prompt_path.rename(final_prompt_path)

            # Save session data on failure
            session_path = self._save_session_on_failure()

            # Log file locations
            click.echo(f"Prompt file kept at: {final_prompt_path}")
            if session_path:
                click.echo(f"Session file saved at: {session_path}")

            raise
        finally:
            # Only remove prompt file on success (if it still exists in cwd)
            if success and prompt_path.exists():
                prompt_path.unlink()

    def _execute_with_retry(
        self,
        prompt_path: Path,
        validator: Callable[[dict], Optional[str]] = None,
    ) -> dict:
        """Execute the agent with retry logic.

        Args:
            prompt_path: Path to the prompt file.
            validator: Optional validation callback.

        Returns:
            The validated result dict.

        Raises:
            AgentExecutionError: If max attempts reached without valid result.
        """
        current_prompt = f"See @{prompt_path} make sure the output is in specified format"
        error = None
        result = None

        for attempt in range(self.max_attempts):
            if error is not None:
                click.echo(
                    f"Attempt {attempt + 1} failed with error: {error}. Retrying..."
                )

            if attempt == self.max_attempts - 1:
                click.echo("Max attempts reached. Failed to obtain a valid result.")
                result = None
                break

            agent_result = self.agent.run(current_prompt)
            if not agent_result.success():
                click.echo(f"Agent execution failed: {agent_result.error()}")
                current_prompt = PromptBuilder.error_to_prompt(str(agent_result.error()))
                error = str(agent_result.error())
                continue

            click.echo("Agent execution succeeded. Checking result...")
            result = agent_result.result()

            # Run validator if provided
            if validator is not None:
                validation_error = validator(result)
                if validation_error is not None:
                    click.echo(f"Validation failed: {validation_error}")
                    current_prompt = PromptBuilder.error_to_prompt(validation_error)
                    error = validation_error
                    continue

            click.echo("Result verified.")
            break

        if result is None:
            raise AgentExecutionError(
                "Failed to obtain a valid result from the agent."
            )

        return result

    def validate_solution_files(self, solution: dict) -> Optional[str]:
        """Validate that solution files have been modified in the repo.

        Checks that all files listed in the solution's 'resolved' dict
        have unstaged changes in the repository.

        Args:
            solution: The solution dict from the agent, expected to have
                     structure: {"response": {"resolved": {path: ...}}}

        Returns:
            None if all files are dirty, or an error message listing
            the files that have no unstaged changes.

        Raises:
            ValueError: If repo was not provided to the executor.
        """
        if self.repo is None:
            raise ValueError("repo is required for validate_solution_files")

        not_dirty_files = []
        modified_files = [item.a_path for item in self.repo.index.diff(None)]

        for path in solution["response"]["resolved"].keys():
            click.echo(
                f"Checking file '{path}': {'dirty' if path in modified_files else 'not dirty'}"
            )
            if path not in modified_files:
                not_dirty_files.append(path)

        if len(not_dirty_files):
            message = "The following files in the solution have no unstaged changes: "
            message += ", ".join(not_dirty_files)
            return message

        return None

    def validate_describe_response(self, response: dict) -> Optional[str]:
        """Validate that describe response has the correct format.

        Checks for required fields: summary, auto_merged, review_notes.
        Also validates that auto_merged is a dictionary.

        Args:
            response: The response dict from the agent.

        Returns:
            None if valid, or an error message string if invalid.
        """
        required_fields = ["summary", "auto_merged", "review_notes"]
        missing_fields = [f for f in required_fields if f not in response]
        if missing_fields:
            return f"Missing required fields: {', '.join(missing_fields)}"

        if not isinstance(response.get("auto_merged"), dict):
            return "'auto_merged' field must be a dictionary"

        return None

    def validate_no_file_modifications(self, was_dirty_before: bool) -> Optional[str]:
        """Validate that no new files were modified during execution.

        Used for operations like 'describe' that should not modify files.

        Args:
            was_dirty_before: Whether the repo was dirty before execution.

        Returns:
            None if no new modifications, or an error message if files were modified.

        Raises:
            ValueError: If repo was not provided to the executor.
        """
        if self.repo is None:
            raise ValueError("repo is required for validate_no_file_modifications")

        is_dirty_after = self.repo.is_dirty(untracked_files=True)

        if is_dirty_after and not was_dirty_before:
            return "Files were modified during operation. No file modifications are allowed."
        elif is_dirty_after and was_dirty_before:
            # Repo was already dirty - we can't verify if new modifications were made
            click.echo(
                "Warning: Repository was already dirty before operation. "
                "Cannot verify if new modifications were made."
            )

        return None

    def create_describe_validator(
        self, was_dirty_before: bool
    ) -> Callable[[dict], Optional[str]]:
        """Create a composite validator for describe operations.

        Combines response format validation and file modification validation.

        Args:
            was_dirty_before: Whether the repo was dirty before execution.

        Returns:
            A validator function suitable for use with run_with_retry.
        """

        def validator(result: dict) -> Optional[str]:
            # Validate response format
            format_error = self.validate_describe_response(result["response"])
            if format_error:
                return format_error

            # Validate no file modifications
            return self.validate_no_file_modifications(was_dirty_before)

        return validator
