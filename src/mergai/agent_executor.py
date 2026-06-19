"""Agent execution with retry logic and result validation.

This module provides the AgentExecutor class which encapsulates the logic
for running AI agents with retry capabilities and result validation.
"""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import git

from .agents.base import Agent
from .utils import git_utils
from .utils.output import echo_err as _echo


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
        repo: git.Repo | None = None,
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

    def _generate_response_filename(self) -> str:
        """Generate unique response filename with timestamp.

        Returns:
            Filename string in format: response_YYYYMMDD_HHMMSS_ffffff.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"response_{timestamp}.json"

    def _generate_session_filename(self, session_id: str) -> str:
        """Generate session filename.

        Args:
            session_id: The session ID from the agent.

        Returns:
            Filename string in format: session_{id}_{timestamp}.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{session_id}_{timestamp}.json"

    def _save_session_on_failure(self) -> Path | None:
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
        validator: Callable[[dict], str | None] | None = None,
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
        # Generate unique filename and write prompt to state_dir
        # state_dir is typically .cache/mergai which is gitignored
        prompt_filename = self._generate_prompt_filename()
        prompt_path = self.state_dir / prompt_filename
        prompt_path.write_text(prompt)

        success = False
        try:
            result = self._execute_with_retry(prompt_path, validator)
            success = True
            return result
        except AgentExecutionError:
            # Save session data on failure
            session_path = self._save_session_on_failure()

            # Log file locations (prompt is already in state_dir)
            _echo(f"Prompt file kept at: {prompt_path}")
            if session_path:
                _echo(f"Session file saved at: {session_path}")

            raise
        finally:
            # Only remove prompt file on success (if it still exists in cwd)
            if success and prompt_path.exists():
                prompt_path.unlink()

    def _execute_with_retry(
        self,
        prompt_path: Path,
        validator: Callable[[dict], str | None] | None = None,
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
        # Create response file path in state_dir (same location as prompt)
        # state_dir is typically .cache/mergai which is gitignored
        response_filename = self._generate_response_filename()
        response_path = self.state_dir / response_filename

        # Base prompt with file instructions - preserved across retries
        base_prompt = (
            f"Read @{prompt_path} and write your JSON response to @{response_path}"
        )
        current_prompt = base_prompt
        result = None

        try:
            for attempt in range(self.max_attempts):
                _echo(f"Attempt {attempt + 1} of {self.max_attempts}...")

                # Remove response file before each attempt to ensure fresh response
                if response_path.exists():
                    response_path.unlink()

                agent_result = self.agent.run(
                    current_prompt,
                    response_file=response_path,
                    allowed_write_paths=[response_path],
                )
                if not agent_result.success():
                    error_msg = str(agent_result.error())
                    _echo(f"Agent execution failed: {error_msg}")
                    # Preserve file instructions and append error context for retry
                    current_prompt = (
                        f"{base_prompt}\n\n"
                        f"Previous attempt failed with error:\n{error_msg}\n\n"
                        "Please fix the issue and try again."
                    )
                    continue

                _echo("Agent execution succeeded. Checking result...")
                result = agent_result.result()

                # Clean up response file before validation to avoid false positives
                # in validators that check for file modifications (e.g., describe)
                if response_path.exists():
                    response_path.unlink()

                # Run validator if provided
                if validator is not None and result is not None:
                    validation_error = validator(result)
                    if validation_error is not None:
                        _echo(f"Validation failed: {validation_error}")
                        # Check if validation failed due to file modifications
                        # In this case, retrying won't help since the repo is now dirty
                        if "Files were modified" in validation_error:
                            _echo(
                                "Error: Agent modified files when it should not have. "
                                "Failing immediately as retries would continue to fail."
                            )
                            raise AgentExecutionError(
                                f"Validation failed: {validation_error}"
                            )
                        # Preserve file instructions and append validation error for retry
                        current_prompt = (
                            f"{base_prompt}\n\n"
                            f"Previous attempt failed validation:\n{validation_error}\n\n"
                            "Please fix the issue and try again."
                        )
                        continue

                _echo("Result verified.")
                return result  # type: ignore[return-value]

            # All attempts exhausted
            _echo("Max attempts reached. Failed to obtain a valid result.")
            raise AgentExecutionError("Failed to obtain a valid result from the agent.")
        finally:
            # Clean up response file
            if response_path.exists():
                response_path.unlink()

    def validate_solution_files(self, solution: dict) -> str | None:
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
        dirty_files = [item.a_path for item in self.repo.index.diff(None)]

        # Check resolved files
        for path in solution["response"]["resolved"]:
            _echo(
                f"Checking file '{path}': {'dirty' if path in dirty_files else 'not dirty'}"
            )
            if path not in dirty_files:
                not_dirty_files.append(path)

        # Also check modified files (non-conflict files that were changed)
        for path in solution["response"].get("modified", {}):
            _echo(
                f"Checking modified file '{path}': {'dirty' if path in dirty_files else 'not dirty'}"
            )
            if path not in dirty_files:
                not_dirty_files.append(path)

        if len(not_dirty_files):
            message = "The following files in the solution have no unstaged changes: "
            message += ", ".join(not_dirty_files)
            return message

        return None

    def validate_resolved_files_have_no_markers(self, solution: dict) -> str | None:
        """Validate that files reported as 'resolved' have no conflict markers left.

        The agent is allowed to leave conflict markers in files listed under
        `response.unresolved`, but any file listed under `response.resolved`
        must be free of markers.

        Args:
            solution: The solution dict from the agent, expected to have
                     structure: {"response": {"resolved": {path: ...}}}.

        Returns:
            None if all resolved files are free of conflict markers, or an
            error message listing the offending files.
        """
        offending: list[str] = []
        for path in solution["response"].get("resolved", {}):
            if git_utils.file_has_conflict_markers_in_workdir(path):
                offending.append(path)

        if offending:
            return (
                "The following files were marked as resolved but still contain "
                "conflict markers: "
                + ", ".join(offending)
                + ". Remove the conflict markers from these files, or move them "
                "to the 'unresolved' section of your response."
            )
        return None

    def validate_solution(self, solution: dict) -> str | None:
        """Combined validator for agent solutions.

        Runs `validate_solution_files` (files listed as resolved/modified were
        actually changed on disk) followed by
        `validate_resolved_files_have_no_markers` (no conflict markers remain
        in files the agent claimed to have resolved).

        Returns the first error encountered, or None if both checks pass.
        """
        error = self.validate_solution_files(solution)
        if error:
            return error
        return self.validate_resolved_files_have_no_markers(solution)

    def validate_describe_response(self, response: dict) -> str | None:
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

    def validate_describe_files_in_changeset(
        self, response: dict, allowed_files: set[str] | None
    ) -> str | None:
        """Validate that every described file is actually part of the merge.

        Guards against the agent describing a file that is not in the changeset
        (a fabricated or misattributed path). ``allowed_files`` is the set of
        files the agent is permitted to describe (typically the merge's
        auto-merged files); ``None`` skips the check (no ground truth available).

        Args:
            response: The describe response dict (expects an ``auto_merged`` map).
            allowed_files: Set of file paths legitimately in the changeset, or
                None to skip the check.

        Returns:
            None if all described files are in the changeset, or an error message
            naming the offending files otherwise.
        """
        if allowed_files is None:
            return None

        auto_merged = response.get("auto_merged")
        if not isinstance(auto_merged, dict):
            return None

        unknown = [path for path in auto_merged if path not in allowed_files]
        if unknown:
            return (
                "Described files are not part of this merge's changeset: "
                f"{', '.join(sorted(unknown))}. Only describe files listed in "
                "merge_context.auto_merged.files, and read each file's diff "
                "before describing it."
            )
        return None

    def validate_describe_verify_response(self, response: dict) -> str | None:
        """Validate the format of a describe-verifier verdict.

        Enforces a self-consistent verdict so a malformed or contradictory
        response can't be silently treated as "accurate":

        - ``accurate`` must be a boolean;
        - ``issues`` must be a list of objects, each with **present**,
          non-empty string ``location`` / ``claim`` / ``reason`` fields;
        - ``accurate: false`` requires at least one issue (otherwise the
          verdict says "inaccurate" but gives nothing to act on, and the
          description would be accepted unchanged);
        - ``accurate: true`` requires an empty issue list (otherwise the
          issues would be silently ignored, defeating the fact-check).

        Args:
            response: The verdict dict from the verifier agent.

        Returns:
            None if valid, or an error message string if invalid.
        """
        accurate = response.get("accurate")
        if not isinstance(accurate, bool):
            return "'accurate' field must be a boolean"

        issues = response.get("issues", [])
        if not isinstance(issues, list):
            return "'issues' field must be a list"

        for index, issue in enumerate(issues):
            if not isinstance(issue, dict):
                return f"issues[{index}] must be an object"
            for key in ("location", "claim", "reason"):
                value = issue.get(key)
                if not isinstance(value, str) or not value.strip():
                    return f"issues[{index}].{key} must be a non-empty string"

        if not accurate and not issues:
            return (
                "'accurate' is false but 'issues' is empty; list each "
                "unsupported claim, or set 'accurate' to true."
            )
        if accurate and issues:
            return (
                "'accurate' is true but 'issues' is non-empty; either drop the "
                "issues or set 'accurate' to false."
            )
        return None

    def validate_no_file_modifications(self, was_dirty_before: bool) -> str | None:
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
            # Collect modified, staged, and untracked files for the error message
            modified_files = []

            # Get modified (unstaged) files
            for item in self.repo.index.diff(None):
                modified_files.append(f"M {item.a_path}")

            # Get staged files (guard against empty/unborn HEAD)
            try:
                staged_diff = self.repo.index.diff("HEAD")
                for item in staged_diff:
                    modified_files.append(f"S {item.a_path}")
            except (git.exc.GitCommandError, git.BadName, ValueError):
                # HEAD doesn't exist (unborn branch) - no staged files to report
                pass

            # Get untracked files
            for path in self.repo.untracked_files:
                modified_files.append(f"? {path}")

            files_list = ", ".join(modified_files) if modified_files else "unknown"
            return f"Files were modified during operation. No file modifications are allowed. Modified files: {files_list}"
        elif is_dirty_after and was_dirty_before:
            # Repo was already dirty - we can't verify if new modifications were made
            _echo(
                "Warning: Repository was already dirty before operation. "
                "Cannot verify if new modifications were made."
            )

        return None

    def create_describe_validator(
        self, was_dirty_before: bool, allowed_files: set[str] | None = None
    ) -> Callable[[dict], str | None]:
        """Create a composite validator for describe operations.

        Combines response format validation, a programmatic changeset check
        (described files must be part of the merge), and file modification
        validation.

        Args:
            was_dirty_before: Whether the repo was dirty before execution.
            allowed_files: Set of file paths the agent is allowed to describe
                (the merge's auto-merged files). None skips the changeset check.

        Returns:
            A validator function suitable for use with run_with_retry.
        """

        def validator(result: dict) -> str | None:
            # Validate response format
            format_error = self.validate_describe_response(result["response"])
            if format_error:
                return format_error

            # Reject descriptions referencing files outside the changeset
            changeset_error = self.validate_describe_files_in_changeset(
                result["response"], allowed_files
            )
            if changeset_error:
                return changeset_error

            # Validate no file modifications
            return self.validate_no_file_modifications(was_dirty_before)

        return validator
