"""Shell-command handler for CI workflow fixes."""

import logging
import os
import subprocess

from ...app import AppContext
from ...config import WorkflowConfig
from ..context_builders.base import WorkflowContext
from .base import WorkflowHandler

log = logging.getLogger(__name__)


class CommandHandler(WorkflowHandler):
    """Runs a shell command configured per-workflow.

    Suitable for deterministic auto-fixers like ``bazel run format`` that
    simply reformat files in place. The command is expected to leave the
    working tree dirty on success; the outer loop commits and pushes.
    """

    def __init__(self, app: AppContext, config: WorkflowConfig):
        if not config.command:
            raise ValueError(
                "CommandHandler requires config.command to be a non-empty string"
            )
        self.app = app
        self.config = config

    def execute(self, context: WorkflowContext) -> dict | None:
        env = os.environ.copy()
        env["TARGET_BRANCH"] = self.app.note.merge_info.target_branch
        env["PR_NUMBER"] = str(context.pr_number)
        env["WORKFLOW_NAME"] = context.workflow_name

        # `config.command` is enforced non-empty by __init__.
        command = str(self.config.command)
        log.info("Running fix command for %s: %s", context.workflow_name, command)

        try:
            result = subprocess.run(  # noqa: S602 — command comes from trusted config
                command,
                shell=True,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                cwd=self.app.repo.working_tree_dir,
            )
        except OSError as e:
            log.error("Failed to spawn fix command: %s", e)
            return None

        if result.stdout:
            log.debug("stdout:\n%s", result.stdout)
        if result.stderr:
            log.debug("stderr:\n%s", result.stderr)

        if result.returncode != 0:
            log.warning(
                "Fix command exited with %s; tree-state will still be checked",
                result.returncode,
            )

        # Success = the command changed something we can commit. A non-zero
        # exit with dirty tree still counts (some formatters return non-zero
        # when they *do* reformat). An exit of 0 with a clean tree means
        # the command ran but made no changes — treat as "no fix applied".
        if not self.app.repo.is_dirty(untracked_files=True):
            return None

        return self._synthesize_solution(context)

    def _synthesize_solution(self, context: WorkflowContext) -> dict:
        """Build a solution dict from the working tree state after the command.

        Command-driven fixers don't produce structured output, so we
        enumerate the dirty / untracked files and treat them as the
        ``resolved`` set. Keeps the recorded shape aligned with what
        :class:`ResolveHandler` returns, so the orchestrator and commit
        logic don't have to special-case action types.
        """
        repo = self.app.repo
        dirty: list[str] = sorted(
            item.a_path for item in repo.index.diff(None) if item.a_path
        )
        untracked: list[str] = sorted(repo.untracked_files)
        changed: list[str] = sorted(set(dirty) | set(untracked))

        explanation = f"changed by '{context.workflow_name}' auto-fix"
        return {
            "response": {
                "summary": (
                    f"{context.workflow_name} auto-fix: {len(changed)} "
                    f"file{'s' if len(changed) != 1 else ''} changed"
                ),
                "resolved": dict.fromkeys(changed, explanation),
                "unresolved": {},
                "modified": {},
                "review_notes": "",
            },
        }
