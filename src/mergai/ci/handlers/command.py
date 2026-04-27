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

    def execute(self, context: WorkflowContext) -> bool:
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
            return False

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
        return self.app.repo.is_dirty(untracked_files=True)
