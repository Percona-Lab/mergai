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
        super().__init__(app, config)

    def execute(self, context: WorkflowContext) -> dict | None:
        env = os.environ.copy()
        # The note supplies the merge target branch, but a command-type
        # workflow can legitimately run without one (e.g. a standalone
        # auto-fixer). Only expose TARGET_BRANCH when a note exists; mirror
        # ResolveHandler's `self.app.note if self.app.has_note else None`.
        if self.app.has_note:
            env["TARGET_BRANCH"] = self.app.note.merge_info.target_branch
        env["PR_NUMBER"] = str(context.pr_number)
        env["WORKFLOW_NAME"] = context.workflow_name
        # Where the failing run's artifacts are extracted (one subdir per
        # artifact). Lets deterministic auto-fixers apply a pre-computed
        # patch (`git apply $MERGAI_ARTIFACTS_DIR/<artifact>/diff.patch`)
        # instead of re-running the underlying tool.
        if context.artifacts_dir:
            env["MERGAI_ARTIFACTS_DIR"] = context.artifacts_dir

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
            # Command fixes are produced by a deterministic tool, not the AI
            # agent or a human, so attribute them to the auto-fix itself
            # (renders as "<workflow> auto-fix" via format_solution_author).
            "author": {
                "name": f"{context.workflow_name} auto-fix",
                "type": "command",
            },
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
