"""Context builder for workflows that upload a git diff artifact.

Used by the ``format`` workflow: format.yml writes ``diff.patch``
(unified diff) and ``files.txt`` (one path per line) into the
``format-results`` artifact.
"""

from pathlib import Path

from ...config import WorkflowContextConfig
from .artifacts import require_single_artifact_name
from .base import WorkflowContext, WorkflowContextBuilder


class DiffContextBuilder(WorkflowContextBuilder):
    """Reads ``diff.patch`` + ``files.txt`` from a workflow artifact."""

    def build_context(
        self,
        config: WorkflowContextConfig,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        artifacts_dir: str | None,
        head_sha: str | None = None,
    ) -> WorkflowContext:
        if artifacts_dir is None:
            raise FileNotFoundError(
                f"Workflow '{workflow_name}' needs artifacts_dir (diff context)"
            )
        artifact_name = require_single_artifact_name(
            config, workflow_name, context_label="diff"
        )
        artifact_path = Path(artifacts_dir) / artifact_name
        diff_file = artifact_path / "diff.patch"
        files_file = artifact_path / "files.txt"

        diff_content = diff_file.read_text() if diff_file.exists() else ""
        files_affected: list[str] = []
        if files_file.exists():
            files_affected = [
                line.strip()
                for line in files_file.read_text().splitlines()
                if line.strip()
            ]

        count = len(files_affected)
        summary = (
            f"{workflow_name} failed: {count} file{'s' if count != 1 else ''} "
            f"need changes"
        )

        return WorkflowContext(
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            summary=summary,
            files_affected=files_affected,
            details=diff_content,
            raw_data={"diff": diff_content, "files": files_affected},
            artifacts_dir=artifacts_dir,
        )
