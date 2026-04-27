"""Context builder for workflows that upload a SARIF file artifact.

Used by the ``clang-tidy`` workflow: clang-tidy.yml uploads
``clang-tidy-results.sarif`` as the ``clang-tidy-results`` artifact
(alongside the Code Scanning upload).

SARIF (Static Analysis Results Interchange Format) is a JSON schema for
reporting static-analysis findings. The parser here extracts enough to
build an AI prompt: affected files and a per-finding summary of
rule-id + location + message. It is deliberately tolerant of missing or
odd fields — real-world SARIF files vary.
"""

import json
from pathlib import Path
from typing import Any

from ...config import WorkflowContextConfig
from .base import WorkflowContext, WorkflowContextBuilder


class SARIFContextBuilder(WorkflowContextBuilder):
    """Parses a SARIF JSON file from a workflow artifact."""

    def build_context(
        self,
        config: WorkflowContextConfig,
        workflow_name: str,
        run_id: str,
        pr_number: int,
        artifacts_dir: str | None,
    ) -> WorkflowContext:
        if config.source != "artifact":
            raise NotImplementedError(
                f"SARIF source '{config.source}' is not supported yet "
                f"(only 'artifact'). See PSMDB-1972 follow-ups."
            )
        if artifacts_dir is None:
            raise FileNotFoundError(
                f"Workflow '{workflow_name}' needs artifacts_dir (sarif context)"
            )
        if not config.artifact_name:
            raise ValueError(
                f"Workflow '{workflow_name}' sarif context requires "
                f"'context.artifact_name' to be set"
            )

        sarif_path = self._find_sarif(
            Path(artifacts_dir) / config.artifact_name, workflow_name
        )
        sarif_data = json.loads(sarif_path.read_text())
        findings = self._flatten_findings(sarif_data)

        files_affected = sorted({f["file"] for f in findings if f["file"]})
        details = self._format_details(findings)

        count = len(findings)
        summary = (
            f"{workflow_name} reported {count} finding{'s' if count != 1 else ''} "
            f"in {len(files_affected)} file{'s' if len(files_affected) != 1 else ''}"
        )

        return WorkflowContext(
            workflow_name=workflow_name,
            run_id=run_id,
            pr_number=pr_number,
            summary=summary,
            files_affected=files_affected,
            details=details,
            raw_data={"findings": findings},
        )

    @staticmethod
    def _find_sarif(artifact_dir: Path, workflow_name: str) -> Path:
        """Find the SARIF file inside an extracted artifact directory.

        Accepts either a well-known filename (e.g.
        ``clang-tidy-results.sarif``) or any file ending in ``.sarif``.
        """
        # Preferred name ``<workflow_name>-results.sarif`` first.
        candidate = artifact_dir / f"{workflow_name}-results.sarif"
        if candidate.exists():
            return candidate

        for entry in sorted(artifact_dir.iterdir()) if artifact_dir.exists() else []:
            if entry.is_file() and entry.suffix == ".sarif":
                return entry

        raise FileNotFoundError(
            f"No SARIF file found in {artifact_dir}. "
            f"Expected {workflow_name}-results.sarif or any *.sarif file."
        )

    @staticmethod
    def _flatten_findings(sarif: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ``{rule_id, level, message, file, line}`` per finding."""
        findings: list[dict[str, Any]] = []
        for run in sarif.get("runs", []):
            for result in run.get("results", []):
                rule_id = result.get("ruleId", "")
                level = result.get("level", "warning")
                message = (result.get("message") or {}).get("text", "")

                file_path = ""
                line = 0
                locations = result.get("locations") or []
                if locations:
                    phys = (locations[0] or {}).get("physicalLocation") or {}
                    file_path = (phys.get("artifactLocation") or {}).get("uri", "")
                    line = (phys.get("region") or {}).get("startLine", 0)

                findings.append(
                    {
                        "rule_id": rule_id,
                        "level": level,
                        "message": message,
                        "file": file_path,
                        "line": line,
                    }
                )
        return findings

    @staticmethod
    def _format_details(findings: list[dict[str, Any]]) -> str:
        """Render findings as a Markdown list suitable for AI prompts."""
        if not findings:
            return "(no findings reported)"

        lines = []
        for f in findings:
            loc = f"{f['file']}:{f['line']}" if f["file"] else "(unknown location)"
            header = f"- [{f['level']}] {f['rule_id']} at {loc}"
            lines.append(header)
            if f["message"]:
                lines.append(f"    {f['message']}")
        return "\n".join(lines)
