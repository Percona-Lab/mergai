"""Tests for the Bazel BEP context builder.

Focus on multi-BEP discovery: the jstests job uploads one
``bazel-bep*.json`` per resmoke invocation (``bazel-bep.json`` for the
reliable batch plus ``bazel-bep-<suite>.json`` per load-sensitive
suite), so a failure isolated to a load-sensitive suite only shows up in
a suffixed file. The builder must glob and concatenate all of them.
"""

import json
import types

from mergai.ci.context_builders.bazel import BazelContextBuilder
from mergai.config import WorkflowContextConfig


def _app_without_github():
    """Minimal AppContext stub: no GitHub token, so job-log fetch is skipped."""
    return types.SimpleNamespace(gh=None, gh_repo=None)


def _write_bep(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _test_result_event(label, status):
    return {"id": {"testResult": {"label": label}}, "testResult": {"status": status}}


def test_parse_bep_extracts_test_failure(tmp_path):
    bep = tmp_path / "bazel-bep.json"
    _write_bep(
        bep,
        [
            _test_result_event("//jstests/audit:audit", "PASSED"),
            _test_result_event("//jstests/oidc:oidc", "FAILED"),
        ],
    )

    failures = BazelContextBuilder._parse_bep(bep)

    assert failures == [
        {"kind": "test", "label": "//jstests/oidc:oidc", "message": "FAILED"}
    ]


def test_build_context_globs_multiple_bep_files(tmp_path):
    """A failure only in a load-sensitive suite's BEP must still surface."""
    artifacts_dir = tmp_path
    artifact_dir = artifacts_dir / "jstest-failure-logs"
    artifact_dir.mkdir()

    # Reliable batch: everything passed.
    _write_bep(
        artifact_dir / "bazel-bep.json",
        [_test_result_event("//jstests/audit:audit", "PASSED")],
    )
    # Load-sensitive suite, run on its own invocation, failed.
    _write_bep(
        artifact_dir / "bazel-bep-_jstests_backup_backup.json",
        [_test_result_event("//jstests/backup:backup", "FAILED")],
    )

    builder = BazelContextBuilder(_app_without_github())
    config = WorkflowContextConfig(
        type="bazel", source="artifact", artifact_name=["jstest-failure-logs"]
    )

    ctx = builder.build_context(
        config=config,
        workflow_name="build-and-test",
        run_id="123",
        pr_number=7,
        artifacts_dir=str(artifacts_dir),
    )

    # The failing target lives only in the suffixed BEP; it must be found.
    assert ctx.raw_data["failure_count"]["test"] == 1
    assert "//jstests/backup:backup" in ctx.details
    assert "1 test failure" in ctx.summary
    # Both BEP streams are pointed at in the agent-facing details.
    assert "bazel-bep.json" in ctx.details
    assert "bazel-bep-_jstests_backup_backup.json" in ctx.details


def test_build_context_no_bep_falls_back_to_summary(tmp_path):
    """No bazel-bep*.json -> no BEP failures, but context still builds."""
    artifacts_dir = tmp_path
    artifact_dir = artifacts_dir / "jstest-failure-logs"
    artifact_dir.mkdir()
    (artifact_dir / "reliable.log").write_text("some console output\n")

    builder = BazelContextBuilder(_app_without_github())
    config = WorkflowContextConfig(
        type="bazel", source="artifact", artifact_name=["jstest-failure-logs"]
    )

    ctx = builder.build_context(
        config=config,
        workflow_name="build-and-test",
        run_id="123",
        pr_number=7,
        artifacts_dir=str(artifacts_dir),
    )

    assert ctx.raw_data["failure_count"] == {"aborted": 0, "action": 0, "test": 0}
    assert "no parsable failure detail" in ctx.summary
