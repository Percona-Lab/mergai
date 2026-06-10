"""Regression tests for parsing git merge output.

Covers the bug where the first ``Auto-merging`` line was silently dropped on a
conflicting merge: GitPython decorates ``GitCommandError.stdout`` as
``"\\n  stdout: '<content>'"``, gluing ``  stdout: '`` onto the first output
line and breaking the ``^Auto-merging`` anchor for that line only. Because git
emits these lines in sorted path order, the alphabetically-first auto-merged
file (e.g. ``.bazelrc``) was always the casualty.
"""

from git.exc import GitCommandError

from mergai.utils import git_utils

# Real `git merge --no-commit --no-ff` stdout for a conflicting merge: all
# "Auto-merging" lines (sorted, so ".bazelrc" is first) and CONFLICT lines go
# to stdout; stderr is empty.
MERGE_STDOUT = (
    "Auto-merging .bazelrc\n"
    "Auto-merging buildscripts/resmokelib/configure_resmoke.py\n"
    "Auto-merging buildscripts/sbom_linter.py\n"
    "CONFLICT (content): Merge conflict in buildscripts/sbom_linter.py\n"
    "Auto-merging sbom.json\n"
    "CONFLICT (content): Merge conflict in sbom.json\n"
    "Auto-merging src/mongo/db/BUILD.bazel\n"
    "Automatic merge failed; fix conflicts and then commit the result.\n"
)


def _decorated_stdout(raw: str) -> str:
    """Mimic GitPython's GitCommandError.stdout decoration of a raw stream."""
    err = GitCommandError("git merge", 1, stderr=b"", stdout=raw.encode())
    return err.stdout


def test_undecorate_strips_gitpython_stdout_wrapper():
    decorated = _decorated_stdout(MERGE_STDOUT)
    assert decorated != MERGE_STDOUT  # sanity: it really is wrapped
    assert git_utils.undecorate_git_stream(decorated) == MERGE_STDOUT


def test_undecorate_strips_stderr_wrapper():
    err = GitCommandError("git merge", 1, stderr=b"boom", stdout=b"")
    assert git_utils.undecorate_git_stream(err.stderr) == "boom"


def test_undecorate_passes_through_plain_and_empty():
    assert git_utils.undecorate_git_stream(MERGE_STDOUT) == MERGE_STDOUT
    assert git_utils.undecorate_git_stream("") == ""
    assert git_utils.undecorate_git_stream(None) == ""


def test_parse_plain_output_captures_first_auto_merged_file():
    parsed = git_utils.parse_git_merge_output(MERGE_STDOUT)
    assert ".bazelrc" in parsed.auto_merged_files
    assert len(parsed.auto_merged_files) == 5
    assert set(parsed.conflicting_files) == {
        "buildscripts/sbom_linter.py",
        "sbom.json",
    }
    assert parsed.success is False


def test_parse_decorated_output_does_not_drop_first_auto_merged_file():
    # This is the regression: the GitPython-decorated string must parse the
    # same as the raw stream, including the alphabetically-first file.
    decorated = _decorated_stdout(MERGE_STDOUT)
    parsed = git_utils.parse_git_merge_output(decorated)
    assert ".bazelrc" in parsed.auto_merged_files
    assert len(parsed.auto_merged_files) == 5
    assert set(parsed.conflicting_files) == {
        "buildscripts/sbom_linter.py",
        "sbom.json",
    }
    assert parsed.success is False
