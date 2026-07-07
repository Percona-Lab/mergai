"""Tests for commit_ci_fix_solution staging behavior.

Regression coverage mirroring tests/test_commit_solution.py: when the agent
resolves a CI failure by deleting a file, staging must use index.remove
instead of index.add, since the file no longer exists on disk for add to
read.
"""

from types import SimpleNamespace

import git

from mergai.ci.commit import commit_ci_fix_solution
from mergai.solution_types import CI_FIX


def _init_repo(path):
    repo = git.Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "test")
        cw.set_value("user", "email", "test@example.com")
    (path / "a.txt").write_text("base a\n")
    repo.index.add(["a.txt"])
    repo.index.commit("init")
    return repo


def _fake_app(repo, *, resolved):
    head = repo.head.commit.hexsha
    note = SimpleNamespace(
        has_solutions=True,
        solutions=[
            {
                "type": CI_FIX,
                "request": {"workflow": "format", "run_id": "1", "attempt_number": 1},
                "response": {"summary": "fix", "resolved": resolved, "modified": {}},
            }
        ],
        merge_info=SimpleNamespace(target_branch="master", merge_commit_sha=head),
    )
    config = SimpleNamespace(
        workflows={},
        commit=SimpleNamespace(
            ci_fix_title_format="Fix '%(workflow)' failure for merge commit "
            "'%(merge_commit_short_sha)' into '%(target_branch)'"
        ),
    )
    return SimpleNamespace(
        note=note,
        repo=repo,
        config=config,
        commit_footer="",
        add_selective_note=lambda *a, **k: None,
    )


def test_commit_ci_fix_solution_stages_deleted_resolved_file(tmp_path):
    repo = _init_repo(tmp_path)

    # The agent resolved the CI failure by deleting the offending file.
    (tmp_path / "a.txt").unlink()

    app = _fake_app(repo, resolved={"a.txt": "removed stale file"})

    commit_ci_fix_solution(app, 0)

    assert not repo.is_dirty()
    assert "a.txt" not in [item[0] for item in repo.index.entries]
    assert not (tmp_path / "a.txt").exists()
