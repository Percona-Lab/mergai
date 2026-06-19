"""Tests for AppContext.commit_solution staging behavior.

Regression coverage for the case where the agent also has to touch a
non-conflict file (e.g. a file git auto-merged) and declares it under the
solution's ``modified`` section: commit_solution must stage it and record it in
the commit body rather than aborting, while still rejecting a dirty file that
the solution declared nowhere.
"""

import git
import pytest

from mergai.app import AppContext
from mergai.models import MergaiNote


def _init_repo(path):
    repo = git.Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "test")
        cw.set_value("user", "email", "test@example.com")
    (path / "a.txt").write_text("base a\n")
    (path / "version.h").write_text("base v\n")
    repo.index.add(["a.txt", "version.h"])
    repo.index.commit("init")
    return repo


def _note(repo, *, resolved, modified):
    head = repo.head.commit.hexsha
    data = {
        "mergai_version": "test",
        "merge_info": {
            "target_branch": "master",
            "target_branch_sha": head,
            "merge_commit": head,
        },
        "solutions": [
            {
                "response": {
                    "summary": "resolve conflicts",
                    "resolved": resolved,
                    "modified": modified,
                }
            }
        ],
    }
    return MergaiNote.from_dict(data, repo)


def test_commit_solution_accepts_declared_modified_files(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    # A conflict-resolved file and a non-conflict file the agent also adjusted.
    (tmp_path / "a.txt").write_text("resolved a\n")
    (tmp_path / "version.h").write_text("adjusted v\n")

    app = AppContext()
    app._note = _note(repo, resolved={"a.txt": {}}, modified={"version.h": {}})

    app.commit_solution()

    assert not repo.is_dirty()
    message = repo.head.commit.message
    assert "Resolved" in message and "a.txt" in message
    assert "Modified" in message and "version.h" in message
    assert (tmp_path / "version.h").read_text() == "adjusted v\n"


def test_commit_solution_rejects_undeclared_dirty_file(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "a.txt").write_text("resolved a\n")
    # version.h is dirty but declared in neither resolved nor modified.
    (tmp_path / "version.h").write_text("stray change\n")

    app = AppContext()
    app._note = _note(repo, resolved={"a.txt": {}}, modified={})

    with pytest.raises(Exception, match="neither 'resolved' nor 'modified'"):
        app.commit_solution()
