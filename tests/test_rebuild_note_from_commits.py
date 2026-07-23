"""Tests for ``AppContext.rebuild_note_from_commits``.

Regression coverage for the merge_pick field being dropped on rebuild. When
``mergai context init`` (no args) reconstructs the note from commit notes, every
combined field must survive — including ``merge_pick``. It previously did not,
so any PR body built after a rebuild (e.g. the semantic PR opened by the
ci-handle job, or the main PR refreshed after a solution/semantic merge) lost
its "Merge Pick" section even though the merge commit's note carried it.
"""

import json

import git

from mergai.app import AppContext


def _init_repo(path):
    repo = git.Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "test")
        cw.set_value("user", "email", "test@example.com")
    (path / "a.txt").write_text("base\n")
    repo.index.add(["a.txt"])
    repo.index.commit("init")
    return repo


def _add_note(repo, sha, data):
    repo.git.notes("--ref", "mergai", "add", "-f", "-m", json.dumps(data), sha)


def _commit(repo, path, name, content, message):
    (path / name).write_text(content)
    repo.index.add([name])
    return repo.index.commit(message).hexsha


def test_rebuild_preserves_merge_pick(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    base_sha = repo.head.commit.hexsha

    # Merge commit carries the merge-level fields, including merge_pick.
    merge_sha = _commit(repo, tmp_path, "b.txt", "merged\n", "merge upstream")
    _add_note(
        repo,
        merge_sha,
        {
            "mergai_version": "test",
            "merge_info": {
                "target_branch": "master",
                "target_branch_sha": base_sha,
                "merge_commit": merge_sha,
            },
            "merge_pick": {
                "type": "ai",
                "sha": merge_sha,
                "short_sha": merge_sha[:11],
                "strategy": "huge_commit",
                "summary": "Cutting here yields ~48 commits.",
            },
        },
    )

    # A later fix commit (as a ci_fix / semantic fix would land) carries a
    # solution but no merge_pick of its own.
    fix_sha = _commit(repo, tmp_path, "b.txt", "fixed\n", "ci fix")
    _add_note(
        repo,
        fix_sha,
        {
            "mergai_version": "test",
            "solutions": [
                {"response": {"summary": "fix build", "resolved": {}, "unresolved": {}}}
            ],
        },
    )

    app = AppContext()
    note = app.rebuild_note_from_commits()

    # merge_pick must survive the rebuild alongside the solution.
    assert note.has_merge_pick
    assert note.merge_pick is not None
    assert note.merge_pick["strategy"] == "huge_commit"
    assert note.merge_pick["type"] == "ai"
    assert note.has_solutions
