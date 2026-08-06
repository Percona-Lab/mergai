"""Regression: squash must collect solutions from a Conflict Solution PR merge.

When a merge's conflicts are resolved via a solution->conflict PR, the
resolution/fix commits live on the *second parent* of that PR merge commit.
``_collect_commits_for_squash`` previously walked the first-parent chain only,
so those commits' ``solutions`` were dropped from the squashed note -- the
combined note ended up with ``conflict_context`` but no ``solutions``, which
later breaks ``mergai rebase`` auto-resolution (it finds a recorded conflict
but no solution to reapply).

This builds that exact topology and asserts the solutions are collected and
combined. See AppContext._collect_commits_for_squash.
"""

import json

import git

from mergai.app import AppContext
from mergai.models import MergaiNote


def _init_repo(path):
    repo = git.Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "test")
        cw.set_value("user", "email", "test@example.com")
    (path / "a.txt").write_text("base\n")
    repo.index.add(["a.txt"])
    repo.index.commit("init")
    return repo


def _note(repo, sha, data):
    repo.git.notes("--ref", "mergai", "add", "-f", "-m", json.dumps(data), sha)


def _commit(repo, path, name, content, message, parents):
    (path / name).write_text(content)
    repo.index.add([name])
    return repo.index.commit(message, parent_commits=parents, head=False)


def _solution(summary):
    return {
        "mergai_version": "test",
        "solutions": [
            {
                "response": {
                    "summary": summary,
                    "resolved": {"x.cpp": "..."},
                    "unresolved": {},
                }
            }
        ],
    }


def test_squash_collects_solutions_from_solution_pr(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    target = repo.head.commit  # target-branch base

    # Upstream commit being merged == the mergai merge commit's second parent.
    upstream = _commit(
        repo, tmp_path, "u.txt", "upstream\n", "upstream commit", [target]
    )

    # Raw mergai merge commit (second parent == upstream): carries conflict_context.
    merge = _commit(
        repo,
        tmp_path,
        "m.txt",
        "merge\n",
        "Merge commit 'up' into master",
        [target, upstream],
    )
    _note(
        repo,
        merge.hexsha,
        {
            "mergai_version": "test",
            "merge_info": {
                "target_branch": "master",
                "target_branch_sha": target.hexsha,
                "merge_commit": upstream.hexsha,
            },
            "conflict_context": {
                "ours_commit": target.hexsha,
                "theirs_commit": upstream.hexsha,
                "base_commit": target.hexsha,
                "files": ["x.cpp"],
                "conflict_types": {"x.cpp": "both modified"},
            },
        },
    )

    # Solution branch off the raw merge: a resolution commit and a CI-fix commit,
    # each carrying a solution -- this is what a Conflict Solution PR brings in.
    resolve = _commit(
        repo,
        tmp_path,
        "x.cpp",
        "resolved\n",
        "Resolve conflicts for merge 'up' into master",
        [merge],
    )
    _note(repo, resolve.hexsha, _solution("resolve x.cpp"))
    cifix = _commit(
        repo,
        tmp_path,
        "x.cpp",
        "resolved+fixed\n",
        "Fix 'build-and-test' failure for merge 'up' into 'master'",
        [resolve],
    )
    _note(repo, cifix.hexsha, _solution("fix build"))

    # The Conflict Solution PR merge: first parent = conflict tip (raw merge),
    # second parent = solution tip. GitHub creates this; it has no note.
    pr_merge = _commit(
        repo,
        tmp_path,
        "m.txt",
        "merge\n",
        "Merge pull request #1 from x/solution",
        [merge, cifix],
    )

    # Point HEAD at the PR merge (this is what finalize squashes from).
    repo.git.reset("--hard", pr_merge.hexsha)

    app = AppContext()
    app._note = MergaiNote.from_dict(
        {
            "mergai_version": "test",
            "merge_info": {
                "target_branch": "master",
                "target_branch_sha": target.hexsha,
                "merge_commit": upstream.hexsha,
            },
        },
        repo,
    )

    collected = app._collect_commits_for_squash(target.hexsha)
    shas = {c.hexsha for c, _ in collected}

    # Raw merge + BOTH solution-side commits collected; the PR merge and the
    # upstream commit are not treated as squashable content.
    assert merge.hexsha in shas
    assert resolve.hexsha in shas
    assert cifix.hexsha in shas
    assert pr_merge.hexsha not in shas
    assert upstream.hexsha not in shas

    combined = MergaiNote.combine_from_dicts(collected, repo)
    assert combined.has_conflict_context
    assert combined.has_solutions
    assert len(combined.solutions) == 2
