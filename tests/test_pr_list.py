"""Tests for the ``mergai pr list`` command.

Exercises the namespace gate (only mergai-managed PRs), the ``--sha`` /
``--state`` / ``--type`` filters, and the ``-q`` / ``--format json`` output
shapes against hand-built PR stubs - no network. ``get_pulls`` is faked to
honor the GitHub ``state`` filter the command passes through.
"""

from datetime import datetime
from types import SimpleNamespace

from click.testing import CliRunner

from mergai.commands.pr import pr
from mergai.config import BranchConfig

SHA = "deadbeef000" + "0" * 29  # 40-char full SHA; short form is SHA[:11]
SHORT_SHA = SHA[:11]  # "deadbeef000"
OTHER_SHA = "cafebabe111" + "1" * 29


def _pr(number, head_ref, *, state="open", merged=False, head_sha=None):
    return SimpleNamespace(
        number=number,
        head=SimpleNamespace(ref=head_ref, sha=head_sha or f"headsha{number}"),
        base=SimpleNamespace(ref="master"),
        state=state,
        merged_at=datetime(2026, 6, 1) if merged else None,
        html_url=f"https://github.com/o/r/pull/{number}",
        title=f"PR {number}",
        user=SimpleNamespace(login="mergai-bot"),
        created_at=datetime(2026, 6, 1),
    )


class _FakeRepo:
    """Stub for ``app.gh_repo`` whose ``get_pulls`` mimics GitHub state filtering."""

    def __init__(self, pulls):
        self._pulls = pulls
        self.last_state = None

    def get_pulls(self, state="open", sort=None):  # noqa: A002 - mirror PyGithub
        self.last_state = state
        if state == "open":
            return [p for p in self._pulls if p.state == "open"]
        if state == "closed":
            return [p for p in self._pulls if p.state == "closed"]
        return list(self._pulls)


def _note(merge_commit_sha):
    """A MergaiNote-like stub exposing ``.merge_info.merge_commit_sha``."""
    return SimpleNamespace(
        merge_info=SimpleNamespace(merge_commit_sha=merge_commit_sha)
    )


def _app(pulls, *, notes=None, branch_config=None):
    # notes: optional {head_sha: full_merge_commit_sha} for note enrichment.
    notes = notes or {}

    def try_get_note_from_commit(commit):
        full = notes.get(commit)
        return _note(full) if full is not None else None

    return SimpleNamespace(
        gh_repo=_FakeRepo(pulls),
        config=SimpleNamespace(branch=branch_config or BranchConfig()),
        try_get_note_from_commit=try_get_note_from_commit,
    )


def _run(app, args):
    return CliRunner().invoke(pr, ["--repo", "o/r", "list", *args], obj=app)


def test_no_prs_outputs():
    app = _app([])
    # text
    res = _run(app, [])
    assert res.exit_code == 0
    assert res.output.strip() == "No matching PRs."
    # quiet -> nothing
    res = _run(app, ["-q"])
    assert res.exit_code == 0
    assert res.output.strip() == ""
    # json -> empty array
    res = _run(app, ["--format", "json"])
    assert res.exit_code == 0
    assert res.output.strip() == "[]"


def test_lists_open_mergai_pr():
    app = _app([_pr(1, f"mergai/master-{SHORT_SHA}/solution")])
    res = _run(app, ["-q"])
    assert res.exit_code == 0
    assert res.output.split() == ["1"]


def test_excludes_non_mergai_pr():
    app = _app(
        [
            _pr(1, f"mergai/master-{SHORT_SHA}/main"),
            _pr(2, "feature/foo"),
        ]
    )
    res = _run(app, ["-q"])
    assert res.output.split() == ["1"]


def test_sha_scoping_short_and_full():
    app = _app([_pr(7, f"mergai/master-{SHORT_SHA}/conflict")])
    assert _run(app, ["-q", "--sha", SHORT_SHA]).output.split() == ["7"]
    assert _run(app, ["-q", "--sha", SHA]).output.split() == ["7"]
    assert _run(app, ["-q", "--sha", OTHER_SHA]).output.split() == []


def test_type_filter():
    app = _app(
        [
            _pr(1, f"mergai/master-{SHORT_SHA}/main"),
            _pr(2, f"mergai/master-{SHORT_SHA}/conflict"),
            _pr(3, f"mergai/master-{SHORT_SHA}/solution"),
        ]
    )
    assert _run(app, ["-q", "--type", "solution"]).output.split() == ["3"]


def test_state_merged_keeps_only_merged():
    app = _app(
        [
            _pr(1, f"mergai/master-{SHORT_SHA}/main", state="closed", merged=True),
            _pr(2, f"mergai/master-{SHORT_SHA}/conflict", state="closed", merged=False),
            _pr(3, f"mergai/master-{SHORT_SHA}/solution", state="open"),
        ]
    )
    res = _run(app, ["-q", "--state", "merged"])
    assert res.output.split() == ["1"]
    # merged maps to the GitHub "closed" query
    assert app.gh_repo.last_state == "closed"


def test_json_shape_with_note():
    app = _app(
        [_pr(5, f"mergai/master-{SHORT_SHA}/main", head_sha="hs5")],
        notes={"hs5": SHA},
    )
    res = _run(app, ["--format", "json"])
    assert res.exit_code == 0
    import json

    data = json.loads(res.stdout)
    assert len(data) == 1
    row = data[0]
    assert row["number"] == 5
    assert row["type"] == "main"
    assert row["target_branch"] == "master"
    assert row["merged"] is False
    # short SHA comes from the branch name; full SHA comes from the note
    assert row["branch_merge_commit_sha"] == SHORT_SHA
    assert row["merge_commit_sha"] == SHA


def test_json_full_sha_null_without_note():
    app = _app([_pr(5, f"mergai/master-{SHORT_SHA}/main")])
    import json

    row = json.loads(_run(app, ["--format", "json"]).stdout)[0]
    assert row["branch_merge_commit_sha"] == SHORT_SHA
    assert row["merge_commit_sha"] is None


def test_text_shows_merge_sha_from_note():
    app = _app(
        [_pr(5, f"mergai/master-{SHORT_SHA}/main", head_sha="hs5")],
        notes={"hs5": SHA},
    )
    res = _run(app, [])
    assert res.exit_code == 0
    assert f"Merge SHA  : {SHA}" in res.output


def test_text_marks_unavailable_note():
    app = _app([_pr(5, f"mergai/master-{SHORT_SHA}/main")])
    res = _run(app, [])
    assert "Merge SHA  : (note unavailable)" in res.output


def test_missing_note_hint_emitted():
    app = _app(
        [
            _pr(1, f"mergai/master-{SHORT_SHA}/main", head_sha="hs1"),
            _pr(2, f"mergai/master-{SHORT_SHA}/conflict", head_sha="hs2"),
        ],
        notes={"hs1": SHA},  # PR 2 has no note
    )
    res = _run(app, [])
    assert res.exit_code == 0
    assert "1 PR(s) have no local mergai note" in res.stderr
    assert "refs/notes/mergai:refs/notes/mergai" in res.stderr


def test_no_hint_when_all_notes_present():
    app = _app(
        [_pr(1, f"mergai/master-{SHORT_SHA}/main", head_sha="hs1")],
        notes={"hs1": SHA},
    )
    res = _run(app, [])
    assert res.stderr.strip() == ""


# Branch format that embeds the FULL merge SHA: no note is needed to report it.
_FULL_FMT = BranchConfig(
    name_format="mergai/%(target_branch)-%(merge_commit_sha)/%(type)"
)


def test_full_sha_from_branch_without_note():
    app = _app(
        [_pr(1, f"mergai/master-{SHA}/main")],  # no note provided
        branch_config=_FULL_FMT,
    )
    # text: full SHA shown, no "(note unavailable)", no hint
    res = _run(app, [])
    assert f"Merge SHA  : {SHA}" in res.output
    assert "(note unavailable)" not in res.output
    assert res.stderr.strip() == ""
    # json: merge_commit_sha is the full SHA even without a note
    import json

    row = json.loads(_run(app, ["--format", "json"]).stdout)[0]
    assert row["merge_commit_sha"] == SHA


def test_state_and_sha_are_case_insensitive():
    app = _app([_pr(9, f"mergai/master-{SHORT_SHA}/main")])
    # --state accepts mixed case
    assert _run(app, ["-q", "--state", "Open"]).output.split() == ["9"]
    # --sha matches case-insensitively (uppercase request)
    assert _run(app, ["-q", "--sha", SHORT_SHA.upper()]).output.split() == ["9"]
