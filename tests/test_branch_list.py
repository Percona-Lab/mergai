"""Tests for the ``mergai branch list`` command.

Exercises the namespace gate (only mergai-managed branches), the
``--sha`` / ``--type`` filters, scope selection (``--local`` / ``--remote`` /
``--all``), and the ``-q`` / ``--format json`` output shapes against hand-built
ref stubs - no network. ``ls_remote`` and ``repo.heads`` are faked.
"""

from types import SimpleNamespace

from click.testing import CliRunner

from mergai.commands.branch import branch
from mergai.config import BranchConfig

SHA = "deadbeef000" + "0" * 29  # 40-char full SHA; short form is SHA[:11]
SHORT_SHA = SHA[:11]  # "deadbeef000"
OTHER_SHA = "cafebabe111" + "1" * 29


class _Git:
    def __init__(self, remote_names):
        self._remote_names = remote_names

    def ls_remote(self, *args):  # args: ("--heads", remote, pattern)
        return "\n".join(f"{'a' * 40}\trefs/heads/{n}" for n in self._remote_names)


def _app(*, local=None, remote=None, branch_config=None):
    return SimpleNamespace(
        repo=SimpleNamespace(
            git=_Git(remote or []),
            heads=[SimpleNamespace(name=n) for n in (local or [])],
        ),
        config=SimpleNamespace(branch=branch_config or BranchConfig()),
    )


def _run(app, args):
    return CliRunner().invoke(branch, ["list", *args], obj=app)


def test_no_branches_outputs():
    app = _app()
    res = _run(app, [])
    assert res.exit_code == 0
    assert res.output.strip() == "No matching branches."
    assert _run(app, ["-q"]).output.strip() == ""
    assert _run(app, ["--format", "json"]).output.strip() == "[]"


def test_lists_remote_branch():
    app = _app(remote=[f"mergai/master-{SHORT_SHA}/conflict"])
    res = _run(app, ["-q"])
    assert res.exit_code == 0
    assert res.output.split() == [f"mergai/master-{SHORT_SHA}/conflict"]


def test_excludes_non_mergai_refs():
    app = _app(
        local=["master", "feature/foo"],
        remote=[f"mergai/master-{SHORT_SHA}/main", "release/1.0"],
    )
    res = _run(app, ["-q"])
    assert res.output.split() == [f"mergai/master-{SHORT_SHA}/main"]


def test_sha_scoping_short_and_full():
    app = _app(remote=[f"mergai/master-{SHORT_SHA}/conflict"])
    name = f"mergai/master-{SHORT_SHA}/conflict"
    assert _run(app, ["-q", "--sha", SHORT_SHA]).output.split() == [name]
    assert _run(app, ["-q", "--sha", SHA]).output.split() == [name]
    assert _run(app, ["-q", "--sha", OTHER_SHA]).output.split() == []


def test_type_filter():
    app = _app(
        remote=[
            f"mergai/master-{SHORT_SHA}/main",
            f"mergai/master-{SHORT_SHA}/conflict",
            f"mergai/master-{SHORT_SHA}/solution",
        ]
    )
    assert _run(app, ["-q", "--type", "solution"]).output.split() == [
        f"mergai/master-{SHORT_SHA}/solution"
    ]


def test_scope_selection():
    local_only = f"mergai/master-{SHORT_SHA}/main"
    remote_only = f"mergai/master-{SHORT_SHA}/conflict"
    app = lambda: _app(local=[local_only], remote=[remote_only])  # noqa: E731
    assert _run(app(), ["-q", "--local"]).output.split() == [local_only]
    assert _run(app(), ["-q", "--remote"]).output.split() == [remote_only]
    assert sorted(_run(app(), ["-q", "--all"]).output.split()) == sorted(
        [local_only, remote_only]
    )
    # default is --all
    assert sorted(_run(app(), ["-q"]).output.split()) == sorted(
        [local_only, remote_only]
    )


def test_branch_in_both_scopes_is_deduped():
    name = f"mergai/master-{SHORT_SHA}/main"
    app = _app(local=[name], remote=[name])
    res = _run(app, ["-q"])
    assert res.output.split() == [name]  # listed once
    # text shows the combined scope
    assert "[local+remote]" in _run(app, []).output


def test_mutually_exclusive_scopes_error():
    app = _app(remote=[f"mergai/master-{SHORT_SHA}/main"])
    res = _run(app, ["--local", "--remote"])
    assert res.exit_code != 0
    assert "only one of" in res.output.lower()


def test_json_shape():
    app = _app(remote=[f"mergai/master-{SHORT_SHA}/semantic"])
    import json

    data = json.loads(_run(app, ["--format", "json"]).stdout)
    assert len(data) == 1
    row = data[0]
    assert row["name"] == f"mergai/master-{SHORT_SHA}/semantic"
    assert row["scope"] == "remote"
    assert row["target_branch"] == "master"
    assert row["merge_commit_sha"] == SHORT_SHA
    assert row["type"] == "semantic"


def test_sha_case_insensitive():
    app = _app(remote=[f"mergai/master-{SHORT_SHA}/main"])
    assert _run(app, ["-q", "--sha", SHORT_SHA.upper()]).output.split() == [
        f"mergai/master-{SHORT_SHA}/main"
    ]
