"""Command-level tests for ``mergai fork merge-pick --record`` / ``--manual``.

``--record`` persists the chosen pick's metadata (type, sha, strategy, summary)
to the state store so ``mergai context init`` can attach it to the note as
``merge_pick``. ``--manual`` records an externally-chosen sha without evaluating
the gate. These tests exercise the command wiring and the recorded payload;
``context init``'s read side is covered by ``test_attach_recorded_pick``.
"""

from types import SimpleNamespace

from click.testing import CliRunner

import mergai.agent_executor
import mergai.commands.fork as fork_mod
import mergai.prompt_builder
from mergai.commands.context import _attach_recorded_pick
from mergai.config import AiPickConfig, MergaiConfig
from mergai.merge_pick_strategies.gate import GateDecision
from mergai.models import MergaiNote, MergeInfo
from mergai.utils.state_store import StateStore

fork = fork_mod.fork
AgentExecutionError = mergai.agent_executor.AgentExecutionError

FULL = "abc123" + "0" * 34
OTHER = "def456" + "0" * 34
UPSTREAM = "upstream/master"


def _app(tmp_path, *, ai_pick=None):
    """Fake AppContext backed by a real StateStore in ``tmp_path``."""
    config = MergaiConfig()
    config.fork.ai_pick = ai_pick or AiPickConfig()
    return SimpleNamespace(
        config=config,
        repo=SimpleNamespace(),
        state=StateStore(str(tmp_path)),
        get_agent=lambda agent_desc=None, yolo=False: SimpleNamespace(),
    )


def _run(app, args):
    return CliRunner().invoke(fork, ["merge-pick", *args], obj=app)


def _pick_commit(sha, strategy):
    return SimpleNamespace(commit=SimpleNamespace(hexsha=sha), strategy_name=strategy)


def _patch_fork_status(monkeypatch, *, up_to_date=False):
    fs = SimpleNamespace(
        is_up_to_date=up_to_date,
        commits_behind=0 if up_to_date else 60,
        unmerged_oldest_age_days=None if up_to_date else 1.0,
        unmerged_commit_shas=[FULL, OTHER],
        unmerged_commits=[],
    )
    monkeypatch.setattr(fork_mod.git_utils, "get_fork_status", lambda *a, **k: fs)
    return fs


def _patch_gate(monkeypatch, *, open_, window, prioritized=None):
    decision = GateDecision(open=open_, reason="min_commits")
    monkeypatch.setattr(
        fork_mod,
        "compute_gate",
        lambda *a, **k: (decision, window, 0, prioritized or []),
    )
    monkeypatch.setattr(
        fork_mod, "resolve_upstream_ref", lambda app, ref: ref or UPSTREAM
    )
    return decision


# --- --manual -------------------------------------------------------------


def test_manual_records_pick_with_actor(tmp_path):
    app = _app(tmp_path)
    res = _run(app, ["--manual", FULL, "--actor", "pawel", "--record"])
    assert res.exit_code == 0, res.output
    assert res.output.strip() == FULL  # sha echoed for capture
    pick = app.state.load_pick()
    assert pick["type"] == "manual"
    assert pick["sha"] == FULL
    assert pick["short_sha"] == FULL[:11]
    assert pick["strategy"] is None
    assert "@pawel" in pick["summary"]
    assert "gate bypassed" in pick["summary"]


def test_manual_without_record_writes_no_file(tmp_path):
    app = _app(tmp_path)
    res = _run(app, ["--manual", FULL])
    assert res.exit_code == 0
    assert res.output.strip() == FULL
    assert not app.state.pick_exists()


def test_manual_mutually_exclusive_with_gate(tmp_path):
    res = _run(_app(tmp_path), ["--manual", FULL, "--gate"])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.stderr.lower()


def test_actor_requires_manual(tmp_path):
    res = _run(_app(tmp_path), ["--actor", "pawel", "--gate"])
    assert res.exit_code != 0
    assert "--actor requires --manual" in res.stderr.lower()


def test_manual_mutually_exclusive_with_next(tmp_path):
    res = _run(_app(tmp_path), ["--manual", FULL, "--next"])
    assert res.exit_code != 0
    assert "mutually exclusive with --next" in res.stderr.lower()


def test_record_requires_a_pick_mode(tmp_path):
    # --record on a read-only path is rejected so it can't clear the pick file
    # without writing a replacement.
    app = _app(tmp_path)
    app.state.save_pick({"type": "gate", "sha": FULL, "summary": "keep me"})
    res = _run(app, ["--record", "--list", UPSTREAM])
    assert res.exit_code != 0
    assert "--record requires" in res.stderr.lower()
    # The pre-existing pick file was left untouched.
    assert app.state.load_pick()["summary"] == "keep me"


# --- --gate --record ------------------------------------------------------


def test_gate_records_pick_with_strategy(monkeypatch, tmp_path):
    app = _app(tmp_path)
    _patch_fork_status(monkeypatch)
    _patch_gate(
        monkeypatch,
        open_=True,
        window=[FULL],
        prioritized=[_pick_commit(FULL, "important_files")],
    )
    res = _run(app, ["--gate", "--record", UPSTREAM])
    assert res.exit_code == 0, res.output
    assert res.output.strip() == FULL
    pick = app.state.load_pick()
    assert pick["type"] == "gate"
    assert pick["sha"] == FULL
    assert pick["strategy"] == "important_files"
    assert "important_files" in pick["summary"]


def test_gate_records_window_tip_when_no_strategy(monkeypatch, tmp_path):
    app = _app(tmp_path)
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL], prioritized=[])
    res = _run(app, ["--gate", "--record", UPSTREAM])
    assert res.exit_code == 0, res.output
    pick = app.state.load_pick()
    assert pick["type"] == "gate"
    assert pick["strategy"] is None
    assert "window tip" in pick["summary"]


def test_gate_record_clears_stale_file_when_gate_closed(monkeypatch, tmp_path):
    app = _app(tmp_path)
    app.state.save_pick({"type": "gate", "sha": OTHER})  # stale from a prior run
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=False, window=[FULL], prioritized=[])
    res = _run(app, ["--gate", "--record", UPSTREAM])
    assert res.exit_code == 0
    # Gate closed -> no sha echoed and no stale file left behind.
    assert FULL not in res.output
    assert not app.state.pick_exists()


# --- --next --record ------------------------------------------------------


def test_next_records_pick(monkeypatch, tmp_path):
    app = _app(tmp_path)
    _patch_fork_status(monkeypatch)
    monkeypatch.setattr(
        fork_mod, "resolve_upstream_ref", lambda app, ref: ref or UPSTREAM
    )
    monkeypatch.setattr(
        fork_mod,
        "get_prioritized_commits",
        lambda *a, **k: [_pick_commit(FULL, "conflict")],
    )
    res = _run(app, ["--next", "--record", UPSTREAM])
    assert res.exit_code == 0, res.output
    assert res.output.strip() == FULL
    pick = app.state.load_pick()
    assert pick["type"] == "next"
    assert pick["strategy"] == "conflict"


# --- --ai --record --------------------------------------------------------


class _FakeExecutor:
    result = {"response": {"sha": FULL, "reasoning": "best boundary"}}

    def __init__(self, *args, **kwargs):
        pass

    def run_with_retry(self, prompt, validator=None):
        if validator is not None:
            error = validator(self.result)
            if error is not None:
                raise AgentExecutionError(error)
        return self.result


class _FailingExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def run_with_retry(self, prompt, validator=None):
        raise AgentExecutionError("agent boom")


def _patch_ai_machinery(monkeypatch, executor_cls):
    monkeypatch.setattr(fork_mod, "build_merge_pick_input", lambda *a, **k: {})
    monkeypatch.setattr(
        mergai.prompt_builder, "build_merge_pick_prompt", lambda *a, **k: "prompt"
    )
    monkeypatch.setattr(mergai.agent_executor, "AgentExecutor", executor_cls)


def test_ai_records_pick_with_reasoning(monkeypatch, tmp_path):
    app = _app(tmp_path)
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])
    _patch_ai_machinery(monkeypatch, _FakeExecutor)
    res = _run(app, ["--ai", "--json", "--record", UPSTREAM])
    assert res.exit_code == 0, res.output
    pick = app.state.load_pick()
    assert pick["type"] == "ai"
    assert pick["sha"] == FULL
    assert pick["summary"] == "best boundary"


def test_ai_fallback_records_ai_type_with_fallback_summary(monkeypatch, tmp_path):
    app = _app(tmp_path)
    _patch_fork_status(monkeypatch)
    _patch_gate(
        monkeypatch,
        open_=True,
        window=[FULL],
        prioritized=[_pick_commit(FULL, "important_files")],
    )
    _patch_ai_machinery(monkeypatch, _FailingExecutor)
    res = _run(app, ["--ai", "--json", "--record", UPSTREAM])
    assert res.exit_code == 0, res.output
    pick = app.state.load_pick()
    # The mechanism is still "ai"; the fallback is noted in the summary only.
    assert pick["type"] == "ai"
    assert "fell back to deterministic" in pick["summary"]
    assert pick["strategy"] == "important_files"


# --- context init read side ------------------------------------------------


def _note():
    mi = MergeInfo(
        target_branch="v8.0", target_branch_sha="b" * 40, merge_commit_sha=FULL
    )
    return MergaiNote.create(mi)


def test_attach_recorded_pick_sets_merge_pick(tmp_path):
    app = _app(tmp_path)
    pick = {
        "type": "gate",
        "sha": FULL,
        "short_sha": FULL[:11],
        "strategy": "conflict",
        "summary": "s",
    }
    app.state.save_pick(pick)
    note = _note()
    _attach_recorded_pick(app, note, FULL)
    assert note.merge_pick == pick


def test_attach_recorded_pick_noop_without_file(tmp_path):
    app = _app(tmp_path)
    note = _note()
    _attach_recorded_pick(app, note, FULL)
    assert note.merge_pick is None


def test_attach_recorded_pick_ignores_stale_sha(tmp_path):
    app = _app(tmp_path)
    app.state.save_pick({"type": "gate", "sha": OTHER, "summary": "s"})
    note = _note()
    _attach_recorded_pick(app, note, FULL)  # note is for FULL, pick is for OTHER
    assert note.merge_pick is None


def test_attach_recorded_pick_ignores_missing_sha(tmp_path):
    app = _app(tmp_path)
    app.state.save_pick({"type": "gate", "summary": "s"})  # corrupt: no sha
    note = _note()
    _attach_recorded_pick(app, note, FULL)
    assert note.merge_pick is None


def test_attach_recorded_pick_matches_short_sha(tmp_path):
    app = _app(tmp_path)
    # Recorded short sha vs full merge_commit_sha still matches on the prefix.
    app.state.save_pick({"type": "gate", "sha": FULL[:11], "summary": "s"})
    note = _note()
    _attach_recorded_pick(app, note, FULL)
    assert note.merge_pick is not None


# --- show-pick -------------------------------------------------------------


def _run_fork(app, args):
    return CliRunner().invoke(fork, args, obj=app)


def test_show_pick_markdown(tmp_path):
    app = _app(tmp_path)
    app.state.save_pick(
        {
            "type": "gate",
            "sha": FULL,
            "short_sha": FULL[:11],
            "strategy": "important_files",
            "summary": "picked the boundary",
        }
    )
    res = _run_fork(app, ["show-pick", "--format", "markdown"])
    assert res.exit_code == 0, res.output
    assert "## Merge Pick" in res.output
    assert "gate" in res.output
    assert "important_files" in res.output
    assert "picked the boundary" in res.output


def test_show_pick_no_file_is_noop(tmp_path):
    app = _app(tmp_path)
    res = _run_fork(app, ["show-pick", "--format", "markdown"])
    assert res.exit_code == 0
    assert res.output.strip() == ""
