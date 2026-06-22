"""Command-level tests for ``mergai fork merge-pick --plan`` and ``--ai``.

The gate evaluation and candidate-window construction are stubbed (covered by
``test_merge_gate``); these tests focus on the command wiring: the ``--plan``
gate decision (merge / wait) and the ``--ai`` flow's sha validation,
deterministic fallback, and ``fallback: error`` behavior. A fake
``AgentExecutor`` runs the real validator against a canned agent result.
"""

import json
from types import SimpleNamespace

from click.testing import CliRunner

import mergai.agent_executor
import mergai.commands.fork as fork_mod
import mergai.prompt_builder
from mergai.config import AiPickConfig, MergaiConfig
from mergai.merge_pick_strategies.gate import GateDecision

# Reference these via their modules (fork_mod.fork,
# mergai.agent_executor.AgentExecutionError) rather than re-importing with
# `from ... import`, which would import the same module two ways (CodeQL).
fork = fork_mod.fork
AgentExecutionError = mergai.agent_executor.AgentExecutionError

FULL = "abc123" + "0" * 34  # 40-char candidate sha
OTHER = "def456" + "0" * 34
UPSTREAM = "upstream/master"


def _app(*, ai_pick=None):
    config = MergaiConfig()
    config.fork.ai_pick = ai_pick or AiPickConfig()
    return SimpleNamespace(
        config=config,
        repo=SimpleNamespace(),
        state=SimpleNamespace(path="/tmp"),
        get_agent=lambda agent_desc=None, yolo=False: SimpleNamespace(),
    )


def _patch_fork_status(monkeypatch, *, up_to_date=False):
    fs = SimpleNamespace(
        is_up_to_date=up_to_date,
        commits_behind=0 if up_to_date else 60,
        unmerged_oldest_age_days=None if up_to_date else 1.0,
    )
    monkeypatch.setattr(fork_mod.git_utils, "get_fork_status", lambda *a, **k: fs)
    return fs


def _patch_gate(monkeypatch, *, open_, window, prioritized=None, reason="min_commits"):
    decision = GateDecision(open=open_, reason=reason)
    monkeypatch.setattr(
        fork_mod,
        "compute_gate",
        lambda *a, **k: (decision, window, 0, prioritized or []),
    )
    return decision


def _run(app, args):
    return CliRunner().invoke(fork, ["merge-pick", *args], obj=app)


# --- --plan ---------------------------------------------------------------


def test_plan_wait_when_gate_closed(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(
        monkeypatch,
        open_=False,
        window=[FULL],
        reason="wait (12 < 50 commits; oldest 0.3d < 2d)",
    )
    res = _run(_app(), ["--plan", UPSTREAM])
    assert res.exit_code == 0
    assert json.loads(res.output) == {
        "action": "wait",
        "reason": "wait (12 < 50 commits; oldest 0.3d < 2d)",
    }


def test_plan_merge_when_gate_open(monkeypatch):
    # --plan is the go/no-go decision only: action + reason, no mode/sha (which
    # commit to merge to is a separate, explicit pick step).
    _patch_fork_status(monkeypatch)
    _patch_gate(
        monkeypatch, open_=True, window=[FULL, OTHER], reason="min_commits (60 >= 50)"
    )
    res = _run(_app(), ["--plan", UPSTREAM])
    assert res.exit_code == 0
    assert json.loads(res.output) == {
        "action": "merge",
        "reason": "min_commits (60 >= 50)",
    }


def test_plan_up_to_date(monkeypatch):
    _patch_fork_status(monkeypatch, up_to_date=True)
    res = _run(_app(), ["--plan", UPSTREAM])
    assert json.loads(res.output) == {
        "action": "wait",
        "reason": "up to date (0 commits)",
    }


def test_plan_and_ai_mutually_exclusive():
    res = _run(_app(), ["--plan", "--ai", UPSTREAM])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.stderr.lower()


# --- --ai -----------------------------------------------------------------


class _FakeExecutor:
    """Stub AgentExecutor: runs the real validator against a canned result.

    Mirrors the real contract closely enough for the command: a result that
    fails validation raises ``AgentExecutionError`` (as the real executor does
    once retries are exhausted).
    """

    result = {"response": {"sha": FULL, "reasoning": "best boundary"}}

    def __init__(self, *args, **kwargs):
        pass

    def run_with_retry(self, prompt, validator=None):
        if validator is not None:
            error = validator(self.result)
            if error is not None:
                raise AgentExecutionError(error)
        return self.result


def _patch_ai_machinery(monkeypatch, executor_cls):
    monkeypatch.setattr(fork_mod, "build_merge_pick_input", lambda *a, **k: {})
    monkeypatch.setattr(
        mergai.prompt_builder, "build_merge_pick_prompt", lambda *a, **k: "prompt"
    )
    monkeypatch.setattr(mergai.agent_executor, "AgentExecutor", executor_cls)


def test_ai_emits_chosen_sha_and_reasoning(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])
    _patch_ai_machinery(monkeypatch, _FakeExecutor)

    res = _run(_app(), ["--ai", UPSTREAM])
    assert res.exit_code == 0
    # Styled human output: the full sha and the reasoning are both visible.
    assert FULL in res.output
    assert "Merge pick" in res.output
    assert "Reasoning" in res.output
    assert "best boundary" in res.output


def test_ai_json_output(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])
    _patch_ai_machinery(monkeypatch, _FakeExecutor)

    res = _run(_app(), ["--ai", "--json", UPSTREAM])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data == {
        "sha": FULL,
        "short_sha": FULL[:11],
        "reasoning": "best boundary",
        "source": "ai",
    }


def test_ai_json_output_on_deterministic_fallback(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])

    class _BadShaExecutor(_FakeExecutor):
        result = {"response": {"sha": "9" * 40, "reasoning": "nope"}}

    _patch_ai_machinery(monkeypatch, _BadShaExecutor)

    res = _run(_app(), ["--ai", "--json", UPSTREAM])
    assert res.exit_code == 0
    # The fallback note must land on stderr, keeping stdout's JSON payload clean.
    assert "falling back to deterministic" in res.stderr
    # In this Click version res.output interleaves stderr into stdout, so the
    # JSON is the last line; the stderr assertion above guards against the note
    # actually leaking onto stdout.
    data = json.loads(res.output.strip().splitlines()[-1])
    assert data["sha"] == FULL  # window tip
    assert data["source"] == "deterministic"


def test_ai_next_prints_only_sha(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])
    _patch_ai_machinery(monkeypatch, _FakeExecutor)

    res = _run(_app(), ["--ai", "--next", UPSTREAM])
    assert res.output.strip() == FULL


def test_ai_invalid_sha_falls_back_deterministic(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])

    class _BadShaExecutor(_FakeExecutor):
        result = {"response": {"sha": "9" * 40, "reasoning": "nope"}}

    _patch_ai_machinery(monkeypatch, _BadShaExecutor)

    res = _run(_app(), ["--ai", "--next", UPSTREAM])
    assert res.exit_code == 0
    # Deterministic fallback -> window tip (newest) emitted to stdout.
    assert FULL in res.output
    assert "falling back to deterministic" in res.stderr


def test_ai_invalid_sha_with_error_fallback_exits_nonzero(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])

    class _BadShaExecutor(_FakeExecutor):
        result = {"response": {"sha": "9" * 40}}

    _patch_ai_machinery(monkeypatch, _BadShaExecutor)

    app = _app(ai_pick=AiPickConfig(fallback="error"))
    res = _run(app, ["--ai", UPSTREAM])
    assert res.exit_code != 0
    assert "AI pick failed" in res.stderr


def test_ai_bails_when_gate_closed(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=False, window=[FULL], reason="wait (...)")
    _patch_ai_machinery(monkeypatch, _FakeExecutor)

    res = _run(_app(), ["--ai", "--next", UPSTREAM])
    assert res.exit_code == 0
    # No sha emitted; the closed-gate note goes to stderr only.
    assert FULL not in res.output
    assert "Merge gate closed" in res.stderr


def test_ai_force_skips_gate_recheck(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=False, window=[FULL, OTHER], reason="wait (...)")
    _patch_ai_machinery(monkeypatch, _FakeExecutor)

    res = _run(_app(), ["--ai", "--force", "--next", UPSTREAM])
    assert res.exit_code == 0
    assert res.output.strip() == FULL


def test_ai_up_to_date_returns_quietly(monkeypatch):
    _patch_fork_status(monkeypatch, up_to_date=True)
    res = _run(_app(), ["--ai", "--next", UPSTREAM])
    assert res.exit_code == 0
    assert res.output.strip() == ""


def test_ai_missing_reasoning_falls_back(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])

    class _NoReasoningExecutor(_FakeExecutor):
        result = {"response": {"sha": FULL}}  # valid sha, no reasoning

    _patch_ai_machinery(monkeypatch, _NoReasoningExecutor)

    res = _run(_app(), ["--ai", "--next", UPSTREAM])
    assert res.exit_code == 0
    assert FULL in res.output  # deterministic fallback (window tip)
    assert "falling back to deterministic" in res.stderr


def test_ai_next_keeps_stdout_clean(monkeypatch):
    """Executor progress (stdout chatter) must not pollute the captured sha."""
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])

    import click

    class _NoisyExecutor(_FakeExecutor):
        def run_with_retry(self, prompt, validator=None):
            click.echo("Attempt 1 of 3...")  # the kind of noise the real one emits
            return super().run_with_retry(prompt, validator)

    _patch_ai_machinery(monkeypatch, _NoisyExecutor)

    res = _run(_app(), ["--ai", "--next", UPSTREAM])
    assert res.exit_code == 0
    assert FULL in res.output
    # The chatter lands in stderr only because the redirect moved it off stdout;
    # without the redirect click.echo would write it to stdout (not stderr).
    assert "Attempt 1 of 3" in res.stderr


def test_force_without_ai_or_gate_errors():
    res = _run(_app(), ["--force", UPSTREAM])
    assert res.exit_code != 0
    assert "--force" in res.stderr and "--ai or --gate" in res.stderr


def test_json_without_ai_errors():
    res = _run(_app(), ["--json", UPSTREAM])
    assert res.exit_code != 0
    assert "--json" in res.stderr and "requires --ai" in res.stderr


# --- --gate ---------------------------------------------------------------


def test_gate_emits_window_tip_when_no_match(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER])
    res = _run(_app(), ["--gate", UPSTREAM])
    assert res.exit_code == 0
    # No prioritized match -> window tip (newest, capped to the window).
    assert res.output.strip() == FULL


def test_gate_emits_forced_pick(monkeypatch):
    _patch_fork_status(monkeypatch)
    # A force-strategy (conflict) match is the cut point even before min_commits.
    pick = SimpleNamespace(
        commit=SimpleNamespace(hexsha=OTHER), strategy_name="conflict"
    )
    _patch_gate(monkeypatch, open_=True, window=[FULL, OTHER], prioritized=[pick])
    res = _run(_app(), ["--gate", UPSTREAM])
    assert res.exit_code == 0
    assert res.output.strip() == OTHER


def test_gate_closed_emits_no_pick(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=False, window=[FULL], reason="wait (5 < 50)")
    res = _run(_app(), ["--gate", UPSTREAM])
    assert res.exit_code == 0
    # No pick on stdout (the closed-gate note goes to stderr).
    assert FULL not in res.output and OTHER not in res.output


def test_gate_force_picks_despite_closed_gate(monkeypatch):
    _patch_fork_status(monkeypatch)
    _patch_gate(monkeypatch, open_=False, window=[FULL], reason="wait")
    res = _run(_app(), ["--gate", "--force", UPSTREAM])
    assert res.exit_code == 0
    assert res.output.strip() == FULL


def test_gate_up_to_date_emits_nothing(monkeypatch):
    _patch_fork_status(monkeypatch, up_to_date=True)
    res = _run(_app(), ["--gate", UPSTREAM])
    assert res.exit_code == 0
    assert res.output.strip() == ""


def test_gate_and_ai_mutually_exclusive():
    res = _run(_app(), ["--gate", "--ai", UPSTREAM])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.stderr.lower()


def test_ai_pick_invalid_fallback_raises():
    import pytest

    with pytest.raises(ValueError, match="ai_pick.fallback"):
        AiPickConfig.from_dict({"fallback": "nope"})
    # ...and surfaced through the top-level config parser too.
    with pytest.raises(ValueError, match="ai_pick.fallback"):
        MergaiConfig.from_dict({"fork": {"ai_pick": {"fallback": "nope"}}})
