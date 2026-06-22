"""Tests for the deterministic merge gate and its supporting helpers.

Covers the pure ``evaluate_merge_gate`` decision (wait / count / age / force /
disabled), the ``max_commits`` candidate-window capping, the deterministic
pick resolution, agent-sha resolution against the window, and config parsing
for ``merge_gate`` / ``ai_pick``. All hand-built stubs - no git repo.
"""

from types import SimpleNamespace

from mergai.commands.fork import (
    resolve_deterministic_sha,
    restrict_to_window,
)
from mergai.config import AiPickConfig, MergaiConfig, MergeGateConfig
from mergai.merge_pick_strategies.gate import evaluate_merge_gate


def _fork_status(commits_behind, age_days):
    return SimpleNamespace(
        commits_behind=commits_behind,
        unmerged_oldest_age_days=age_days,
    )


def _pick(strategy_name, sha="x"):
    return SimpleNamespace(
        strategy_name=strategy_name,
        commit=SimpleNamespace(hexsha=sha),
    )


# --- evaluate_merge_gate --------------------------------------------------


def test_gate_waits_below_thresholds():
    cfg = MergeGateConfig()  # min 50, age 2
    decision = evaluate_merge_gate(_fork_status(12, 0.3), [], cfg)
    assert decision.open is False
    assert decision.reason == "wait (12 < 50 commits; oldest 0.3d < 2d)"


def test_gate_opens_on_min_commits():
    cfg = MergeGateConfig()
    decision = evaluate_merge_gate(_fork_status(63, 0.3), [], cfg)
    assert decision.open is True
    assert decision.reason == "min_commits (63 >= 50)"


def test_gate_opens_on_max_age():
    cfg = MergeGateConfig()
    decision = evaluate_merge_gate(_fork_status(12, 3.5), [], cfg)
    assert decision.open is True
    assert decision.reason == "max_age (3.5d >= 2d)"


def test_gate_opens_on_force_strategy():
    cfg = MergeGateConfig()
    decision = evaluate_merge_gate(
        _fork_status(12, 0.3),
        [_pick("huge_commit"), _pick("conflict")],
        cfg,
    )
    assert decision.open is True
    assert decision.reason == "force:conflict"


def test_gate_force_takes_priority_over_count():
    # Even above min_commits, a forced strategy gives the more specific reason.
    cfg = MergeGateConfig()
    decision = evaluate_merge_gate(
        _fork_status(99, 9.0), [_pick("important_files")], cfg
    )
    assert decision.reason == "force:important_files"


def test_gate_ignores_non_force_strategies():
    cfg = MergeGateConfig(force_strategies=["conflict"])
    decision = evaluate_merge_gate(
        _fork_status(12, 0.3), [_pick("huge_commit"), _pick("branching_point")], cfg
    )
    assert decision.open is False


def test_gate_empty_force_list_disables_force():
    cfg = MergeGateConfig(force_strategies=[])
    decision = evaluate_merge_gate(_fork_status(12, 0.3), [_pick("conflict")], cfg)
    assert decision.open is False


def test_gate_handles_missing_age():
    cfg = MergeGateConfig()
    decision = evaluate_merge_gate(_fork_status(12, None), [], cfg)
    assert decision.open is False
    assert "oldest n/a" in decision.reason


# --- restrict_to_window ---------------------------------------------------


def test_window_no_capping_when_under_max():
    shas = ["c", "b", "a"]  # newest-first
    window, omitted = restrict_to_window(shas, 10)
    assert window == shas
    assert omitted == 0


def test_window_keeps_oldest_max_commits():
    # newest-first: e d c b a ; oldest two are a, b -> window tail ["b", "a"]
    shas = ["e", "d", "c", "b", "a"]
    window, omitted = restrict_to_window(shas, 2)
    assert window == ["b", "a"]
    assert omitted == 3


def test_window_disabled_with_none_or_zero():
    shas = ["c", "b", "a"]
    assert restrict_to_window(shas, None) == (shas, 0)
    assert restrict_to_window(shas, 0) == (shas, 0)


# --- resolve_deterministic_sha --------------------------------------------


def test_deterministic_forced_match_overrides_min():
    # A force-strategy match is honored even before min_commits.
    cfg = MergeGateConfig(min_commits=50)  # force defaults include "conflict"
    window = ["c", "b", "a"]  # newest-first -> oldest-first a, b, c
    prioritized = [_pick("conflict", sha="a")]  # 'a' at count 1
    assert resolve_deterministic_sha(window, prioritized, cfg) == "a"


def test_deterministic_forced_at_or_after_min_has_no_priority():
    # A forced match is special only *below* min_commits. At/after min it is
    # just an in-band boundary, so an earlier in-band match wins and the later
    # forced commit is excluded from this batch (picked up by a later one).
    cfg = MergeGateConfig(min_commits=2, force_strategies=["conflict"])
    window = ["c", "b", "a"]  # oldest-first a(1), b(2), c(3)
    prioritized = [_pick("huge_commit", sha="b"), _pick("conflict", sha="c")]
    assert resolve_deterministic_sha(window, prioritized, cfg) == "b"


def test_deterministic_skips_non_forced_match_before_min():
    # A non-forced match before min_commits is skipped; the first match at/after
    # min_commits is the cut point.
    cfg = MergeGateConfig(min_commits=2, force_strategies=[])
    window = ["c", "b", "a"]  # oldest-first a(1), b(2), c(3)
    prioritized = [_pick("huge_commit", sha="a"), _pick("huge_commit", sha="c")]
    assert resolve_deterministic_sha(window, prioritized, cfg) == "c"


def test_deterministic_window_tip_when_no_match_at_or_after_min():
    # Only a pre-min non-forced match exists -> cut at the window tip (newest).
    cfg = MergeGateConfig(min_commits=2, force_strategies=[])
    window = ["c", "b", "a"]  # oldest-first a(1), b(2), c(3)
    prioritized = [_pick("huge_commit", sha="a")]
    assert resolve_deterministic_sha(window, prioritized, cfg) == "c"


def test_deterministic_falls_back_to_window_tip():
    # No prioritized match -> cap at the window's newest commit (element 0).
    cfg = MergeGateConfig()
    window = ["c", "b", "a"]
    assert resolve_deterministic_sha(window, [], cfg) == "c"


def test_deterministic_empty_window():
    assert resolve_deterministic_sha([], [], MergeGateConfig()) is None


# --- config parsing -------------------------------------------------------


def test_merge_gate_config_defaults():
    cfg = MergeGateConfig()
    assert cfg.min_commits == 50
    assert cfg.max_age_days == 2
    assert cfg.max_commits == 150
    assert cfg.force_strategies == ["conflict", "important_files"]


def test_fork_config_parses_gate_and_ai_pick():
    data = {
        "fork": {
            "merge_gate": {
                "min_commits": 30,
                "max_age_days": 5,
                "max_commits": 100,
                "force_strategies": ["conflict"],
            },
            "ai_pick": {
                "enabled": True,
                "agent": "claude-cli:claude-opus-4-5",
                "rules_file": ".mergai/merge_pick_rules.md",
                "fallback": "error",
            },
        }
    }
    cfg = MergaiConfig.from_dict(data)
    assert cfg.fork.merge_gate == MergeGateConfig(
        min_commits=30,
        max_age_days=5,
        max_commits=100,
        force_strategies=["conflict"],
    )
    assert cfg.fork.ai_pick == AiPickConfig(
        enabled=True,
        agent="claude-cli:claude-opus-4-5",
        rules_file=".mergai/merge_pick_rules.md",
        fallback="error",
    )


def test_fork_config_uses_defaults_when_sections_absent():
    cfg = MergaiConfig.from_dict({"fork": {}})
    assert cfg.fork.merge_gate == MergeGateConfig()
    assert cfg.fork.ai_pick == AiPickConfig()


def test_merge_gate_force_strategies_null_falls_back_to_defaults():
    # `force_strategies: null` must not raise (list(None)); it uses defaults.
    cfg = MergeGateConfig.from_dict({"force_strategies": None})
    assert cfg.force_strategies == ["conflict", "important_files"]


def test_merge_gate_force_strategies_empty_list_disables():
    # An explicit empty list intentionally disables force strategies.
    cfg = MergeGateConfig.from_dict({"force_strategies": []})
    assert cfg.force_strategies == []


def test_merge_gate_force_strategies_single_string_normalized():
    # A bare string is a single strategy name, not split into characters.
    cfg = MergeGateConfig.from_dict({"force_strategies": "conflict"})
    assert cfg.force_strategies == ["conflict"]


def test_merge_gate_force_strategies_invalid_type_raises():
    import pytest

    with pytest.raises(ValueError, match="force_strategies"):
        MergeGateConfig.from_dict({"force_strategies": 123})
    with pytest.raises(ValueError, match="force_strategies"):
        MergeGateConfig.from_dict({"force_strategies": [1, 2]})


def test_merge_gate_int_fields_null_fall_back_to_defaults():
    # `min_commits: null` etc. must not pass None through (which would crash
    # gate evaluation later with a TypeError); they fall back to the defaults.
    cfg = MergeGateConfig.from_dict(
        {"min_commits": None, "max_age_days": None, "max_commits": None}
    )
    assert (cfg.min_commits, cfg.max_age_days, cfg.max_commits) == (50, 2, 150)


def test_merge_gate_int_fields_invalid_type_raises():
    import pytest

    for field in ("min_commits", "max_age_days", "max_commits"):
        with pytest.raises(ValueError, match=field):
            MergeGateConfig.from_dict({field: "50"})
        with pytest.raises(ValueError, match=field):
            MergeGateConfig.from_dict({field: True})
