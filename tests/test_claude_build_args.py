"""Tests for ClaudeCLIAgent.build_args guardrails.

yolo maps to a constrained acceptEdits mode (never --dangerously-skip-
permissions), and the remote-write / bypass deny-list is applied on every
invocation regardless of permission mode.
"""

from pathlib import Path

from mergai.agents.claude_cli import (
    GUARDRAIL_ALLOWED_TOOLS,
    GUARDRAIL_DISALLOWED_TOOLS,
    ClaudeCLIAgent,
)


def _flag_value(args, flag):
    return args[args.index(flag) + 1]


def test_never_skips_permissions():
    for yolo in (True, False):
        args = ClaudeCLIAgent(model="opus", yolo=yolo).build_args("go")
        assert "--dangerously-skip-permissions" not in args


def test_allows_bash_broadly():
    # A broad Bash allow is required so the agent can run read commands in
    # --print mode (CI's CLI denies un-allowed Bash there).
    args = ClaudeCLIAgent(model="opus", yolo=True).build_args("go")
    allow = _flag_value(args, "--allowedTools").split(",")
    assert "Bash" in allow and "Read" in allow


def test_allow_present_on_every_invocation_and_disjoint_from_deny():
    for yolo, paths in ((True, None), (False, [Path("r.json")]), (False, None)):
        args = ClaudeCLIAgent(model="opus", yolo=yolo).build_args(
            "go", allowed_write_paths=paths
        )
        assert "--allowedTools" in args
        assert "--disallowedTools" in args
    # a tool must not be both allowed and denied
    assert set(GUARDRAIL_ALLOWED_TOOLS).isdisjoint(GUARDRAIL_DISALLOWED_TOOLS)


def test_disallowed_tools_always_present():
    # Applied on every invocation: yolo, non-yolo write, and read-only.
    for yolo, paths in ((True, None), (False, [Path("r.json")]), (False, None)):
        args = ClaudeCLIAgent(model="opus", yolo=yolo).build_args(
            "go", allowed_write_paths=paths
        )
        assert "--disallowedTools" in args
        # Exact membership on the comma-split value: a substring check would let
        # "Task" pass on "TaskCreate" and mask a missing entry.
        deny = set(_flag_value(args, "--disallowedTools").split(","))
        for expected in (
            "Bash(git push:*)",
            "Bash(gh:*)",
            "WebFetch",
            "WebSearch",
            "Workflow",
            "Task",
        ):
            assert expected in deny


def test_yolo_uses_accept_edits():
    args = ClaudeCLIAgent(model="opus", yolo=True).build_args("go")
    assert _flag_value(args, "--permission-mode") == "acceptEdits"


def test_accept_edits_when_write_paths_given_non_yolo():
    args = ClaudeCLIAgent(model="opus", yolo=False).build_args(
        "go", allowed_write_paths=[Path("r.json")]
    )
    assert _flag_value(args, "--permission-mode") == "acceptEdits"


def test_read_only_run_has_no_permission_mode_but_still_denies():
    # Neither yolo nor write paths -> default permission mode, but the deny-list
    # is still applied.
    args = ClaudeCLIAgent(model="opus", yolo=False).build_args("go")
    assert "--permission-mode" not in args
    assert "--disallowedTools" in args


def test_prompt_is_last_arg_not_swallowed():
    args = ClaudeCLIAgent(model="opus", yolo=True).build_args("PROMPT-SENTINEL")
    assert args[-1] == "PROMPT-SENTINEL"


def test_disallowed_list_is_single_comma_joined_value():
    # A single value (not variadic), so it cannot consume the trailing prompt.
    args = ClaudeCLIAgent(model="opus", yolo=True).build_args("go")
    deny = _flag_value(args, "--disallowedTools")
    assert deny == ",".join(GUARDRAIL_DISALLOWED_TOOLS)
