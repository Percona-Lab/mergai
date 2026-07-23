"""Tests for the workflow-run footer appended to bot-authored PR comments."""

from mergai.config import MergaiConfig
from mergai.utils import run_link

GITHUB_ENV = {
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "percona/percona-server-mongodb",
    "GITHUB_RUN_ID": "12345",
}


def _set_env(monkeypatch, **overrides):
    """Populate the GitHub Actions run environment, then apply overrides."""
    env = {**GITHUB_ENV, **overrides}
    for key in (
        "GITHUB_SERVER_URL",
        "GITHUB_REPOSITORY",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if value is not None:
            monkeypatch.setenv(key, value)


def test_run_url_from_actions_env(monkeypatch):
    _set_env(monkeypatch)
    assert run_link.run_url() == (
        "https://github.com/percona/percona-server-mongodb/actions/runs/12345"
    )


def test_run_url_deep_links_to_rerun_attempt(monkeypatch):
    _set_env(monkeypatch, GITHUB_RUN_ATTEMPT="3")
    assert run_link.run_url().endswith("/actions/runs/12345/attempts/3")


def test_run_url_ignores_first_attempt(monkeypatch):
    _set_env(monkeypatch, GITHUB_RUN_ATTEMPT="1")
    assert run_link.run_url().endswith("/actions/runs/12345")


def test_run_url_none_outside_actions(monkeypatch):
    _set_env(monkeypatch, GITHUB_RUN_ID=None)
    assert run_link.run_url() is None


def test_append_run_footer_adds_link_when_enabled(monkeypatch):
    _set_env(monkeypatch)
    out = run_link.append_run_footer("Fixed the build.", enabled=True)
    assert out.startswith("Fixed the build.")
    assert "/actions/runs/12345" in out
    assert "mergai workflow run" in out


def test_append_run_footer_noop_when_disabled(monkeypatch):
    # Even in Actions, the footer is strictly opt-in.
    _set_env(monkeypatch)
    assert (
        run_link.append_run_footer("Fixed the build.", enabled=False)
        == "Fixed the build."
    )


def test_append_run_footer_noop_outside_actions(monkeypatch):
    _set_env(monkeypatch, GITHUB_RUN_ID=None)
    assert (
        run_link.append_run_footer("Fixed the build.", enabled=True)
        == "Fixed the build."
    )


def test_run_link_disabled_by_default():
    assert MergaiConfig().run_link.enabled is False
    assert MergaiConfig.from_dict({}).run_link.enabled is False


def test_run_link_opt_in_via_config():
    cfg = MergaiConfig.from_dict({"run_link": {"enabled": True}})
    assert cfg.run_link.enabled is True
