"""Tests for agent subprocess environment sanitization."""

from mergai.agents.env import agent_subprocess_env


def test_strips_github_credentials(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "write-token")
    monkeypatch.setenv("GH_TOKEN", "write-token")
    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", "write-token")

    env = agent_subprocess_env()

    assert "GITHUB_TOKEN" not in env
    assert "GH_TOKEN" not in env
    assert "GH_ENTERPRISE_TOKEN" not in env


def test_preserves_non_github_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = agent_subprocess_env()

    assert env["ANTHROPIC_API_KEY"] == "sk-ant"
    assert env["PATH"] == "/usr/bin"
