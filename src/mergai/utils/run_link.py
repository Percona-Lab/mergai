"""Link a bot-authored PR comment back to the workflow run that posted it.

When mergai runs in CI it posts comments (CI-fix explanations, review replies,
acknowledgements) as the bot. Appending a link to the workflow run that
authored each comment lets a maintainer click straight through to the job that
made the change and inspect its logs.

The run URL is reconstructed from the standard GitHub Actions environment
(``GITHUB_SERVER_URL`` / ``GITHUB_REPOSITORY`` / ``GITHUB_RUN_ID``, plus
``GITHUB_RUN_ATTEMPT`` for re-runs). Outside Actions those are unset, so
``run_url`` returns ``None`` and the footer helpers are no-ops - local runs
post a bare body.
"""

from __future__ import annotations

import os


def run_url() -> str | None:
    """Return the current GitHub Actions run URL, or ``None`` outside Actions.

    Deep-links to the specific attempt when this is a re-run
    (``GITHUB_RUN_ATTEMPT`` > 1) so the link points at the logs that actually
    posted the comment.
    """
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not (server and repo and run_id):
        return None
    url = f"{server}/{repo}/actions/runs/{run_id}"
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if attempt and attempt != "1":
        url = f"{url}/attempts/{attempt}"
    return url


def run_footer() -> str:
    """Markdown footer linking to the current workflow run (empty if none)."""
    url = run_url()
    if not url:
        return ""
    return f'<sub>Posted by <a href="{url}">mergai workflow run</a>.</sub>'


def append_run_footer(body: str, enabled: bool) -> str:
    """Append the workflow-run footer to ``body`` when enabled and in Actions.

    ``enabled`` is the ``run_link.enabled`` config flag (off by default): when
    false the body is returned unchanged, so the footer is strictly opt-in.
    Even when enabled, the footer is empty outside Actions (no run to link),
    so callers can wrap every comment body unconditionally.
    """
    if not enabled:
        return body
    footer = run_footer()
    if not footer:
        return body
    return f"{body.rstrip()}\n\n{footer}"
