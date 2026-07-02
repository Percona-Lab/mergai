"""Environment sanitization for spawned agent subprocesses.

The agent must never be able to modify remote state (push to origin or issue
GitHub write APIs). Because the agent runs as a subprocess of mergai, it would
otherwise inherit mergai's environment -- which in CI carries the GitHub write
token used by mergai's own deterministic push/PR steps.

`agent_subprocess_env` strips those credentials from the environment handed to
the agent, so that even a fully compromised or prompt-injected agent cannot
authenticate to GitHub at all. The agent only needs local access (edit files,
local git reads and commits), none of which requires a credential.

The parent mergai process keeps its own (credentialed) environment, so its
push/PR steps continue to work -- only the spawned agent is de-privileged.

NOTE: This closes the *environment* path only. `actions/checkout` also persists
a credential into the repo's ``.git/config`` (``http.extraheader``); a bare
`git push` authenticates via that on-disk token regardless of the environment,
and the agent can even read it. That path must be closed separately with
``persist-credentials: false`` in the workflow (plus a credential helper so
mergai's own push still authenticates from the parent environment).
"""

import os

# Environment variables that authenticate to GitHub: the gh CLI reads these,
# and a credential-helper-based `git push` derives its token from them.
_GITHUB_CREDENTIAL_VARS = ("GITHUB_TOKEN", "GH_TOKEN", "GH_ENTERPRISE_TOKEN")


def agent_subprocess_env() -> dict[str, str]:
    """Return a copy of the environment with GitHub credentials removed.

    The agent is left with no GitHub token, so it cannot push or call GitHub
    APIs; local file and git operations need no credential and are unaffected.

    Returns:
        A sanitized environment dict suitable for the agent subprocess.
    """
    env = os.environ.copy()
    for var in _GITHUB_CREDENTIAL_VARS:
        env.pop(var, None)
    return env
