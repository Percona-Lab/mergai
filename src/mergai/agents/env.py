"""Environment sanitization for spawned agent subprocesses.

The agent must never be able to modify remote state (push to origin or issue
GitHub write APIs). Because the agent runs as a subprocess of mergai, it would
otherwise inherit mergai's environment -- which in CI carries the GitHub *write*
token used by mergai's own deterministic push/PR steps.

`agent_subprocess_env` strips those write credentials from the environment
handed to the agent, so that even a fully compromised or prompt-injected agent
cannot authenticate a write to GitHub. Local operations (file edits, local git
reads and commits) need no credential and are unaffected. If a read-only token
is provided via ``MERGAI_AGENT_GH_TOKEN``, it is substituted under the standard
variable names so the agent can still *read* GitHub.

The parent mergai process keeps its own (write-capable) environment, so its
push/PR steps continue to work -- only the spawned agent is de-privileged.

NOTE: This closes the *environment* path only. `actions/checkout` also persists
a credential into the repo's ``.git/config`` (``http.extraheader``); a bare
`git push` authenticates via that on-disk token regardless of the environment,
and the agent can even read it. That path must be closed separately with
``persist-credentials: false`` in the workflow (plus a credential helper so
mergai's own push still authenticates from the parent environment).
"""

import os

# Environment variables that authenticate GitHub writes: the gh CLI reads these,
# and a credential-helper-based `git push` derives its token from them.
_WRITE_CREDENTIAL_VARS = ("GITHUB_TOKEN", "GH_TOKEN", "GH_ENTERPRISE_TOKEN")

# Optional read-only token the workflow may pass so the agent can still read
# GitHub (PRs, issues) without being able to write.
_READONLY_TOKEN_VAR = "MERGAI_AGENT_GH_TOKEN"


def agent_subprocess_env() -> dict[str, str]:
    """Return a copy of the environment with GitHub write credentials removed.

    Any ``MERGAI_AGENT_GH_TOKEN`` is substituted under ``GH_TOKEN`` /
    ``GITHUB_TOKEN`` so the agent retains read-only GitHub access.

    Returns:
        A sanitized environment dict suitable for the agent subprocess.
    """
    env = os.environ.copy()
    readonly_token = env.get(_READONLY_TOKEN_VAR)

    for var in _WRITE_CREDENTIAL_VARS:
        env.pop(var, None)

    if readonly_token:
        env["GH_TOKEN"] = readonly_token
        env["GITHUB_TOKEN"] = readonly_token

    return env
