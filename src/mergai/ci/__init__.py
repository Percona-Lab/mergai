"""CI workflow handling for mergai.

This package implements the automated fix loop for CI workflow failures
on mergai PRs (PSMDB-1972). Top-level pieces:

- :mod:`mergai.ci.context_builders` — turn workflow failure artifacts
  (git diffs, SARIF files, logs) into a structured
  :class:`~mergai.ci.context_builders.base.WorkflowContext`.
- :mod:`mergai.ci.handlers` — execute a fix given a
  ``WorkflowContext``: either a shell command (``command``) or an AI
  agent run (``resolve``).

The ``mergai ci fix`` click command in :mod:`mergai.commands.ci`
wires these together and is the public entry point.
"""
