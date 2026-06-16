"""Code-review handling for ``mergai review``.

Fetches a PR's review threads, filters them down to the ones an agent should
act on, builds an agent prompt context, runs the agent over the working tree,
and posts a reply to each addressed / unaddressed thread. Mirrors the
structure of :mod:`mergai.ci`.
"""
