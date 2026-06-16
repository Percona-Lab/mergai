"""Canonical solution ``type`` values recorded on mergai notes.

A solution dict on a :class:`~mergai.models.MergaiNote` carries a ``type``
field identifying how it was produced. Centralizing the values here documents
the union in one place and keeps the literals from drifting across the commit,
note, CI-fix, and review-fix paths.
"""

# A conflict resolution produced during a merge (by the AI agent or a human).
CONFLICT_RESOLUTION = "conflict_resolution"

# A CI-failure fix produced by `mergai ci fix`.
CI_FIX = "ci_fix"

# A code-review fix produced by `mergai review fix`.
REVIEW_FIX = "review_fix"
