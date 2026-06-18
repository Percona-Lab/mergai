"""Tests for ``get_note_from_commit`` SHA resolution.

The key regression: a note must be readable for a SHA whose commit object is
not present locally (e.g. a PR head branch that was never fetched, while its
note was). ``git notes show`` derives the note path from the SHA hex and does
not need the commit object, so we must not hard-require ``repo.commit()`` to
resolve.
"""

from types import SimpleNamespace

from mergai.utils import git_utils

FULL_SHA = "2ed9e1aa664b8a9a4d95079fd0758a2dcd7aacf7"


class _Git:
    def __init__(self, note):
        self._note = note
        self.shown = None

    def notes(self, *args):
        # args: ("--ref", ref, "show", target)
        self.shown = args[-1]
        return self._note


def test_resolves_via_commit_when_object_present():
    git = _Git('{"k": 1}')
    repo = SimpleNamespace(
        git=git,
        commit=lambda c: SimpleNamespace(hexsha=FULL_SHA),
    )
    assert git_utils.get_note_from_commit_as_dict(repo, "mergai", "HEAD") == {"k": 1}
    # the resolved full SHA is what gets passed to `git notes show`
    assert git.shown == FULL_SHA


def test_falls_back_to_raw_sha_when_commit_object_missing():
    git = _Git('{"k": 2}')

    def _missing(_c):
        raise ValueError("bad object - commit not present locally")

    repo = SimpleNamespace(git=git, commit=_missing)
    # commit object unresolvable, but the note still reads via the raw SHA
    assert git_utils.get_note_from_commit_as_dict(repo, "mergai", FULL_SHA) == {"k": 2}
    assert git.shown == FULL_SHA


def test_returns_none_when_no_note():
    git = _Git("")  # `git notes show` empties / raises -> treated as no note
    repo = SimpleNamespace(git=git, commit=lambda c: SimpleNamespace(hexsha=FULL_SHA))
    assert git_utils.get_note_from_commit_as_dict(repo, "mergai", FULL_SHA) is None
