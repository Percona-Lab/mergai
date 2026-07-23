"""Tests for ``git_utils.git_output_to_text``.

GitPython command methods return ``str`` by default, but their type hints also
permit ``bytes`` and a ``(status, stdout, stderr)`` tuple. The helper normalizes
all of these to text without the lossy ``str(bytes)`` repr.
"""

from mergai.utils.git_utils import git_output_to_text


def test_str_passthrough():
    assert git_output_to_text("refs/heads/foo") == "refs/heads/foo"


def test_empty_str_stays_empty():
    assert git_output_to_text("") == ""


def test_bytes_are_decoded():
    assert git_output_to_text(b"refs/heads/foo") == "refs/heads/foo"


def test_empty_bytes_stay_empty_not_repr():
    # The bug the normalization guards against: str(b"") == "b''" (truthy).
    assert git_output_to_text(b"") == ""


def test_extended_output_tuple_uses_stdout():
    assert git_output_to_text((0, "refs/heads/foo", "")) == "refs/heads/foo"


def test_extended_output_tuple_with_bytes_stdout():
    assert git_output_to_text((0, b"refs/heads/foo", b"")) == "refs/heads/foo"


def test_unknown_type_is_empty():
    assert git_output_to_text(None) == ""
