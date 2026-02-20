"""Version handling for MergAI.

This module provides version information derived from git tags via setuptools-scm.
When installed from a tagged release (e.g., v0.1.0), the version will be "0.1.0".
When running from development (after a tag), version will be like "0.1.0.dev5+g1234abc".

The version is primarily obtained from package metadata (set at install time by
setuptools-scm). As a fallback for development without installation, it uses
git describe directly.
"""

import subprocess
from importlib.metadata import version, PackageNotFoundError


def get_version() -> str:
    """Get the mergai version.

    Returns the version string from package metadata (populated by setuptools-scm
    at install time). If the package is not installed, falls back to running
    `git describe` directly.

    Returns:
        Version string, e.g., "0.1.0", "0.1.0.dev5+g1234abc", or "unknown" if
        version cannot be determined.
    """
    try:
        return version("mergai")
    except PackageNotFoundError:
        # Development mode without install - try git describe
        return _get_version_from_git()


def _get_version_from_git() -> str:
    """Get version from git describe.

    Uses `git describe --tags --dirty --always` to determine the version
    from git history.

    Returns:
        Version string from git describe, or "unknown" if git is not available
        or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# Module-level version string for convenience
__version__ = get_version()
