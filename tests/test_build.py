"""Test building the package."""

import sys
import subprocess

import pytest
from nanoutils import delete_finally

try:
    import build  # noqa: F401
except ImportError:
    BUILD_EX: None | ImportError = ImportError("Test requires the `build` package")
else:
    BUILD_EX = None


@pytest.mark.skipif(BUILD_EX is not None, reason=str(BUILD_EX))
@delete_finally('dist', 'build')
def test_build() -> None:
    """Test if the package is properly build."""
    subprocess.run([sys.executable, '-m', 'build', '--sdist', '--wheel'], check=True)
