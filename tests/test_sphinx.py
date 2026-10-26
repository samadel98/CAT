"""Test the :mod:`sphinx` documentation generation."""

import warnings
from os.path import join

import pytest
from nanoutils import delete_finally

try:
    from sphinx.application import Sphinx
    from sphinx.errors import SphinxWarning
except ImportError:
    HAS_SPHINX = False
else:
    HAS_SPHINX = True

# The documentation includes the API of Nano-CAT and Data-CAT
try:
    import nanoCAT  # noqa: F401
    import dataCAT  # noqa: F401
except Exception:
    HAS_NANO_DATA_CAT = False
else:
    HAS_NANO_DATA_CAT = True

SRCDIR = CONFDIR = 'docs'
OUTDIR = join('tests', 'test_files', 'build')
DOCTREEDIR = join('tests', 'test_files', 'build', 'doctrees')


@delete_finally(OUTDIR)
@pytest.mark.skipif(not HAS_SPHINX, reason="Requires Sphinx")
@pytest.mark.skipif(not HAS_NANO_DATA_CAT, reason="Requires Nano-CAT and Data-CAT")
def test_sphinx_build() -> None:
    """Test :meth:`sphinx.application.Sphinx.build`."""
    try:
        app = Sphinx(SRCDIR, CONFDIR, OUTDIR, DOCTREEDIR, buildername='html', warningiserror=True)
        app.build(force_all=True)
    except SphinxWarning as ex:
        str_ex = str(ex)
        if "requests.exceptions.HTTPError" in str_ex:
            warnings.warn(str_ex, stacklevel=2)
            pytest.xfail(str_ex)
        else:
            raise
