"""A pytest ``conftest.py`` file."""

import os
import re
import sys
import logging
import warnings

import pytest
import rdkit
from packaging.version import Version

from CAT import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
from rdkit_references import RDKIT_REF_VERSION  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def rdkit_version() -> None:
    """Validate the rdkit version."""
    version = Version(rdkit.__version__)
    if version < Version("2021.03.4"):
        warnings.warn(
            "The CAT test suite requires RDKit >= 2021.03.4 to properly run; "
            f"observed version: {rdkit.__version__}", stacklevel=2
        )
    elif version >= Version("2024.03") and (
        rdkit.__version__.split(".")[:2] != RDKIT_REF_VERSION.split(".")
    ):
        warnings.warn(
            "The reference geometries of the CAT test suite were made with RDKit 2021.03 - "
            f"2023.09 and {RDKIT_REF_VERSION}, so some tests can fail with other versions; "
            f"observed version: {rdkit.__version__}", stacklevel=2
        )


class TestFilter(logging.Filter):
    """Filter out the warnings about missing optional packages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether **record** should be logged."""
        return not re.fullmatch(r"The optional [\-a-zA-Z]+ package was not found", record.msg)


@pytest.fixture(scope="session", autouse=True)
def modify_logger() -> None:
    """Suppres the loggers ``stderr`` output and filter out all optional-package warnings."""
    for handler in logger.handlers.copy():
        logger.removeHandler(handler)

    filter = TestFilter(logger.name)
    logger.addFilter(filter)
