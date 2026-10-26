"""A script for running mypy on the ``dataCAT`` directory.

Examples
--------

.. code-block:: bash

    python tools/run_mypy.py

"""

import sys
import textwrap
from pathlib import Path

from mypy import api

ROOT = Path(__file__).parent.parent
INI_PATH = str(ROOT / "mypy.ini")
PKG_PATH = str(ROOT / "dataCAT")

TEMPLATE = """\
{}

exit code:
    {}

stderr:
{}

stdout:
{}

"""


def main() -> None:
    """Run mypy on the ``dataCAT`` directory."""
    _stdout, _stderr, exit_code = api.run(["--config-file", INI_PATH, PKG_PATH])
    stdout = "    None" if not _stdout else textwrap.indent(_stdout.strip("\n"), 4 * " ")
    stderr = "    None" if not _stderr else textwrap.indent(_stderr.strip("\n"), 4 * " ")

    if exit_code not in (0, 1):
        msg = TEMPLATE.format("Unexpected mypy exit code", exit_code, stdout, stderr)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(exit_code)
    elif stderr != "    None":
        msg = TEMPLATE.format("Unexpected mypy stderr output", exit_code, stdout, stderr)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)
    else:
        msg = TEMPLATE.format("Successful mypy run", exit_code, stdout, stderr)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
