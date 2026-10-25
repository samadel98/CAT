"""Tests for the ``--template``, ``--validate`` and ``--dry-run`` options of ``init_cat``."""

import shutil
from pathlib import Path

import yaml
import pytest
from scm.plams import Settings

from CAT.data_handling.entry_points import main, check_input, get_input_template

CORE = Path('tests') / 'test_files' / 'core' / 'Cd68Se55.xyz'


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    """Write the template input file (and its core) to a temporary directory."""
    (tmp_path / 'core').mkdir()
    shutil.copy(CORE, tmp_path / 'core')
    s = yaml.safe_load(get_input_template())
    s['path'] = str(tmp_path)
    ret = tmp_path / 'input.yaml'
    ret.write_text(yaml.safe_dump(s))
    return ret


def test_template(capsys: pytest.CaptureFixture[str]) -> None:
    """``--template`` prints a valid input file."""
    main(['--template'])
    out = capsys.readouterr().out
    assert out == get_input_template()
    assert check_input(Settings(yaml.safe_load(out))) == 'The input is valid.'


def test_validate(input_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--validate`` checks the input without creating any files."""
    main(['--validate', str(input_file)])
    assert capsys.readouterr().out.strip() == 'The input is valid.'
    assert sorted(f.name for f in input_file.parent.iterdir()) == ['core', 'input.yaml']


def test_dry_run(input_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--dry-run`` reads all molecules and lists the workflows, without creating any files."""
    main(['--dry-run', str(input_file)])
    out = capsys.readouterr().out
    assert 'Read 1 core(s) and 2 ligand(s): 2 quantum dot(s)' in out
    assert '  - Ligand geometry optimization (RDKit UFF)' in out
    assert '  - Quantum dot construction' in out
    assert 'Activation strain' not in out
    assert sorted(f.name for f in input_file.parent.iterdir()) == ['core', 'input.yaml']


def test_invalid(input_file: Path) -> None:
    """Invalid input and missing arguments."""
    s = yaml.safe_load(input_file.read_text())
    s['optional']['qd']['optimize'] = 'bob'
    input_file.write_text(yaml.safe_dump(s))
    with pytest.raises(Exception):
        main(['--validate', str(input_file)])
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(['--validate', '--dry-run', str(input_file)])
