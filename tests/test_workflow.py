"""Tests for :mod:`CAT.workflows.workflow`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CAT.workflows import WorkFlow


@pytest.mark.parametrize("dtype", [object, "str"])
def test_isnull_str(dtype: object) -> None:
    """Empty strings count as missing values, whether stored as ``object`` or ``str``."""
    df = pd.DataFrame({"a": pd.Series(["", "0-1", None], dtype=dtype)})
    out = WorkFlow._isnull(df, ["a"])
    assert out["a"].tolist() == [True, False, True]


def test_isnull() -> None:
    """Test :meth:`WorkFlow._isnull` for the other dtypes."""
    df = pd.DataFrame({
        "int": np.array([-1, 0, 5]),
        "bool": np.array([False, True, True]),
        "float": np.array([np.nan, 0.0, 1.0]),
    })
    out = WorkFlow._isnull(df, ["int", "bool", "float"])
    assert out.values.tolist() == [[True, True, True], [False, False, False], [False, False, False]]
