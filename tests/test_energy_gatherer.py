"""Tests for :class:`nanoCAT.asa.energy_gatherer.EnergyGatherer`."""

import operator
from collections.abc import Callable

import pytest
import numpy as np
import pandas as pd

from nanoCAT.asa.energy_gatherer import EnergyGatherer

OPERATORS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "floordiv": operator.floordiv,
    "truediv": operator.truediv,
    "pow": operator.pow,
    "mod": operator.mod,
}


@pytest.mark.parametrize("func", OPERATORS.values(), ids=OPERATORS)
def test_arithmetic(func: Callable[[object, float], object]) -> None:
    df = pd.DataFrame({"a": [1.5, 7.0, 10.25]})
    gatherer = EnergyGatherer(inter_lj=df.copy())

    ret = func(gatherer, 2.0)
    assert isinstance(ret, EnergyGatherer)
    assert ret.inter_elstat is None
    np.testing.assert_allclose(ret.inter_lj, func(df, 2.0))
    np.testing.assert_allclose(gatherer.inter_lj, df)  # The original is unchanged
