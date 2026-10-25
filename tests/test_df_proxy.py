"""Tests for the :class:`dataCAT.DFProxy` class."""

import warnings
from itertools import chain

import numpy as np
import pandas as pd

from assertionlib import assertion
from dataCAT import DFProxy
from dataCAT.df_proxy import _DFMeta

_DF = pd.DataFrame(np.random.rand(3, 10))
DF = DFProxy(_DF)

# Accessors which (depending on the pandas version) return a new instance upon every access
IGNORE = frozenset({'sparse', 'style', 'plot'})


def test_dfproxy() -> None:
    """Test :class:`dataCAT.DFProxy`."""
    name_iterator = (i for i in dir(DFProxy.NDTYPE) if not i.startswith('_'))
    iterator = ((name, getattr(DF, name), getattr(_DF, name)) for
                name in chain(_DFMeta.MAGIC, name_iterator) if name not in IGNORE)

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        for name, func1, func2 in iterator:
            if hasattr(func1, '__self__'):
                assertion.is_(func1.__self__, func2.__self__, message=name)
            elif isinstance(getattr(pd.DataFrame, name), property):
                continue
            else:
                assertion.is_(func1, func2, message=name)
