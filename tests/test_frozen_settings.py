"""Tests for :mod:`CAT.frozen_settings`."""

from assertionlib import assertion

from scm.plams import Settings

from CAT.frozen_settings import FrozenSettings, thaw_settings

SETTINGS = FrozenSettings({'a': True, 'b': False, 'c': [1, 2, 3, 4]})


def test_missing() -> None:
    """Tests for :meth:`CAT.frozen_settings.FrozenSettings.__missing__`."""
    item = SETTINGS.d
    assertion.eq(item, FrozenSettings())


def test_delitem() -> None:
    """Tests for :meth:`CAT.frozen_settings.FrozenSettings.__delitem__`."""
    args = ('a')
    assertion.assert_(SETTINGS.__delitem__, args, exception=TypeError)


def test_setitem() -> None:
    """Tests for :meth:`CAT.frozen_settings.FrozenSettings.__setitem__`."""
    args = ('d', True)
    assertion.assert_(SETTINGS.__setitem__, args, exception=TypeError)


def test_copy() -> None:
    """Tests for :meth:`CAT.frozen_settings.FrozenSettings.copy`."""
    settings = SETTINGS.copy()
    assertion.eq(settings, SETTINGS)
    assertion.is_not(settings, SETTINGS)


def test_thaw_settings() -> None:
    """Test :func:`CAT.frozen_settings.thaw_settings`."""
    frozen = FrozenSettings({"a": {"b": {"c": 1}}, "d": [{"e": 2}, 3]})
    ret = thaw_settings(frozen)
    assert ret == frozen
    assert type(ret) is Settings
    assert type(ret.a.b) is Settings
    assert type(ret.d[0]) is Settings

    ret.a.b.f = 4
    ret.d[0].g = 5
    assert "f" not in frozen.a.b
    assert "g" not in frozen.d[0]
    assert thaw_settings(None) == Settings()
