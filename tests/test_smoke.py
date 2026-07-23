"""Phase 0 smoke tests: the package imports and core constants are sane."""

from sfsim.constants import FEE_CAP_PERCENT, NUM_AGENTS, REAL_YES_PCT


def test_package_imports():
    import sfsim  # noqa: F401


def test_core_constants():
    assert NUM_AGENTS == 30
    assert FEE_CAP_PERCENT == 15
    assert REAL_YES_PCT == 60.8
