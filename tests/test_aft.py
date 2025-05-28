"""Test file for aft.py."""

import numpy as np
import pytest

from gdtchron import aft

# Dummy constants used for some tests
TEST_CONSTS = {
    "c0": -19.844,
    "c1": 0.38951,
    "c2": -51.253,
    "c3": -7.6423,
    "alpha": -2.,
    "beta": -1.,
    "r_kappa_sum": 1.,
    "l_slope": 0.35,        # Value taken from HeFTy
    "l_intercept": 15.72
}

def test_g():
    """Unit tests for g (length transform)."""
    # confirm NaNs stay NaN
    assert np.isnan(aft.g(np.nan))

    # Confirm calculations performed properly for np array
    # Using Ketcham et al. (1999) fanning curvilinear model:
    assert aft.g(np.array([np.nan, 0.9, 0.1]))[1:] == \
            pytest.approx(np.array([-1.711884, 7.745533]))
    assert np.isnan(aft.g(np.array([np.nan, 0.9, 0.1]))[0])

    # Confirm calculations work for dummy constants
    assert aft.g(0.5, TEST_CONSTS) == 0.