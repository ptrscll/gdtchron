"""Test file for aft.py."""

import numpy as np
import pytest

from gdtchron import aft

# Dummy constants used for some tests
TEST_CONSTS = {
    "c0": -1.,
    "c1": 1. / 3.,
    "c2": -1.,
    "c3": -2.,
    "alpha": -2.,
    "beta": -1.,
    "r_kappa_sum": 1.,
    "l_slope": 0.35,        # Value taken from HeFTy
    "l_intercept": 15.72
}

# This Dpar yields r_mr0 = 0.5
TEST_DPAR = (1.834 - np.log(2)) / 0.647 + 1.75


def test_g():
    """Unit tests for g (length transform)."""
    # Confirm NaNs stay NaN
    assert np.isnan(aft.g(np.nan))

    # Confirm calculations performed properly for np array
    # Using Ketcham et al. (1999) fanning curvilinear model:
    assert aft.g(np.array([np.nan, 0.9, 0.1]))[1:] == \
            pytest.approx(np.array([-1.711884, 7.745533]))
    assert np.isnan(aft.g(np.array([np.nan, 0.9, 0.1]))[0])

    # Confirm calculations work for dummy constants
    assert aft.g(0.5, TEST_CONSTS) == 0.


def test_f():
    """Unit tests for f (right-hand side of main annealing equation)."""
    # Confirm NaNs stay NaN
    assert np.isnan(aft.f(T=450., t=np.nan))

    # Confirm calculations performed properly for np array
    # Using Ketcham et al. (1999) fanning curvilinear model:
    assert aft.f(T=450, t=np.array([np.nan, 1e10, 1e11]))[1:] == \
            pytest.approx(np.array([-0.971615, -0.386586]))
    assert np.isnan(aft.f(T=450, t=np.array([np.nan, 0.9, 0.1]))[0])

    # Confirm calculations work for dummy constants
    assert aft.f(T=np.e, t=np.e ** 2, constants=TEST_CONSTS) == 0.


def test_get_equiv_time():
    """Unit tests for get_equiv_time."""
    # Confirm calculations performed properly for np array with NaNs
    # Using Ketcham et al. (1999) fanning curvilinear model:
    equiv_times = aft.get_equiv_time(np.array([np.nan, 0.9, 0.1]), T=450.)[1:]
    assert equiv_times == pytest.approx(np.array([5.428073e8, 7.950111e24]))
    assert np.isnan(aft.get_equiv_time(np.array([np.nan, 0.9, 0.1]), T=450.)[0])
    
    # Confirming that Equation 5 of Ketcham (2005) still holds
    assert aft.g(np.array([0.9, 0.1])) == \
        pytest.approx(aft.f(T=450., t=equiv_times))

    # Confirm calculations work for dummy constants
    assert aft.get_equiv_time(np.array([0.5]), T=np.e, constants=TEST_CONSTS) \
        == pytest.approx(np.array([np.e ** 2]))


def test_get_next_r():
    """Unit tests for get_next_r."""
    # Confirm calculations performed properly for np array with NaNs
    # Using Ketcham et al. (1999) fanning curvilinear model:
    rs = aft.get_next_r(450., np.array([np.nan, 1e10, 1e11]))[1:]
    assert rs == pytest.approx(np.array([0.863753, 0.830875]))
    assert np.isnan(aft.get_next_r(450., np.array([np.nan, 1e10, 1e11]))[0])
    
    # Confirming that Equation 5 of Ketcham (2005) still holds
    assert aft.g(rs) == pytest.approx(aft.f(T=450., t=np.array([1e10, 1e11])))

    # Confirm calculations work for dummy constants
    assert aft.get_next_r(T=np.e, 
                          cumulative_t=np.e ** 2, 
                          constants=TEST_CONSTS) == 0.5
    
    # Confirm code works for fully annealed tracks
    assert aft.get_next_r(T=550., cumulative_t=np.array([1.87165e19])) == \
        np.array([0.])
    assert aft.get_next_r(T=550., cumulative_t=np.array([1e21])) == \
        np.array([0.])
    

def test_calc_annealing():
    """Unit tests for calc_annealing."""
    # Using Ketcham et al. (1999) fanning curvilinear model

    # Confirm calculations performed properly for initial timestep
    assert aft.calc_annealing(r_initial=np.array([np.nan, np.nan]),
                              T=450.,
                              start=1e10 / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=0)[0] == pytest.approx(0.863753)
    assert np.isnan(aft.calc_annealing(r_initial=np.array([np.nan, np.nan]),
                                       T=450.,
                                       start=1e10 / aft.SECONDS_PER_YEAR,
                                       end=0.,
                                       next_nan_index=0)[1])
    
    # Confirm calculations performed properly for intermediate timestep
    r1 = aft.calc_annealing(r_initial=np.array([np.nan, np.nan, np.nan]),
                            T=450.,
                            start=1e11 / aft.SECONDS_PER_YEAR,
                            end=1e10 / aft.SECONDS_PER_YEAR,
                            next_nan_index=0)
    r2 = aft.calc_annealing(r_initial=r1,
                            T=450.,
                            start=1e10 / aft.SECONDS_PER_YEAR,
                            end=0. / aft.SECONDS_PER_YEAR,
                            next_nan_index=1)
    assert r2[:2] == pytest.approx(np.array([0.830875, 0.863753]))
    assert np.isnan(r2[2])
    
    # Confirm calculations performed properly for final timestep
    # Also confirm that full annealing of some (but not all tracks) works
    r1 = aft.calc_annealing(r_initial=np.array([np.nan, np.nan]),
                            T=650.,
                            start=1.8e19 / aft.SECONDS_PER_YEAR,
                            end=1e19 / aft.SECONDS_PER_YEAR,
                            next_nan_index=0)
    assert aft.calc_annealing(r_initial=r1,
                              T=550.,
                              start=1e19 / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=1) \
                                == pytest.approx(np.array([0., 0.0625308]))
    
    # Confirm calculations work when all tracks fully anneal at end
    assert aft.calc_annealing(r_initial=r1,
                              T=950.,
                              start=1e19 / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=1) \
                                == pytest.approx(np.array([0., 0.]))

    # Confirm calculations work for dummy constants
    assert aft.calc_annealing(r_initial=np.array([np.nan]),
                              T=np.e,
                              start=(np.e ** 2) / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=0,
                              constants=TEST_CONSTS)[0] == pytest.approx(0.5)
    

def test_dpar_conversion():
    """Unit tests for dpar_conversion."""
    # Test that dpar conversion works for the following cases:
    #   Normal conversion (r_mr = 0.625)
    #   When r_mr = r_mr0 (r_mr = 0.5)
    #   When r_mr < r_mr0 (r_mr = 0.2)
    #   When r_mr = 0
    assert aft.dpar_conversion(r_mr=np.array([0.625, 0.5, 0.2, 0.]),
                               Dpar=TEST_DPAR) == \
        pytest.approx(np.array([0.5, 0., 0., 0.]), abs=1e-6)
    

def test_r_to_rho():
    """Unit tests for r_to_rho."""
    # Test that r_to_rho works for the following cases:
    #   r = 0
    #   0 < r < 0.13
    #   r < 0.765
    #   r = 0.765
    #   r > 0.765
    #   r = 1
    assert aft.r_to_rho(np.array([0., 0., 0.1, 0.73, 0.8, 1.])) == \
        pytest.approx(np.array([0., 0., 0.054176125, 0.624, 0.84, 1.]))
    