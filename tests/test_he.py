"""Test file for aft.py."""

import numpy as np
import pytest

from gdtchron import he


def test_tridiag_banded():
    """Unit tests for tridiag_banded."""
    # Test for diag_length 1
    assert (he.tridiag_banded(a=1, b=2, c=3, diag_length=1) == 
            np.array([[0.], 
                      [2.], 
                      [0.]], dtype=np.float32)).all()
    
    # Test for diag_length 2
    assert (he.tridiag_banded(a=1, b=2, c=3, diag_length=2) == 
           np.array([[0., 1.], 
                     [2., 2.], 
                     [3., 0.]], dtype=np.float32)).all()
    
    # Test for diag_length 3
    assert (he.tridiag_banded(a=1, b=0, c=3, diag_length=3) == 
            np.array([[0., 1., 1.], 
                      [0., 0., 0.], 
                      [3., 3., 0.]], dtype=np.float32)).all()
    
    # Test for diag_length 4
    assert (he.tridiag_banded(a=0, b=2, c=3, diag_length=4) == 
            np.array([[0., 0., 0., 0.], 
                      [2., 2., 2., 2.], 
                      [3., 3., 3., 0.]], dtype=np.float32)).all()
    
    # Confirm datatypes work correctly
    assert he.tridiag_banded(a=1, b=2, c=3, diag_length=3).dtype == 'float32'
    assert he.tridiag_banded(1, 2, 3, 3, dtype=np.int32).dtype == 'int32'


def test_calc_diffusivity():
    """Unit tests for calc_diffusivity."""
    assert he.calc_diffusivity(165.985, 'AHe') == \
        pytest.approx(152.529 * np.e ** -100)
    
    assert he.calc_diffusivity(203.272, 'ZHe') == \
        pytest.approx(1.45847 * np.e ** -100)
    

def test_calc_beta():
    """Unit test for calc_beta."""
    assert he.calc_beta(node_spacing=0.5,
                        diffusivity=0.025,
                        time_interval=200.) == pytest.approx(0.1)


def test_u_th_ppm_to_molg():
    """Unit test for u_th_ppm_to_molg."""
    u238_molg, u235_molg, th_molg = he.u_th_ppm_to_molg(u_ppm=138.88, 
                                                        th_ppm=232.)
    
    assert u238_molg == 137.88e-6 / 238.
    assert u235_molg == 1e-6 / 235.
    assert th_molg == 1e-6


def test_calc_he_production_rate():
    """Unit tests for calc_he_production_rate."""
    # Make sure rates are correct when only one kind of isotope present
    assert he.calc_he_production_rate(u238_molg=0.001, 
                                      u235_molg=0., 
                                      th_molg=0.) == pytest.approx(1.241e-12)
    assert he.calc_he_production_rate(u238_molg=0., 
                                      u235_molg=0.0005, 
                                      th_molg=0.) == pytest.approx(3.446e-12)
    assert he.calc_he_production_rate(u238_molg=0., 
                                      u235_molg=0., 
                                      th_molg=0.005) == pytest.approx(1.480e-12)
    
    # Test rates when all isotopes present
    assert he.calc_he_production_rate(u238_molg=0.001, 
                                      u235_molg=0.0005, 
                                      th_molg=0.005) == pytest.approx(6.167e-12)
    

def test_calc_node_positions():
    """Unit test for calc_node_positions."""
    assert he.calc_node_positions(node_spacing=0.1, radius=0.75) == \
        pytest.approx(np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]))
    

def test_sum_he_shells():
    """Unit test for sum_he_shells."""
    # Get components of x
    node_positions = he.calc_node_positions(node_spacing=0.1, radius=0.35)
    v = np.array([0.343, 0.026385, 0.007])

    # Call function
    he_molg, v_from_fn = he.sum_he_shells(x=node_positions * v, 
                                          node_positions=node_positions,
                                          radius=0.35)
    
    # Check output
    assert v == pytest.approx(v_from_fn)
    assert he_molg == pytest.approx(0.003666706)
    

def test_calculate_he_age():
    """Unit tests for calculate_he_age."""
    # Testing individual isotopes
    assert he.calc_age(he_molg=0.001,
                       u238_molg=0.005,
                       u235_molg=0.00,
                       th_molg=0.00) == pytest.approx(159.168, abs=1.)
    assert he.calc_age(he_molg=0.001,
                       u238_molg=0.00,
                       u235_molg=0.001,
                       th_molg=0.00) == pytest.approx(135.622, abs=1.)
    assert he.calc_age(he_molg=0.001,
                       u238_molg=0.00,
                       u235_molg=0.00,
                       th_molg=0.01) == pytest.approx(333.854, abs=1.)
    
    # Testing all three isotopes at once
    assert he.calc_age(he_molg=0.001,
                       u238_molg=0.005,
                       u235_molg=0.001,
                       th_molg=0.01) == pytest.approx(61.2952, abs=1.)


def test_alpha_correction():
    """Unit test for alpha_correction."""
    assert he.alpha_correction(stopping_distance=0.4, 
                               radius=3.) == pytest.approx(0.90)


def test_model_alpha_ejection():
    """Unit tests for modle_alpha_ejection."""
    # Testing when the intersection plane is located atone of the node positions
    r = 75.
    s = 20.
    x = he.calc_node_positions(node_spacing=10, radius=75)
    fracs = he.model_alpha_ejection(node_positions=x,
                                    stopping_distance=s,
                                    radius=r)
    # First 6 values should be 1
    assert fracs[:6] == pytest.approx(np.ones(6))
    assert fracs[6] == pytest.approx(0.692308)

    # Testing when intersection plane is in between nodes
    s = 25
    fracs = he.model_alpha_ejection(node_positions=x,
                                    stopping_distance=s,
                                    radius=r)
    # First 5 values should be 1
    assert fracs[:5] == pytest.approx(np.ones(5))
    assert fracs[5:] == pytest.approx(np.array([0.859091, 0.619231]))
