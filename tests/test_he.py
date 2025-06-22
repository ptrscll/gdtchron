"""Test file for he.py."""

import numpy as np
import pytest

from gdtchron import he

# Constants for tests using T-t series from Ketcham (2005) Figure 10
# Constants for all Ketcham (2005) tests
U = 100
TH = 100
RADIUS = 100
NODES = 513
TIMES = np.arange(60, -0.001, -0.1)

# Constants for test 1
EARLY_TEMPS_1 = np.linspace(120, 20, 101)
LATE_TEMPS_1 = np.linspace(20, 20, 500)
TEMPS_1 = np.append(EARLY_TEMPS_1, LATE_TEMPS_1) + 273

# Constants for test 2
TEMPS_2 = np.linspace(120, 20, 601) + 273

# Constants for test 3
EARLY_TEMPS_3 = np.linspace(120, 65, 576) + 273
LATE_TEMPS_3 = np.linspace(65, 20, 26) + 273
TEMPS_3 = np.append(EARLY_TEMPS_3, LATE_TEMPS_3[1:])


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
                                      th_molg=0.) == pytest.approx(1.241e-6,
                                                                   rel=1e-4)
    assert he.calc_he_production_rate(u238_molg=0., 
                                      u235_molg=0.0005, 
                                      th_molg=0.) == pytest.approx(3.446e-6,
                                                                   rel=1e-4)
    assert he.calc_he_production_rate(u238_molg=0., 
                                      u235_molg=0., 
                                      th_molg=0.005) == pytest.approx(1.480e-6,
                                                                      rel=5e-3)
    
    # Test rates when all isotopes present
    assert he.calc_he_production_rate(u238_molg=0.001, 
                                      u235_molg=0.0005, 
                                      th_molg=0.005) == pytest.approx(6.167e-6,
                                                                      rel=1e-3)
    

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
    assert he.alpha_correction(stop_distance=0.4, 
                               radius=3.) == pytest.approx(0.90)


def test_model_alpha_ejection():
    """Unit tests for modle_alpha_ejection."""
    # Testing when the intersection plane is located atone of the node positions
    r = 75.
    s = 20.
    x = he.calc_node_positions(node_spacing=10, radius=75)
    fracs = he.model_alpha_ejection(node_positions=x,
                                    stop_distance=s,
                                    radius=r)
    # First 6 values should be 1
    assert fracs[:6] == pytest.approx(np.ones(6))
    assert fracs[6] == pytest.approx(0.692308)

    # Testing when intersection plane is in between nodes
    s = 25
    fracs = he.model_alpha_ejection(node_positions=x,
                                    stop_distance=s,
                                    radius=r)
    # First 5 values should be 1
    assert fracs[:5] == pytest.approx(np.ones(5))
    assert fracs[5:] == pytest.approx(np.array([0.859091, 0.619231]))


def test_he_profile():
    """Unit tests for he_profile.
    
    The time-temperature series for these tests are taken from Ketcham (2005)
    Figure 10, and the comparison data comes from HeFTy v1.9.3.
    """
    # Getting values for all tests
    uth_molg = he.u_th_ppm_to_molg(U, TH)

    node_spacing = RADIUS / NODES
    
    node_positions = he.calc_node_positions(node_spacing, RADIUS)

    node_info = (NODES, node_spacing, node_positions)
    
    avg_temps_1 = np.convolve(TEMPS_1, np.ones(2), 'valid') / 2.
    avg_temps_2 = np.convolve(TEMPS_2, np.ones(2), 'valid') / 2.
    avg_temps_3 = np.convolve(TEMPS_3, np.ones(2), 'valid') / 2.

    # Test 1
    x = he.he_profile(avg_temps=avg_temps_1, 
                      tsteps=TIMES, 
                      system='AHe',
                      radius=RADIUS,
                      uth_molg=uth_molg,
                      node_information=node_info)
    
    v = x / node_positions
    v_normalized = v / np.max(v)

    assert v_normalized[0] == pytest.approx(1)
    assert v_normalized[np.argmin(np.abs(node_positions - 80))] == \
        pytest.approx(0.957, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 84))] == \
        pytest.approx(0.868, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 90))] == \
        pytest.approx(0.694, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 94))] == \
        pytest.approx(0.584, rel=1e-2)
    assert np.argmin(v_normalized) == 512

    # Test 2
    x = he.he_profile(avg_temps=avg_temps_2,
                      tsteps=TIMES, 
                      system='AHe',
                      radius=RADIUS,
                      uth_molg=uth_molg,
                      node_information=node_info)
    
    v = x / node_positions
    v_normalized = v / np.max(v)

    assert v_normalized[0] == pytest.approx(1)
    assert v_normalized[np.argmin(np.abs(node_positions - 28))] == \
        pytest.approx(0.985, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 50))] == \
        pytest.approx(0.945, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 68))] == \
        pytest.approx(0.876, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 84))] == \
        pytest.approx(0.672, rel=5e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 90))] == \
        pytest.approx(0.508, rel=5e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 94))] == \
        pytest.approx(0.385, rel=5e-2)
    assert np.argmin(v_normalized) == 512

    # Test 3
    x = he.he_profile(avg_temps=avg_temps_3, 
                      tsteps=TIMES,
                      system='AHe',
                      radius=RADIUS,
                      uth_molg=uth_molg,
                      node_information=node_info)
    
    v = x / node_positions
    v_normalized = v / np.max(v)

    assert v_normalized[0] == pytest.approx(1)
    assert v_normalized[np.argmin(np.abs(node_positions - 20))] == \
        pytest.approx(0.976, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 28))] == \
        pytest.approx(0.953, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 50))] == \
        pytest.approx(0.840, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 68))] == \
        pytest.approx(0.683, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 84))] == \
        pytest.approx(0.444, rel=5e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 92))] == \
        pytest.approx(0.267, rel=5e-2)
    assert np.argmin(v_normalized) == 512


def test_profile_to_age():
    """Unit test for profile_to_age."""
    # Setting up x
    node_positions = he.calc_node_positions(node_spacing=0.1, radius=0.35)
    v = np.array([0.343, 0.026385, 0.007])
    x = v * node_positions

    (age_corr, 
     age_uncorr,
     he_molg,
     pos_norm, 
     v_norm) = he.profile_to_age(x=x, 
                                 node_positions=node_positions,
                                 radius=0.35,
                                 uth_molg=(0.01, 0.002, 0.05),
                                 stop_distances=np.array([0.07, 0.14, 0.]))
    
    assert age_uncorr == pytest.approx(87.7654)
    assert age_corr == pytest.approx(102.77)
    assert he_molg == pytest.approx(0.003666706)
    assert pos_norm == pytest.approx(np.array([1 / 7, 3 / 7, 5 / 7]))
    assert v_norm == pytest.approx(np.array([1., 0.0769242, 0.02040816]))
    

def test_forward_model_he():
    """Unit tests for forward_model_he.
    
    The time-temperature series for these tests are taken from Ketcham (2005)
    Figure 10, and the comparison data comes from HeFTy v1.9.3.
    """
    # Getting values for all tests
    node_spacing = RADIUS / NODES
    
    node_positions = he.calc_node_positions(node_spacing, RADIUS)

    # Test 1
    (age_corrected, 
     age_uncorrected, 
     he_nmolg, 
     position_normalized, 
     v_normalized, 
     x) = he.forward_model_he(temps=TEMPS_1,
                              tsteps=TIMES, 
                              system='AHe',
                              u=U,
                              th=TH,
                              radius=RADIUS,
                              return_all=True)

    assert v_normalized[0] == pytest.approx(1)
    assert v_normalized[np.argmin(np.abs(node_positions - 80))] == \
        pytest.approx(0.957, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 84))] == \
        pytest.approx(0.868, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 90))] == \
        pytest.approx(0.694, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 94))] == \
        pytest.approx(0.584, rel=1e-2)
    assert np.argmin(v_normalized) == 512

    assert he_nmolg == pytest.approx(3.115e1, rel=5e-3)
    assert age_corrected == pytest.approx(54.4, rel=5e-3)
    assert age_uncorrected == pytest.approx(46.5, rel=5e-3)

    # Test 2
    (age_corrected, 
     age_uncorrected, 
     he_nmolg, 
     position_normalized, 
     v_normalized, 
     x) = he.forward_model_he(temps=TEMPS_2,
                              tsteps=TIMES, 
                              system='AHe',
                              u=U,
                              th=TH,
                              radius=RADIUS,
                              return_all=True)

    assert v_normalized[0] == pytest.approx(1)
    assert v_normalized[np.argmin(np.abs(node_positions - 28))] == \
        pytest.approx(0.985, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 50))] == \
        pytest.approx(0.945, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 68))] == \
        pytest.approx(0.876, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 84))] == \
        pytest.approx(0.672, rel=5e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 90))] == \
        pytest.approx(0.508, rel=5e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 94))] == \
        pytest.approx(0.385, rel=5e-2)
    assert np.argmin(v_normalized) == 512

    assert he_nmolg == pytest.approx(15.01, rel=5e-2)
    assert age_corrected == pytest.approx(26.3, rel=5e-2)
    assert age_uncorrected == pytest.approx(22.5, rel=5e-2)

    # Test 3
    (age_corrected, 
     age_uncorrected, 
     he_nmolg, 
     position_normalized, 
     v_normalized, 
     x) = he.forward_model_he(temps=TEMPS_3,
                              tsteps=TIMES, 
                              system='AHe',
                              u=U,
                              th=TH,
                              radius=RADIUS,
                              return_all=True)

    assert v_normalized[0] == pytest.approx(1)
    assert v_normalized[np.argmin(np.abs(node_positions - 20))] == \
        pytest.approx(0.976, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 28))] == \
        pytest.approx(0.953, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 50))] == \
        pytest.approx(0.840, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 68))] == \
        pytest.approx(0.683, rel=1e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 84))] == \
        pytest.approx(0.444, rel=5e-2)
    assert v_normalized[np.argmin(np.abs(node_positions - 92))] == \
        pytest.approx(0.267, rel=5e-2)
    assert np.argmin(v_normalized) == 512

    assert he_nmolg == pytest.approx(3.894, rel=1e-1)
    assert age_corrected == pytest.approx(6.83, rel=1e-1)
    assert age_uncorrected == pytest.approx(5.84, rel=1e-1)
