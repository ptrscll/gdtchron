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
    