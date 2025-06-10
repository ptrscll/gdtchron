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