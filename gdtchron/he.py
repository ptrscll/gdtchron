"""Module for forward modeling of (U-Th)/He ages.

This code follows the workflow from Ketcham (2005) and includes the 
alpha correction from Ketcham et al. (2011)
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import romb
from scipy.interpolate import griddata
from scipy.linalg import solve_banded
from scipy.optimize import fsolve


def tridiag_banded(a, b, c, diag_length):
    """Set up tridiagonal matrix in banded form from values for the 3 diagonals.

    Parameters
    ----------
    a : float
        First diagonal value
    b : float
        Second diagonal value
    c : float
        Third diagonal value
    diag_length : float
        Length of principal diagonal 

    Returns
    -------
    tridiag_matrix : numpy array
        Tridiagonal matrix

    """
    dtype = np.float32
    
    a_array = np.ones(diag_length, dtype=dtype) * a
    b_array = np.ones(diag_length, dtype=dtype) * b
    c_array = np.ones(diag_length, dtype=dtype) * c
    
    banded_matrix = np.vstack((a_array, b_array, c_array))
    banded_matrix[0, 0] = 0
    banded_matrix[-1, -1] = 0
    
    # del a_array,b_array,c_array

    return banded_matrix