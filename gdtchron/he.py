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

# Constants
SYSTEM_PARAMS = {'AHe': {'freq_factor': 50e8 * 3.154e7,  # micrometers^2 / yr
                         'activ_energy': 13800,        # J * mol^-1
                         'R_238U': 18.81,  # Stopping distances (micrometers)
                         'R_235U': 21.80,
                         'R_232Th': 22.25},
                 'ZHe': {'freq_factor': 0.46e8 * 3.154e7,  # micrometers^2 / yr
                         'activ_energy': 169000,        # J * mol^-1
                         'R_238U': 15.55,  # Stopping distances (micrometers)
                         'R_235U': 18.05,
                         'R_232Th': 18.43}}


def tridiag_banded(a, b, c, diag_length, dtype=np.float32):
    """Set up tridiagonal matrix in banded form from values for the 3 diagonals.

    For example, a tridiagonal matrix in banded form with values a, b, c for
    its diagonals and with a principal diagonal of length 6 appears as follows:
    [0, a, a, a, a, a]
    [b, b, b, b, b, b]
    [c, c, c, c, c, 0]

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
    dtype : type
        Type of numbers in the matrix (default: np.float32). 32-bit floats are
        preferred to save memory when running large numbers of forward models.

    Returns
    -------
    tridiag_matrix : numpy array
        Tridiagonal matrix

    """
    a_array = np.ones(diag_length, dtype=dtype) * a
    b_array = np.ones(diag_length, dtype=dtype) * b
    c_array = np.ones(diag_length, dtype=dtype) * c
    
    banded_matrix = np.vstack((a_array, b_array, c_array))
    banded_matrix[0, 0] = 0
    banded_matrix[-1, -1] = 0

    return banded_matrix