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

# Frequency factors and activation energies are from Reiners and Brandon (2006)
# and references therein. Stopping distances are from Ketcham et al. (2011).
SYSTEM_PARAMS = {'AHe': {'freq_factor': 50e8 * 3.154e7,  # micrometers^2 / yr
                         'activ_energy': 138000,        # J * mol^-1
                         'R_238U': 18.81,  # Stopping distances (micrometers)
                         'R_235U': 21.80,
                         'R_232Th': 22.25},
                 'ZHe': {'freq_factor': 0.46e8 * 3.154e7,  # micrometers^2 / yr
                         'activ_energy': 169000,        # J * mol^-1
                         'R_238U': 15.55,  # Stopping distances (micrometers)
                         'R_235U': 18.05,
                         'R_232Th': 18.43}}

IDEAL_GAS_CONST = 8.3144598  # (J * K^-1 * mol^-1)
U238_PER_U235 = 137.88  # Number of U-238 atoms for every U-235 atom (unitless)


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


def calc_diffusivity(temperature, system):
    """Calculate diffusivity from temperature and system diffusion parameters.
    
    After Reiners and Brandon (2006), with PV_a term assumed to be 0.

    Parameters
    ----------
    temperature : float
        Temperature (K)
    system : string
        String indicating whether to use parameters for the apatite system 
        ('AHe') or the zircon system ('ZHe')

    Returns
    -------
    kappa : float
        Diffusivity (micrometers^2 / yr)

    """
    freq_factor = SYSTEM_PARAMS[system]['freq_factor']
    activ_energy = SYSTEM_PARAMS[system]['activ_energy']

    exponent = np.exp(-activ_energy / (IDEAL_GAS_CONST * temperature))
    kappa = freq_factor * exponent

    return kappa


def calc_beta(diffusivity, node_spacing, time_interval):
    """Calculate beta, a substitution term from Ketcham (2005).
    
    The equation uses diffusivity, the spacing of nodes within the modeled 
    grain, and the timestep duration. It comes from the in-text equation
    between equations 20 and 21 in Ketcham (2005).

    Parameters
    ----------
    diffusivity : float
        Diffusivity (micrometers^2 / yr)
    node_spacing : float
        Spacing of nodes in the modeled crystal (micrometers)
    time_interval : float
        Timestep in the thermal model (yr)

    Returns
    -------
    beta : float
        Beta (unitless), after Ketcham (2005).

    """
    beta = (2 * (node_spacing ** 2)) / (diffusivity * time_interval)
    return beta


def u_th_ppm_to_molg(u_ppm, th_ppm):
    """Convert concentrations of U and Th from ppm to mol/g.

    Parameters
    ----------
    u_ppm : float
        U concentration (ppm)
    th_ppm : float
        Th concentration (ppm)

    Returns
    -------
    u238_molg : float
        U-238 (mol / g)
    u235_molg : float
        U-235 (mol / g)
    th_molg : float
        Th-232 (mol / g)
    """
    u238_ppm = (U238_PER_U235 / (1 + U238_PER_U235)) * u_ppm
    u235_ppm = (1 / (1 + U238_PER_U235)) * u_ppm
    
    u238_molg = u238_ppm * 1e-6 / 238
    u235_molg = u235_ppm * 1e-6 / 235
    th_molg = th_ppm * 1e-6 / 232

    return (u238_molg, u235_molg, th_molg)