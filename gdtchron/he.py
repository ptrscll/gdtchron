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

# Half lives (yr)
U238_HALF_LIFE = 4.468e9
U235_HALF_LIFE = 7.04e8
TH232_HALF_LIFE = 1.40e10

# Decay constants (1 / yr)
LAMBDA_U238 = np.log(2) / U238_HALF_LIFE
LAMBDA_U235 = np.log(2) / U235_HALF_LIFE
LAMBDA_TH232 = np.log(2) / TH232_HALF_LIFE

# Misc constants
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


def calc_he_production_rate(u238_molg, u235_molg, th_molg):
    """Calculate instantaneous He production rate as a function of U and Th.

    Parameters
    ----------
    u238_molg : float
        U238 (mol / g)
    u235_molg : float
        U235 (mol / g)
    th_molg : float
        Th232 (mol / g)

    Returns
    -------
    he_production : float
        He production (mol * g^-1 * yr^-1)

    """
    term238 = 8 * LAMBDA_U238 * u238_molg     # mol * g^-1 * yr^-1
    term235 = 7 * LAMBDA_U235 * u235_molg     # mol * g^-1 * yr^-1
    term232 = 6 * LAMBDA_TH232 * th_molg      # mol * g^-1 * yr^-1
    
    he_production = term238 + term235 + term232  # mol * g^-1 * yr^-1
    
    return he_production


def calc_node_positions(node_spacing, radius):
    """Calculate node positions given spacing and radius.
    
    Follows Ketcham (2005); see Figure 8 for an example. The first
    node is a half-spacing away from the center and the last node is a
    full spacing away from the edge of the grain.

    Parameters
    ----------
    node_spacing : float
        Distance between nodes in the crystal (micrometers)
    radius : float
        Radius of the grain (micrometers)  
    
    Returns
    -------
    node_positions : NumPy array of floats
        Radial positions of each modeled node (micrometers)

    """
    node_positions = np.arange(node_spacing / 2, radius, node_spacing)
    
    return node_positions


def sum_he_shells(x, node_positions, radius):
    """Sum He produced within all nodes of the modeled crystal.
    
    Uses substition for He concentration after Ketcham (2005). Converts radial 
    profile of He to system of shells, so He is weighted by volume of shell.

    Parameters
    ----------
    x : NumPy array of floats
        Matrix x solved for using matrices A and B after Ketcham (2005). 
        Equivalent to the concentration times the node position. 
        In Ketcham (2005), this variable is referred to as u, but x is used here
        to distinguish this variable from the uranium-related variables.
    node_positions : NumPy array of floats
        Radial positions of each modeled node (micrometers)
    radius : float
        Radius of the grain (micrometers)

    Returns
    -------
    he_molg : float
        Total amount (mol/g) of He within the modeled crystal.
    v : NumPy array
        Radial profile of He (mol/g)

    """
    # Back-substitute u=vr to get radial profile
    v = x / node_positions
    
    # Get volumes of spheres at each node
    sphere_volumes = node_positions ** 3 * (4 * np.pi / 3)
    
    # Get total volume of the sphere
    total_volume = radius ** 3 * (4 * np.pi / 3)
    
    # Calculate volumes for the shell corresponding to each node
    shell_volumes = np.empty(sphere_volumes.size)
    
    shell_volumes[0] = sphere_volumes[0]
    shell_volumes[1:] = np.diff(sphere_volumes)
    
    # Get shell as fraction of total volume
    shell_fraction = shell_volumes / total_volume
    
    # Scale He within radial profile by shell fraction
    v_shells = v * shell_fraction
    
    # Integrate weighted radial profile
    he_molg = romb(v_shells)

    return (he_molg, v)


def calc_age(he_molg, u238_molg, u235_molg, th_molg):
    """Calculate (U-Th)/He age from U, Th, and He concentrations.

    Uses Equation 15 from Ketcham (2005).
    Note that no alpha correction is applied here. Instead, the alpha 
    correction is applied to the amounts of each parent isotope fed into 
    this function, following Ketcham et al. (2011).

    Parameters
    ----------
    he_molg : float
        Amount of He (mol/g)
    u238_molg : float
        Amount of U238 (mol/g)
    u235_molg : float
        Amount of U235 (mol/g)
    th_molg : float
        Amount of Th232 (mol/g)

    Returns
    -------
    age_ma : float
        Calculated (U-Th)/He age (Ma)

    """        
    ageterm_238 = 8 * u238_molg
    ageterm_235 = 7 * u235_molg
    ageterm_232 = 6 * th_molg
    
    def age_equation(t):
        root = (
            ageterm_238 * (np.exp(LAMBDA_U238 * t) - 1)
            + ageterm_235 * (np.exp(LAMBDA_U235 * t) - 1)
            + ageterm_232 * (np.exp(LAMBDA_TH232 * t) - 1) - he_molg
            ) 
    
        return root
    
    warnings.filterwarnings('ignore',
                            'The iteration is not making good progress')
    
    age = fsolve(age_equation, 1e6)[0]
    age_ma = age / 1e6
    
    return age_ma


def alpha_correction(stopping_distance, radius):
    """
    Calculate alpha ejection correction factor, after Ketcham et al. (2011).

    Uses Equation 2 of Ketcham et al. (2011)

    Parameters
    ----------
    stopping_distance : float
        Stopping distance for particular isotopic system (micrometers).
    radius : float
        Radius of the grain (micrometers)

    Returns
    -------
    tau : float
        Alpha correction factor (F_T in Ketcham et al. (2011))

    """
    volume = (4 / 3) * np.pi * radius ** 3
    surface_area = 4 * np.pi * radius ** 2
    
    tau = 1 - 0.25 * ((surface_area * stopping_distance) / volume)
    
    return tau


def model_alpha_ejection(node_positions, stopping_distance, radius):
    """Model retained fraction of He after alpha ejection.

    Calculations from in-text equations in Ketcham (2005).

    Parameters
    ----------
    node_positions : NumPy array
        Radial positions of each modeled node (micrometers)
    stopping_distance : float
        Stopping distance for particular isotopic system (micrometers).
    radius : float
        Radius of the grain (micrometers)

    Returns
    -------
    retained_fraction_edge : NumPy array
        Fraction of He retained after alpha ejection for each node position

    """
    # Find edge nodes based on stopping distance and radius
    edge_nodes = node_positions >= radius - stopping_distance
    
    # Calculate location of the intersection planes for all nodes
    intersection_planes = (
        (node_positions ** 2 + radius ** 2 - stopping_distance ** 2) /
        (2 * node_positions)
        )
    
    # Calculate retained fractions for all nodes hypothetically
    retained_fractions_all = (
        0.5 + (intersection_planes - node_positions) / (2 * stopping_distance)
        )
    
    # Only apply retained fraction to edge nodes
    retained_fractions_edge = np.where(edge_nodes, retained_fractions_all, 1)
    
    return retained_fractions_edge

