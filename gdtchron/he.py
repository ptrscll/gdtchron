"""Module for forward modeling of (U-Th)/He ages.

This code follows the workflow from Ketcham (2005) and includes the 
alpha correction from Ketcham et al. (2011)
"""

import warnings

import numpy as np
from scipy.integrate import romb
from scipy.linalg import solve_banded
from scipy.optimize import fsolve

# Constants

# Frequency factors and activation energies are from Reiners and Brandon (2006)
# and references therein. Stopping distances are from Ketcham et al. (2011).
SYSTEM_PARAMS = {'AHe': {'freq_factor': 50e8 * 3.154e13,  # micrometers^2 / Myr
                         'activ_energy': 138000,        # J * mol^-1
                         'S_238U': 18.81,  # Stopping distances (micrometers)
                         'S_235U': 21.80,
                         'S_232Th': 22.25},
                 'ZHe': {'freq_factor': 0.46e8 * 3.154e13,
                         'activ_energy': 169000,        # J * mol^-1
                         'S_238U': 15.55,  # Stopping distances (micrometers)
                         'S_235U': 18.05,
                         'S_232Th': 18.43}}

# Half lives (Myr)
U238_HALF_LIFE = 4.468e3
U235_HALF_LIFE = 7.04e2
TH232_HALF_LIFE = 1.40e4

# Decay constants (1 / Myr)
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
    tridiag_matrix : NumPy ndarray
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
        Diffusivity (micrometers^2 / Myr)

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
        Diffusivity (micrometers^2 / Myr)
    node_spacing : float
        Spacing of nodes in the modeled crystal (micrometers)
    time_interval : float
        Timestep in the thermal model (Myr)

    Returns
    -------
    beta : float
        Beta (unitless), after Ketcham (2005).

    """
    beta = (2 * (node_spacing ** 2)) / (diffusivity * time_interval)
    return beta


def u_th_ppm_to_molg(u_ppm, th_ppm):
    """Convert concentrations of U and Th from ppm to mol / g.

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
        He production (mol * g^-1 * Myr^-1)

    """
    term238 = 8 * LAMBDA_U238 * u238_molg     # mol * g^-1 * Myr^-1
    term235 = 7 * LAMBDA_U235 * u235_molg     # mol * g^-1 * Myr^-1
    term232 = 6 * LAMBDA_TH232 * th_molg      # mol * g^-1 * Myr^-1
    
    he_production = term238 + term235 + term232  # mol * g^-1 * Myr^-1
    
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
        Matrix x (mol * micrometers / g) solved for using Equation 21 from 
        Ketcham (2005). 
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
        Total amount (mol / g) of He within the modeled crystal.
    v : NumPy array of floats
        Radial profile of He (mol / g)

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
        Amount of He (mol / g)
    u238_molg : float
        Amount of U238 (mol / g)
    u235_molg : float
        Amount of U235 (mol / g)
    th_molg : float
        Amount of Th232 (mol / g)

    Returns
    -------
    age : float
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
    
    age = fsolve(age_equation, 1)[0]
    
    return age


def alpha_correction(stop_distance, radius):
    """
    Calculate alpha ejection correction factor, after Ketcham et al. (2011).

    Uses Equation 2 of Ketcham et al. (2011)

    Parameters
    ----------
    stop_distance : float or NumPy array of floats
        Stopping distance(s) for particular isotopic system(s) (micrometers).
    radius : float
        Radius of the grain (micrometers)

    Returns
    -------
    tau : float or NumPy array of floats
        Alpha correction factor(s) (F_T in Ketcham et al. (2011))

    """
    volume = (4 / 3) * np.pi * radius ** 3
    surface_area = 4 * np.pi * radius ** 2
    
    tau = 1 - 0.25 * ((surface_area * stop_distance) / volume)
    
    return tau


def model_alpha_ejection(node_positions, stop_distance, radius):
    """Model retained fraction of He after alpha ejection.

    Calculations from in-text equations in Ketcham (2005).

    Parameters
    ----------
    node_positions : NumPy array of floats
        Radial positions of each modeled node (micrometers)
    stop_distance : float
        Stopping distance for particular isotopic system (micrometers).
    radius : float
        Radius of the grain (micrometers)

    Returns
    -------
    retained_fraction_edge : NumPy array of floats
        Fraction of He retained after alpha ejection for each node position

    """
    # Find edge nodes based on stopping distance and radius
    edge_nodes = node_positions >= radius - stop_distance
    
    # Calculate location of the intersection planes for all nodes
    intersection_planes = (
        (node_positions ** 2 + radius ** 2 - stop_distance ** 2) /
        (2 * node_positions)
        )
    
    # Calculate retained fractions for all nodes hypothetically
    retained_fractions_all = (
        0.5 + (intersection_planes - node_positions) / (2 * stop_distance)
        )
    
    # Only apply retained fraction to edge nodes
    retained_fractions_edge = np.where(edge_nodes, retained_fractions_all, 1)
    
    return retained_fractions_edge


def he_profile(avg_temps, tsteps, system, radius, uth_molg, 
               node_information, initial_x=None):    
    """Solve for the helium profile of the grain at the present day.
    
    Solves for final x in Equation 21 of Ketcham (2005). Note that
    in Ketcham (2005), this variable is referred to as u, but x is used here
    to distinguish this variable from the uranium-related variables.

    Parameters
    ----------
    avg_temps : NumPy array of floats with length n
        List of average temperatures (K) for each timestep for the 
        time-temperature path
    tsteps : NumPy array of floats with length n + 1
        Array of times (Myrs BP) in chronological (descending) order. First 
        time is start of first timestep, last time is end of last timestep. 
        Each pair of adjacent times composes a timestep.
    system : string
        Isotopic system. Current options are 'AHe' and 'ZHe'.
    radius : float
        Radius of the grain (micrometers)
    uth_molg : tuple
        u238_molg : float
            The concentration of U238 (mol / g),
        u235_molg : float
            The concentration of U235 (mol / g)
        th_molg : float
            The concentration of Th (mol / g)
    node_information : tuple
        nodes : float
            Number of nodes to model within the crystal. The default is 513.
        node_spacing : float
            Spacing of nodes in the modeled crystal (micrometers)
        node_positions : NumPy array of floats
            Radial positions of each modeled node (micrometers)
    initial_x : NumPy array of floats or None, optional
        Array containing the initial values for x (mol * micrometers / g). If 
        all values in the array are np.nan or initial_x is set to None, this 
        function assumes that the initial values for x are 0 for all nodes 
        (default: None)

    Returns
    -------
    x : NumPy array of floats
        Matrix x solved for using Equation 21 Ketcham (2005). 
        Equivalent to the concentration times the node position.

    """ 
    # Unpack parameters
    nodes, node_spacing, node_positions = node_information
    u238_molg, u235_molg, th_molg = uth_molg
    stop_dist_u238 = SYSTEM_PARAMS[system]['S_238U']
    stop_dist_u235 = SYSTEM_PARAMS[system]['S_235U']
    stop_dist_th232 = SYSTEM_PARAMS[system]['S_232Th']

    # Modify mol / g of U,Th for alpha ejection
    u238_alpha = u238_molg * model_alpha_ejection(node_positions,
                                                  stop_dist_u238,
                                                  radius)
    u235_alpha = u235_molg * model_alpha_ejection(node_positions,
                                                  stop_dist_u235,
                                                  radius)
    th_alpha = th_molg * model_alpha_ejection(node_positions,
                                              stop_dist_th232,
                                              radius)
        
    # Calculate He production based on U and Th, adjusted for alpha ejection.
    he_production = calc_he_production_rate(u238_alpha, u235_alpha, th_alpha)
    
    if initial_x is None or np.all(np.isnan(initial_x)):
        # Set initial x (u in Ketcham (2005)) equal to 0
        x = np.zeros(nodes)
    else:
        x = initial_x
    
    # Loop through each step of the T-t path, solving Equation 21 in Ketcham
    # (2005) using each temperature
    for i in range(0, len(avg_temps)):

        time_interval = tsteps[i] - tsteps[i + 1]
        
        # Use temperature to calculate diffusivity and beta
        diffusivity = calc_diffusivity(avg_temps[i], system)
        beta = calc_beta(diffusivity, node_spacing, time_interval)
        
        # Calculate A (banded) and use current x to calculate new B
        # (In Equation 21, A represents the coefficients on the lefthand side of
        # the equation, while B represents the righthand side of the equation)
        ab = tridiag_banded(1, -2 - beta, 1, nodes)
        ab[1, 0] = -3 - beta  # To satisfy Neumann condition
        b = np.zeros(nodes)

        # For first node, use boundary condition (Neumann)
        b[0] = (
            x[0] + (2 - beta) * x[0] - x[1]
            - he_production[0] * node_positions[0] * beta * time_interval
            )
        
        # Use previous and subsequent nodes for remaining nodes
        b[1:-1] = (
            -x[0:-2] + (2 - beta) * x[1:-1] - x[2:]
            - he_production[1:-1] * node_positions[1:-1] * beta * time_interval
            )           
        
        # For last node, use boundary condition (Drichlet)
        b[-1] = (
            -x[-2] + (2 - beta) * x[-1] - 0
            - he_production[-1] * node_positions[-1] * beta * time_interval
            )
        
        # Solve for x using banded A and B
        x = solve_banded((1, 1), ab, b)
        
    return x


def profile_to_age(x, node_positions, radius, uth_molg, stop_distances):
    """Calculate age based on final helium profile.

    Calculate corrected and uncorrected age based on final He profile. Also
    calculates normalized He profile and node positions.

    Parameters
    ----------
    x : NumPy array of floats
        Matrix x (mol * micrometers / g) solved for using Equation 21 from 
        Ketcham (2005). 
        Equivalent to the concentration times the node position. 
        In Ketcham (2005), this variable is referred to as u, but x is used here
        to distinguish this variable from the uranium-related variables.
    node_positions : NumPy array of floats
        Radial positions of each modeled node (micrometers)
    radius : float
        Radius of the grain (micrometers)
    uth_molg : tuple
        u238_molg : float
            The concentration of U238 (mol / g),
        u235_molg : float
            The concentration of U235 (mol / g)
        th_molg : float
            The concentration of Th (mol / g)
    stop_distances : NumPy array of floats
        Stopping distances (micrometers) for (in order) U238, U235, and Th232

    Returns
    -------
    age_corrected : float
        Corrected (U-Th) / He age
    age_uncorrected : Float
        (U-Th) / He age without alpha correction
    he_molg : float
        Total amount (mol / g) of He within the modeled crystal.
    position_normalized : NumPy array of floats
        Radial positions of each modeled node, normalized so that the radius has
        a radial position of 1
    v_normalized : NumPy array of floats
        Radial profile of He, normalized so that the position with the highest
        concentration has a value of 1

    """
    he_molg, v = sum_he_shells(x, node_positions, radius)
    
    u238_molg, u235_molg, th_molg = uth_molg
    
    # Because alpha ejection modeled, model age is an "uncorrected" age.
    age_uncorrected = calc_age(he_molg, u238_molg, u235_molg, th_molg)

    # "Corrected" age uses alpha-adjusted U-Th values
    
    # Get array of correction values (238,235,232)
    tau = alpha_correction(stop_distances, radius)
    
    # Correct U and Th accordingly
    u238_corr = u238_molg * tau[0]
    u235_corr = u235_molg * tau[1]
    th_corr = th_molg * tau[2]
    
    age_corrected = calc_age(he_molg, u238_corr, u235_corr, th_corr)
    
    # Make diffusional profile
    v_normalized = v / np.max(v)
    position_normalized = node_positions / radius
    
    return (age_corrected, 
            age_uncorrected, 
            he_molg,
            position_normalized,
            v_normalized)


def forward_model_he(temps, tsteps, system, u, th, radius, nodes=513,
                     initial_x=None, return_all=False):
    """Forward model a (U-Th)/He age for a particular time-temperature path.
    
    Uses finite difference method for diffusion within a sphere as described 
    in Ketcham (2005). Applies alpha ejection correction after Ketcham et al.
    (2011). Returns corrected age and optionally additional relevant values.

    Parameters
    ----------
    temps : NumPy array of length n, where n > 1
        List of temperatures (K) for the time-temperature path
    tsteps : NumPy array of floats with length n, where n > 1
        Array of times (Myrs BP) in chronological (descending) order. First 
        time is start of first timestep, last time is end of last timestep. 
        Each pair of adjacent times composes a timestep. Each time corresponds
        to a temperature in temps.
    system : string
        Isotopic system. Current options are 'AHe' and 'ZHe'.
    u : float
        U concentration (ppm)
    th : float
        Th concentration (ppm)
    radius : float
        Radius of the grain (micrometers)
    nodes : float, optional
        Number of nodes to model within the crystal. (default: 513)
    initial_x : NumPy array of floats or None, optional
        Array containing the initial values for x (mol * micrometers / g). If 
        all values in the array are np.nan or initial_x is set to None, this 
        function assumes that the initial values for x are 0 for all nodes. Note
        that x is referred to as u in Ketcham (2005). (default: None)
    return_all : bool
        Boolean indicating whether to return additional values calculated by
        the functions as a part of the forward model. If False, only the 
        corrected age is returned. (default: False)

    Returns
    -------
    age_corrected : float
        Corrected (U-Th) / He age
    age_uncorrected : float (optional)
        (U-Th) / He age without alpha correction. 
        Returned if return_all is True.
    he_nmolg : float (optional)
        Total amount (nmol / g) of He within the modeled crystal.
        Returned if return_all is True.
    position_normalized : NumPy array of floats (optional)
        Radial positions of each modeled node, normalized so that the radius has
        a radial position of 1.
        Returned if return_all is True.
    v_normalized : NumPy array of floats (optional)
        Radial profile of He, normalized so that the position with the highest
        concentration has a value of 1.
        Returned if return_all is True.
    x : NumPy array of floats (optional)
        Matrix x solved for using Equation 21 Ketcham (2005). 
        Equivalent to the concentration times the node position.
        Returned if return_all is True.

    """
    # Find node spacing and time interval based on radius and T-t path
    node_spacing = radius / nodes
    
    node_positions = calc_node_positions(node_spacing, radius)
    
    # Get mol/g of U,Th
    u238_molg, u235_molg, th_molg = u_th_ppm_to_molg(u, th)
    
    # Package parameters to pass to subsequent functions    
    node_information = (nodes, node_spacing, node_positions)
    uth_molg = (u238_molg, u235_molg, th_molg)
    stop_distances = np.array([SYSTEM_PARAMS[system]['S_238U'],
                               SYSTEM_PARAMS[system]['S_235U'],
                               SYSTEM_PARAMS[system]['S_232Th']])
    
    # Calculate He profile
    # Getting avg temperatures for each interval
    avg_temps = np.convolve(temps, np.ones(2), 'valid') / 2.
    x = he_profile(avg_temps, tsteps, system, radius, uth_molg, 
                   node_information, initial_x)
    
    (age_corrected, 
     age_uncorrected, 
     he_molg,
     position_normalized,
     v_normalized) = profile_to_age(x=x,
                                    node_positions=node_positions,
                                    radius=radius,
                                    uth_molg=uth_molg,
                                    stop_distances=stop_distances)
    
    he_nmolg = 1e9 * he_molg
    
    if return_all:
        return (age_corrected, 
                age_uncorrected, 
                he_nmolg, 
                position_normalized, 
                v_normalized, 
                x)
    else:
        return age_corrected
