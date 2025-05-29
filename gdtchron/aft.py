################################################################################
# aft.py
#
# Functions to execute the forward modeling of the apatite fission track system,
# following workflow from Ketcham (2005). 
#
# The current code supports producing ages and distributions of c-axis projected
# lengths. This code uses the fanning curvilinear (FC) model from Ketcham et al.
# (1999). 
################################################################################

import numpy as np
from scipy.stats import norm

#################
### Constants ###
#################

# Constants for the fanning curvilinear model from Ketcham et al. (1999)
KETCHAM_99_FC = {
    "c0": -19.844,
    "c1": 0.38951,
    "c2": -51.253,
    "c3": -7.6423,
    "alpha": -0.12327,
    "beta": -11.988,
    "r_kappa_sum": 1.,
    "l_slope": 0.35,        # Value taken from HeFTy
    "l_intercept": 15.72
}

# Other constants
# #TODO: Check if changing from 365.25 to 365.2422 substantially changed results
SECONDS_PER_YEAR = 365.2422 * 24 * 60 * 60 
# TODO: Make this a parameter of its corresponding function
LENGTH_DIST_SPACING = 100 
# 50 is acceptable for sd/mean calcs, larger nums needed for pretty graphs


###########################
### Annealing Functions ###
###########################


def g(r, constants=KETCHAM_99_FC):
    """Implement the length transform from Ketcham et al. (1999) (Equation 6).

    Args:
        r: float (or numpy array of floats)
            Reduced length (unitless)
        constants: dictionary 
            Dictionary of constants associated with annealing model being used
            Default dictionary: KETCHAM_99_FC
    
    Returns:
        Result of length transform as a float or numpy array of floats
    
    """
    alpha = constants["alpha"]
    beta = constants["beta"]

    return (((1- r ** beta) / beta) ** alpha - 1) / alpha


def f(T, t, constants=KETCHAM_99_FC):
    """Calculate f following Equation 4 from Ketcham et al. (1999).

    Args:
        T: float
            Temperature (K)
        t: float or numpy array of floats
            How long the crystal annealed at a given temperature (s)
        constants: dictionary   
            Dictionary of constants associated with annealing model being used
            Default dictionary: KETCHAM_99_FC

    Returns:
        float or numpy array of floats containing value(s) of f for each value 
        of t

    """
    c0 = constants["c0"]
    c1 = constants["c1"]
    c2 = constants["c2"]
    c3 = constants["c3"]

    return c0 + c1 * ((np.log(t) - c2) / (np.log(1 / T) - c3))


def get_equiv_time(r_initial, T, constants = KETCHAM_99_FC):
    """Calculate time it would take to reach a reduced length at temperature T.
     
    This function solves Equation 5 from Ketcham (2005) for t (time).

    Args:
        r_initial: numpy array of floats
            reduced length (unitless)
        T: float
            Temperature (K)
        constants: dictionary   
            Dictionary of constants associated with annealing model being used
            Default dictionary: KETCHAM_99_FC

    Returns:
        numpy array of floats containing time(s) (in seconds) it would take to
        anneal to the given reduced length(s) if T remained constant

    """
    c0 = constants["c0"]
    c1 = constants["c1"]
    c2 = constants["c2"]
    c3 = constants["c3"]

    exponent =((g(r_initial, constants) - c0) / c1) * (np.log(1 / T) - c3) + c2

    return np.exp(exponent)


def get_next_r(T, cumulative_t, constants=KETCHAM_99_FC):
    """Calculate reduced lengths after annealing over a given time period.
     
    This function solves Equation 5 from Ketcham (2005) for r (reduced length).

    Args:
        T: float
            Temperature (K)
        cumulative_t: numpy array of floats
            How long the crystal annealed at a given temperature (s)
        constants: dictionary   
            Dictionary of constants associated with annealing model being used
            Default dictionary: KETCHAM_99_FC

    Returns:
        numpy array of floats containing mean reduced length(s) of fission
        tracks that annealed at the given temperature for the given period(s)
        of time

    """
    alpha = constants["alpha"]
    beta = constants["beta"]

    inner_base = alpha * f(T, cumulative_t, constants) + 1
    
    # Anywhere the inner base is negative has high enough temperatures that it
    # got fully annealed
    # Any number that would cause an integer overflow (i.e, any number below
    # approximately 0.00002 for Ketcham 1999 constants) is also excluded (and
    # effectively corresponds to all tracks being annealed)
    fully_annealed = inner_base < 0.00002

    inner_root = np.zeros(np.size(inner_base))

    # Only take root of inner base if not fully annealed
    # If fully annealed, leave inner root as 0 (and ultimately make r = 0)
    inner_root[~fully_annealed] = inner_base[~fully_annealed] ** (1 / alpha)

    outer_base = (1 - beta * inner_root)

    r = np.zeros(np.size(inner_root))
    r[~fully_annealed] = outer_base[~fully_annealed] ** (1 / beta)
    r[fully_annealed] = 0

    return r

# Function to calc annealing at each tstep
# start time, end time measured in yrs BP
# T measured in K
# next_nan_index corresponds to the index of the current timestep in the r array
def calc_annealing(r_initial, T, start, end, next_nan_index, 
                   constants=KETCHAM_99_FC):
    """Calculate the annealing of fission tracks across a timestep.

    Args:
        r_initial: numpy array of floats
            Initial mean reduced lengths (unitless) of fission tracks at start 
            of timestep. The value at index 0 corresponds to fission tracks
            produced at the first timestep, the value at index 1 corresponds to
            fission tracks produced at the second timstep, and so on. np.nan
            should be stored at indices corresponding to fission tracks
            produced at the current timestep or at future timesteps.
        T: float
            Temperature (K)
        start: float
            Start time of timestep (yrs BP)
        end: float
            End time of timestep (yrs BP)
        next_nan_index: int
            First index of r_initial to have a value of np.nan. This index
            corresponds to fission tracks that will be produced at the current
            timestep.
        constants: dictionary   
            Dictionary of constants associated with annealing model being used
            Default dictionary: KETCHAM_99_FC

    Returns:
        numpy array of floats containing the updated mean reduced length(s) of 
        fission tracks at the end of this timestep. 
        
    """
    # Convert timesteps to seconds
    start *= SECONDS_PER_YEAR
    end   *= SECONDS_PER_YEAR

    # Getting equivalent time it would have taken to reach current reduced
    # lengths if the temperature had always been at its current value
    # Note - we can't call get_equiv_time on r_initial = 0 (or very small r),
    # so we need to check for that
    fully_annealed = r_initial < 0.0007
    t_before = np.zeros(np.size(r_initial))
    t_before[~fully_annealed] = \
        get_equiv_time(r_initial=r_initial[~fully_annealed], 
                       T=T, constants=constants)
    t_before[next_nan_index] = 0  # Accounting for FTs formed at this timestep

    # Adding the duration of the current timestep to get the new cumulative
    # duration of annealing
    cumulative_t = t_before + start - end

    # Calculating next r
    r = np.zeros(np.size(r_initial))
    r[~fully_annealed] = get_next_r(T=T, 
                                    cumulative_t=cumulative_t[~fully_annealed],
                                    constants=constants)
    return r