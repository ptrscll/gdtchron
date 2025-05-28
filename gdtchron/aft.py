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
            reduced length (unitless)
        constants: dictionary 
            dictionary of constants associated with annealing model being used
            default dictionary: KETCHAM_99_FC
    
    Returns:
        Result of length transform as a float or numpy array of floats
    
    """
    alpha = constants["alpha"]
    beta = constants["beta"]

    return (((1- r ** beta) / beta) ** alpha - 1) / alpha

