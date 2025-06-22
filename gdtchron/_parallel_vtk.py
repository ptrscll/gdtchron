"""Module for running forward models across multiple T-t paths and VTK meshes.

This module includes a function for running forward models across multiple
user-provided t-T paths (run_tt_paths) and a function to run forward models
across the T-t paths experienced by different particles across a series of
VTK meshes (run_vtk)
"""

import os
import shutil

import numpy as np
import pyvista as pv
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from tqdm import tqdm

from gdtchron import aft, he


def run_particle_he(particle_id, inputs, calc_age, interpolate_profile,
                        dtype=np.float32):  
    """Calculate profile of x values for a particular ASPECT particle.

    Function to calculate the profile of x values (He concentrations times
    node positions) across all nodes within a hypothetical grain found in 
    a given particle. This function's primary purpose is to be run in
    parallel by run_vtk.

    Parameters
    ----------
    particle_id : int
        ID corresponding to the particle to get the He profile of
    inputs : tuple
        k : any
            Unused parameter (included here for symmetry with the inputs of
            run_particle_aft)
        positions : TODO
            Locations of each particle
        tree : scipy.spatial.KDTree
            K-d tree ... TODO (and TODO: resume from here)
    calc_age : bool
        Boolean indicating whether to calculate age of particle. If False,
        age is returned as np.nan. Note that setting this to False will not
        substantially improve the speed of calculations for this function.
    interpolate_profile : bool
        Boolean indicating whether to interpolate He data from nearest neighbor
        of the particle if the particle itself lacks He data. If False and the
        particle is missing He data, an age of np.nan and a profile filled with 
        np.inf are returned.
    dtype : type
        Type of numbers used for calculations (default: np.float32). 32-bit 
        floats are preferred to save memory.

    Returns
    -------
    age : float
        Age of the particle. This equals np.nan if calc_age is False or there
        is an issue obtaining the age of the particle
    x : NumPy array of floats
        Matrix x solved for using Equation 21 in Ketcham (2005) (in that paper,
        x is referred to as u).
        Equivalent to the He concentration (mol / g) times the node position
        (micrometers).


    """
    # Unpack inputs
    (k, positions, tree, ids, old_ids, temps, old_temps, old_profiles,
     time_interval, other_particles,
     system, he_profile_nodes, (u, th, radius)) = inputs
    
    # Get old profile and temperature for current particle if present
    array = old_profiles[particle_id == old_ids]
    particle_start_temp = old_temps[particle_id == old_ids]

    # Create variable to track if missing old data for particle
    missing = False
     
    # If array is empty, assign np.nan
    # If the initial array is filled with np.inf (i.e, we have a bugged
    # particle), then return (NaN, initial array)
    # Otherwise, assign new value from old profile
    if array.size == 0:
        profile = np.empty(he_profile_nodes, dtype=dtype)
        profile.fill(np.nan)
        missing = True
    elif array[0][0] == dtype(np.inf):
        age = np.nan
        return (age, array[0])
    else:
        profile = array 
    
    # Get particle temperature
    particle_end_temp = temps[ids == particle_id]
       
    # If particle not found, don't attempt to calculate profile or age
    if particle_end_temp.size == 0:            
        x = np.empty(he_profile_nodes, dtype=dtype)
        x.fill(dtype(np.inf))
        age = np.nan
        return (age, x)
    
    # Use previous He from nearest neighbor in previous timestep if none present
    
    if missing:
        
        if interpolate_profile:
        
            # Get particle position
            particle_position = positions[ids == particle_id]
            
            # Find closest particle
            distance, index = tree.query(particle_position)
            
            # Get id of closest particle
            neighbor_id = other_particles[index]

            # Get profile of closest particle
            try:
                profile = old_profiles[neighbor_id == old_ids]
            except Exception:
                print("problem - likely multi-dimensional id!")
                x = np.empty(he_profile_nodes, dtype=dtype)
                x.fill(dtype(np.inf))
                age = np.nan
                return (age, x)
            
            # Get temp of closest particle
            particle_start_temp = old_temps[neighbor_id == old_ids]
        
        # If turned off, return original profile of np.inf
        elif not interpolate_profile:
            x = np.empty(he_profile_nodes, dtype=dtype)
            x.fill(dtype(np.inf))
            age = np.nan
            return (age, x)
    
    # Double checking that interpolated particle isn't bugged
    if profile[0][0] == dtype(np.inf):
        age = np.nan
        return (age, profile[0])
        
    # passing start and end temperatures to forward model
    particle_temps = np.array([particle_start_temp[0], particle_end_temp[0]])
    particle_tsteps = np.array([time_interval, 0])

    age, age_unc, he_tot, pos, v, x = \
        he.forward_model(temps=particle_temps, 
                         tsteps=particle_tsteps,
                         system=system,
                         u=u,
                         th=th,
                         radius=radius,
                         nodes=he_profile_nodes,
                         initial_x=profile.flatten(),
                         return_all=True)
    
    if calc_age:
        return (age, x)
    else:
        age = np.nan
        return (age, x)
    

def run_tt_paths(temp_paths, tsteps, system, 
                 u=100, th=100, radius=50,
                 dpar=1.75, annealing_model='Ketcham99',
                 batch_size=100, processes=None,
                 **kwargs):
    """Run forward model of a given isotopic system across multiple t-T paths.

    Parameters
    ----------
    temp_paths : list of NumPy arrays of floats
        List of NumPy arrays of floats containing the temperatures (K) at each
        timestep in tsteps. Each array corresponds to a different grain
        to obtain a thermochronometric age for.
    tsteps : Numpy array of floats
        Array of times (Ma) in chronological (descending) order. First 
        time is start of first timestep, last time is end of last timestep. 
        Each pair of adjacent times composes a timestep. The time at a given
        index i corresponds to the temperatures at index i of each of the NumPy
        arrays in temp_paths.
    system : string
        Isotopic system to model. Current options are:
            'AHe': Apatite (U-Th) / He
            'ZHe': Zircon (U-Th) / He
            'AFT': Apatite Fission Track
    u : float, optional
        U concentration (ppm). Default is 100 ppm. Only used if system is 'AHe'
        or 'ZHe'.
    th : float, optional
        Th concentration (ppm). Default is 100 ppm. Only used if system is 'AHe'
        or 'ZHe'.
    radius : float, optional
        Radius of the grain (micrometers). Default is 50 micrometers. Only used 
        if system is 'AHe' or 'ZHe'.
    dpar : float, optional
        Etch figure length (micrometers). Default is 1.75 micrometers. Only used
        if system is 'AFT'.
    annealing_model : string, optional
        Annealing model to use. Currently, the only acceptable value is
        'Ketcham99', which corresponds to the fanning curvilinear model from
        Ketcham et al. (1999). Default is 'Ketcham99'. Only used if system is
        'AFT'.
    batch_size : int or 'auto', optional
        Number of batches to be dispatched to each worker at a time during 
        parallel computation (default: 100). If set to 'auto', this value is
        dynamically adjusted during computations to try to optimize efficiency.
    processes : int or None, optional
        Maximum number of processes that can run concurrently. If None, this
        parameter is internally set to two less than the number of CPUs on the
        user's system (default: None).
    **kwargs : optional
        Additional arguments to pass to the forward model function of the
        corresponding isotopic system

    Returns
    -------
    ages : list of floats
        Thermochronometric ages for the given isotopic system for grains that 
        experienced each of the provided time series. All (U-Th) / He ages 
        returned are corrected ages. 
    
    """
    if processes is None:
        processes = os.cpu_count() - 2

    # Setting how many tasks to initially dispatch to workers
    pre_dispatch = 2 * processes if batch_size == 'auto' else 2 * batch_size

    model_fn = {'AHe': he.forward_model_he,
                'ZHe': he.forward_model_he,
                'AFT': aft.forward_model_aft}
    ft_constants = {'Ketcham99': aft.KETCHAM_99_FC}

    he_inputs = (tsteps, system, u, th, radius)
    ft_inputs = (tsteps, dpar, ft_constants[annealing_model])

    model_inputs = {'AHe': he_inputs,
                    'ZHe': he_inputs,
                    'AFT': ft_inputs}
    
    output = Parallel(n_jobs=processes,
                  batch_size=batch_size,
                  pre_dispatch=pre_dispatch)(
                    delayed(model_fn[system])(path, 
                                              *model_inputs[system], 
                                              **kwargs)
                    for path in tqdm(temp_paths, position=0))
    
    return output
