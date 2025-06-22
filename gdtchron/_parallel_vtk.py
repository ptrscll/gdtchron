"""Module for running forward models across multiple T-t paths and VTK meshes.

This module includes a function for running forward models across multiple
user-provided t-T paths (run_tt_paths) and a function to run forward models
across the T-t paths experienced by different particles across a series of
VTK meshes (run_vtk)
"""

import gc
import os
import shutil

import numpy as np
import pyvista as pv
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from tqdm import tqdm

from gdtchron import aft, he


def run_tt_paths(temp_paths, tsteps, system, 
                 u=100, th=100, radius=50,
                 dpar=1.75, annealing_model='Ketcham99',
                 batch_size=100, processes=None,
                 **kwargs):
    """Run forward model of a given isotopic system across multiple t-T paths.

    TODO

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
    TODO : TODO
        TODO.
    
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
