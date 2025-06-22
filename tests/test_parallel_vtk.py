"""Tests for parallel_vtk module."""
import numpy as np
import pytest
import pyvista as pv

from gdtchron import _parallel_vtk, run_tt_paths

"""
# Generate dummy VTK files for testing
for x in range(10):
    # Create very small mesh with 16 points and assign each an id and temp 
    mesh = pv.ImageData(dimensions=(4, 4, 1)).cast_to_unstructured_grid()
    mesh['id'] = np.arange(16)
    # Make temperature the same for all points so it cools over time
    mesh['T'] = 100 - (x * 10 * np.ones(16))

    mesh.save('file_' + str(x) + '.vtu')

# Define times (Ma) for the 10 files
times = np.arange(9, -1, 1)
"""


def test_run_tt_paths():
    """Unit tests for run_tt_paths."""
    # Test (U-Th) / He system
    he_times = np.arange(60, -0.001, -0.1)
    
    # Constants for test 1
    early_temps_1 = np.linspace(120, 20, 101)
    late_temps_1 = np.linspace(20, 20, 500)
    temps_1 = np.append(early_temps_1, late_temps_1) + 273

    # Constants for test 2
    temps_2 = np.linspace(120, 20, 601) + 273

    # Constants for test 3
    early_temps_3 = np.linspace(120, 65, 576) + 273
    late_temps_3 = np.linspace(65, 20, 26) + 273
    temps_3 = np.append(early_temps_3, late_temps_3[1:])

    temp_paths = [temps_1, temps_2, temps_3]

    ages = run_tt_paths(temp_paths=temp_paths, 
                        tsteps=he_times, 
                        system='AHe',
                        radius=100,
                        processes=3)

    assert ages[0] == pytest.approx(54.4, rel=5e-3)
    assert ages[1] == pytest.approx(26.3, rel=5e-2)
    assert ages[2] == pytest.approx(6.83, rel=1e-1)
    assert len(ages) == 3

    # Test AFT system (and a singleton list of temps)
    temps = np.linspace(190.1, 20, 500, endpoint=True)
    temps += 273.15
    ft_times = np.linspace(93, 0, 500, endpoint=True)

    age_list = run_tt_paths(temp_paths=[temps], tsteps=ft_times, system='AFT')
    assert age_list[0] == pytest.approx(39.8, rel=5e-3)
    assert len(age_list) == 1