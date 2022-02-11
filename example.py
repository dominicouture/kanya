# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" example.py: Example script to show Traceback functionalities. """

from Traceback import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Data example (from Miret-Roig et al. 2020)
values_xyz = [
    [   'X',    'Y',    'Z',    'U',    'V',    'W'],
    [ 14.60, -18.59, -28.21, -10.51, -16.09,  -8.60],
    [-21.28,  -6.77,  -9.83, -12.42,  -16.3,  -9.28],
    [ -1.54, -21.32, -16.33, -11.17,  -16.5,  -9.09],
    [ -3.43, -16.65, -10.06, -11.07, -15.79,  -9.21],
    [  7.59, -33.75, -18.58,  -10.5, -16.46,  -8.71],
    [ 32.00, -23.67, -10.97,  -8.20, -16.62,  -9.09],
    [ 30.33, -10.06,  -1.24,  -8.67, -16.51, -10.21],
    [ 48.62, -18.83, -11.34,  -7.17, -16.93, -10.12],
    [ 45.37, -21.65,  -5.88,  -7.35, -15.87, -10.49],
    [ 70.36, -26.64, -16.96,  -7.44, -16.81,  -9.20],
    [ 54.93,   0.82,  -4.03,  -7.83, -14.52,  -8.45],
    [ 49.55, -21.31, -16.18,  -8.58, -15.54,  -8.09],
    [ 34.91, -13.14, -10.76, -10.23, -15.13,  -8.05],
    [ 26.21, -11.84,  -9.68,  -7.78, -17.09, -10.26],
    [ 77.46, -13.26, -20.69,  -6.84, -16.37,  -8.99],
    [ 79.32, -14.36, -21.66,  -6.45, -16.22,  -8.95],
    [ 45.01, -16.58, -18.35,  -8.52,  -15.4,  -8.28],
    [ 41.30, -12.80, -21.32,  -8.89, -15.43,  -8.06],
    [ 64.02,  -9.10,  -29.5,  -6.73, -16.37,  -9.76],
    [ 48.13,   3.22, -18.92,  -9.47, -16.14,  -8.88],
    [ 57.78,  13.52, -26.17,  -7.69, -14.99,  -9.20],
    [ 52.34,   7.26, -28.47,  -7.68, -15.78,  -9.13],
    [ 42.62,  12.23, -23.41,  -6.96, -14.69, -10.27],
    [ 34.55,  11.39, -23.62,  -8.44, -14.85,  -9.59],
    [ 21.87,  12.25, -20.26,  -9.91, -15.17,  -9.78],
    [ 19.57, -18.94, -24.54, -10.25, -15.84,  -7.98]]

# Initialize a series of moving groups
example_series = Series(
    path='config.py',
    from_data={'value': True},
    jackknife_number={'value': 2},
    jackknife_fraction={'value': 1.0},
    rv_offset={'value': 0.0},
    data={
        'value': values_xyz,
        'units': ['pc', 'pc', 'pc', 'km/s', 'km/s', 'km/s'],
        'system': 'cartesian',
        'axis': 'equatorial'})

# Compute traceback of moving groups
example_series.create()

# Select the first (only) moving group
example_group = example_series[0]

# Create an array with the XYZ positions of all stars in the moving group
position_xyz = np.array([star.position_xyz for star in example_group])

# Create trajectory xyz figure
example_group.trajectory_xyz(forced=True, age=24.0)
