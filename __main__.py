# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback package. It executes the following steps:

        -   Series creation: arguments, parameters, data and configuration file import, format and
            check, and conversions

        -   Groups creation: star traceback from data or a model, and age computation by minimizing
            the scater, median absolute deviation (MAD), minimum spanning tree (MST) mean branch
            length, MST branch length MAD and covariances

        -   Output creation: series and groups output
"""

from Traceback import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Series creation
Series(path='config.py', args=True)

# Groups creation
collection.create()

# Output creation
for series in collection:
    series.show_errors() # Valid
    # series.create_scatter_xyz_plot(title=False, forced=True)
    # series.create_scatter_ξηζ_plot(title=False, forced=True)
    # series.create_mad_xyz_plot(title=False, forced=True)
    series.create_mad_ξηζ_plot(title=False, forced=True) # Valid
    # series.create_covariances_xyz_plot(forced=True, title=False)
    series.create_covariances_ξηζ_plot(forced=True, title=False) # Valid
    series.create_covariances_ξηζ_2_plot(forced=True, title=False) # Valid
    series.create_robust_covariances_ξηζ_plot(forced=True, title=False) # Valid
    # series.create_cross_covariances_xyz_plot(forced=True, title=False)
    series.create_cross_covariances_ξηζ_plot(forced=True, title=False) # Valid
    # series.create_scatter_mad_mst_plot(title=False, forced=True)

    # Group output
    for group in series:
        if group.number == 0:
            group.trajectory(title=False, forced=True)
            group.trajectory_xyz(title=False, forced=True)
            group.trajectory_ξηζ(title=False, forced=True) # Valid
            # group.create_map(title=False, forced=True, labels=False)
            # group.create_2D_and_3D_scatter([0,  5,  10], title=False, forced=True)
            # group.create_2D_and_3D_scatter([15, 20, 25], title=False, forced=True)
            # group.create_cross_covariances_scatter('x', 'u', age=10, title=False, errors=True, forced=True)
