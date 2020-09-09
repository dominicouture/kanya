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
    series.create_scatter_mad_mst_plot(title=False, forced=True)
    series.create_scatter_mad_mst_cross_covariances_plots(series, title=False, forced=True)
    series.create_scatter_mad_xyz_plot(title=False, forced=True)
    series.create_scatter_mad_rθz_plot(title=False, forced=True)
    series.create_covariances_xyz_plot(forced=True, title=False)
    series.create_covariances_rθz_plot(forced=True, title=False)
    series.create_cross_covariances_xyz_plot(forced=True, title=False)
    series.create_cross_covariances_rθz_plot(forced=True, title=False)

    # Group output
    for group in series:
        group.trajectory_XYZR(title=False, forced=True)
        group.create_map(title=False, forced=True, labels=False)
        group.create_2D_and_3D_scatter([0,  5,  10], title=False, forced=True)
        group.create_2D_and_3D_scatter([15, 20, 25], title=False, forced=True)
        group.trajectory(forced=True)
        group.create_cross_covariances_scatter('x', 'u', age=0, title=False, errors=True, forced=True)
