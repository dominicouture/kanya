# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback package. It executes the following steps:

    1 - Series creation: arguments, parameters, data and configuration file import, format and
        check, and conversions.
    2 - Groups creation: traceback members' positions from data or a model, and age computation
        by minimizing association size metrics.
    3 - Output creation: series and groups output.
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
    series.create_covariances_xyz_plot(forced=True)
    series.create_covariances_ξηζ_plot(forced=True) # Valid
    series.create_covariances_xyz_plot(robust=True, forced=True)
    series.create_covariances_ξηζ_plot(robust=True, forced=True)
    series.create_covariances_xyz_plot(sklearn=True, forced=True)
    series.create_covariances_ξηζ_plot(sklearn=True, forced=True)
    series.create_cross_covariances_xyz_plot(forced=True)
    series.create_cross_covariances_ξηζ_plot(forced=True) # Valid
    series.create_cross_covariances_xyz_plot(robust=True, forced=True)
    series.create_cross_covariances_ξηζ_plot(robust=True, forced=True)
    series.create_mad_xyz_plot(forced=True)
    series.create_mad_ξηζ_plot(forced=True) # Valid
    series.create_xyz_mst_plot(forced=True)
    series.create_ξηζ_mst_plot(forced=True)
    series.create_age_distribution(forced=True)
    series.show_metrics() # Valid

    # Group output
    for group in series:
        if group.number == 0:
            group.trajectory_xyz(forced=True, metric='covariances_xyz', index=0) # Valid
            group.trajectory_ξηζ(forced=True, metric='covariances_ξηζ', index=0) # Valid
            group.trajectory_txyz(forced=True, metric='covariances_xyz') # Valid
            group.trajectory_tξηζ(forced=True, metric='covariances_ξηζ') # Valid
            group.create_map(forced=True, labels=False)
            group.create_2D_and_3D_scatter([0,  5,  10], forced=True)
            group.create_2D_and_3D_scatter([15, 20, 25], forced=True)
            group.create_cross_covariances_scatter('x', 'u', age=10, errors=True, forced=True)
            group.create_age_distribution(forced=True, metric='covariances_ξηζ', index=0)
            for metric in ('covariances_ξηζ_matrix_det', 'covariances_ξηζ', 'mad_ξηζ', 'mst_ξηζ_mean'):
                group.create_age_distribution(forced=True, metric=metric, index=0)
