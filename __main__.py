# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback package. It executes the following steps:

     -  Series creation: arguments, parameters, data and configuration file import, format and
        check, and conversions.
     -  Groups creation: traceback stars from data or a model, and age computation by minimizing
        size indicators.
     -  Output creation: series and groups output.
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
    series.show_metrics() # Valid
    series.create_mad_xyz_plot(forced=True)
    series.create_mad_ξηζ_plot(forced=True) # Valid
    series.create_covariances_xyz_plot(forced=True)
    series.create_covariances_xyz_plot(robust=True, forced=True)
    series.create_covariances_ξηζ_plot(forced=True) # Valid
    series.create_covariances_ξηζ_plot(robust=True, forced=True)
    series.create_cross_covariances_xyz_plot(forced=True)
    series.create_cross_covariances_xyz_plot(robust=True, forced=True)
    series.create_cross_covariances_ξηζ_plot(forced=True) # Valid
    series.create_cross_covariances_ξηζ_plot(robust=True, forced=True)
    # series.create_covariances_ξηζ_sklearn_plot(forced=True)
    # series.create_det_mad_mst_plot(forced=True)

    # Group output
    for group in series:
        if group.number == 0:
            group.trajectory_xyz(forced=True, indicator='covariances_xyz', index=0) # Valid
            group.trajectory_ξηζ(forced=True, indicator='covariances_xyz', index=0) # Valid
            group.trajectory_txyz(forced=True, indicator='covariances_xyz', index=0) # Valid
            group.trajectory_tξηζ(forced=True, indicator='covariances_xyz', index=0) # Valid
            # group.create_map(title=False, forced=True, labels=False)
            # group.create_2D_and_3D_scatter([0,  5,  10], title=False, forced=True)
            # group.create_2D_and_3D_scatter([15, 20, 25], title=False, forced=True)
            # group.create_cross_covariances_scatter('x', 'u', age=10, title=False, errors=True, forced=True)
            group.create_age_distribution(forced=True, title=False, indicator='covariances_ξηζ', index=0)
            # for indicator in ('covariances_ξηζ_matrix_det', 'covariances_ξηζ', 'mad_ξηζ', 'mst_ξηζ_mean'):
            #     group.create_age_distribution(forced=True, title=False, indicator=indicator, index=0)
