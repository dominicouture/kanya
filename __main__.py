# !/usr/local/bin/env python3
# -*- coding: utf-8 -*-

"""
__main__.py: Example pipeline of the kanya package. The following steps are executed:

     -  Series creation: arguments, parameters, data and configuration file import, format and
        checks, and unit conversions.

      - Groups creation: Galactic orbit computation from data or a model, and age computation
        by minimizing association size metrics.

     -  Output creation: figures and tables generation for series, groups and stars.
"""

import kanya

# Series creation
kanya.Series(file_path='config.py', args=True)

# Groups creation
kanya.collection.create()

# Output creation
for series in kanya.collection:
    series.create_metrics_table(forced=True, show=True, save=True, machine=False)
    series.create_metrics_table(show=False, save=True, forced=True, machine=False)
    series.create_covariances_xyz_plot(forced=True)
    series.create_covariances_ξηζ_plot(forced=True) # Valid
    series.create_mahalanobis_plot(forced=True)
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
    series.create_covariances_mad_ξηζ_plot(forced=True)
    series.create_age_distribution(forced=True)

    # Group output
    for group in series:
        if group.number == 0:
            group.create_kinematics_table(forced=True, save=True, machine=True, age=-30.0)
            group.create_kinematics_time_table(forced=True, save=True, machine=True)
            group.create_trajectory_xyz(forced=True, metric='covariances_xyz', index=0)
            group.create_trajectory_ξηζ(forced=True, metric='covariances_ξηζ', index=0) # Valid
            group.create_trajectory_time_xyz('1x3', forced=True, metric='covariances_xyz')
            group.create_trajectory_time_ξηζ('1x3', forced=True, metric='covariances_ξηζ') # Valid
            group.create_map(forced=True, labels=False)
            group.create_2D_scatter('x', 'y', age=-10, errors=True, forced=True)
            group.create_3D_scatter(age=-10, errors=True, forced=True)
            group.create_2D_and_3D_scatter([0,  -10,  -20], errors=True, forced=True)
            group.create_cross_covariances_scatter('x', 'u', age=-10, errors=True, forced=True)
            group.create_age_distribution(forced=True, metric='covariances_ξηζ', index=0)
            for star in group:
                if star.name == 'HR 8799':
                    star.create_kinematics_time_table(show=False, save=True, machine=True)
