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
kanya.collection.create(forced=True)

# Output creation
for series in kanya.collection:
    series.draw_covariances('ξηζ', forced=True) # Valid
    series.draw_covariances('xyz', robust=True, forced=True)
    series.draw_cross_covariances('ξηζ', forced=True) # Valid
    series.draw_cross_covariances('xyz', robust=True, forced=True)
    series.draw_mad('ξηζ', forced=True) # Valid
    series.draw_mad('xyz', forced=True)
    series.draw_mst('ξηζ', forced=True)
    series.draw_mst('xyz', forced=True)
    series.draw_covariances_mad('ξηζ', '2x1', forced=True)
    series.draw_covariances_mad('xyz', '1x2', forced=True)
    series.draw_covariances_cross_mad_mst('ξηζ', forced=True)
    series.draw_covariances_cross_mad_mst('xyz', forced=True)
    series.draw_age_distribution('covariances_ξηζ', index=0, fit='normal', adjusted=False, forced=True)
    series.create_metrics_table(forced=True, show=True, save=True, machine=False)
    series.create_metrics_table(show=False, save=True, forced=True, machine=False)

    # Group output
    for group in series:
        if group.number == 0:
            group.draw_trajectory('position', 'xyz', metric='covariances_xyz', index=0, forced=True) # Valid
            group.draw_trajectory('position', 'ξηζ', metric='covariances_ξηζ', index=0, forced=True) # Valid
            group.draw_time('position', 'xyz', '3x1', metric='covariances_xyz', forced=True) # Valid
            group.draw_time('velocity', 'xyz', '1x3', metric='covariances_xyz', forced=True) # Valid
            group.draw_time('position', 'ξηζ', '2x2', metric='covariances_ξηζ', forced=True) # Valid
            group.draw_scatter('velocity', 'xyz', '2x2', age=-5, errors=True, mst=True, forced=True) # Valid
            group.draw_scatter('velocity', 'xyz', '1x3', age=-5, errors=True, mst=True, forced=True) # Valid
            group.draw_scatter('velocity', 'xyz', '3x1', age=-5, errors=True, mst=True, forced=True) # Valid
            group.draw_scatter('velocity', 'xyz', '4x1', age=-5, errors=True, mst=True, forced=True) # Valid
            group.draw_scatter('position', 'ξηζ', '2x2', age=-5, errors=True, mst=True, forced=True) # Valid
            group.draw_scatter('velocity', 'xyz', '4x1', age=-5, errors=True, forced=True) # Valid
            group.draw_cross_scatter('xyz', age=-5, errors=True, forced=True)
            group.draw_cross_scatter('ξηζ', age=-5, errors=True, forced=True)
            group.draw_time_scatter('position', 'xyz', '4x2', ages=[0,  -5], errors=True, mst=True, forced=True)
            group.draw_time_scatter('velocity', 'xyz', '4x3', ages=[0,  -5,  -10], errors=True, mst=True, forced=True)
            group.draw_corner_scatter('xyz', age=-5, errors=True, forced=True)
            group.draw_corner_scatter('ξηζ', age=-5, errors=True, forced=True)
            group.draw_age_distribution('covariances_ξηζ', index=0, fit='skewnormal', number_of_bins=80, forced=True)
            group.draw_map(labels=False, forced=True)
            group.create_kinematics_table(forced=True, save=True, machine=True, age=-30.0)
            group.create_kinematics_time_table(forced=True, save=True, machine=True)

            # Star output
            for star in group:
                if star.name == 'HR 8799':
                    star.create_kinematics_time_table(show=False, save=True, machine=True)