# !/usr/local/bin/env python3
# -*- coding: utf-8 -*-

"""
__main__.py: pipeline of the kanya package. The following steps are executed:

     -  Group : arguments, parameters, data and configuration file import, format and
        checks, and unit conversions.
      - Traceback : Galactic orbit computation from data or a model
      - Metrics : age computation by minimizing association size metrics.
      - Output : figures and tables generation.
"""

import kanya

# Group
kanya.Group(file_path='config.py', args=True)

# Traceback and metrics
kanya.collection.create(forced=True)

# Output
exit()
for group in kanya.collection:
    group.draw_covariances('ξηζ', forced=True)
    group.draw_covariances('xyz', robust=True, forced=True)
    group.draw_cross_covariances('ξηζ', forced=True)
    group.draw_cross_covariances('xyz', robust=True, forced=True)
    group.draw_mad('ξηζ', forced=True)
    group.draw_mad('xyz', forced=True)
    group.draw_tree('xyz', forced=True)
    group.draw_tree('ξηζ', forced=True)
    group.draw_covariances_mad('xyz', '2x1', forced=True)
    group.draw_covariances_mad('ξηζ', '1x2', forced=True)
    group.draw_covariances_cross_mad_tree('ξηζ', forced=True)
    group.draw_covariances_cross_mad_tree('xyz', forced=True)
    group.draw_age_distribution('covariances_ξηζ', index=0, fit='normal', adjusted=False, forced=True)
    group.create_metrics_table(forced=True, show=True, save=False, machine=False)
    group.create_metrics_table(show=False, save=True, forced=True, machine=True)
    group.draw_trajectory('position', 'xyz', metric='covariances_xyz', index=0, forced=True)
    group.draw_trajectory('position', 'ξηζ', metric='covariances_ξηζ', index=0, forced=True)
    group.draw_time('position', 'ξηζ', '1x3', metric='covariances_ξηζ', forced=True)
    group.draw_time('velocity', 'xyz', '1x3', metric='covariances_xyz', forced=True)
    group.draw_time('position', 'ξηζ', '2x2', metric='covariances_ξηζ', forced=True)
    group.draw_scatter('position', 'xyz', '2x2', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_scatter('velocity', 'xyz', '2x3', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_scatter('position', 'ξηζ', '3x2', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_scatter('velocity', 'ξηζ', '3x3', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_scatter('position', 'xyz', '1x3', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_scatter('velocity', 'xyz', '3x1', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_scatter('position', 'ξηζ', '4x1', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_cross_scatter('xyz', '3x3', age=-5, errors=True, ellipses=True, forced=True)
    group.draw_cross_scatter('ξηζ', '4x4', age=-5, errors=True, ellipses=True, forced=True)
    group.draw_time_scatter('position', 'xyz', '4x2', ages=[0,  -5], errors=True, mst=True, ellipses=True, forced=True)
    group.draw_time_scatter('velocity', 'ξηζ', '4x3', ages=[0,  -5,  -10], errors=True, mst=True, ellipses=True, forced=True)
    group.draw_corner_scatter('xyz', age=-5, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_corner_scatter('ξηζ', age=0, errors=True, mst=True, ellipses=True, forced=True)
    group.draw_age_distribution('covariances_ξηζ', index=0, fit='skewnormal', number_of_bins=80, forced=True)
    group.draw_map(metric='covariances_ξηζ', index=0, labels=False, forced=True)
    group.create_kinematics_table(forced=True, save=True, machine=True, age=0.0)
    group.create_kinematics_time_table(forced=True, save=True, machine=True)
    group.create_kinematics_time_table_star(show=True, save=False, machine=False)
