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
collection.new(path='config.py', args=True)

# Groups creation
collection.create()

# Series output
for series in collection:
    series.create_size_indicators_plot(forced=True)
    series.create_covariances_plot(forced=True)

    # Group output
    for group in series:
        group.create_2D_scatter('x', 'z', age=15.0, errors=True, labels=True, mst=False, forced=True)
        group.create_3D_scatter(age=15.0, errors=False, labels=False, mst=True, forced=True)
        group.create_covariances_scatter('x', 'u', age=0, errors=True, forced=True)
