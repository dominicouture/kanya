# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback package. It executes the following steps:

        Series: arguments, parameters, data and file import, format and check, and conversions
        Groups: star traceback from data or a model, and age computation
        Output: figures creation
"""

from series import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Series
Series(path='Traceback/config.py', args=True)

# Groups
collection.create()

# Output
for series in collection:
    series.create_size_indicators_plot()
    series.create_covariances_plot()
    for group in series:
        group.create_2D_scatter('x', 'z', age=15.0, errors=True, labels=True, mst=False)
        group.create_3D_scatter(age=15.0, errors=False, labels=False, mst=True)
        group.create_covariances_scatter('x', 'u', age=0)
