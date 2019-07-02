# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback package. It executes the following steps:

        Series: arguments, parameters, data and file import, format and check, and conversions
        Groups: star traceback from data or a model, and age computation
        Output: figures creation
"""

from series import *
from output import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Series
Series(path='Traceback/config.py', args=True)

# Groups
collection.create()

# Output
for series in collection:
    create_size_indicators_plot(series)
    create_covariances_plot(series)
    for group in series:
        create_covariances_scatter(group, 'x', 'u', age=0)
        create_2D_scatter(group, 'x', 'z', age=15.0, errors=True, labels=True, mst=False)
        create_3D_scatter(group, age=15.0, errors=False, labels=False, mst=True)
