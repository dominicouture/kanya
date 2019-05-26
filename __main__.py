# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback algorithm. It executes the following steps:

        Series configuration:  arguments, parameters, data and database import, format and check.
        Group creation: data import or simulation, traceback, and scatter and MST calculation.
        Output creation: figures creation.
"""

from init import *
from series import *
from output import *
from quantity import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Series configuration
Series(Config('config.py', args=True))

# Groups creation
groups.create()

# Output creation
for name, series in groups.items():
    create_size_indicators_plot(series)
    create_covariance_plot(series)
    for group in series:
        create_2D_scatter(group, 'x', 'y', age=15.0, errors=True, labels=True, mst=False)
        create_3D_scatter(group, age=15.0, errors=False, labels=False, mst=True)
