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

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Series configuration
Series(Config('config.py', args=True))

# Groups creation
groups.create()

# Output creation
for name, series in groups.items():
    create_size_indicators_plot(series)
    # for group in series:
        # for frame in [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]:
            # create_2D_scatter(group, 'x', 'z', age=frame, errors=True, labels=True, mst=True)
            # create_3D_scatter(group, step=frame, errors=False, labels=False, mst=True)
