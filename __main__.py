# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback algorithm. It executes the following steps:

        Series configuration:  arguments, parameters, data and database import, format and check.
        Group creation: data import or simulation, traceback, scatter calculation.
        Output export: figures creation.
"""

from init import *
from series import *
from output import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Series configuration
a = Config('config.py', args=True)

Series(a)

# Groups creation
groups.create()

# Output export
for name, series in groups.items():
    # create_scatter_plot(series[0], 100)
    create_scatter_graph(series)
