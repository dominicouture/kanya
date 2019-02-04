# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main pipeline of the Traceback algorithm. It executes the following steps:

        Configuration:  arguments, parameters, data and database import, format and check.
        Traceback: data import or simulation, traceback, scatter calculation.
        Output: figures creation.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Configuration
from init import *
Series(Config('config.py', args=True))

# Traceback
groups.traceback()

# Output
from output import *
for series in groups.keys():
    create_scatter_graph(groups[series])
