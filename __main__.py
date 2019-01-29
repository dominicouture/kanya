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
config = Config('config.py')

# Traceback
from group import *
series = {config.series: Group.create(config)}

# Output
from output import *
create_scatter_graph(series[config.series], config.series)
