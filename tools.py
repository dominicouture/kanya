# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Provide the necesary tools to transform coordinates and other stuff

import numpy as np
from traceback import format_exc
from logging import basicConfig, info, warning, INFO
from time import strftime
from os.path import join
from config import *

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

def xyz_to_αδr(x, y, z):
    """ Convert a
    """

def αδr_to_xyz(α, δ, r):
    """ Convert a
    """

def uvw_to_μαμδvr(u, v, w):
    """ Convert a
    """

def μαμδvr_to_uvw(μα, μδ, vr):
    """ Convert a
    """
