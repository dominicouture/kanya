# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kanya (Kinematic Age for Nearby Young Associations) is a Python package that performs
three-dimensional traceback analysis of members of a nearby young association (NYA) and
computes their kinematic age by finding the epoch of minimal spatial extent.
"""

from .__version__ import *
from .series import Series, collection, Config

__all__ = [
    'Series',
    'collection',
    'Config'
]