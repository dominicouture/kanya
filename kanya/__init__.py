# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kanya (Kinematic Age for Nearby Young Associations) is a Python package that performs
three-dimensional traceback analysis of members of a nearby young association (NYA) and
computes their kinematic age by finding the epoch when the spatial extent of the NYA was
minimal.
"""

from .__version__ import *
from .series import Series, collection

__all__ = [
    'Series',
    'collection'
]