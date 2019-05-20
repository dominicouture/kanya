# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" tools.py: Defines various useful functions. """

import numpy as np
from series import info

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

def montecarlo(function, values, errors, n=200):
    """ Wraps a function to output both its value and errors, calculated with Monte Carlo
        algorithm with n iterations. The inputs and outputs are Quantity objects with values,
        units and errors.
    """

    values, errors = [i if type(i) in (tuple, list, np.ndarray) else [i] for i in (values, errors)]
    outputs = function(*values)
    output_errors = np.std(
        np.array([function(*arguments) for arguments in np.random.normal(
            values, errors, (n, len(values)))]),
        axis=0)

    return (outputs, output_errors)

def montecarlo2(function, values, errors, n=10000):
    """ Wraps a function to output both its value and errors, calculated with Monte Carlo
        algorithm with n iterations. The inputs and outputs are Quantity objects with values,
        units and errors.
    """

    outputs = function(values)
    output_errors = np.std(
        np.array([function(arguments) for arguments in np.random.normal(
            values, errors, (n, len(values)))]),
        axis=0)

    return (outputs, output_errors)
