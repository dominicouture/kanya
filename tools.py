# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" tools.py: Defines various useful functions to handle array manipulation and to wrap functions
    around a Monte Carlo algorithm.
"""

import numpy as np

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

def squeeze(array):
    """ Squeezes a np.ndarray and adds one dimension if the resulting array has no dimension. """

    array = np.squeeze(array)

    return array if array.ndim > 0 else np.expand_dims(array, 0)

def broadcast(name, array1, array2):
    """ Returns the resulting of two arrays that can be broacast together or raises an error. """

    try:
        return np.broadcast(array1, array2).shape
    except Exception as error:
        error.args = ("{} ({} and {}) with shapes {} and {} cannot be broadcast together.".format(
            name, array1, array2, array1.shape, array2.shape),)
        raise

def full(name, shape, array):
    """ Broadcasts an array to a specified shape or raises an error. """

    try:
        return np.full(shape, array)
    except Exception as error:
        error.args = ("{} ({}) cannot be broadcast to the shape {}.".format(name, array, shape),)
        raise

def montecarlo(function, values, errors, n=10000):
    """ Wraps a 'function' to output both its value and error, calculated with Monte Carlo
        algorithm with 'n' iterations. 'values' and 'errors' are a list or tuple or values to be
        used as arguments in 'function' and errors representating the standard deviation of those
        values.
    """

    outputs = function(*values)
    output_errors = np.std(np.array(
        [function(*arguments) for arguments in np.random.normal(values, errors, (n, len(values)))]
    ), axis=0)

    return (outputs, output_errors)
