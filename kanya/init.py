# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
init.py: Imports information from config.py, command line arguments and parameters into a
Config object. This script must be run first to create a Series object.
"""

from copy import deepcopy
from .collection import *
from .coordinate import *

class Config():
    """
    Contains the parameters imported from a configuration file (which must be a Python file),
    command line arguments, parameters in the __init__ function call or another Config object,
    as well as related methods, a Parameter class and a dictionary of default values. A Config
    object can then be used as the input of a Series object.
    """

    class Metric():
        """Basic parameters of an association size metric."""

        def __init__(self, label, name, latex_short, latex_long, valid, age_shift, order):
            """Initializes an association size metric."""

            self.label = label
            self.name = np.atleast_1d(name)
            self.latex_short = np.atleast_1d(latex_short)
            self.latex_long = np.atleast_1d(latex_long)
            self.valid = np.atleast_1d(valid)
            self.age_shift = np.atleast_1d(age_shift)
            self.order = order

    # Initializes association size metrics
    metrics = {metric.label: metric for metric in (

        # xyz spatial covariances matrix
        Metric(
            'covariances_xyz',
            np.array([
                'X Variance',
                'Y Variance',
                'Z Variance']),
            np.array([
                'Var.${{}}_{{X}}$',
                'Var.${{}}_{{Y}}$',
                'Var.${{}}_{{Z}}$']),
            np.array([
                '$X$ Variance',
                '$Y$ Variance',
                '$Z$ Variance']),
            np.array([True, False, False]),
            np.array([0.94, 0.36, -0.15]), 0),
        Metric(
            'covariances_xyz_matrix_det',
            'XYZ Covariance Matrix Determinant',
            'Det.${{}}_{{XYZ}}$',
            'Determinant${{}}_{{XYZ}}$',
            False, 0.26, 1),
        Metric(
            'covariances_xyz_matrix_trace',
            'XYZ Covariance Matrix Trace',
            'Trace${{}}_{{XYZ}}$',
            'Trace${{}}_{{XYZ}}$',
            False, 0.4, 2),

        # xyz spatial robust covariances matrix
        Metric(
            'covariances_xyz_robust',
            np.array([
                'X Variance (robust)',
                'Y Variance (robust)',
                'Z Variance (robust)']),
            np.array([
                'Var.${{}}_{{X}}$ (robust)',
                'Var.${{}}_{{Y}}$ (robust)',
                'Var.${{}}_{{Z}}$ (robust)']),
            np.array([
                '$X$ Variance (robust)',
                '$Y$ Variance (robust)',
                '$Z$ Variance (robust)']),
            np.array([True, False, False]),
            np.array([0.97, 0.28, -0.01]), 3),
        Metric(
            'covariances_xyz_matrix_det_robust',
            'XYZ Covariance Matrix Determinant (robust)',
            'Det.${{}}_{{XYZ}}$ (robust)',
            'Determinant${{}}_{{XYZ}}$ (robust)',
            False, 0.32, 4),
        Metric(
            'covariances_xyz_matrix_trace_robust',
            'XYZ Covariance Matrix Trace (robust)',
            'Trace${{}}_{{XYZ}}$ (robust)',
            'Trace${{}}_{{XYZ}}$ (robust)',
            False, 0.44, 5),

        # xyz spatial sklearn covariances matrix
        Metric(
            'covariances_xyz_sklearn',
            np.array([
                'X Variance (sklearn)',
                'Y Variance (sklearn)',
                'Z Variance (sklearn)']),
            np.array([
                'Var.${{}}_{{X}}$ (sklearn)',
                'Var.${{}}_{{Y}}$ (sklearn)',
                'Var.${{}}_{{Z}}$ (sklearn)']),
            np.array([
                '$X$ Variance (sklearn)',
                '$Y$ Variance (sklearn)',
                '$Z$ Variance (sklearn)']),
            np.array([False, False, False]),
            np.array([0.42, -0.9, 0.16]), 6),
        Metric(
            'covariances_xyz_matrix_det_sklearn',
            'XYZ Covariance Matrix Determinant (sklearn)',
            'Det.${{}}_{{XYZ}}$ (sklearn)',
            'Determinant${{}}_{{XYZ}}$ (sklearn)',
            False, 0.22, 7),
        Metric(
            'covariances_xyz_matrix_trace_sklearn',
            'XYZ Covariance Matrix Trace (sklearn)',
            'Trace${{}}_{{XYZ}}$ (sklearn)',
            'Trace${{}}_{{XYZ}}$ (sklearn)',
            False, 0.21, 8),

        # ξηζ position covariances matrix
        Metric(
            'covariances_ξηζ',
            np.array([
                'ξ Variance',
                'η Variance',
                'ζ Variance']),
            np.array([
                'Var.${{}}_{{ξ^\\prime}}$',
                'Var.${{}}_{{η^\\prime}}$',
                'Var.${{}}_{{ζ^\\prime}}$']),
            np.array([
                '$ξ^\\prime$ Variance',
                '$η^\\prime$ Variance',
                '$ζ^\\prime$ Variance']),
            np.array([True, False, False]),
            np.array([0.60, 0.70, -0.15]), 9),
        Metric(
            'covariances_ξηζ_matrix_det',
            'ξηζ Covariance Matrix Determinant',
            'Det.${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$',
            'Determinant${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$',
            False, 0.26, 10),
        Metric(
            'covariances_ξηζ_matrix_trace',
            'ξηζ Covariance Matrix Trace',
            'Trace${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$',
            'Trace${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$',
            False, 0.40, 11),

        # ξηζ position robust covariances matrix, determinant and trace
        Metric(
            'covariances_ξηζ_robust',
            np.array([
                'ξ Variance (robust)',
                'η Variance (robust)',
                'ζ Variance (robust)']),
            np.array([
                'Var.${{}}_{{ξ^\\prime}}$ (robust)',
                'Var.${{}}_{{η^\\prime}}$ (robust)',
                'Var.${{}}_{{ζ^\\prime}}$ (robust)']),
            np.array([
                '$ξ^\\prime$ Variance (robust)',
                '$η^\\prime$ Variance (robust)',
                '$ζ^\\prime$ Variance (robust)']),
            np.array([True, False, False]),
            np.array([0.53, 0.71, -0.01]), 12),
        Metric(
            'covariances_ξηζ_matrix_det_robust',
            'ξηζ Covariance Matrix Determinant (robust)',
            'Det.${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (robust)',
            'Determinant${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (robust)',
            False, 0.32, 13),
        Metric(
            'covariances_ξηζ_matrix_trace_robust',
            'ξηζ Covariance Matrix Trace (robust)',
            'Trace${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (robust)',
            'Trace${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (robust)',
            False, 0.44, 14),

        # ξηζ position sklearn covariances matrix, determinant and trace
        Metric(
            'covariances_ξηζ_sklearn',
            np.array([
                'ξ Variance (sklearn)',
                'η Variance (sklearn)',
                'ζ Variance (sklearn)']),
            np.array([
                'Var.${{}}_{{ξ^\\prime}}$ (sklearn)',
                'Var.${{}}_{{η^\\prime}}$ (sklearn)',
                'Var.${{}}_{{ζ^\\prime}}$ (sklearn)']),
            np.array([
                '$ξ^\\prime$ Variance (sklearn)',
                '$η^\\prime$ Variance (sklearn)',
                '$ζ^\\prime$ Variance (sklearn)']),
            np.array([False, False, False]),
            np.array([1.23, 1.52, 0.64]), 15),
        Metric(
            'covariances_ξηζ_matrix_det_sklearn',
            'ξηζ Covariance Matrix Determinant (sklearn)',
            'Det.${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (sklearn)',
            'Determinant${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (sklearn)',
            False, 0.13, 16),
        Metric(
            'covariances_ξηζ_matrix_trace_sklearn',
            'ξηζ Covariance Matrix Trace (sklearn)',
            'Trace${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (sklearn)',
            'Trace${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ (sklearn)',
            False, 0.18, 17),

        # xyz position and velocity cross-covariances matrix, determinant and trace
        Metric(
            'cross_covariances_xyz',
            np.array([
                'X-U Cross-Covariance',
                'Y-V Cross-Covariance',
                'Z-W Cross-Covariance']),
            np.array([
                '$X-U$',
                '$Y-V$',
                '$Z-W$']),
            np.array([
                '$X-U$ Cross-Covariance',
                '$Y-V$ Cross-Covariance',
                '$Z-W$ Cross-Covariance']),
            np.array([False, False, False]),
            np.array([0.96, 0.4, 10.31]), 18),
        Metric(
            'cross_covariances_xyz_matrix_det',
            'XYZ Cross-Covariance Matrix Determinant',
            'Det.${{}}_{{XUYVZW}}$', 'Determinant${{}}_{{XUYVZW}}$',
            False, 0.28, 19),
        Metric(
            'cross_covariances_xyz_matrix_trace',
            'XYZ Cross-Covariance Matrix Trace',
            'Trace${{}}_{{XUYVZW}}$', 'Trace${{}}_{{XUYVZW}}$',
            False, 0.42, 20),

        # xyz position and velocity robust cross-covariances matrix, determinant and trace
        Metric(
            'cross_covariances_xyz_robust',
            np.array([
                'X-U Cross-Covariance (robust)',
                'Y-V Cross-Covariance (robust)',
                'Z-W Cross-Covariance (robust)']),
            np.array([
                '$X-U$ (robust)',
                '$Y-V$ (robust)',
                '$Z-W$ (robust)']),
            np.array([
                '$X-U$ Cross-Covariance (robust)',
                '$Y-V$ Cross-Covariance (robust)',
                '$Z-W$ Cross-Covariance (robust)']),
            np.array([False, False, False]),
            np.array([0.70, 0.24, 6.37]), 21),
        Metric(
            'cross_covariances_xyz_matrix_det_robust',
            'XYZ Cross-Covariance Matrix Determinant (robust)',
            'Det.${{}}_{{XUYVZW}}$ (robust)',
            'Determinant${{}}_{{XUYVZW}}$ (robust)',
            False, 0.29, 22),
        Metric(
            'cross_covariances_xyz_matrix_trace_robust',
            'XYZ Cross-Covariance Matrix Trace (robust)',
            'Trace${{}}_{{XUYVZW}}$ (robust)',
            'Trace${{}}_{{XUYVZW}}$ (robust)',
            False, 0.35, 23),

       # xyz position and velocity sklearn cross-covariances matrix, determinant and trace
        Metric(
            'cross_covariances_xyz_sklearn',
            np.array([
                'X-U Cross-Covariance (sklearn)',
                'Y-V Cross-Covariance (sklearn)',
                'Z-W Cross-Covariance (sklearn)']),
            np.array([
                '$X-U$ (sklearn)',
                '$Y-V$ (sklearn)',
                '$Z-W$ (sklearn)']),
            np.array([
                '$X-U$ Cross-Covariance (sklearn)',
                '$Y-V$ Cross-Covariance (sklearn)',
                '$Z-W$ Cross-Covariance (sklearn)']),
            np.array([False, False, False]),
            np.array([0.0, 0.0, 0.0]), 24),
        Metric(
            'cross_covariances_xyz_matrix_det_sklearn',
            'XYZ Cross-Covariance Matrix Determinant (sklearn)',
            'Det.${{}}_{{XUYVZW}}$ (sklearn)',
            'Determinant${{}}_{{XUYVZW}}$ (sklearn)',
            False, 0.0, 25),
        Metric(
            'cross_covariances_xyz_matrix_trace_sklearn',
            'XYZ Cross-Covariance Matrix Trace (sklearn)',
            'Trace${{}}_{{XUYVZW}}$ (sklearn)',
            'Trace${{}}_{{XUYVZW}}$ (sklearn)',
            False, 0.0, 26),

        # ξηζ position and velocity cross-covariances matrix, determinant and trace
        Metric(
            'cross_covariances_ξηζ',
            np.array([
                'ξ-vξ Cross-Covariance',
                'η-vη Cross-Covariance',
                'ζ-vζ Cross-Covariance']),
            np.array([
                '$ξ^\\prime-\\dot{ξ}^\\prime$',
                '$η^\\prime-\\dot{η}^\\prime$',
                '$ζ^\\prime-\\dot{ζ}^\\prime$']),
            np.array([
                '$ξ^\\prime-\\dot{ξ}^\\prime$ Cross-Covariance',
                '$η^\\prime-\\dot{η}^\\prime$ Cross-Covariance',
                '$ζ^\\prime-\\dot{ζ}^\\prime$ Cross-Covariance']),
            np.array([False, False, False]),
            np.array([0.63, 0.58, 10.31]), 27),
        Metric(
            'cross_covariances_ξηζ_matrix_det',
            'ξηζ Cross-Covariance Matrix Determinant',
            'Det.${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$',
            'Determinant${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$',
            False, 0.30, 28),
        Metric(
            'cross_covariances_ξηζ_matrix_trace',
            'ξηζ Cross-Covariance Matrix Trace',
            'Trace${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$',
            'Trace${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$',
            False, 0.01, 29),

        # ξηζ position and velocity robust cross-covariances matrix, determinant and trace
        Metric(
            'cross_covariances_ξηζ_robust',
            np.array([
                'ξ-vξ Cross-Covariance (robust)',
                'η-vη Cross-Covariance (robust)',
                'ζ-vζ Cross-Covariance (robust)']),
            np.array([
                '$ξ^\\prime-\\dot{ξ}^\\prime$ (robust)',
                '$η^\\prime-\\dot{η}^\\prime$ (robust)',
                '$ζ^\\prime-\\dot{ζ}^\\prime$ (robust)']),
            np.array([
                '$ξ^\\prime-\\dot{ξ}^\\prime$ Cross-Covariance (robust)',
                '$η^\\prime-\\dot{η}^\\prime$ Cross-Covariance (robust)',
                '$ζ^\\prime-\\dot{ζ}^\\prime$ Cross-Covariance (robust)']),
            np.array([False, False, False]),
            np.array([0.52, 0.51, 6.16]), 30),
        Metric(
            'cross_covariances_ξηζ_matrix_det_robust',
            'ξηζ Cross-Covariance Matrix Determinant (robust)',
            'Det.${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (robust)',
            'Determinant${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (robust)',
            False, 0.30, 31),
        Metric(
            'cross_covariances_ξηζ_matrix_trace_robust',
            'ξηζ Cross-Covariance Matrix Trace (robust)',
            'Trace${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (robust)',
            'Trace${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (robust)',
            False, 0.14, 32),

        # ξηζ position and velocity sklearn cross-covariances matrix, determinant and trace
        Metric(
            'cross_covariances_ξηζ_sklearn',
            np.array([
                'ξ-vξ Cross-Covariance (sklearn)',
                'η-vη Cross-Covariance (sklearn)',
                'ζ-vζ Cross-Covariance (sklearn)']),
            np.array([
                '$ξ^\\prime-\\dot{ξ}^\\prime$ (sklearn)',
                '$η^\\prime-\\dot{η}^\\prime$ (sklearn)',
                '$ζ^\\prime-\\dot{ζ}^\\prime$ (sklearn)']),
            np.array([
                '$ξ^\\prime-\\dot{ξ}^\\prime$ Cross-Covariance (sklearn)',
                '$η^\\prime-\\dot{η}^\\prime$ Cross-Covariance (sklearn)',
                '$ζ^\\prime-\\dot{ζ}^\\prime$ Cross-Covariance (sklearn)']),
            np.array([False, False, False]),
            np.array([0.0, 0.0, 0.0]), 33),
        Metric(
            'cross_covariances_ξηζ_matrix_det_sklearn',
            'ξηζ Cross-Covariance Matrix Determinant (sklearn)',
            'Det.${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (sklearn)',
            'Determinant${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (sklearn)',
            False, 0.0, 34),
        Metric(
            'cross_covariances_ξηζ_matrix_trace_sklearn',
            'ξηζ Cross-Covariance Matrix Trace (sklearn)',
            'Trace${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (sklearn)',
            'Trace${{}}_{{ξ^\\prime \\dot{ξ}^\\prime η^\\prime \\dot{η}^\\prime ζ^\\prime \\dot{ζ}^\\prime}}$ (sklearn)',
            False, 0.0, 35),

        # xyz median absolution deviation
        Metric(
            'mad_xyz',
            np.array(['X MAD', 'Y MAD', 'Z MAD']),
            np.array([
                'MAD${{}}_{{X}}$',
                'MAD${{}}_{{Y}}$',
                'MAD${{}}_{{Z}}$']),
            np.array([
                'MAD${{}}_{{X}}$',
                'MAD${{}}_{{Y}}$',
                'MAD${{}}_{{Z}}$']),
            np.array([False, False, False]),
            np.array([0.93, 0.28, 0.13]), 36),
        Metric(
            'mad_xyz_total', 'Total XYZ MAD',
            'MAD${{}}_{{XYZ}}$', 'MAD${{}}_{{XYZ}}$',
            False, 0.40, 37),

        # ξηζ median absolution deviation
        Metric(
            'mad_ξηζ',
            np.array(['ξ MAD', 'η MAD', 'ζ MAD']),
            np.array([
                'MAD${{}}_{{ξ^\\prime}}$',
                'MAD${{}}_{{η^\\prime}}$',
                'MAD${{}}_{{ζ^\\prime}}$']),
            np.array([
                'MAD${{}}_{{ξ^\\prime}}$',
                'MAD${{}}_{{η^\\prime}}$',
                'MAD${{}}_{{ζ^\\prime}}$']),
            np.array([False, False, False]),
            np.array([0.53, 0.84, 0.13]), 38),
        Metric(
            'mad_ξηζ_total', 'Total ξηζ MAD',
            'MAD${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$',
            'MAD${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$',
            False, 0.48, 39),

        # xyz Mahalanobis distance mean
        Metric(
            'mahalanobis_xyz_mean', 'Mahalanobis XYZ Mean',
            'MD${{}}_{{XYZ}}$ Mean', 'MD${{}}_{{XYZ}}$ Mean',
            False, 0.0, 46),

        # xyz Mahalanobis distance median
        Metric(
            'mahalanobis_xyz_median', 'Mahalanobis XYZ Median',
            'MD${{}}_{{XYZ}}$ Median', 'MD${{}}_{{XYZ}}$ Median',
            False, 0.0, 47),

        # ξηζ Mahalanobis distance mean
        Metric(
            'mahalanobis_ξηζ_mean', 'Mahalanobis ξηζ Mean',
            'MD${{}}_{{ξηζ}}$ Mean', 'MD${{}}_{{ξηζ}}$ Mean',
            False, 0.0, 48),

        # ξηζ Mahalanobis distance median
        Metric(
            'mahalanobis_ξηζ_median', 'Mahalanobis ξηζ Median',
            'MD${{}}_{{ξηζ}}$ Median', 'MD${{}}_{{ξηζ}}$ Median',
            False, 0.0, 49),

        # xyz minimum spanning tree average branch length and median absolute deviation
        Metric(
            'mst_xyz_mean', 'XYZ MST Mean',
            'MST${{}}_{{XYZ}}$ Mean', 'MST${{}}_{{XYZ}}$ Mean',
            False, 0.26, 40),
        Metric(
            'mst_xyz_mean_robust', 'XYZ MST Mean (robust)',
            'MST${{}}_{{XYZ}}$ Mean (robust)', 'MST${{}}_{{XYZ}}$ Mean (robust)',
            False, 0.31, 41),
        Metric(
            'mst_xyz_mad', 'XYZ MST MAD',
            'MST${{}}_{{XYZ}}$ MAD', 'MST${{}}_{{XYZ}}$ MAD',
            False, -0.04, 42),

        # ξηζ minimum spanning tree average branch length and median absolute deviation
        Metric(
            'mst_ξηζ_mean', 'ξηζ MST Mean',
            'MST${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ Mean',
            'MST${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ Mean',
            False, 0.26, 43),
        Metric(
            'mst_ξηζ_mean_robust', 'ξηζ MST Mean (robust)',
            'MST${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ Mean (robust)',
            'MST${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ Mean (robust)',
            False, 0.33, 44),
        Metric(
            'mst_ξηζ_mad', 'ξηζ MST MAD',
            'MST${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ MAD',
            'MST${{}}_{{ξ^\\prime η^\\prime ζ^\\prime}}$ MAD',
            False, 0.0, 45))}

    class Parameter():
        """Components of a configuration parameter."""

        # Default components
        default_components = {component: None for component in
            ('label', 'name', 'values', 'units', 'system', 'axis', 'origin')}

        def __init__(self, **components):
            """ Initializes a Parameter object with the given 'components'. """

            # Initialization
            vars(self).update(deepcopy(self.default_components))

            # Update
            self.update(components.copy())

        def update(self, parameter):
            """
            Updates the components of 'self' with those of another 'parameter' or a dictionary
            of components, only if those new components are part of the default components
            tuple or singular forms of default components.
            """

            # Parameter conversion into a dictionary
            if type(parameter) == type(self):
                parameter = vars(parameter)

            # Check if the parameter is a dictionary here
            stop(
                type(parameter) != dict, 'TypeError',
                "A parameter must be a Config.Parameter object or a dictionary ({} given).",
                type(parameter)
            )

            # Component conversion from singular to plural form
            for component in ('value', 'unit'):
                components = component + 's'
                if component in parameter.keys():
                    parameter[components] = parameter[component]
                    parameter.pop(component)

            # Component conversion from observables to standard form
            if 'axis' in parameter.keys() and parameter['axis'] is not None:
                if type(parameter['axis']) == str and parameter['axis'].lower() == 'observables':
                    parameter['axis'] = 'equatorial'
            if 'origin' in parameter.keys() and parameter['origin'] is not None:
                if type(parameter['origin']) == str and parameter['origin'].lower() == 'observables':
                    parameter['origin'] = 'sun'

            # Parameter update if present in self.default_components
            if type(parameter) == dict:
                vars(self).update({
                    key: component for key, component in parameter.items()
                    if key in self.default_components
                })

        def __repr__(self):
            """Returns a string with all the components of the parameter."""

            return '({})'.format(
                ', '.join(['{}: {}'.format(key, value) for key, value in vars(self).items()])
            )

    # Null parameters
    null_position = dict(
        values=(0.0, 0.0, 0.0), system='cartesian', axis='galactic', origin='sun',
        units=tuple(variable.unit.label for variable in systems['cartesian'].position)
    )
    null_velocity = dict(
        values=(0.0, 0.0, 0.0), system='cartesian', axis='galactic', origin='sun',
        units=tuple(variable.unit.label for variable in systems['cartesian'].velocity)
    )
    null_time = dict(units=System.default_units['time'].label)

    # Default parameters
    default_parameters = {
        parameter.label: parameter for parameter in (
            Parameter(label='name', name='Name'),
            Parameter(label='file_path', name='Series path'),
            Parameter(label='from_data', name='From data', values=False),
            Parameter(label='from_model', name='From model', values=False),
            Parameter(label='from_file', name='From file', values=False),
            Parameter(label='to_file', name='To file', values=False),
            Parameter(label='size_metrics', name='Association Size Metrics', values=True),
            Parameter(label='cov_metrics', name='Covariance Metrics', values=True),
            Parameter(label='cov_robust_metrics', name='Covariance Robust Metrics', values=True),
            Parameter(label='cov_sklearn_metrics', name='Covariance Sklearn Metrics', values=True),
            Parameter(label='mad_metrics', name='Median Absolute Deviation Metrics', values=True),
            Parameter(label='mst_metrics', name='Minimum Spanning Tree Metrics', values=True),
            Parameter(label='number_of_groups', name='Number of groups', values=1),
            Parameter(label='number_of_steps', name='Number of steps', values=1),
            Parameter(label='number_of_stars', name='Number of star'),
            Parameter(label='initial_time', name='Initial time', values=0.0, **null_time),
            Parameter(label='final_time', name='Final time', **null_time),
            Parameter(label='age', name='Age', **null_time),
            Parameter(label='position', name='Average position', **null_position),
            Parameter(label='position_error', name='Average position error', **null_position),
            Parameter(label='position_scatter', name='Average position scatter', **null_position),
            Parameter(label='velocity', name='Average velocity', **null_velocity),
            Parameter(label='velocity_error', name='Average velocity error', **null_velocity),
            Parameter(label='velocity_scatter', name='Average velocity scatter', **null_velocity),
            Parameter(label='data', name='Data', system='cartesian', axis='galactic', origin='sun'),
            Parameter(label='data_errors', name='Data errors', values=False),
            Parameter(
                label='rv_shift', name='Radial velocity shift', values=0.0,
                units=System.default_units['speed'].label, system='cartesian'
            ),
            Parameter(label='data_rv_shifts', name='Data radial velocity shifts', values=False),
            Parameter(label='jackknife_number', name='Jack-knife number', values=1),
            Parameter(label='jackknife_fraction', name='Jack-knife fraction', values=1.0, units=''
            ),
            Parameter(
                label='mst_fraction', name='Minimum spanning tree fraction', values=1.0, units=''
            ),
            Parameter(label='cutoff', name='Cutoff', values=None),
            Parameter(label='sample', name='Sample', values=None),
            Parameter(label='potential', name='Galactic potential', values=None),
            Parameter(label='pca', name='Principal Component Analysis', values=False)
        )
    }

    # Position and velocity paramaters
    position_parameters = ('position', 'position_error', 'position_scatter')
    velocity_parameters = ('velocity', 'velocity_error', 'velocity_scatter')

    def __init__(self, parent=None, path=None, args=False, **parameters):
        """
        Configures a Config objects from, in order, 'parent', an existing Config object,
        'path', a string representing a path to a configuration file, 'args' a boolean value
        that sets whether command line arguments are used, and '**parameters', a dictionary
        of dictionaries, where keys must match values in Parameter.default_components, or
        Config.Parameter objects. Only values that match a key in self.default_parameters are
        used. If no value are given the default parameter is used instead.
        """

        # Default or parent's parameters import
        if parent is None:
            self.initialize_from_parameters(deepcopy(self.default_parameters))
        elif type(parent) == Config:
            self.initialize_from_parameters(deepcopy(vars(parent)))
        else:
            stop(
                True, 'TypeError',
                "'parent' can either be a Config object or None ({} given).", type(parent)
            )

        # Parameters import
        if path is not None:
            self.initialize_from_path(path)
        if args:
            self.initialize_from_arguments(args)
        if len(parameters) > 0:
            self.initialize_from_parameters(parameters)

    def initialize_from_path(self, config_path):
        """
        Initializes a Config object from a configuration file located at 'config_path', and
        checks for NameError, TypeError and ValueError exceptions. 'config_path' can be an
        absolute path or relative to the current working directory.
        """

        # Check if config_path is a string
        stop(
            type(config_path) != str, 'TypeError',
            "The path to the configuration file must be a string ('{}' given).", type(config_path)
        )

        # Absolute path
        config_path = directory(collection.base_dir, config_path, 'config_path')

        # Check if the configuration file exists
        stop(
            not path.exists(config_path), 'FileNotFoundError',
            "No configuration file located at '{}'.", config_path
        )

        # Check if the configuration is a Python file
        stop(
            path.splitext(config_path)[1] != '.py', 'TypeError',
            "'{}' is not a Python file. (with a .py extension)", path.basename(config_path)
        )

        # Configuration file import
        from importlib.util import spec_from_file_location, module_from_spec
        try:
            spec = spec_from_file_location(
                path.splitext(path.basename(config_path))[0], config_path
            )
            parameters = module_from_spec(spec)
            vars(parameters).update(vars(self))
            spec.loader.exec_module(parameters)

        # Check if all names are valid
        except NameError as error:
            stop(
                True, 'NameError',
                "{}, only values in 'default_parameters' are configurable.", error.args[0]
            )

        # Parameters import
        self.initialize_from_parameters(vars(parameters))

    def initialize_from_arguments(self, args):
        """
        Parses arguments from the commmand line, creates an arguments object and adds these
        new values to the Config object. Also checks if 'args' is a boolean value. Overwrites
        values given in a configuration file.
        """

        # Check if 'args' is a boolean value
        stop(
            type(args) != bool, 'TypeError',
            "'args' must be a boolean value ({} given).", type(args)
        )

        # Arguments parsing
        from argparse import ArgumentParser
        parser = ArgumentParser(
            prog='kanya',
            description='traces given or simulated moving groups of stars back to their origin.'
        )
        parser.add_argument(
            '-n', '--name', action='store', type=str,
            help='name of the series of tracebacks.'
        )
        parser.add_argument(
            '-d', '--data', action='store_true',
            help='use the data parameter in the configuration file as input.'
        )
        parser.add_argument(
            '-m', '--model', action='store_true',
            help='model an input based on simulation parameters in the configuration file.'
        )
        parser.add_argument(
            '-l', '--from_file', action='store_true',
            help='load the input data from a file.'
        )
        parser.add_argument(
            '-s', '--to_file', action='store_true',
            help='save the output data to a file.'
        )
        args = parser.parse_args()

        # Series name import if not None
        if args.name is not None:
            self.name.values = args.name

        # Mode import, overwrites any value imported from a path
        self.from_data.values = args.data
        self.from_model.values = args.model
        self.from_file.values = args.from_file
        self.to_file.values = args.to_file

    def initialize_from_parameters(self, parameters):
        """
        Initializes a Config object from a parameters dictionary. Overwrites values given in
        a configuration file or as arguments in command line. Values in the dictionary can
        either be Parameter objects or dictionaries of components.
        """

        # Filter parameters that don't match a default parameter
        for key, parameter in filter(
                lambda item: item[0] in self.default_parameters.keys(), parameters.items()
            ):

            # Parameter update from a Parameter object or dictionary
            if key in vars(self).keys() and type(vars(self)[key]) == Config.Parameter:
                vars(self)[key].update(parameter)

            # Parameter object import from default parameters or parent configuration
            elif type(parameter) == Config.Parameter:
                vars(self)[key] = parameter

    def __repr__(self):
        """Returns a string of name of the configuration."""

        return '{} Configuration'.format(self.name.values)
