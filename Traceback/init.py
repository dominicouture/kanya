# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" init.py: Imports information from config.py, command line arguments and parameters into a
    Config object. This script must be run first to create a Series object.
"""

from copy import deepcopy
from Traceback.collection import *
from Traceback.coordinate import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Config():
    """ Contains the parameters imported from a configuration file (which must be a Python file),
        command line arguments, parameters in the __init__() function call or another Config
        object, as well as related methods, a Parameter class and a dictionary of default values.
        A config object can then be used as the input of a Series object.
    """

    class Indicator():
        """ Contains the data of a given size indicator. """

        def __init__(self, label, name, valid, order):
            """ Initializes an Indicator object. """

            self.label = label
            self.name = np.atleast_1d(name)
            self.valid = np.atleast_1d(valid)
            self.order = order

    # Size indicators
    indicators = {indicator.label: indicator for indicator in (

        # xyz median absolution deviation indicators
        Indicator('mad_xyz', np.array(['X MAD', 'Y MAD', 'Z MAD']), False, 26),
        Indicator('mad_xyz_total', 'Total XYZ MAD', False, 27),
        Indicator('mad_ξηζ', np.array(['ξ MAD', 'η MAD', 'ζ MAD']), False, 28),
        Indicator('mad_ξηζ_total', 'Total ξηζ MAD', False, 29),

        # xyz position covariance matrix, determinant and trace indicators
        Indicator(
            'covariances_xyz',
            np.array(['X Variance', 'Y Variance', 'Z Variance']),
            np.array([True, False, False]), 0),
        Indicator(
            'covariances_xyz_matrix_det',
            'XYZ Covariance Matrix Determinant', False, 2),
        Indicator(
            'covariances_xyz_matrix_trace',
            'XYZ Covariance Matrix Trace', False, 4),

        # ξηζ position covariance matrix, determinant and trace indicators
        Indicator(
            'covariances_ξηζ',
            np.array(['ξ Variance', 'η Variance', 'ζ Variance']),
            np.array([True, False, False]), 6),
        Indicator(
            'covariances_ξηζ_matrix_det',
            'ξηζ Covariance Matrix Determinant', False, 8),
        Indicator(
            'covariances_ξηζ_matrix_trace',
            'ξηζ Covariance Matrix Trace', False, 10),

        # xyz position and velocity cross covariance matrix, determinant and trace indicators
        Indicator(
            'cross_covariances_xyz',
            np.array(['X-U Cross Covariance', 'Y-V Cross Covariance', 'Z-W Cross Covariance']),
            np.array([False, False, False]), 12),
        Indicator(
            'cross_covariances_xyz_matrix_det',
            'XYZ Cross Covariance Matrix Determinant', False, 14),
        Indicator(
            'cross_covariances_xyz_matrix_trace',
            'XYZ Cross Covariance Matrix Trace', False, 16),

        # ξηζ position and velocity cross covariance matrix, determinant and trace indicators
        Indicator(
            'cross_covariances_ξηζ',
            np.array(['ξ-vξ Cross Covariance', 'η-vη Cross Covariance', 'ζ-vζ Cross Covariance']),
            np.array([False, False, False]), 18),
        Indicator(
            'cross_covariances_ξηζ_matrix_det',
            'ξηζ Cross Covariance Matrix Determinant', False, 20),
        Indicator(
            'cross_covariances_ξηζ_matrix_trace',
            'ξηζ Cross Covariance Matrix Trace', False, 22),

        # xyz position robust covariance matrix, determinant and trace indicators
        Indicator(
            'covariances_xyz_robust',
            np.array(['X Variance (robust)', 'Y Variance (robust)', 'Z Variance (robust)']),
            np.array([True, False, False]), 1),
        Indicator(
            'covariances_xyz_matrix_det_robust',
            'XYZ Covariance Matrix Determinant (robust)', False, 3),
        Indicator(
            'covariances_xyz_matrix_trace_robust',
            'XYZ Covariance Matrix Trace (robust)', False, 5),

        # ξηζ position robust covariance matrix, determinant and trace indicators
        Indicator(
            'covariances_ξηζ_robust',
            np.array(['ξ Variance (robust)', 'η Variance (robust)', 'ζ Variance (robust)']),
            np.array([True, False, False]), 7),
        Indicator(
            'covariances_ξηζ_matrix_det_robust',
            'ξηζ Covariance Matrix Determinant (robust)', False, 9),
        Indicator(
            'covariances_ξηζ_matrix_trace_robust',
            'ξηζ Covariance Matrix Trace (robust)', False, 11),

        # xyz position and velocity robust cross covariance matrix, determinant and trace indicators
        Indicator(
            'cross_covariances_xyz_robust',
            np.array(['X-U Cross Covariance (robust)', 'Y-V Cross Covariance (robust)',
                'Z-W Cross Covariance (robust)']),
            np.array([False, False, False]), 13),
        Indicator(
            'cross_covariances_xyz_matrix_det_robust',
            'XYZ Cross Covariance Matrix Determinant (robust)', False, 15),
        Indicator(
            'cross_covariances_xyz_matrix_trace_robust',
            'XYZ Cross Covariance Matrix Trace (robust)', False, 17),

        # ξηζ position and velocity robust cross covariance matrix, determinant and trace indicators
        Indicator(
            'cross_covariances_ξηζ_robust',
            np.array(['ξ-vξ Cross Covariance (robust)', 'η-vη Cross Covariance (robust)',
                'ζ-vζ Cross Covariance (robust)']),
            np.array([False, False, False]), 21),
        Indicator(
            'cross_covariances_ξηζ_matrix_det_robust',
            'ξηζ Cross Covariance Matrix Determinant (robust)', False, 23),
        Indicator(
            'cross_covariances_ξηζ_matrix_trace_robust',
            'ξηζ Cross Covariance Matrix Trace (robust)', False, 25),

        # ξηζ position sklearn covariance matrix, determinant and trace indicators
        Indicator(
            'covariances_ξηζ_sklearn',
            np.array(['ξ Variance (sklearn)', 'η Variance (sklearn)', 'ζ Variance (sklearn)']),
            np.array([False, False, False]), 7.5),
        Indicator(
            'covariances_ξηζ_matrix_det_sklearn',
            'ξηζ Covariance Matrix Determinant (sklearn)', False, 9.5),
        Indicator(
            'covariances_ξηζ_matrix_trace_sklearn',
            'ξηζ Covariance Matrix Trace (sklearn)', False, 11.5),

        # Minimum spanning tree average branch length indicators
        Indicator('mst_xyz_mean', 'XYZ MST Mean', False, 30),
        Indicator('mst_ξηζ_mean', 'ξηζ MST Mean', False, 31),

        # Minimum spanning tree robust average branch length indicators
        Indicator('mst_xyz_mean_robust', 'XYZ MST Mean (robust)', False, 32),
        Indicator('mst_ξηζ_mean_robust', 'ξηζ MST Mean (robust)', False, 33),

        # Minimum spanning tree branch length median absolute deviation indicators
        Indicator('mst_xyz_mad', 'XYZ MST MAD', False, 34),
        Indicator('mst_ξηζ_mad', 'ξηζ MST MAD', False, 35))}

    class Parameter():
        """ Contains the components of a given configuration parameter. """

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
            """ Updates the components of 'self' with those of another 'parameter' or a dictionary
                of components, only if those new components are part of the default components
                tuple or singular forms of default components.
            """

            # Parameter conversion into a dictionary
            if type(parameter) == type(self):
                parameter = vars(parameter)

            # Check if the parameter is a dictionary here
            stop(type(parameter) != dict, 'TypeError', "A parameter must be a Config.Parameter "
                "object or a dictionary ({} given).", type(parameter))

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
                vars(self).update({key: component for key, component in parameter.items() \
                    if key in self.default_components})

        def __repr__(self):
            """ Returns a string with all the components of the parameter. """

            return '({})'.format(', '.join(
                ['{}: {}'.format(key, value) for key, value in vars(self).items()]))

    # Null parameters
    null_position = dict(values=(0.0, 0.0, 0.0), system='cartesian', axis='galactic', origin='sun',
        units=tuple(variable.unit.label for variable in systems['cartesian'].position))
    null_velocity = dict(values=(0.0, 0.0, 0.0), system='cartesian', axis='galactic', origin='sun',
        units=tuple(variable.unit.label for variable in systems['cartesian'].velocity))
    null_time = dict(units=System.default_units['time'].label)

    # Default parameters
    default_parameters = {parameter.label: parameter for parameter in (
        Parameter(label='name', name='Name'),
        Parameter(label='file_path', name='Series path'),
        Parameter(label='from_data', name='From data', values=False),
        Parameter(label='from_model', name='From model', values=False),
        Parameter(label='from_file', name='From file', values=False),
        Parameter(label='to_file', name='To file', values=False),
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
        Parameter(label='rv_offset', name='Radial velocity offset', values=0.0,
            units=System.default_units['speed'].label, system='cartesian'),
        Parameter(label='data_rv_offsets', name='Data radial velocity offset', values=False),
        Parameter(label='jackknife_number', name='Jackknife number', values=1),
        Parameter(label='jackknife_fraction', name='Jackknife fraction', values=1.0, units=''),
        Parameter(label='mst_fraction', name='Minimum spanning tree fraction', values=1.0, units=''),
        Parameter(label='cutoff', name='Cutoff', values=None),
        Parameter(label='sample', name='Sample', values=None),
        Parameter(label='potential', name='Galactic potential', values=None))}

    # Position and velocity paramaters
    position_parameters = ('position', 'position_error', 'position_scatter')
    velocity_parameters = ('velocity', 'velocity_error', 'velocity_scatter')

    def __init__(self, parent=None, path=None, args=False, **parameters):
        """ Configures a Config objects from, in order, 'parent', an existing Config object,
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
            stop(True, 'TypeError', "'parent' can either be a Config object or None ({} given).",
                type(parent))

        # Parameters import
        if path is not None:
            self.initialize_from_path(path)
        if args:
            self.initialize_from_arguments(args)
        if len(parameters) > 0:
            self.initialize_from_parameters(parameters)

    def initialize_from_path(self, config_path):
        """ Initializes a Config object from a configuration file located at 'config_path', and
            checks for NameError, TypeError and ValueError exceptions. 'config_path' can be an
            absolute path or relative to the current working directory.
        """

        # Check if config_path is a string
        stop(type(config_path) != str, 'TypeError',
            "The path to the configuration file must be a string ('{}' given).", type(config_path))

        # Absolute path
        config_path = directory(collection.base_dir, config_path, 'config_path')

        # Check if the configuration file exists
        stop(not path.exists(config_path), 'FileNotFoundError',
            "No configuration file located at '{}'.", config_path)

        # Check if the configuration is a Python file
        stop(path.splitext(config_path)[1] != '.py', 'TypeError', "'{}' is not a Python file. "
            "(with a .py extension)", path.basename(config_path))

        # Configuration file import
        from importlib.util import spec_from_file_location, module_from_spec
        try:
            spec = spec_from_file_location(path.splitext(path.basename(config_path))[0], config_path)
            parameters = module_from_spec(spec)
            vars(parameters).update(vars(self))
            spec.loader.exec_module(parameters)

        # Check if all names are valid
        except NameError as error:
            stop(True, 'NameError', "{}, only values in 'default_parameters' are configurable.",
                error.args[0])

        # Parameters import
        self.initialize_from_parameters(vars(parameters))

    def initialize_from_arguments(self, args):
        """ Parses arguments from the commmand line, creates an arguments object and adds these
            new values to the Config object. Also checks if 'args' is a boolean value. Overwrites
            values given in a configuration file.
        """

        # Check if 'args' is a boolean value
        stop(type(args) != bool, 'TypeError', "'args' must be a boolean value ({} given).",
            type(args))

        # Arguments parsing
        from argparse import ArgumentParser
        parser = ArgumentParser(
            prog='Traceback',
            description='traces given or simulated moving groups of stars back to their origin.')
        parser.add_argument(
            '-n', '--name', action='store', type=str,
            help='name of the series of tracebacks.')
        parser.add_argument(
            '-d', '--data', action='store_true',
            help='use the data parameter in the configuration file as input.')
        parser.add_argument(
            '-m', '--model', action='store_true',
            help='model an input based on simulation parameters in the configuration file.')
        parser.add_argument(
            '-l', '--from_file', action='store_true',
            help='load the input data from a file.')
        parser.add_argument(
            '-s', '--to_file', action='store_true',
            help='save the output data to a file.')
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
        """ Initializes a Config object from a parameters dictionary. Overwrites values given in
            a configuration file or as arguments in command line. Values in the dictionary can
            either be Parameter objects or dictionaries of components.
        """

        # Filter parameters that don't match a default parameter
        for key, parameter in filter(
                lambda item: item[0] in self.default_parameters.keys(), parameters.items()):

            # Parameter update from a Parameter object or dictionary
            if key in vars(self).keys() and type(vars(self)[key]) == Config.Parameter:
                vars(self)[key].update(parameter)

            # Parameter object import from default parameters or parent configuration
            elif type(parameter) == Config.Parameter:
                vars(self)[key] = parameter

    def __repr__(self):
        """ Returns a string of name of the configuration. """

        return '{} Configuration'.format(self.name.values)
