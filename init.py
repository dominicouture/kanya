# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" init.py: Imports information from config.py, command line arguments and parameters into a
    Config object. This script must be run first to create a Series object.
"""

from astropy import units as un
from os import path
from importlib.util import spec_from_file_location, module_from_spec
from argparse import ArgumentParser
from copy import deepcopy
from coordinate import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Config():
    """ Contains the parameters imported from a configuration file (which must be a Python file),
        command line arguments, parameters in the __init__() function call or another Config
        object, as well as related methods, a Parameter class and a dictionary of default values.
        A config object can then be used as the input of a Series object.
    """

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
            if type(parameter) != dict:
                raise TypeError("A parameter must be a Config.Parameter object or a dictionary. "
                    "('{}' given).".format(type(parameter)))

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
    null_time = dict(units=System.default_units['time'].label, system='cartesian')

    # Default parameters
    default_parameters = {parameter.label: parameter for parameter in (
        Parameter(label='name', name='Name'),
        Parameter(label='from_data', name='From data', values=False),
        Parameter(label='from_model', name='From model', values=False),
        Parameter(label='from_database', name='From database', values=False),
        Parameter(label='to_database', name='To database', values=False),
        Parameter(label='output_dir', name='Output directory', values=''),
        Parameter(label='logs_dir', name='Logs directory', values='Logs'),
        Parameter(label='db_path', name='Database path'),
        Parameter(label='number_of_groups', name='Number of groups', values=1),
        Parameter(label='number_of_steps', name='Number of steps', values=1),
        Parameter(label='number_of_stars', name='Number of star'),
        Parameter(label='initial_time', name='Initial time', values=0.0, **null_time),
        Parameter(label='final_time', name='Final time', **null_time),
        Parameter(label='age', name='Age', **null_time),
        Parameter(label='avg_position', name='Average position', **null_position),
        Parameter(label='avg_position_error', name='Average position error', **null_position),
        Parameter(label='avg_position_scatter', name='Average position scatter', **null_position),
        Parameter(label='avg_velocity', name='Average velocity', **null_velocity),
        Parameter(label='avg_velocity_error', name='Average velocity error', **null_velocity),
        Parameter(label='avg_velocity_scatter', name='Average velocity scatter', **null_velocity),
        Parameter(label='data', name='Data', system='cartesian', axis='galactic', origin='sun'),
        Parameter(label='rv_offset', name='Radial velocity offset', values=0.0,
            units=System.default_units['speed'].label, system='cartesian'))}

    # Position and velocity paramaters
    position_parameters = ('avg_position', 'avg_position_error', 'avg_position_scatter')
    velocity_parameters = ('avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter')

    def __init__(self, parent=None, path=None, args=False, **parameters):
        """ Configures a Config objects from, in order, 'parent', an existing Config object,
            'path', a string representing a path to a configuration file, 'args' a boolean value
            that sets whether command line arguments are used, and **parameters, a dictionary of
            dictionaries or Config.Parameter objects. Only values that match a key in
            self.default_parameters are used. If no value are given the default parameter is used
            instead.
        """

        # Default or parent's parameters import
        if parent is None:
            self.initialize_from_parameters(deepcopy(self.default_parameters))
        elif type(parent) == Config:
            self.initialize_from_parameters(deepcopy(vars(parent)))

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

        # Check the type of config_path
        if type(config_path) != str:
            raise TypeError("The path to the configuration file must be a string "
                "('{}' given).".format(type(config_path)))

        # Check if the configuration file is present
        abs_config_path = path.abspath(config_path)
        config_name = path.basename(config_path)
        if not path.exists(abs_config_path):
            raise FileNotFoundError(
                "No configuration file found at location '{}'.".format(abs_config_path))

        # Check if the configuration is a Python file
        elif path.splitext(config_name)[1] != '.py':
            raise TypeError("'{}' is not a Python file.".format(config_name))

        # Configuration file import
        else:
            try:
                spec = spec_from_file_location(path.splitext(config_name)[0], config_path)
                parameters = module_from_spec(spec)
                vars(parameters).update(vars(self))
                spec.loader.exec_module(parameters)
            except NameError as error:
                error.args = ( "{}, only values in 'default_parameters' are configurable".format(
                    error.args[0]),)
                raise

        # Parameters import
        self.initialize_from_parameters(vars(parameters))

    def initialize_from_arguments(self, args):
        """ Parses arguments from the commmand line, creates an arguments object and adds these
            new values to the Config object. Also checks if 'args' is a boolean value. Overwrites
            values given in a configuration file.
        """

        # Check if 'args' is a boolean value
        if type(args) != bool:
            raise TypeError("'args' must be a boolean value ('{}' given).".format(type(args)))

        # Arguments parsing
        parser = ArgumentParser(
            prog='Traceback',
            description='traces given or simulated moving groups of stars back to their origin.')
        parser.add_argument(
            '-n', '--name', action='store', type=str,
            help='name of the series of tracebacks, used in the database and output.')
        parser.add_argument(
            '-d', '--data', action='store_true',
            help='use the data parameter in the configuration file as input.')
        parser.add_argument(
            '-m', '--model', action='store_true',
            help='model an input based on simulation parameters in the configuration file.')
        parser.add_argument(
            '-l', '--from_database', action='store_true',
            help='load the input data from a database file.')
        parser.add_argument(
            '-s', '--to_database', action='store_true',
            help='save the output data to a database file.')
        args = parser.parse_args()

        # Series name import if not None
        if args.name is not None:
            self.name.values = args.name

        # Mode import, overwrites any value imported from a path
        self.from_data.values = args.data
        self.from_model.values = args.model
        self.from_database.values = args.from_database
        self.to_database.values = args.to_database

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
