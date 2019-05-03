# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" init.py: Imports information from config.py, command line arguments and parameters into a
    Config object. This script must be run first to create a Series object. It also defines
    variables and their default units and values.
"""

from astropy import units as un
from os import path
from importlib.util import spec_from_file_location, module_from_spec
from argparse import ArgumentParser
from copy import deepcopy

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class System():
    """ Defines a Coordinates system object with Variable objects. """

    def __init__(self, name: str, position: tuple, velocity: tuple):
        """ Initializes a System from position and velocity tuples with 3 Variables objects. """

        # Name and index
        self.name = name

        # Values
        self.position = [self.variables[label] for label in position]
        self.velocity = [self.variables[label] for label in velocity]

        # Errors
        self.position_error = [self.variables['Δ' + label] for label in position]
        self.velocity_error = [self.variables['Δ' + label] for label in velocity]

    class Unit():
        """ Defines default units per physical type, and their label and name. """

        def __init__(self, label, name=None):
            """ Initializes a Unit from name, label and an Astropy Units object.
                !!! System.Unit class must be used during unit conversion instead of unit
                label (input of Quantity ?) and un.Unit() calls. !!!
            """

            # Initialization
            self.label = label
            self.unit = un.Unit(label)
            self.physical_type = self.unit.physical_type

            # Unit name
            self.name = name if name is not None and type(name) == str else self.unit.to_string()

    # Default units per physical type dictionary
    default_units = {
        'time': Unit('Myr', 'megayear'),
        'length': Unit('pc', 'parsec'),
        'speed': Unit('pc/Myr', 'parsec per megayear'),
        'angle': Unit('rad', 'radian'),
        'angular speed': Unit('rad/Myr', 'radian per megayear')}

    class Axis():
        """ Defines a Coordinate system axis."""

        def __init__(self, name):
            """ Initializes an Axis object. """

            # Initialization
            self.name = name

    # Coordinate system axis
    axis = {axis.name: axis for axis in (Axis('galactic'), Axis('equatorial'))}

    class Origin(Axis):
        """ Defines a Coordinate system origin."""

        pass

    # Coordinate system axis
    origins = {origin.name: origin for origin in (Origin('sun'), Axis('galaxy'))}

    class Variable():
        """ Defines a Variable object and required variables from all systems. """

        def __init__(self, label, name, unit):
            """ Initializes a Variable from a name, label and Unit object. """

            # Initialization
            self.label = label
            self.name = name
            self.unit = unit
            self.physical_type = unit.physical_type

    # Variables dictionary
    variables = {variable.label: variable for variable in (

        # Castesian coordinate system variables
        Variable('x', 'y position', default_units['length']),
        Variable('y', 'y position', default_units['length']),
        Variable('z', 'z position', default_units['length']),
        Variable('u', 'u velocity', default_units['speed']),
        Variable('v', 'v velocity', default_units['speed']),
        Variable('w', 'w velocity', default_units['speed']),

        # Spherical coordinates system variables
        Variable('r', 'distance', default_units['length']),
        Variable('δ', 'declination', default_units['angle']),
        Variable('α', 'right ascension', default_units['angle']),
        Variable('rv', 'radial velocity', default_units['speed']),
        Variable('μδ', 'declination proper motion', default_units['angular speed']),
        Variable('μα', 'right ascension proper motion', default_units['angular speed']),

        # Observable coordinates system variables
        Variable('p', 'paralax', default_units['angle']),
        Variable('μαcosδ', 'right ascension proper motion * cos(declination)',
            default_units['angular speed']))}

    # Error variables creation
    for label, variable in variables.copy().items():
        variables['Δ' + label] = Variable(
            'Δ' + label, variable.name + ' error', variable.unit)

class Config():
    """ Contains the parameters imported from a configuration file (which must be a Python file),
        command line arguments, parameters in the __init__() function call or another Config
        object, related methods, a Parameter class and a dictionary of default values. A config
        object can then be used as the input of a Series object.
    """

    # Coordinates systems
    systems = {system.name: system for system in (
        System('cartesian', ('x', 'y', 'z'), ('u', 'v', 'w')),
        System('spherical', ('r', 'δ', 'α'), ('rv', 'μδ', 'μα')),
        System('observables', ('p', 'δ', 'α'), ('rv', 'μδ', 'μαcosδ')))}

    class Parameter():
        """ Contains the components of a given configuration parameter. """

        # Default components
        default_components = {component: None for component in
            ('label', 'name', 'values', 'units', 'system', 'axis', 'origin')}

        def __init__(self, **components):
            """ Initializes a Parameter object with the given components. """

            # Initialization
            vars(self).update(deepcopy(self.default_components))

            # Update
            self.update(components.copy())

        def update(self, parameter):
            """ Updates the components of self with those of another parameter or a dictionary
                of components, only if those new components are part of the components tuple.
            """

            # Parameter conversion into a dictionary
            if type(parameter) == type(self):
                parameter = vars(parameter)

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
        Parameter(label='to_database', name='To database', values=False),
        Parameter(label='from_data', name='From data', values=False),
        Parameter(label='from_simulation', name='From simulation', values=False),
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

    def __init__(self, path=None, args=False, parent=None, **parameters):
        """ Initializes a Config object from a configuration file, command line arguments
            and parameters, in that order. 'path' must be a string and 'args' a boolean value
            that causes arguments in command line to be imported if true. Only values that
            match a key in self.default_parameters are used. If no value are given the default
            parameter is used instead. Items in'parameters' dictionary must be dictionaries or
            Config.Parameter objects.
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
                "('{}' given.)".format(type(config_path)))

        # Check if the configuration file is present
        abs_config_path = path.abspath(config_path)
        config_name = path.basename(config_path)
        if not path.exists(abs_config_path):
            raise FileNotFoundError(
                "No configuration file found at location '{}'.".format(abs_config_path))

        # Check if the configuraiton is a Python file
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
            '-b', '--to_database', action='store_true',
            help='save the output data to a database file.')
        parser.add_argument(
            '-d', '--data', action='store_true',
            help='use data parameter in the configuration file as input.')
        parser.add_argument(
            '-s', '--simulation', action='store_true',
            help='simulate an input based on parameters in the configuration file.')
        parser.add_argument(
            'name', action='store', type=str,
            help='name of the series of tracebacks, used in the database and output.')
        args = parser.parse_args()

        # Series name import
        self.name.values = args.name

        # Mode import, overwrites any value imported from a path
        self.to_database.values = args.to_database
        self.from_data.values = args.data
        self.from_simulation.values = args.simulation

    def initialize_from_parameters(self, parameters):
        """ Initializes a Config object from a parameters dictionary. Overwrites values given in
            a configuration file or as arguments in command line. The values in the dictionary
            can either be a Parameter object or a dictionary of components.
        """

        # Import only if it matches a default parameter
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
