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

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'


class System():
    """ Defines a Coordinate system object with Variable objects.
    """
    def __init__(self, name: str, index: int, position: tuple, velocity: tuple):
        """ Initializes a System from two tuples with 3 Variables objects, position and velocity.
        """
        # Name and index
        self.name = name
        self.index = index

        # Values
        self.position = [self.variables[label] for label in position]
        self.velocity = [self.variables[label] for label in velocity]

        # Errors
        self.position_error = [self.variables['Δ' + label] for label in position]
        self.velocity_error = [self.variables['Δ' + label] for label in velocity]

    class Unit():
        """ Defines default units per physical type, and their label and name.
        """
        def __init__(self, label, name, unit):
            """ Initializes a Unit from name, label and an Astropy Units object.
            """
            self.label = label
            self.name = name
            self.unit = unit
            self.physical_type = unit.physical_type

    # Default units per physical type dictionary
    default_units = {
        'time': Unit('Myr', 'megayear', un.Myr),
        'length': Unit('pc', 'parsec', un.pc),
        'speed': Unit('pc/Myr', 'parsec per megayear', un.pc/un.Myr),
        'angle': Unit('rad', 'radian', un.rad),
        'angular_speed': Unit('rad/Myr', 'radian per megayear', un.rad/un.Myr)}

    class Axis():
        """Defines a Coordinate system axis."""
        def __init__(self, name, index):
            """Initializes an Axis object."""
            # Name and index
            self.name = name
            self.index = index

    # Coordinate system axis
    axis = {axis.name: axis for axis in (Axis('galactic', 0), Axis('equatorial', 1))}

    class Variable():
        """ Defines a Variable object and required variables and default units per physical type."""
        def __init__(self, label, name, unit):
            """ Initializes a Variable from a name, label and Unit object.
            """
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
        Variable('μδ', 'declination proper motion', default_units['angular_speed']),
        Variable('μα', 'right ascension proper motion', default_units['angular_speed']),
        # Observable coordinates system variables
        Variable('p', 'paralax', default_units['angle']),
        Variable('μα_cos_δ', 'right ascension proper motion * cos(declination)',
            default_units['angular_speed'])
    )}

    # Error variables creation
    for label, variable in variables.copy().items():
        variables['Δ' + label] = Variable(
            'Δ' + label, variable.name + ' error', variable.unit)

# Coordinates system variables !!! Put in Config !!!
systems = {system.name: system for system in (
    System('cartesian', 0, ('x', 'y', 'z'), ('u', 'v', 'w')),
    System('spherical', 1, ('r', 'δ', 'α'), ('rv', 'μδ', 'μα')),
    System('observables', 2, ('p', 'δ', 'α'), ('rv', 'μδ', 'μα_cos_δ')))}

class Config():
    """ Contains the parameters imported from a 'config.py' file (which must be a Python file), a
        dictionary of values or another Config objects, and related methods and a dictionary of
        default values. A config object can then be used as the input of a Series object. The
        default units are:

            time: million year (Myr)
            position: parsec (pc)
            velocity: parsec per million year (pc/Myr)
            angle: radian (rad)
            angular velocity: (rad/Myr)
    """
    # Parameters default values
    default_parameters = {
        'name': None,
        'to_database': False,
        'from_data': False,
        'from_simulation': False,
        'output_dir': '',
        'logs_dir': 'Logs',
        'db_path': None,
        'number_of_groups': 1,
        'number_of_steps': None,
        'initial_time': 0.0,
        'final_time': None,
        'number_of_stars': None,
        'age': None,
        'avg_position': (0.0, 0.0, 0.0),
        'avg_position_error': (0.0, 0.0, 0.0),
        'avg_position_scatter': None,
        'avg_velocity': None,
        'avg_velocity_error': (0.0, 0.0, 0.0),
        'avg_velocity_scatter': None,
        'system': None,
        'axis': None,
        'data': None
    }

    # Label names
    names = {
        'r': 'distance',
        'p': 'paralax',
        'δ': 'declination',
        'α': 'right ascension',
        'rv': 'radial velocity',
        'μδ': 'declination proper motion',
        'μα': 'right ascension proper motion',
        'μα_cos_δ': 'right ascension proper motion * cos(declination)',
        'x': 'x position',
        'y': 'y position',
        'z': 'z position',
        'u': 'u velocity',
        'v': 'v velocity',
        'w': 'w velocity'
    }

    # Label physical types
    physical_types = {
        'r': 'length',
        'p': 'angle',
        'δ': 'angle',
        'α': 'angle',
        'rv': 'speed',
        'μδ': 'angular speed',
        'μα': 'angular speed',
        'μα_cos_δ': 'angular speed',
        'x': 'length',
        'y': 'length',
        'z': 'length',
        'u': 'speed',
        'v': 'speed',
        'w': 'speed'
    }

    # Default units per physical type: speed and angular speed must match time, lengh and angle
    default_units = {
        'time': un.Myr,
        'length': un.pc,
        'speed': (un.pc/un.Myr),
        'angle': un.rad,
        'angular speed': (un.rad/un.Myr)
    }

    # Coordinates system labels
    systems = {
        'cartesian':   (('x', 'y', 'z'), ('u', 'v', 'w')),
        'spherical':   (('r', 'δ', 'α'), ('rv', 'μδ', 'μα')),
        'observables': (('p', 'δ', 'α'), ('rv', 'μδ', 'μα_cos_δ'))
    }

    def __init__(self, path=None, args=False, **parameters):
        """ Initializes a Config object from a configuration file, command line arguments and
            parameters, in that order. 'path' must be a string and 'args' a boolean value that
            causes arguments in command line to be imported if true. Only values that match a
            key in self.default_parameters are used. If no value are given the default parameter
            is used instead.
        """
        # Parameters import
        if path is not None:
            self.initialize_from_path(path)
        if args:
            self.initialize_from_arguments(args)
        if len(parameters) > 0:
            self.initialize_from_parameters(parameters)

        # Default parameters if none were given
        for key, default_value in self.default_parameters.items():
            if key not in vars(self):
                vars(self)[key] = default_value

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
            spec = spec_from_file_location(path.splitext(config_name)[0], config_path)
            parameters = module_from_spec(spec)
            spec.loader.exec_module(parameters)

        # Parameters import
        self.initialize_from_parameters(vars(parameters))

    def initialize_from_arguments(self, args):
        """ Parses arguments from the commmand line, creates an arguments object and adds these new
            values to the Config object. Also checks if 'args' is a boolean value. Overwrites
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

        # Series name import, if given
        self.name = args.name
        # Mode import, overwrites any value imported from a path
        self.to_database = args.to_database
        self.from_data = args.data
        self.from_simulation = args.simulation

    def initialize_from_parameters(self, parameters):
        """ Initializes a Config object from a parameters dictionary. Overwrites values given in
            a configuration file or as arguments in command line.
        """
        # Parameters import
        for key in self.default_parameters.keys():
            if key in parameters.keys():
                vars(self)[key] = parameters[key]

    def copy(self):
        """ Returns a copy of a Config object.
        """
        return Config(vars(config).copy())
