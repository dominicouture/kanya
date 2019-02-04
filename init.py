# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" init.py: Imports information from config.py, handles exceptions (type, value, shape), handles
    unit conversions, checks for the presence of output and logs directories and creates them if
    necessary, recursively creates a list of configurations to be executed by the main Traceback
    algorithm. This script must be run first before the rest of the package.
"""

import numpy as np
from astropy import units as un
from time import strftime
from os import path, makedirs, remove, getcwd, chdir
from logging import basicConfig, info, warning, INFO
from importlib.util import spec_from_file_location, module_from_spec
from argparse import ArgumentParser

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Config():
    """ Contains the parameters imported from a 'config.py' file (which must be a Python file), a
        dictionary of values or another Config objects, and related methods and a dictionary of
        default values.

        A config object can then be used as the input of a series of Group objects. The number
        of groups defines how many objects are in a series in which all groups have the same
        Config object as their progenitor.
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
        'data': None,
        'representation': None,
        'system': None
    }

    default_units = {
        'time': un.Myr,
        'length': un.pc,
        'speed': (un.pc/un.Myr),
        'angle': un.rad,
        'angular speed': (un.rad/un.Myr)
    }

    def __init__(self, path=None, args=False, **parameters):
        """ Initializes a Config object from a configuration file, command line arguments and
            parameters, in that order. 'path' must be a string and 'args' a boolean value. Only
            values that match a key in self.default_parameters are used. If no value are given
            the default parameter is used instead.
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
            checks for NameError, TypeError and ValueError exceptions.
        """
        # Check the type of config_path
        if type(config_path) != str:
            raise TypeError("The path to the configuration file must be a string "
                "('{}' given.)".format(type(config_path)))

        # Check if the configuration file is present and a Python file
        abs_config_path = path.abspath(config_path)
        config_name = path.basename(config_path)
        if not path.exists(abs_config_path):
            raise FileNotFoundError(
                "No configuration file found at location '{}'.".format(abs_config_path))
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

class Series(list):
    """ Contains a series of Groups and their embedded Star objects and the methods required to
        initialize a series from a Config object. Values and data in the Config object are,
        copied, checked and converted into Quantity objects and default units:

            time: million year (Myr)
            position: parsec (pc)
            velocity: parsec per million year (pc/Myr)
            angle: radian (rad)
            angular velocity: (rad/Myr)
    """

    def __init__(self, config):
        """ Configures a Config object from database, data or simulation, and creates output and
            Logs directory if necessary.
        """
        # Copy of config values
        vars(self).update(vars(config).copy())

        # Check 'name' parameter
        if self.name is None:
            raise NameError("Required parameter 'name' is missing.")
        if type(self.name) != str:
            raise TypeError("'name' must be a string ('{}' given).".format(type(self.name)))

        # Output directory creation
        self.output_dir = self.create_directory(
            path.abspath(path.join(path.dirname(path.realpath(__file__)), '..')),
                self.output_dir, 'output_dir')

        # Logs configuration
        self.configure_logs()

        # Check mode
        for argument in ('to_database', 'from_data', 'from_simulation'):
            if type(vars(self)[argument]) != bool:
                self.stop("'{}' must be a boolean value ({} given).",
                    'TypeError', argument, type(vars(self)[argument]))
        # Check if traceback and output from both data and simulation
        if self.from_data and self.from_simulation:
            self.stop("Either traceback series '{}' from data or a simulation, not both.",
                'ValueError', self.name)
        # Set 'from_database' parameter
        self.from_database = True if not self.from_data and not self.from_simulation else False
        if self.from_database and self.to_database:
            info("Output of '{}' from and to database. "
                "The database will not be updated with its own data.".format(self.name))
            self.to_database = False

        # Database configuration
        if self.from_database or self.to_database:
            self.configure_database()

        # Traceback configuration
        if self.from_data or self.from_simulation:
            self.configure_traceback()
            # Data configuration
            if self.from_data:
                self.configure_data()
            # Simulation configuration
            if self.from_simulation:
                self.configure_simulation()

        # groups dictionary creation if necessary
        import __main__ as main
        if 'name' not in vars(main):
            vars(main)['groups'] = Groups()
        # Preivous entry deleted if it already exists
        elif self.name in main.groups.keys():
            info("Existing series '{}' deleted and replaced.".format(self.name))
        # Self addition to groups dictionary
        main.groups[self.name] = self
        info("Series '{}' ready for traceback.".format(self.name))

    def configure_logs(self):
        """ Checks if the logs directory already exists, creates it if necessary and configures
            the Logs file. 'logs_dir' is redefined as the absolution logs directory
        """
        # Logs directory creation
        self.logs_dir = self.create_directory(self.output_dir, self.logs_dir, 'logs_dir')

        # Logs configuration
        basicConfig(
            filename=path.join(
                self.logs_dir, '{}_{}.log'.format(self.name, strftime('%Y-%m-%d_%H-%M-%S'))),
            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

    def configure_database(self):
        """ Checks if the database directory and the database file itself exist and creates them
            if necessary. 'database_path' is redefined as the absolute path. The database Model
            is initiated and errors handled based on whether the input or output from and to the
            database is required.
        """
        # Creation or import of database
        if self.db_path is None:
            self.db_path = '{}.db'.format(self.name)
        self.db_name = '{}.db'.format(self.name) if path.basename(self.db_path) == '' \
            else path.basename(self.db_path)
        self.db_dir = self.create_directory(self.output_dir, path.dirname(self.db_path), 'db_path')
        self.db_path = path.join(self.db_dir, self.db_name) # Absolute path
        # Check if a database exists
        if self.from_database and not path.exists(self.db_path):
            self.stop("No existing database located at '{}'.", 'NameError', self.db_path)

        # Check if output from database is possible
        Config.db_path = self.db_path # Necessary for Database object definition
        from model import GroupModel
        if self.from_database:
            # Group names creation
            info("Output of '{}' from database.".format(self.name))
            self.groups = [group.name
                for group in GroupModel.select().where(GroupModel.series == self.name)]

            # Check if data is present in the database
            if len(self.groups) == 0:
                self.stop("No existing series '{}' in the database '{}'.",
                    'ValueError', self.name, self.db_name)

    def configure_traceback(self):
        """ Check configuration parameters for traceback and output from data or simulation.
        """
        # Check if all the necessary parameters are present
        for parameter in ('number_of_groups', 'number_of_steps', 'initial_time', 'final_time'):
            if vars(self)[parameter] is None:
                self.stop("Required traceback parameter '{}' is missing in the configuration.",
                    'NameError', parameter)

        # Check number_of_groups parameter
        if type(self.number_of_groups) not in (int, float):
            self.stop("'number_of_groups' must be an integer or a float ({} given).",
                'TypeError', type(self.number_of_groups))
        if self.number_of_groups % 1 != 0:
            self.stop("'number_of_groups' must be convertible to an integer ({} given).",
                'ValueError', self.number_of_groups)
        if not self.number_of_groups > 0:
            self.stop("'number_of_groups' must be greater than 0 ({} given).",
                'ValueError', self.number_of_groups)
        self.number_of_groups = int(self.number_of_groups)

        # Check number_of_steps parameter
        if type(self.number_of_steps) not in (int, float):
            self.stop("'number_of_steps' must be an integer or a float ({} given).",
                'TypeError', type(self.number_of_steps))
        if self.number_of_steps % 1 != 0:
            self.stop("'number_of_steps' must be convertible to an integer ({} given).",
                'ValueError', self.number_of_steps)
        if not self.number_of_steps > 0:
            self.stop("'number_of_steps' must be greater than 0 ({} given).",
                'ValueError', self.number_of_steps)
        self.number_of_steps = int(self.number_of_steps)

        # Check initial_time parameter
        if type(self.initial_time) not in (int, float):
            self.stop("'initial_time' must be an integer or a float ({} given).",
                'TypeError', type(initial_time))

        # Check final_time parameter
        if type(self.final_time) not in (int, float):
            self.stop("'final_time' must be an integer or a float ({} given).",
                'TypeError', type(self.final_time))
        if not self.final_time > self.initial_time:
            self.stop("'final_time' must be greater than initial_time ({} and {} given).",
             'ValueError', self.final_time, self.initial_time)

        # Group names creation
        self.groups = ['{}_{}'.format(self.name, i) for i in range(1, self.number_of_groups + 1)]

    def configure_data(self):
        """ Check if traceback and output from data is possible and creates a Data object from a
            CSV file, or a Python dictionary, list, tuple or np.ndarray.
        """
        info("Traceback and output of '{}' from data.".format(self.name))

        # Check if data is present
        if self.data is None:
            self.stop("No data provided for traceback '{}'.", 'NameError', self.name)

        # Data import
        from data import Data
        self.stars = Data(self.name, self.representation, self.data)

    def configure_simulation(self):
        """ Check if traceback and output from simulation is possible.
        """
        info("Traceback and output of '{}' from simulation.".format(self.name))

        # Check if all the necessary parameters are present
        for parameter in (
                'number_of_stars', 'age',
                'avg_position', 'avg_position_error', 'avg_position_scatter',
                'avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
            if vars(self)[parameter] is None:
                self.stop("Required simulation parameter '{}' is missing in the configuration.",
                    'NameError', parameter)

        # Check number_of_stars parameter
        if type(self.number_of_stars) not in (int, float):
            self.stop("'number_of_stars' must be an integer or a float ({} given).",
                'TypeError', type(self.number_of_stars))
        if self.number_of_stars % 1 != 0:
            self.stop("'number_of_stars' must be convertible to an integer ({} given).",
                'ValueError', self.number_of_stars)
        if not self.number_of_stars > 0:
            self.stop("'number_of_stars' must be greater than 0 ({} given).",
                'ValueError', self.number_of_stars)
        self.number_of_stars = int(self.number_of_stars)

        # Check age parameter
        if type(self.age) not in (int, float):
            self.stop("'age' must be an integer or a float ({} given).",
                'TypeError', type(self.age))
        if not self.age >= 0.0:
            self.stop("'age' must be greater than or equal to 0.0 ({} given).",
                'ValueError', self.age)

        # !!! Add check for 'avg_position', 'avg_position_error', 'avg_position_scatter', !!!
        # !!! 'avg_velocity', 'avg_velocity_error' and 'avg_velocity_scatter' here !!!

    def traceback(self):
        """ Creates Group and embeded Star objects for all group names in self.groups.
        """
        from group import Group
        for name in self.groups:
            message = "Tracing back {}.".format(name.replace('_', ' '))
            info(message)
            print(message)
            self.append(Group(name, self))

    def create_directory(self, base, directory, name):
        """ Checks for type errors, checks if the directory already exists, creates it if
            necessary and returns the absolute directory path. The directory can either be
            relative the base or an absolute.
        """
        # Check the type of directory
        if type(directory) is not str:
            raise TypeError("'{}' parameter must be a string ({} given).".format(
                name, type(directory)))
        # Output directory formatting
        working_dir = getcwd()
        chdir(path.abspath(base))
        abs_dir = path.abspath(directory)
        chdir(working_dir)
        # Output directory creation
        if not path.exists(abs_dir):
            makedirs(abs_dir)
        return abs_dir

    def stop(self, message, error, *values):
        """ Raises an exception and logs it into the log file and stops the execution.
        """
        message = message.format(*values)
        warning('{}: {}'.format(error, message))
        exec("""raise {}("{}")""".format(error, message))

class Groups(dict):
    """ Contains a dictionary of series created from a configuration as well a traceback method
        that computes a traceback for all or selected series.
    """

    def traceback(self, *series):
        """ Creates a series of Groups object for all series in self if no series name is given
            or selected series given in argument.
        """
        # Selects all series if none are provided
        selected_series = list(self.keys()) if len(series) == 0 else series

        # Computes a traceback for every groups in every selected series
        for series in selected_series:
            if series not in self.keys():
                Series.stop("Series '{}' does not exist.", "NameError", series)
            elif len(self[series]) == self[series].number_of_groups:
                info("Series '{}' has already been tracebacked.".format(series))
            else:
                self[series].traceback()
