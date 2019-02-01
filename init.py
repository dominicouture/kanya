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
        default values. Values are converted into Quantity objects and default units:

            time: million year (Myr)
            position: parsec (pc)
            velocity: parsec per million year (pc/Myr)
            angle: radian (rad)
            angular velocity: (rad/Myr)

        A config object can then be used as the input of a series of Group objects. The number
        of groups defines how many objects are in a series in which all groups have the same
        Config object as their progenitor.
    """
    # Arguments default values
    default_arguments = {
        'to_database': False,
        'from_data': False,
        'from_simulation': False,
        'series': None
    }

    # Parameters default values
    default_parameters = {
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
        'representation': None
    }

    default_units = {
        'time': un.Myr,
        'length': un.pc,
        'speed': (un.pc/un.Myr),
        'angle': un.rad,
        'angular speed': (un.rad/un.Myr)
    }

    def __init__(self, parameters):
        """ Initializes a Config object from a configuration file, a Python dictionary or another
            Config object.
        """
        if type(parameters) == str:
            self.init_from_path(parameters)
        elif type(parameters) == dict:
            self.init_from_parameters(parameters)
        elif type(parameters) == Config:
            self.init_from_Config(parameters)
        else:
            raise TypeError("Config cannot be initialized from '{}'.".format(parameters))

    def init_from_path(self, config_path):
        """ Initializes a Config object from a configuration file located at 'config_path' with
            default values if necessary and arguments in the command line. Also checks for
            NameError, TypeError and ValueError exceptions.
        """
        # Arguments import
        self.import_arguments()

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
            config = module_from_spec(spec)
            spec.loader.exec_module(config)
        # Parameters dictionary creation
        for value in self.default_parameters.keys():
            if value in vars(config):
                vars(self)[value] = vars(config)[value]
            else:
                vars(self)[value] = self.default_parameters[value]

        # Config configuration
        self.configure_Config()

    def init_from_parameters(self, parameters):
        """ Initializes a Config object from a Python dictionary with default values if necessary.
        """
        # Arguments import
        for value in self.default_arguments.keys():
            if value in parameters:
                vars(self)[value] = parameters[value]
            else:
                vars(self)[value] = self.default_parameters[value]
        self.from_database = True if not self.from_data and not self.from_simulation else False

        # Parameters dictionary creation
        for value in self.default_parameters.keys():
            if value in parameters:
                vars(self)[value] = parameters[value]
            else:
                vars(self)[value] = self.default_parameters[value]

        # Config configuration
        self.configure_Config()

    def init_from_Config(self, config):
        """ Creates a copy of a Config object.
        """
        vars(self).update(vars(config).copy())

    def import_arguments(self):
        """ Parses arguments from the commmand line, creates an arguments object, checks for errors
            and adds new values to Config object.
        """
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
            'series', action='store', type=str,
            help='series of the series of tracebacks, used in the database and output.')
        # Arguments object creation
        args = parser.parse_args()
        self.to_database = args.to_database
        self.from_database = True if not args.data and not args.simulation else False
        self.from_data = args.data
        self.from_simulation = args.simulation
        self.series = args.series

    def configure_Config(self):
        """ Configures a Config object from database, data or simulation, and creates output and
            Logs directory if necessary.
        """
        # Output directory creation
        self.output_dir = self.create_directory(
            path.abspath(path.join(path.dirname(path.realpath(__file__)), '..')),
                self.output_dir, 'output_dir')
        # Logs configuration
        self.configure_logs()

        # Check arguments
        # !!! Add check if arguments are present and compatible value !!!
        for argument in ('to_database', 'from_data', 'from_simulation'):
            if type(vars(self)[argument]) != bool:
                self.stop("'{}' must be a boolean value ({} given).",
                    'TypeError', argument, type(vars(self)[argument]))
        # Check if traceback and output from both data and simulation
        if self.from_data and self.from_simulation:
            self.stop("Either traceback series '{}' from data or a simulation, not both.",
                'ValueError', self.series)
        # Check series name
        if self.series is None:
            raise NameError("Required argument 'series' is missing.")
        if type(self.series) != str:
            raise TypeError("'series' must be a string ('{}' given).".format(type(self.series)))

        # Database configuration
        if self.from_database or self.to_database:
            self.configure_database()
        # Data configuration
        if self.from_data:
            self.configure_data()
        # Simulation configuration
        if self.from_simulation:
            self.configure_simulation()

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

    def configure_logs(self):
        """ Checks if the logs directory already exists, creates it if necessary and configures
            the Logs file. 'logs_dir' is redefined as the absolution logs directory
        """
        # Logs directory creation
        self.logs_dir = self.create_directory(self.output_dir, self.logs_dir, 'logs_dir')
        # Logs configuration
        basicConfig(
            filename=path.join(
                self.logs_dir, '{}_{}.log'.format(self.series, strftime('%Y-%m-%d_%H-%M-%S'))),
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S -',
            level=INFO
        )

    def configure_database(self):
        """ Checks if the database directory and the database file itself exist and creates them
            if necessary. 'database_path' is redefined as the absolute path. The database Model
            is initiated and errors handled based on whether the input or output from and to the
            database is required.
        """
        # Check if output from and to database
        if self.from_database and self.to_database:
            info("Output of '{}' from and to database. "
                "The database will not be updated with its own data.".format(self.series))
            self.to_database = False
        # Creation or import of database
        if self.db_path is None:
            self.db_path = '{}.db'.format(self.series)
        self.db_name = '{}.db'.format(self.series) if path.basename(self.db_path) == '' \
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
            info("Output of '{}' from database.".format(self.series))
            self.groups = [group.name
                for group in GroupModel.select().where(GroupModel.series == self.series)]

            # Check if data is present in the database
            if len(self.groups) == 0:
                self.stop("No existing series '{}' in the database '{}'.",
                    'ValueError', self.series, self.db_name)

    def configure_data(self):
        """ Check if traceback and output from data is possible.
        """
        # General configuration
        info("Traceback and output of '{}' from data.".format(self.series))
        self.configure_traceback()

        # Check if data is present
        if self.data is None:
            self.stop("No data provided for traceback '{}'.", 'NameError', self.series)

        # Data import
        from data import Data
        self.data_series = Data(self.series, self.representation, self.data)

    def configure_simulation(self):
        """ Check if traceback and output from simulation is possible.
        """
        # General configuration
        info("Traceback and output of '{}' from simulation.".format(self.series))
        self.configure_traceback()

        # Check if all the necessary parameters are present
        for parameter in (
                'number_of_stars', 'age',
                'avg_position', 'avg_position_error', 'avg_position_scatter',
                'avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
            if vars(self)[parameter] is None:
                self.stop("Required parameter '{}' is missing in the configuration file.",
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

    def configure_traceback(self):
        """ Check configuration parameters for traceback and output from data or simulation.
        """
        # Check if all the necessary parameters are present
        for parameter in ('number_of_groups', 'number_of_steps', 'initial_time', 'final_time'):
            if vars(self)[parameter] is None:
                self.stop("Required parameter '{}' is missing in the configuration file.",
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
        self.groups = ['{}_{}'.format(self.series, i) for i in range(1, self.number_of_groups + 1)]

    def stop(self, message, error, *values):
        """ Raises an exception and logs it into the log file and stops the execution.
        """
        message = message.format(*values)
        warning('{}: {}'.format(error, message))
        exec("""raise {}("{}")""".format(error, message))
