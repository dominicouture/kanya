# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" series.py: Defines the class Series which uses a Config object to create a series of traceback
    either from a database, data or simulation. First, it handles exceptions (name, type, value,
    shape) of all parameters, creates or import the database, if needed, handles unit conversions,
    checks for the presence of output and logs directories and creates them if necessary.
"""

import numpy as np
from astropy import units as un
from time import strftime
from os import path, makedirs, remove, getcwd, chdir
from logging import basicConfig, info, warning, INFO
from tools import Quantity
from init import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Series(list):
    """ Contains a series of Groups and their embedded Star objects and the methods required to
        initialize a series from a Config object. Values and data in the Config object are, copied,
        checked and converted into Quantity objects and default units. The number of groups defines
        how many objects are in a series in which all groups have the same Config object as their
        progenitor.
    """

    def __init__(self, config):
        """ Configures a Config object from database, data or simulation, and creates output and
            Logs directory if necessary.
        """

        # Inilialization
        self.config = config
        # vars(self).update(vars(config).copy())

        # Series name
        self.name = config.name.values
        if self.name is None:
            raise NameError("Required parameter 'name' is missing.")
        if type(self.name) != str:
            raise TypeError("'name' must be a string ('{}' given).".format(type(self.name)))

        # Present date and time
        self.date = strftime('%Y-%m-%d %H:%M:%S')

        # Output directory creation
        self.output_dir = self.directory(path.abspath(path.join(
            path.dirname(path.realpath(__file__)), '..')), config.output_dir.values, 'output_dir')

        # Logs configuration
        self.configure_logs()

        # Arguments configuration
        self.configure_arguments()

        # Database configuration
        self.configure_database()

        # Traceback configuration
        if self.from_data or self.from_simulation:
            self.configure_traceback()

        # Groups dictionary creation
        Groups(self)

    def directory(self, base, directory, name, create=True):
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
        if create and not path.exists(abs_dir):
            makedirs(abs_dir)
        return abs_dir

    def stop(self, condition, error, message, *words):
        """ If condition is True, raises an exception and logs it into the log file and stops the
            execution.
        """

        if condition:
            message = message.format(*words)
            warning('{}: {}'.format(error, message))
            exec("""raise {}("{}")""".format(error, message))

    def configure_logs(self):
        """ Checks if the logs directory already exists, creates it if necessary and configures
            the Logs file. 'logs_dir' is defined as the absolute logs directory.
        """

        # Logs directory creation
        self.logs_dir = self.directory(self.output_dir, self.config.logs_dir.values, 'logs_dir')

        # Logs configuration
        basicConfig(
            filename=path.join(
                self.logs_dir, '{}_{}.log'.format(self.name, strftime('%Y-%m-%d_%H-%M-%S'))),
            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

    def configure_arguments(self):
        """ Checks for the type of the arguments provided in the configuration and check if
            their values are compatible. from_database is defined based on the values of
            from_data and from simulation.
        """
        # from_data, from_simulation and to_database parameters
        for argument in ('to_database', 'from_data', 'from_simulation'):
            self.stop(type(vars(self.config)[argument].values) != bool, 'TypeError'
                "'{}' must be a boolean value ({} given).",
                argument, type(vars(self.config)[argument].values))
            vars(self)[argument] = vars(self.config)[argument].values

        # Check if traceback and output from both data and simulation
        self.stop(self.from_data and self.from_simulation, 'ValueError',
            "Either traceback series '{}' from data or a simulation, not both.", self.name)

        # from_database parameter
        self.from_database = True if not self.from_data and not self.from_simulation else False

        # Check if traceback and output from both data and simulation
        if self.from_database and self.to_database:
            info("Output of '{}' from and to database. "
                "The database will not be updated with its own data.".format(self.name))
            self.to_database = False

    def configure_database(self):
        """ Checks if the database directory and the database file itself exist and creates them
            if necessary. 'database_path' is redefined as the absolute path. The database Model
            is initiated and errors handled based on whether the input or output from and to the
            database is required.
        """

        # Database absolute directory and name
        self.db_path = '{}.db'.format(self.name) if self.config.db_path.values is None \
            else self.config.db_path.values
        self.db_name = '{}.db'.format(self.name) if path.basename(self.db_path) == '' \
            else path.basename(self.db_path)
        self.db_dir = self.directory(self.output_dir, path.dirname(self.db_path), 'db_path',
            create=self.from_database or self.to_database)
        self.db_path = path.join(self.db_dir, self.db_name) # Absolute path
        Config.db_path = self.db_path # Necessary for Database object definition

        if self.from_database or self.to_database:
            # Check if a database exists
            self.stop(self.from_database and not path.exists(self.db_path), 'NameError',
                "No existing database located at '{}'.", self.db_path)

            # Database initialization
            from model import SeriesModel, GroupModel
            self.model, self.created = SeriesModel.get_or_create(name=self.name)

            # Check if output from database is possible
            if self.from_database:
                # Logging
                info("Output of '{}' from database.".format(self.name))

                # Check if data is present in the database
                self.stop(len(tuple(GroupModel.select().where(
                    GroupModel.series == self.model))) == 0, 'ValueError',
                    "No existing series '{}' in the database '{}'.", self.name, self.db_name)
        else:
            # Set model and created to None and False beacuse no database is needed
            self.model, self.created = (None, False)

    def configure_traceback(self):
        """ Check configuration parameters for traceback and output from data or simulation. """

        # Check if all the traceback parameters are present in the configuration
        for parameter in (
                'number_of_groups', 'number_of_steps', 'initial_time', 'final_time'):
            self.stop(vars(self.config)[parameter].values is None, 'NameError',
                "Required traceback parameter '{}' is missing in the configuration.", parameter)

        # number_of_groups parameter
        self.number_of_groups = self.integer(self.config.number_of_groups, 'number_of_groups')

        # number_of_steps parameter: one more step to account for t = 0
        self.number_of_steps = self.integer(self.config.number_of_steps, 'number_of_steps') + 1

        # initial_time parameter
        self.stop(type(self.config.initial_time.values) not in (int, float), 'TypeError',
            "'initial_time' must be an integer or a float ({} given).",
            type(self.config.initial_time.values))
        self.initial_time = float(self.config.initial_time.values)

        # final_time parameter
        self.stop(type(self.config.final_time.values) not in (int, float), 'TypeError',
            "'final_time' must be an integer or a float ({} given).",
            type(self.config.final_time.values))
        self.final_time = float(self.config.final_time.values)
        self.stop(not self.final_time > self.initial_time, 'ValueError',
            "'final_time' must be greater than initial_time ({} and {} given).",
            self.final_time, self.initial_time)

        # duration, timesteps and time parameters
        self.duration = self.final_time - self.initial_time
        self.timestep = self.duration / (self.number_of_steps - 1)
        self.time = np.linspace(self.initial_time, self.final_time, self.number_of_steps)

        # Data configuration
        if self.from_data:
            self.configure_data()
        # Simulation configuration
        if self.from_simulation:
            self.configure_simulation()

    def configure_data(self):
        """ Check if traceback and output from data is possible and creates a Data object from a
            CSV file, or a Python dictionary, list, tuple or np.ndarray.
        """

        # Logging
        info("Traceback and output of '{}' from data.".format(self.name))

        # Check if data is present
        self.stop(self.config.data is None, 'NameError',
            "No data provided for traceback '{}'.", self.name)

        # Stars creation from data
        from data import Data
        self.data = Data(self)

        # number_of_stars parameter
        self.number_of_stars = len(self.data)

        # Simulation parameters set to None because stars are imported from data
        for parameter in (
                'age', 'avg_position', 'avg_position_error', 'avg_position_scatter',
                'avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
            vars(self)[parameter] = None

    def configure_simulation(self):
        """ Check if traceback and output from simulation is possible. """

        # Logging
        info("Traceback and output of '{}' from simulation.".format(self.name))

        # Check if all the necessary parameters are present
        for parameter in (
                'number_of_stars', 'age', 'avg_position', 'avg_velocity', 'avg_position_error',
                'avg_velocity_error', 'avg_position_scatter', 'avg_velocity_scatter'):
            self.stop(vars(self.config)[parameter].values is None, 'NameError',
                "Required simulation parameter '{}' is missing in the configuration.", parameter)

        # number_of_stars parameter
        self.number_of_stars = self.integer(self.config.number_of_stars, 'number_of_stars')

        # age parameter
        self.stop(type(self.config.age.values) not in (int, float), 'TypeError',
            "'age' must be an integer or a float ({} given).", type(self.config.age.values))
        self.age = float(self.config.age.values)
        self.stop(not self.age >= 0.0, 'ValueError',
            "'age' must be greater than or equal to 0.0 ({} given).", self.age)

        # avg_position parameter
        self.avg_position = self.quantify('avg_position', self.config.avg_position)

        # avg_position_error parameter
        self.avg_position_error = self.quantify(
            'avg_position_error', self.config.avg_position_error)

        # avg_position_scatter parameter
        self.avg_position_scatter = self.quantify(
            'avg_position_scatter', self.config.avg_position_scatter)

        # avg_velocity parameter
        self.avg_velocity = self.quantify('avg_velocity', self.config.avg_velocity)

        # avg_velocity_error parameter
        self.avg_velocity_error = self.quantify(
            'avg_velocity_error', self.config.avg_velocity_error)

        # avg_velocity_scatter parameter
        self.avg_velocity_scatter = self.quantify(
            'avg_velocity_scatter', self.config.avg_velocity_scatter)

        # Data set to None because stars are simulated
        self.data = None

    def integer(self, number, name):
        """ Checks if an integer value is valid and converts it if needed. """

        # Type check
        self.stop(type(number.values) not in (int, float), 'TypeError',
            "'{}' must be an integer or a float ({} given).", name, type(number.values))

        # Check if the value is convertible to an integer and greater than equal to 1
        self.stop(number.values % 1 != 0, 'ValueError',
            "'{}' must be convertible to an integer ({} given).", name, number.values)
        self.stop(number.values < 1, 'ValueError',
            "'{}' must be greater than or equal to 1 ({} given).", name, number.values)

        # Conversion to integer
        return int(number.values)

    def float(self):
        """ Checks if a float value is valid and converts it if needed. """

        pass

    def quantify(self, name, parameter):
        """ Converts a Parameter into a Quantity object and raises an error if an exception
            occurs in the process. Returns a value converted into the correct units for the
            physical type defined by a Variable object.
        """

        # Check the type of parameter
        self.stop(type(parameter) != Config.Parameter, 'TypeError',
            "'{}' must be a Config.Parameter object ({} given).", name, type(parameter))

        # Check if all necessary components are present
        for component in ('values', 'units', 'system', 'axis'):
            self.stop(component not in vars(parameter).keys(), 'NameError'
                "Component '{}' is missing in '{}' parameter.", component, name)
        # Check for invalid components
        for component in vars(parameter).keys():
            self.stop(component not in ('name', 'label', 'values', 'units', 'system', 'axis'),
                'NameError', "Component '{}' is an invalid parameter for '{}'.", component, name)

        # Default system component
        if parameter.system is None:
            parameter.system = self.system
        # Check the type of system component
        self.stop(type(parameter.system) != str, 'TypeError',
            "'system' component of '{}' must be a string ({}).", name, type(parameter.system))
        # Check the value of system component
        parameter.system = parameter.system.lower()
        self.stop(parameter.system not in systems.keys(), 'ValueError',
            "'system' component of '{}' is not a valid ({} required, {} given).",
            name, list(systems.keys()), parameter.system)
        system = systems[parameter.system]

        # Default axis component
        if parameter.axis is None:
            parameter.axis = self.axis
        # Check the type of axis component
        self.stop(type(parameter.axis) != str, 'TypeError',
            "'axis' component of '{}' must be a string ({}).", name, type(parameter.axis))
        # Check the value of axis component
        parameter.axis = parameter.axis.lower()
        self.stop(parameter.axis not in system.axis.keys(), 'ValueError',
            "'axis' component of '{}' is not a valid ({} required, {} given).",
            name, list(system.axis.keys()), parameter.axis)

        # Variables from name
        if name in ('avg_position', 'avg_position_error', 'avg_position_scatter'):
            variables = system.position
        elif name in ('avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
            variables = system.velocity
        else:
            self.stop(True, 'NameError', "'{}' is not a supported name.", parameter.name())

        # Default units component
        if parameter.units is None:
            parameter.units = [variable.unit.unit for variable in variables]
        # Check the type of units component
        if type(parameter.units) == str:
            parameter.units = [parameter.units]
        self.stop(type(parameter.units) not in (tuple, list, np.ndarray), 'TypeError',
            "'units' component of '{}' must be a string, tuple, list or np.ndarray ({} given).",
                name, type(parameter.values))

        # Check if all elements in units component can be converted to un.Units
        try:
            np.vectorize(un.Unit)(parameter.units)
        except:
            self.stop(True, 'ValueError',
                "'units' components of '{}' must all represent a unit.", name)

        # Check the type of values component
        self.stop(parameter.values is None, 'TypeError',
            "'values' component of '{}' cannot be None ({} given.)", name, parameter.values)
        self.stop(type(parameter.values) not in (tuple, list, np.ndarray), 'TypeError',
            "'values' compoent of '{}' must be a tuple, list or np.ndarray ({} given).'",
                name, type(parameter.values))

        # Check if all elements in values component are numerical
        try:
            np.vectorize(float)(parameter.values)
        except:
            self.stop(True, 'ValueError',
                "'values' component of '{}' contains non-numerical elements.'", name)

        # Check dimensions of values component
        shape = np.array(parameter.values).shape
        ndim = len(shape)
        self.stop(ndim > 2, 'ValueError', "'{}' must have 1 or 2 dimensions ({} given).",
            name, ndim)
        self.stop(shape[-1] != 3, 'ValueError',
            "'{}' last dimension must have a size of 3 ({} given).", name, shape[-1])
        self.stop(ndim == 2 and shape[0] not in (1, self.number_of_stars),
            'ValueError',  "'{}' first dimension ({} given) must have a size of 1 "
            "or equal to the number of stars ({} given).", name, shape[0], self.number_of_stars)

        # Quantity object creation
        try:
            quantity = Quantity(**vars(parameter))
        except:
            self.stop(True, 'ValueError', "'{}' could not be converted to a Quantity object.", name)

        # Physical types
        physical_types = [variable.physical_type for variable in variables]
        self.stop(not (quantity.physical_types == physical_types).all(), 'ValueError',
            "Units in '{}' do not have the correct physical type ({} given, {} required for '{}' "
            "system.)", name, quantity.physical_types.tolist(), physical_types, quantity.system)

        # Units conversion
        return quantity
        # return quantity.to([variable.unit.unit for variable in variables])

    def create(self):
        """ Creates a Group and embeded Star objects for all group names in self.groups either
            from a database, or by tracing back stars imported from data or simulated stars.
        """

        # Creation from database
        if self.from_database:
            from model import SeriesModel
            SeriesModel.load_from_database(SeriesModel, self)

        # Creation from data or simulation
        else:
            from group import Group
            for name in ['{}_{}'.format(self.name, i) for i in range(1, self.number_of_groups + 1)]:
                # Logging
                message = "Tracing back {}.".format(name.replace('_', ' '))
                info(message)
                print(message)
                # Group object creation
                self.append(Group(self, name=name))

        # Save series to database, previous entry deleted if needed
        if self.to_database:
            from model import SeriesModel
            SeriesModel.save_to_database(SeriesModel, self)


class Groups(dict):
    """ Contains a dictionary of series created from a configuration as well a traceback method
        that computes a traceback for all or selected series. Using this class is optional,
        although it is initialized automaticaly when a Series object is created.
    """

    def __init__(self, series):
        """ Initializes a Groups dictionary with a given series in main.py. If it already exists
            and series is already present in it, the previous entry is removed first.
        """

        # Groups dictionary creation if necessary
        import __main__ as main
        if 'groups' not in vars(main):
            vars(main)['groups'] = self

        # Previous entry deleted if it already exists
        if series.name in main.groups.keys():
            info("Existing series '{}' deleted and replaced.".format(series.name))
            del main.groups[series.name]

        # Addition of self to groups dictionary and logging
        main.groups[series.name] = series
        info("Series '{}' ready for traceback.".format(series.name))

    # ??? Fonction pour sélectionner from_data ou from_simulation, override de la valeur dans series et le check si le traceback est déjà fait. ???
    def create(self, *series):
        """ Creates a series of Groups object for all series in self if no series name is given
            or selected series given in argument.
        """

        # Computes a traceback for every groups in every selected series
        for name in list(self.keys()) if len(series) == 0 else series:

            # Check if the series exists
            Series.stop(Series, name not in self.keys(), "NameError",
                "Series '{}' does not exist.", name)

            # Traceback if it hasn't already been done
            if len(self[name]) == self[name].number_of_groups:
                info("Series '{}' has already been tracebacked.".format(name))
            else:
                self[name].create()
