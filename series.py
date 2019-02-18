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
        vars(self).update(vars(config).copy())

        # Series name
        if self.name is None:
            raise NameError("Required parameter 'name' is missing.")
        if type(self.name) != str:
            raise TypeError("'name' must be a string ('{}' given).".format(type(self.name)))

        # Present date and time
        self.date = strftime('%Y-%m-%d %H:%M:%S')

        # Output directory creation
        self.output_dir = self.directory(path.abspath(path.join(
            path.dirname(path.realpath(__file__)), '..')), self.output_dir, 'output_dir')

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
            the Logs file. 'logs_dir' is redefined as the absolution logs directory
        """
        # Logs directory creation
        self.logs_dir = self.directory(self.output_dir, self.logs_dir, 'logs_dir')

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
            self.stop(type(vars(self)[argument]) != bool,
                "'{}' must be a boolean value ({} given).",
                'TypeError', argument, type(vars(self)[argument]))

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
        if self.db_path is None:
            self.db_path = '{}.db'.format(self.name)
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
        """ Check configuration parameters for traceback and output from data or simulation.
        """
        # Check if all the traceback parameters are present
        for parameter in (
                'number_of_groups', 'number_of_steps',
                'initial_time', 'final_time', 'system', 'axis'):
            self.stop(vars(self)[parameter] is None, 'NameError',
                "Required traceback parameter '{}' is missing in the configuration.", parameter)

        # number_of_groups parameter
        self.stop(type(self.number_of_groups) not in (int, float), 'TypeError',
            "'number_of_groups' must be an integer or a float ({} given).",
            type(self.number_of_groups))
        self.stop(self.number_of_groups % 1 != 0, 'ValueError',
            "'number_of_groups' must be convertible to an integer ({} given).",
            self.number_of_groups)
        self.stop(self.number_of_groups < 1, 'ValueError',
            "'number_of_groups' must be greater than 0 ({} given).", self.number_of_groups)
        self.number_of_groups = int(self.number_of_groups)

        # number_of_steps parameter
        self.stop(type(self.number_of_steps) not in (int, float), 'TypeError'
            "'number_of_steps' must be an integer or a float ({} given).",
            type(self.number_of_steps))
        self.stop(self.number_of_steps % 1 != 0, 'ValueError',
            "'number_of_steps' must be convertible to an integer ({} given).",
            self.number_of_steps)
        self.stop(not self.number_of_steps > 0, 'ValueError',
            "'number_of_steps' must be greater than 0 ({} given).", self.number_of_steps)
        self.number_of_steps = int(self.number_of_steps) + 1 # One more step to account for t = 0

        # initial_time parameter
        self.stop(type(self.initial_time) not in (int, float), 'TypeError',
            "'initial_time' must be an integer or a float ({} given).", type(self.initial_time))

        # final_time parameter
        self.stop(type(self.final_time) not in (int, float), 'TypeError',
            "'final_time' must be an integer or a float ({} given).", type(self.final_time))
        self.stop(not self.final_time > self.initial_time, 'ValueError',
            "'final_time' must be greater than initial_time ({} and {} given).",
            self.final_time, self.initial_time)

        # duration, timesteps and time parameters
        self.duration = self.final_time - self.initial_time
        self.timestep = self.duration / (self.number_of_steps - 1)
        self.time = np.linspace(self.initial_time, self.final_time, self.number_of_steps)

        # Coordinates system
        self.stop(type(self.system) != str, 'TypeError',
            "'system' must be a string ({} given).", type(self.system))
        self.system = self.system.lower()
        self.stop(self.system not in Config.systems.keys(), 'ValueError',
            "'{}' is not a valid system value.", self.system)

        # Coordinates system axis
        self.stop(type(self.axis) != str, 'TypeError',
            "'axis' must be a string ({} given).", type(self.axis))
        self.axis = self.axis.lower()
        self.stop(self.axis not in System.axis, 'ValueError',
            "'{}' is not a valid system value.", self.axis)

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
        self.stop(self.data is None, "No data provided for traceback '{}'.", 'NameError', self.name)

        # Stars creation from data
        from data import Data
        self.stars = Data(self)

        # number_of_stars parameter
        self.number_of_stars = len(self.stars)

        # ??? Calculate avg_position, avg_position_error, avg_position_scatter here ???

    def configure_simulation(self):
        """ Check if traceback and output from simulation is possible.
        """
        # Logging
        info("Traceback and output of '{}' from simulation.".format(self.name))

        # Check if all the necessary parameters are present
        for parameter in (
                'number_of_stars', 'age',
                'avg_position', 'avg_position_error', 'avg_position_scatter',
                'avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
            self.stop(vars(self)[parameter] is None, 'NameError',
                "Required simulation parameter '{}' is missing in the configuration.", parameter)

        # number_of_stars parameter
        self.stop(type(self.number_of_stars) not in (int, float), 'TypeError'
            "'number_of_stars' must be an integer or a float ({} given).",
            type(self.number_of_stars))
        self.stop(self.number_of_stars % 1 != 0, 'ValueError',
            "'number_of_stars' must be convertible to an integer ({} given).",
            self.number_of_stars)
        self.stop(not self.number_of_stars > 0, 'ValueError'
            "'number_of_stars' must be greater than 0 ({} given).", self.number_of_stars)
        self.number_of_stars = int(self.number_of_stars)

        # age parameter
        self.stop(type(self.age) not in (int, float), 'TypeError',
            "'age' must be an integer or a float ({} given).", type(self.age))
        self.stop(not self.age >= 0.0, 'ValueError',
            "'age' must be greater than or equal to 0.0 ({} given).", self.age)

        # avg_position parameter
        self.avg_position = self.quantify(self.avg_position, 'avg_position')

        # avg_position_error parameter
        self.avg_position_error = self.quantify(self.avg_position_error, 'avg_position_error')

        # avg_position_scatter parameter
        self.avg_position_scatter = self.quantify(self.avg_position_scatter, 'avg_position_scatter')

        # avg_velocity parameter
        self.avg_velocity = self.quantify(self.avg_velocity, 'avg_velocity')

        # avg_velocity_error parameter
        self.avg_velocity_error = self.quantify(self.avg_velocity_error, 'avg_velocity_error')

        # avg_velocity_scatter parameter
        self.avg_velocity_scatter = self.quantify(self.avg_velocity_scatter, 'avg_velocity_scatter')

        # Stars set to None because stars are simulated
        self.stars = None

    def quantify(self, parameter, name):
        """ Converts a variable into a Quantity object and raises an error if an exception occurs
            in that process. Returns a value converted into the correct units for the physical
            type defined in by a variable object.
        """
        def identify(value, name):
            """ Identify the type of a parameter and calls itself recursively if it finds a tuple
                or a list. If it finds a strings, it identifies whether the strings represents a
                unit, system of axis.
            """
            # Units, systems and axis values identification and check
            if type(value) == str:
                try:
                    un.Unit(value)
                    variable_type = 'units'
                except:
                    if value.lower() in systems.keys():
                        variable_type = 'system'
                    elif value.lower() in System.axis:
                        variable_type = 'axis'
                    else:
                        self.stop(True, 'ValueError',
                            "The value '{}' in '{}' could not be identified.", value, name)

            # Identify position and velocity values
            elif type(value) in (int, float):
                variable_type = 'values'

            # Recursive call if the value is an iterable
            elif type(value) in (tuple, list):
                variable_types = set(identify(i, name) for i in value)

                # Check if there's more than one type
                self.stop(len(variable_types) > 1, 'ValueError'
                    "Elements in tuples or lists in '{}' must all have the same type ({} given).",
                    name, list(variable_types))
                variable_type = tuple(variable_types)[0]

            # Unsupported types handling
            else:
                self.stop(True, 'TypeError'
                    "'{}' can use strings, integer, float, tuple or list types ({} given).",
                    name, type(value))

            return variable_type

        # Values from a tuple or list
        if type(parameter) in (tuple, list):
            # Types identification
            variable_types = [identify(value, name) for value in parameter]

            # Identify whether all types are values, or if units, systems or axis are present
            if all([variable_type == 'values' for variable_type in variable_types]):
                values = {'values': parameter}
            else:
                values = {key: parameter[i] for i, key in enumerate(variable_types)}

        # Values from a dictionary
        elif type(parameter) == dict:
            # Values definition
            values = {key.lower(): parameter[key] for key in parameter.keys()}

            # Check if keys in the dictionary are all supported.
            for key in values.keys():
                self.stop(key not in ('values', 'units', 'system', 'axis'), 'ValueError',
                    "'{}' in {} is not a supported value ('values', 'units', 'system' or 'axis').",
                    key, name)

            # Check if all values are identified correctly
            variable_types = {key: identify(values[key], name) for key in values.keys()}
            for key in variable_types.keys():
                self.stop(key != variable_types[key], 'ValueError', "'{}' in {} was identified as '{}'.",
                    key, name, variable_types[key])

        # Unsupported types handling
        else:
            self.stop(True, 'TypeError' "'{}' must be a tuple, list or dictionary ({} given).",
                name, type(parameter))

        # Check if a numerical value is present
        self.stop('values' not in values.keys(), 'NameError'
            "'{}' must contain a numerical value.", name)

        # Default system
        if 'system' not in values.keys():
            values['system'] = self.system
        system = systems[values['system']]

        # Default axis
        if 'axis' not in values.keys():
            values['axis'] = self.axis

        # Variables
        if name in ('avg_position', 'avg_position_error', 'avg_position_scatter'):
            variables = system.position
        elif name in ('avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
            variables = system.velocity
        else:
            self.stop(True, 'NameError', "'{}' is not a supported name.", name)

        # Default units
        if 'units' not in values.keys():
            values['units'] = [variable.unit.unit for variable in variables]

        # Dimensions
        shape = np.array(values['values']).shape
        ndim = len(shape)
        self.stop(ndim > 2, 'ValueError', "'{}' must have 1 or 2 dimensions ({} given).",
            name, ndim)
        self.stop(shape[-1] != 3, 'ValueError',
            "'{}' last dimension must have a size of 3 ({} given).", name, shape[-1])
        self.stop(ndim == 2 and shape[0] not in (1, self.number_of_stars),
            'ValueError',  "'{}' first dimension ({} given) must have a size of 1 or equal to the number of stars ({} given).",
            name, shape[0], self.number_of_stars)

        # Quantity object creation
        try:
            quantity = Quantity(**values)
        except:
            self.stop(True, 'ValueError', "'{}' could not be converted to a Quantity object.", name)

        # Physical types
        physical_types = [variable.physical_type for variable in variables]
        self.stop(not (quantity.physical_types == physical_types).all(), 'ValueError',
            "Units in '{}' do not have the correct physical type ({} given, {} required for '{}' system.)",
            name, quantity.physical_types.tolist(), physical_types, quantity.system)

        # Units conversion
        quantity = quantity.to([variable.unit.unit for variable in variables])

        return quantity

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
