# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" series.py: Defines the class Series which uses a Config object to create a series of tracebacks
    either from a database, data or simulation. First, it handles exceptions (name, type, value,
    shape) of all parameters, creates or imports the database, if needed, handles unit conversions,
    checks for the presence of output and logs directories and creates them, if needed.
"""

import numpy as np
from time import strftime
from os import path, makedirs, remove, getcwd, chdir
from logging import basicConfig, info, warning, INFO
from coordinate import *
from quantity import *
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
        """ Configures a Config object from database, data or simulation. """

        # Inilialization
        self.config = Config(parent=config)
        self.logs_dir = None

        # Parameters configuration
        self.configure_parameters()

        # Series name
        self.name = self.config.name.values
        self.stop(self.name is None, 'NameError',
            "Required parameter 'name' is missing in the configuration.")
        self.stop(type(self.name) != str, 'TypeError', "'name' must be a string ('{}' given).",
            type(self.name))

        # Present date and time
        self.date = strftime('%Y-%m-%d %H:%M:%S')

        # Output directory creation
        self.output_dir = self.directory(path.abspath(path.join(path.dirname(
            path.realpath(__file__)), '..')), self.config.output_dir.values, 'output_dir')

        # Logs configuration
        self.configure_logs()

        # Arguments configuration
        self.configure_arguments()

        # Database configuration
        self.configure_database()

        # Traceback configuration if needed
        if self.from_data or self.from_simulation:
            self.configure_traceback()

        # Groups dictionary creation
        Groups(self)

    def configure_parameters(self):
        """ Checks if all parameters are present and are Config.Parameter objects with all their
            components, and checks for invalid parameters and components. The type of 'label',
            'name', 'system', 'axis' and 'origin' components is checked (str), and 'system',
            'axis' and 'origin' components are converted to their respective classes if they are
            not None.
        """

        # Check if all parameters are present and are Config.Parameter objects with all components
        for parameter in self.config.default_parameters.keys():
            self.stop(parameter not in vars(self.config), 'NameError',
                "Required parameter '{}' is missing in the configuration.", parameter)
            self.stop(type(vars(self.config)[parameter]) != self.config.Parameter, 'TypeError',
                "'{}' must be a Config.Parameter object ({} given).", parameter,
                type(vars(self.config)[parameter]))
            for component in self.config.Parameter.default_components.keys():
                self.stop(component not in vars(vars(self.config)[parameter]).keys(), 'NameError',
                    "Required component '{}' is missing from the '{}' parameter "
                    "in the configuration.", component, parameter)

        # Check for invalid parameters and components
        for parameter_label, parameter in vars(self.config).items():
            self.stop(parameter_label not in self.config.default_parameters.keys(), 'NameError',
                "Parameter '{}' in the configuration is invalid.", parameter_label)
            for component_label, component in vars(parameter).items():
                self.stop(component_label not in self.config.Parameter.default_components.keys(),
                    'NameError', "Parameter's component '{}' in '{}' is invalid.", component_label,
                     parameter_label)

                # Checks whether all component, but 'values' and 'units' are strings or None
                if component_label not in ('values', 'units'):
                    self.stop(component is not None and type(component) != str, 'TypeError',
                        "'{}' component in '{}' parameter must be a string or None ('{}' given.)",
                        component_label, parameter_label, type(component))

            # Default parameter
            default_parameter = self.config.default_parameters[parameter_label]

            # Checks if parameter.label and parameter.name were changed
            if parameter.label != parameter_label:
                parameter.label = parameter_label
            if parameter.name != default_parameter.name:
                parameter.name = default_parameter.name

            # Checks if parameter.system is valid and converts it into a System object
            if parameter.system is not None:
                self.stop(parameter.system.lower() not in systems.keys(), 'ValueError',
                    "'system' component of '{}' is invalid ({} required, {} given).",
                    parameter.label, list(systems.keys()), parameter.system)
                parameter.system = systems[parameter.system.lower()]
            elif default_parameter.system is not None:
                parameter.system = systems[default_parameter.system]

            # Checks if parameter.axis is valid and converts it into a System.Axis object
            if parameter.axis is not None:
                self.stop(parameter.axis.lower() not in parameter.system.axes.keys(), 'ValueError',
                    "'axis' component of '{}' is not a valid ({} required, {} given).",
                    parameter.label, list(parameter.system.axes.keys()), parameter.axis)
                parameter.axis = parameter.system.axes[parameter.axis.lower()]
            elif default_parameter.axis is not None:
                parameter.axis = parameter.system.axes[default_parameter.axis]

            # Checks if parameter.origin is valid and converts it into a System.origin object
            if parameter.origin is not None:
                self.stop(parameter.origin.lower() not in parameter.system.origins.keys(), 'Value'
                    'Error', "'origin' component of '{}' is not a valid ({} required, {} given).",
                    parameter.label, list(parameter.system.origins.keys()), parameter.origin)
                parameter.origin = parameter.system.origins[parameter.origin.lower()]
            elif default_parameter.origin is not None:
                parameter.origin = parameter.system.origins[default_parameter.origin]

    def configure_logs(self):
        """ Checks if the logs directory already exists, creates it if needed and configures
            the Logs file. 'logs_dir' is defined as the absolute logs directory.
        """

        # Logs directory parameter
        self.logs_dir = 'Logs' if self.config.logs_dir.values is None \
            else self.config.logs_dir.values

        # Logs direction creation logs_dir parameter redefinied as the absolute path
        self.logs_dir = self.directory(self.output_dir, self.config.logs_dir.values, 'logs_dir')

        # Logs configuration
        basicConfig(
            filename=path.join(
                self.logs_dir, '{}_{}.log'.format(self.name, strftime('%Y-%m-%d_%H-%M-%S'))),
            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

        # Set logs_configured as True
        self.logs_configured = True

    def configure_arguments(self):
        """ Checks for the type of the arguments provided in the configuration and check if
            their values are compatible. from_database is defined based on the values of
            from_data and from simulation.
        """

        # from_data, from_simulation and to_database parameters
        for argument in ('to_database', 'from_data', 'from_simulation'):
            vars(self)[argument] = vars(self.config)[argument].values
            self.stop(type(vars(self)[argument]) is None, 'NameError'
                "Required parameter '{}' is missing in the configuration.", argument)
            self.stop(type(vars(self)[argument]) != bool, 'TypeError'
                "'{}' must be a boolean value ({} given).", argument, type(vars(self)[argument]))

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
            if needed. 'database_path' is redefined as the absolute path. The database Model
            is initiated and errors handled based on whether the input or output from and to the
            database is required.
        """

        # db_path parameter
        self.db_path = '{}.db'.format(self.name) if self.config.db_path.values is None \
            else self.config.db_path.values

        # Check if db_path parameter is a string, witch must be done before self.directory call
        self.stop(type(self.db_path) != str, 'TypeError', "'db_path' must be a string ({} given).",
            type(self.db_path))

        # Database absolute name and director
        self.db_name = '{}.db'.format(self.name) if path.basename(self.db_path) == '' \
            else path.basename(self.db_path)
        self.db_dir = self.directory(self.output_dir, path.dirname(self.db_path), 'db_path',
            create=self.from_database or self.to_database)

        # db_path parameter redefinition as the absolute path
        self.db_path = path.join(self.db_dir, self.db_name)
        Config.db_path = self.db_path # Necessary for Database object definition

        # Check if the path links to a database file.
        self.stop(path.splitext(self.db_path)[1].lower() != '.db', 'TypeError',
            "'{}' is not a database file (with a .db extension).", self.db_path)

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

        # number_of_groups parameter
        self.number_of_groups = self.configure_integer(self.config.number_of_groups)

        # number_of_steps parameter: one more step to account for t = 0
        self.number_of_steps = self.configure_integer(self.config.number_of_steps) + 1

        # initial_time parameter
        self.initial_time = self.configure_quantity(self.config.initial_time)

        # final_time parameter
        self.final_time = self.configure_quantity(self.config.final_time)
        self.stop(not self.final_time > self.initial_time, 'ValueError',
            "'final_time' must be greater than initial_time ({} and {} given).",
            self.final_time, self.initial_time)

        # duration, timesteps and time parameters
        self.duration = self.final_time - self.initial_time
        self.timestep = self.duration / (self.number_of_steps - 1)
        self.time = np.linspace(
            self.initial_time.values, self.final_time.values, self.number_of_steps)

        # Data configuration
        # if self.from_data:
        #     self.configure_data()

        # Data configuration for error inclusion in simulation
        self.configure_data()

        # Simulation configuration
        if self.from_simulation:
            self.configure_simulation()

        # Radial velocity offset
        self.rv_offset = self.configure_quantity(self.config.rv_offset)

    def configure_data(self):
        """ Check if traceback and output from data is possible and creates a Data object from a
            CSV file, or a Python dictionary, list, tuple or np.ndarray.
        """

        # Logging
        info("Traceback and output of '{}' from data.".format(self.name))

        # Check if data is present
        self.stop(self.config.data is None, 'NameError',
            "Required traceback parameter 'data' is missing in the configuration.")

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
                'number_of_stars', 'age', 'avg_position', 'avg_position_error',
                'avg_position_scatter', 'avg_velocity', 'avg_velocity_error',
                'avg_velocity_scatter'):
            self.stop(vars(self.config)[parameter].values is None, 'NameError',
                "Required simulation parameter '{}' is missing in the configuration.", parameter)

        # number_of_stars parameter
        self.number_of_stars = self.configure_integer(self.config.number_of_stars)

        # age parameter
        self.age = self.configure_quantity(self.config.age)
        self.stop(not self.age.value >= 0.0, 'ValueError',
            "'age' must be greater than or equal to 0.0 Myr ({} given).", self.age)

        # avg_position parameter
        self.avg_position = self.configure_coordinate(self.config.avg_position)

        # avg_position_error parameter
        self.avg_position_error = self.configure_coordinate(self.config.avg_position_error)

        # avg_position_scatter parameter
        self.avg_position_scatter = self.configure_coordinate(self.config.avg_position_scatter)

        # avg_velocity parameter
        self.avg_velocity = self.configure_coordinate(self.config.avg_velocity)

        # avg_velocity_error parameter
        self.avg_velocity_error = self.configure_coordinate(self.config.avg_velocity_error)

        # avg_velocity_scatter parameter
        self.avg_velocity_scatter = self.configure_coordinate(self.config.avg_velocity_scatter)

        # Data set to None because stars are simulated
        # self.data = None

    def configure_integer(self, parameter):
        """ Checks if an integer value is valid and converts it if needed. """

        # Check the presence and type of values component
        self.stop(parameter.values is None, 'NameError',
            "Required traceback parameter '{}' is missing in the configuration.", parameter.label)
        self.stop(type(parameter.values) not in (int, float), 'TypeError', "'{}' must be "
            "an integer or a float ({} given).", parameter.label, type(parameter.values))

        # Check if the value is convertible to an integer and greater than or equal to 1
        self.stop(parameter.values % 1 != 0, 'ValueError', "'{}' must be convertible "
            "to an integer ({} given).", parameter.label, parameter.values)
        self.stop(parameter.values < 1, 'ValueError', "'{}' must be "
            "greater than or equal to 1 ({} given).", parameter.label, parameter.values)

        # Conversion to integer
        return int(parameter.values)

    def configure_quantity(self, parameter):
        """ Checks if a value is valid and converts it to default units if needed. """

        # Check the presence and type of parameter.values component
        self.stop(parameter.values is None, 'NameError',
            "Required traceback parameter '{}' is missing in the configuration.", parameter.label)
        self.stop(type(parameter.values) not in (int, float), 'TypeError', "'{}' must be "
            " an integer or a float ({} given).", parameter.label, type(parameter.values))

        # Default units component
        if parameter.units is None:
            parameter.units = self.config.default_parameters[parameter.label].units

        # Check the type of parameter.units component
        self.stop(type(parameter.units) != str, 'TypeError', "'units' component of '{}' "
            "must be a string ({} given).", parameter.label, type(parameter.units))

        # Check if parameter.units component can be converted to Unit
        try:
            Unit(parameter.units)
        except:
            self.stop(True, 'ValueError', "'units' component of '{}' must represent a unit.",
                parameter.label)

        # Quantity object creation
        try:
            quantity = Quantity(**vars(parameter))
        except:
            self.stop(True, 'ValueError', "'{}' could not be converted to a Quantity object.",
                parameter.label)

        # Check if the physical type is valid
        default_physical_type = Unit(self.config.default_parameters[parameter.label].units).physical_type
        self.stop(quantity.physical_types.flatten()[0] != default_physical_type, 'ValueError',
            "Unit of '{}' does not have the correct physical type ('{}' given, '{}' required).",
            parameter.label, quantity.physical_types.flatten()[0], default_physical_type)

        # Unit conversion
        return quantity.to()

    def configure_coordinate(self, parameter):
        """ Converts a Parameter into a Quantity object and raises an error if an exception
            occurs in the process. Returns a vector converted into the correct units for the
            physical type defined by a Variable object.
        """

        # Variables from label and check for invalid label
        if parameter.label in ('avg_position', 'avg_position_error', 'avg_position_scatter'):
            variables = parameter.system.position
        elif parameter.label in ('avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
            variables = parameter.system.velocity
        else:
            self.stop(True, 'NameError', "'{}' is not a supported name.", parameter.label)

        # Check the presence and type of parameter.values component
        self.stop(parameter.values is None, 'NameError',
            "Required simulation parameter '{}' is missing in the configuration.", parameter.label)
        self.stop(type(parameter.values) not in (tuple, list, np.ndarray), 'TypeError',
            "'values' component of '{}' must be a tuple, list or np.ndarray ({} given).'",
                parameter.label, type(parameter.values))

        # Check if all elements in parameter.values component are numerical
        try:
            np.vectorize(float)(parameter.values)
        except:
            self.stop(True, 'ValueError',
                "'values' component of '{}' contains non-numerical elements.", parameter.label)

        # Check the dimensions of parameter.values component
        shape = np.array(parameter.values).shape
        ndim = len(shape)
        self.stop(ndim > 2, 'ValueError', "'{}' must have 1 or 2 dimensions ({} given).",
            parameter.label, ndim)
        self.stop(shape[-1] != 3, 'ValueError',
            "'{}' last dimension must have a size of 3 ({} given).", parameter.label, shape[-1])
        self.stop(ndim == 2 and shape[0] not in (1, self.number_of_stars),
            'ValueError',  "'{}' first dimension ({} given) must have a size of 1 or equal "
            "to the number of stars ({} given).", parameter.label, shape[0], self.number_of_stars)

        # Default units component
        if parameter.units is None:
            parameter.units = [variable.unit.label for variable in variables]

        # Check the type of parameter.units component
        if type(parameter.units) == str:
            parameter.units = [parameter.units]
        self.stop(type(parameter.units) not in (tuple, list, np.ndarray), 'TypeError',
            "'units' component of '{}' must be a string, tuple, list or np.ndarray ({} given).",
                parameter.label, type(parameter.values))

        # Check if all elements in parameter.units component can be converted to Unit
        try:
            np.vectorize(Unit)(np.array(parameter.units, dtype=object))
        except:
            self.stop(True, 'ValueError',
                "'units' components of '{}' must all represent units.", parameter.label)

        # Quantity object creation
        try:
            quantity = Quantity(**vars(parameter))
        except:
            self.stop(True, 'ValueError', "'{}' could not be converted to a Quantity object.",
                parameter.label)

        # Check if physical types are valid
        physical_types = np.array([variable.physical_type for variable in variables])
        self.stop(not (quantity.physical_types == physical_types).all(), 'ValueError',
            "Units in '{}' do not have the correct physical type "
            "({} given, {} required for '{}' system.)", parameter.label,
             quantity.physical_types.tolist(), physical_types, quantity.system.label)

        # Units conversion
        return quantity.to()

    def directory(self, base, directory, name, create=True):
        """ Checks for type errors, checks if the directory already exists, creates it if
            needed and returns the absolute directory path. The directory can either be
            relative the base or an absolute.
        """

        # Check the type of directory
        self.stop(type(directory) is not str, 'TypeError', "'{}' must be a string ({} given).",
            name, type(directory))

        # Output directory formatting
        working_dir = getcwd()
        chdir(path.abspath(base))
        abs_dir = path.abspath(directory)
        chdir(working_dir)

        # Output directory creation
        if create and not path.exists(abs_dir):
            makedirs(abs_dir)
        return abs_dir

    def stop(self, condition, error, message, *words, extra=1):
        """ Raises an exception if 'condition' is True, logs it into the log file if logs have
            been configured and stops the execution. 'error' is a string representing an exception
            class, 'message' is the error message to be displayed, 'words' is a list of words to
            be inserted into 'message' and 'extra' is the number of superfluous lines in the
            traceback stack. If an exception isn't being handled, the function is recursively
            called within an except statement.
        """

        # Exception information extraction
        from sys import exc_info, exit
        exc_type, exc_value = exc_info()[:2]

        # If no exception is being handled, one is raised if 'conidition' is True
        if exc_type is None and exc_value is None:
            try:
                if condition:
                    exec("raise {}".format(error))
            except:
                self.stop(True, error, message, *words, extra=2)

        # If an exception is being handled, its traceback is formatted and execution is terminated
        else:
            # Traceback message creation
            tb_message = "{} in '{}': {}".format(error, self.name, message.format(*words)) \
                if 'name' in vars(self) else "{}: {}".format(error, message.format(*words))

            # Traceback stack formatting
            from traceback import format_stack
            tb_stack = ''.join(
                ['An exception has been raised: \n'] + format_stack()[:-extra] + [tb_message])

            # Exception logging only if logs have been configured and code termination
            if self.logs_dir is not None:
                warning(tb_stack)
            print(tb_stack)
            exit()

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
