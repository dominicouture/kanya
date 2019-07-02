# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" series.py: Defines the class Series which uses a Config object to create a series of tracebacks
    either from a file, data or a model. First, it handles exceptions (name, type, value,
    shape) of all parameters, then creates or imports the file, if needed, handles unit,
    conversions, and checks for the presence of output and logs directories and creates them, if
    needed.
"""

import numpy as np
from init import *
from coordinate import *
from quantity import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Series(list):
    """ Contains a series of Groups and their embedded Star objects and the methods required to
        initialize a series from a Config object. Values and data in the Config object are, copied,
        checked for errors and converted to usable types such as Quantity objects, and values are
        converted to default units. The number of groups defines how many objects are in a series
        in which all groups have the same Config object as their progenitor.
    """

    def __init__(self, parent=None, path=None, args=False, **parameters):
        """ Initializes a Series object by first configuring it with 'parent', 'path', 'args' and
            '**parameter' and then adding it to the collection.
        """

        # Configuration
        self.configure(parent, path, args, **parameters)

        # Series addition to the collection
        self.add()

    def configure(self, parent=None, path=None, args=False, **parameters):
        """ Configures a Series objects from 'parent', an existing Config object, 'path', a string
            representing a path to a configuration file, 'args' a boolean value that sets whether
            command line arguments are used, and '**parameters', a dictionary of dictionaries or
            Config.Parameter objects, in that order. If no value are given the default parameter is
            used instead. Only values that match a key in self.default_parameters are used. Then,
            parameters are copied, checked for errors and converted, and the series is configured.
        """

        # Configuration inilialization
        self.config = Config(parent, path, args, **parameters)

        # Parameters configuration
        self.configure_parameters()

        # Series configuration
        self.configure_series()

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

                # Check whether all components, but parameter.values and parameter.units are
                # strings or None
                if component_label not in ('values', 'units'):
                    self.stop(component is not None and type(component) not in (
                            str, System, System.Axis, System.Origin), 'TypeError',
                        "'{}' component in '{}' parameter must be a string or None ('{}' given.)",
                        component_label, parameter_label, type(component))

            # Default parameter
            default_parameter = self.config.default_parameters[parameter_label]

            # Check if parameter.label and parameter.name were changed
            if parameter.label != parameter_label:
                parameter.label = parameter_label
            if parameter.name != default_parameter.name:
                parameter.name = default_parameter.name

            # Check if parameter.system is valid and converts it to a System object
            if type(parameter.system) == System:
                pass
            elif parameter.system is not None:
                self.stop(parameter.system.lower() not in systems.keys(), 'ValueError',
                    "'system' component of '{}' is invalid ({} required, {} given).",
                    parameter.label, list(systems.keys()), parameter.system)
                parameter.system = systems[parameter.system.lower()]
            elif default_parameter.system is not None:
                parameter.system = systems[default_parameter.system]

            # Check if parameter.axis is valid and converts it to a System.Axis object
            if type(parameter.axis) == System.Axis:
                pass
            elif parameter.axis is not None:
                self.stop(parameter.axis.lower() not in parameter.system.axes.keys(), 'ValueError',
                    "'axis' component of '{}' is not a valid ({} required, {} given).",
                    parameter.label, list(parameter.system.axes.keys()), parameter.axis)
                parameter.axis = parameter.system.axes[parameter.axis.lower()]
            elif default_parameter.axis is not None:
                parameter.axis = parameter.system.axes[default_parameter.axis]

            # Check if parameter.origin is valid and converts it to a System.Origin object
            if type(parameter.origin) == System.Origin:
                pass
            elif parameter.origin is not None:
                self.stop(parameter.origin.lower() not in parameter.system.origins.keys(), 'Value'
                    'Error', "'origin' component of '{}' is not a valid ({} required, {} given).",
                    parameter.label, list(parameter.system.origins.keys()), parameter.origin)
                parameter.origin = parameter.system.origins[parameter.origin.lower()]
            elif default_parameter.origin is not None:
                parameter.origin = parameter.system.origins[default_parameter.origin]

    def configure_series(self):
        """ Checks basic series parameters. Checks if self.name is a string and creates a default
            value if needed. Checks the type of the modes provided in the configuration and checks
            if their values are valid. self.to_file is set to False if self.from_file is True.
            Moreover, self.date is defined.
        """

        # Series name
        self.name = collection.default('untitled') if self.config.name.values is None \
            else self.config.name.values

        # Current date and time
        from time import strftime
        self.date = strftime('%Y-%m-%d %H:%M:%S')

        # Check if series.name is a string
        self.stop(type(self.name) != str, 'TypeError', "'name' must be a string ('{}' given).",
            type(self.name))

        # from_data, from_model, from_file and to_file parameters
        for argument in ('from_data', 'from_model', 'from_file', 'to_file'):
            vars(self)[argument] = vars(self.config)[argument].values
            self.stop(vars(self)[argument] is None, 'NameError'
                "Required parameter '{}' is missing in the configuration.", argument)
            self.stop(type(vars(self)[argument]) != bool, 'TypeError'
                "'{}' must be a boolean value ({} given).", argument, type(vars(self)[argument]))

        # Check if no more than one mode has been selected
        self.stop(self.from_data == True and self.from_model == True,
            'ValueError', "No more than one traceback mode ('from_data' or 'from_model') can be "
            "selected for '{}'", self.name)

        # self.to_file parameter set to False if self.from_file is True
        if self.from_file and self.to_file:
            log("Output of '{}' from and to file. "
                "The file will not be updated with its own data.", self.name)
            self.to_file = False

    def configure_file(self, series_path=None):
        """ Checks if the file directory and the file itself exist and creates them if needed.
            self.series_path is redefined as the absolute path to the file. Errors are handed
            based on whether the input or output from and to the file is required.
        """

        # series_path parameter
        self.series_path = series_path if series_path is not None \
            else self.config.series_path.values if self.config.series_path.values is not None \
            else path.join(output(check=self.from_file, create=self.to_file),
                '{}.series'.format(self.name))

        # Check if series_path parameter is a string, which must be done before the directory call
        self.stop(type(self.series_path) != str, 'TypeError', "'series_path' must be a string "
            "({} given).", type(self.series_path))

        # self.series_path redefined as the absolute path, default name and directory creation
        self.series_path = path.join(
            directory(collection.base_dir, path.dirname(self.series_path),
                'series_path', check=self.from_file, create=self.to_file),
            '{}.series'.format(self.name) if path.basename(self.series_path) == '' \
                else path.basename(self.series_path))

        # Check if the series file exists, if loading from a file
        self.stop(self.from_file and not path.exists(self.series_path), 'FileNotFoundError',
            "No series file located at '{}'.", self.series_path)

        # Check if the file is a series file
        self.stop(path.splitext(self.series_path)[1].lower() != '.series',
            'TypeError', "'{}' is not a series file (with a .series extension).",
            path.basename(self.series_path))

    def configure_traceback(self):
        """ Checks traceback configuration parameters from data or a model. """

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

        # Radial velocity offset paramaters
        self.rv_offset = self.configure_quantity(self.config.rv_offset)

        # Data configuration for inclusion in a model
        if self.from_data or True:
            self.configure_data()

        # Simulation configuration
        if self.from_model:
            self.configure_model()

        # Logging
        log("Series '{}' ready for traceback.", self.name)

    def configure_data(self):
        """ Checks if traceback and output from data is possible and creates a Data object from a
            CSV file, or a Python dictionary, list, tuple or np.ndarray.
        """

        # Logging
        log("Initializing '{}' from data.", self.name)

        # Check if data is present
        self.stop(self.config.data is None, 'NameError',
            "Required traceback parameter 'data' is missing in the configuration.")

        # Stars creation from data
        from data import Data
        self.data = Data(self)

        # number_of_stars parameter
        self.number_of_stars = len(self.data)

        # Simulation parameters set to None because stars are imported from data
        for parameter in ('age', *self.config.position_parameters,
                *self.config.velocity_parameters):
            vars(self)[parameter] = None

    def configure_model(self):
        """ Checks if traceback and output from a model is possible. """

        # Logging
        log("Initializing '{}' from a model.", self.name)

        # Check if all the necessary parameters are present
        for parameter in ('number_of_stars', 'age', *self.config.position_parameters,
                *self.config.velocity_parameters):
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

        # Check the presence and type of parameter.values
        self.stop(parameter.values is None, 'NameError',
            "Required traceback parameter '{}' is missing in the configuration.", parameter.label)
        self.stop(type(parameter.values) not in (int, float), 'TypeError', "'{}' must be "
            "an integer or a float ({} given).", parameter.label, type(parameter.values))

        # Check if parameter.values is convertible to an integer and greater than or equal to 1
        self.stop(parameter.values % 1 != 0, 'ValueError', "'{}' must be convertible "
            "to an integer ({} given).", parameter.label, parameter.values)
        self.stop(parameter.values < 1, 'ValueError', "'{}' must be "
            "greater than or equal to 1 ({} given).", parameter.label, parameter.values)

        # Conversion to an integer
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

        # Check if parameter.units is a string
        self.stop(type(parameter.units) != str, 'TypeError', "'units' component of '{}' "
            "must be a string ({} given).", parameter.label, type(parameter.units))

        # Check if parameter.units can be converted to Unit
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
        default_physical_type = Unit(
            self.config.default_parameters[parameter.label].units).physical_type
        self.stop(quantity.physical_types.flatten()[0] != default_physical_type, 'ValueError',
            "Unit of '{}' does not have the correct physical type ('{}' given, '{}' required).",
            parameter.label, quantity.physical_types.flatten()[0], default_physical_type)

        # Unit conversion to default units
        return quantity.to()

    def configure_coordinate(self, parameter):
        """ Converts a Parameter into a Quantity object and raises an error if an exception
            occurs in the process. Returns a vector converted to default units for the physical
            type defined by a Variable object.
        """

        # Variables from label and check for invalid label
        if parameter.label in self.config.position_parameters:
            variables = parameter.system.position
        elif parameter.label in self.config.velocity_parameters:
            variables = parameter.system.velocity
        else:
            self.stop(True, 'NameError', "'{}' is not a supported name.", parameter.label)

        # Check the presence and type of parameter.values
        self.stop(parameter.values is None, 'NameError',
            "Required simulation parameter '{}' is missing in the configuration.", parameter.label)
        self.stop(type(parameter.values) not in (tuple, list, np.ndarray), 'TypeError',
            "'values' component of '{}' must be a tuple, list or np.ndarray ({} given).'",
                parameter.label, type(parameter.values))

        # Check if all elements in parameter.values are numerical
        try:
            np.vectorize(float)(parameter.values)
        except:
            self.stop(True, 'ValueError',
                "'values' component of '{}' contains non-numerical elements.", parameter.label)

        # Check the dimensions of parameter.values
        shape = np.array(parameter.values).shape
        ndim = len(shape)
        self.stop(ndim > 2, 'ValueError', "'{}' must have 1 or 2 dimensions ({} given).",
            parameter.label, ndim)
        self.stop(shape[-1] != 3, 'ValueError',
            "'{}' last dimension must have a size of 3 ({} given).", parameter.label, shape[-1])
        self.stop(ndim == 2 and shape[0] not in (1, self.number_of_stars),
            'ValueError',  "'{}' first dimension ({} given) must have a size of 1 or equal "
            "to the number of stars ({} given).", parameter.label, shape[0], self.number_of_stars)

        # Default parameter.units component
        if parameter.units is None:
            parameter.units = [variable.unit.label for variable in variables]

        # Check if parameter.units is a string representing a coordinate system
        if type(parameter.units) == str:
            if parameter.units.lower() in systems.keys():
                if parameter.label in self.config.position_parameters:
                    parameter.units = [variable.usual_unit.unit \
                        for variable in systems[parameter.units.lower()].position]
                elif parameter.label in self.config.velocity_parameters:
                    parameter.units = [variable.usual_unit.unit \
                        for variable in systems[parameter.units.lower()].velocity]
            else:
                parameter.units = [parameter.units]

        # Check the type of parameter.units component
        self.stop(type(parameter.units) not in (tuple, list, np.ndarray), 'TypeError',
            "'units' component of '{}' must be a string, tuple, list or np.ndarray ({} given).",
                parameter.label, type(parameter.values))

        # Check if all elements in parameter.units component can be converted to Unit
        try:
            Unit(np.array(parameter.units, dtype=object))
        except:
            self.stop(True, 'ValueError',
                "'units' components of '{}' must all represent units.", parameter.label)

        # Quantity object creation
        try:
            quantity = Quantity(**vars(parameter))
        except:
            self.stop(True, 'ValueError', "'{}' could not be converted to a Quantity object.",
                parameter.label)

        # Check if physical types are valid based on parameter.system
        physical_types = np.array([variable.physical_type for variable in variables])
        self.stop(not (quantity.physical_types == physical_types).all(), 'ValueError',
            "Units in '{}' do not have the correct physical type "
            "({} given, {} required for '{}' system.)", parameter.label,
             quantity.physical_types.tolist(), physical_types, quantity.system.label)

        # Units conversion to default units
        return quantity.to()

    def add(self, forced=False, default=False, cancel=False):
        """ Adds the series to the collection. If forced is True, any existing series with the
            same name is overwritten, otherwise user input is required to proceed (overwrite, do
            nothing or default). If default is True, then instead of asking for a input, a default
            name is used and the series added to the collection.
        """

        # Series addition and replacement, if needed
        if self.name in collection.series.keys():
            choice = ''
            if not forced:
                if not default:
                    if not cancel:

                        # User input
                        while choice == '':
                            choice = input(
                                "The series '{}' already exists. Do you wish to overwrite"
                                "(Y), keep both (K) or cancel (N)? ".format(self.name))

                            # Loop over question if input could not be interpreted
                            if choice.lower() not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                                print("Could not understand '{}'.".format(choice))
                                choice = ''

                # Default name and series addition to the collection
                if default or choice.lower() in ('k', 'keep'):
                    self.name = collection.default(self.name)
                    self.config.name.values = self.name
                    collection.append(self)

                    # Logging
                    log("Series renamed '{}' and added to the collection.", self.name)

            # Existing series deletion and series addition to the collection
            if forced or choice.lower() in ('y', 'yes'):
                del collection[collection.series[self.name]]
                collection.append(self)

                # Logging
                log("Series '{}' deleted and replaced in the collection.", self.name)

        # Series addition to the collection
        else:
            collection.append(self)

            # Logging
            log("Series '{}' added to the collection.", self.name)

        # Collection series re-initialization
        collection.initialize_series()

    def remove(self):
        """ Removes the series from the collection. """

        # Series deletion from the collection
        if self.name in collection.series.keys():
            del collection[collection.series[self.name]]

        # Collection series re-initialization
        collection.initialize_series()

        # Logging
        log("Series '{}' removed from the collection.", self.name)

    def update(self, parent=None, path=None, args=False, **parameters):
        """ Updates the series by modifying its self.config configuration, re-initializing itself
            and deleting existing groups. The groups are also traced back again if they had been
            traced back before the update. 'parent', 'path', 'args' and '**parameters' are the
            same as in Series._init__.
        """

        # Initialization
        parent = self.config if parent is None else parent
        traceback = True if len(self) > 0 else False

        # Removal of self from the collection, if it exists
        if self.name in collection.series.keys():
            self.remove()

        # Deletion of all parameters and groups
        vars(self).clear()
        self.clear()

        # Series re-initialization and traceback, if needed
        self.configure(parent=parent, path=path, args=args, **parameters)
        if traceback:
            self.traceback()

        # Self re-addition to the collection
        self.add()

        # Logging
        log("Series '{}' updated.", self.name)

        return self

    def copy(self, parent=None, path=None, args=False, **parameters):
        """ Copies the series under a new name. If 'parent', 'path', 'args' or '**parameters' are
            provided, the same as in Series.__init__, this new Series object is updated as well.
            If no new 'name' is provided,  a default name is used instead.
        """

        # Self cloning
        from copy import deepcopy
        clone = deepcopy(self)

        # Clone addition to the collection
        clone.add(default=True)

        # Clone update, if needed
        if parent is not None or path is not None or args == True or len(parameters) > 0:
            clone.update(parent=parent, path=path, args=args, **parameters)

        # Logging
        log("Series '{}' copied.", self.name)

        return clone

    def load(self, series_path=None):
        """ Loads a series from a binary file. self.series_path is defined as the absolute path to
            the file.
        """

        # Logging
        log("Loading '{}' from a file.", self.name)

        # File configuration
        self.from_file = True
        self.to_file = False
        self.configure_file(series_path=series_path)

        # Check if loading has already been done
        if len(self) > 0:
            log("Series '{}' has already been loaded.", self.name)

        # File unpickling
        else:
            from pickle import load
            series_file = open(self.series_path, 'rb')
            parameters, groups = load(series_file)
            series_file.close()

            # Parameters and groups import
            vars(self).update(parameters)
            for group in groups:
                self.append(group)

            # Logging
            log("Series '{}' loaded from '{}'.", self.name, self.series_path)

    def traceback(self):
        """ Traces back Group and embeded Star objects using either imported data or by
            modeling a group based on simulation parameters.
        """

        # Check if at least one mode has been selected
        self.stop(self.from_data == False and self.from_model == False, 'ValueError',
            "Either traceback '{}' from data or a model (none selected).", self.name)

        # Traceback configuration, if traceback is needed
        self.configure_traceback()

        # Check if traceback has already been done
        if len(self) == self.number_of_groups:
            log("Series '{}' has already been traced back.", self.name)

        # Group creation
        else:
            from group import Group
            for name in ['{}-{}'.format(self.name, i) for i in range(1, self.number_of_groups + 1)]:

                # Logging
                log("Tracing back '{}' from {}.", name, 'data' if self.from_data else 'a model')

                # Group traceback
                self.append(Group(self, name=name))

            # Logging
            log("Series '{}' succesfully traced back.", self.name)

    def save(self, series_path=None, forced=False):
        """ Saves a series to a binary file. self.series_path is defined as the actual path to the
            file. If forced, the existing file is overwritten and otherwise user input is required
            to proceed.
        """

        # Logging
        log("Saving '{}' to a file.", self.name)

        # File configuration
        self.from_file = False
        self.to_file = True
        self.configure_file(series_path=series_path)

        # Existing file deletion, if needed
        if path.exists(self.series_path):
            if not forced:
                forced = None
                while forced is None:
                    choice = input("The file located at '{}' already exists. "
                        "Do you wish to overwrite? (Y/N) ".format(self.series_path))
                    forced = True if choice.lower() in ('y', 'yes') \
                        else False if choice.lower() in ('n', 'no') else None
                    if forced is None:
                        print("Could not understand '{}'.".format(choice))
            if forced:
                from os import remove
                remove(self.series_path)
                log("Existing file located at '{}' deleted.", self.series_path)
        else:
            forced = True

        # Parameters export
        if forced:
            parameters = {parameter: vars(self)[parameter] for parameter in vars(self).keys() \
                if parameter not in ('series_path', 'to_file', 'from_file')}

            # File pickling
            from pickle import dump
            series_file = open(self.series_path, 'wb')
            dump((parameters, tuple(group for group in self)), series_file)
            series_file.close()

            # Logging
            log("Series '{}' saved at '{}'.", self.name, self.series_path)

    def create(self):
        """ Either loads a series from a file, or traces a series back from data or a model. If
            needed, the series is also saved. If both self.from_file, and self.from_data or
            self.from_model are True, loading operation supercedes the traceback mode, which is
            ignored and replaced by the value loaded from the file.
        """

        # Load from a file
        if self.from_file:
            self.load()

        # Traceback from data or a model
        elif self.from_data or self.from_model:
            self.traceback()

        # Save to a file
        if self.to_file:
            self.save()

    def stop(self, condition, error, message, *words):
        """ Calls the stop function from tools with self.name, if it has been set. """

        # Addition of series name to stop function call
        stop(condition, error, message, *words,
            name=self.name if 'name' in vars(self) else None, extra=2)
