# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
series.py: Defines the class Series which uses a Config object to create a series of tracebacks
either from a file, data or a model. First, it handles exceptions (name, type, value, shape)
of all parameters, then creates or imports the file, if needed, handles unit, conversions, and
checks for the presence of output and logs directories and creates them, if needed.
"""

import numpy as np
import pandas as pd
from gc import collect
from .init import *
from .output import Output_Series

class Series(list, Output_Series):
    """
    Contains a series of Groups and their embedded Star objects and the methods required to
    initialize a series from a Config object. Values and data in the Config object are, copied,
    checked for errors and converted to usable types such as Quantity objects, and values are
    converted to default units. The number of groups defines how many objects are in a series
    in which all groups have the same Config object as their progenitor.
    """

    def __init__(
            self, parent=None, path=None, args=False, forced=False, default=False,
            cancel=False, logging=True, addition=True, **parameters
        ):
        """
        Initializes a Series object by first configuring it with 'parent', 'path', 'args' and
        '**parameter' and then adding it to the collection. 'forced', 'default', 'cancel' and
        'logging' arguments are passed to self.add function.
        """

        # Configuration
        self.configure(parent, path, args, **parameters)

        # Series addition to the collection
        if addition:
            self.add(forced=forced, default=default, cancel=cancel, logging=logging)

    def configure(self, parent=None, path=None, args=False, **parameters):
        """
        Configures a Series objects from 'parent', an existing Config object, 'path', a string
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
        """
        Checks if all parameters are present and are Config.Parameter objects with all their
        components, and checks for invalid parameters and components. The type of 'label',
        'name' and 'system' components is checked (str), and the 'system' component is converted
        to its corresponding class if it is not None.
        """

        # Check if all parameters are present and are Config.Parameter objects
        for parameter in self.config.default_parameters.keys():
            self.stop(
                parameter not in vars(self.config), 'NameError',
                "Required parameter '{}' is missing in the configuration.", parameter
            )
            self.stop(
                type(vars(self.config)[parameter]) != self.config.Parameter, 'TypeError',
                "'{}' must be a Config.Parameter object ({} given).", parameter,
                type(vars(self.config)[parameter])
            )

            # Check if all components are present
            for component in self.config.Parameter.default_components.keys():
                self.stop(
                    component not in vars(vars(self.config)[parameter]).keys(), 'NameError',
                    "Required component '{}' is missing from the '{}' parameter "
                    "in the configuration.", component, parameter
                )

        # Check for invalid parameters and components
        for parameter_label, parameter in vars(self.config).items():
            self.stop(
                parameter_label not in self.config.default_parameters.keys(), 'NameError',
                "Parameter '{}' in the configuration is invalid.", parameter_label
            )
            for component_label, component in vars(parameter).items():
                self.stop(
                    component_label not in self.config.Parameter.default_components.keys(),
                    'NameError', "Parameter's component '{}' in '{}' is invalid.",
                    component_label, parameter_label
                )

                # Check whether all components, but parameter.values and parameter.units, are
                # strings or None
                if component_label not in ('values', 'units'):
                    self.stop(
                        component is not None and type(component) not in (str, System), 'TypeError',
                        "'{}' component in '{}' parameter must be a string or None ('{}' given.)",
                        component_label, parameter_label, type(component)
                    )

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
                self.stop(
                    parameter.system.lower() not in systems.keys(), 'ValueError',
                    "'system' component of '{}' is invalid ({} required, {} given).",
                    parameter.label, list(systems.keys()), parameter.system
                )
                parameter.system = systems[parameter.system.lower()]
            elif default_parameter.system is not None:
                parameter.system = systems[default_parameter.system]

    def configure_series(self):
        """
        Checks basic series parameters. Checks if self.name is a string and creates a default
        value if needed. Checks the type of the modes provided in the configuration and checks
        if their values are valid. self.to_file is set to False if self.from_file is True.
        Moreover, self.date is defined and self.metrics configured.
        """

        # Check if name is a string
        if self.config.name.values is not None:
            self.stop(
                type(self.config.name.values) != str, 'TypeError',
                "'name' must be a string ('{}' given).", type(self.config.name.values)
            )

        # Series name
        self.name = (
            collection.default_name() if self.config.name.values is None
            else self.config.name.values
        )

        # Current date and time
        from time import strftime
        self.date = strftime('%Y-%m-%d %H:%M:%S')

        # from_data, from_model, from_file, to_file and size_metrics parameters
        for argument in (
            'from_data', 'from_model', 'from_file', 'to_file',
            'size_metrics', 'cov_metrics', 'cov_robust_metrics',
            'cov_sklearn_metrics', 'mad_metrics', 'mst_metrics',
            'pca', 'timer'
        ):
            vars(self)[argument] = vars(self.config)[argument].values
            self.stop(
                vars(self)[argument] is None, 'NameError',
                "Required parameter '{}' is missing in the configuration.", argument
            )
            self.stop(
                type(vars(self)[argument]) != bool, 'TypeError',
                "'{}' must be a boolean ({} given).", argument, type(vars(self)[argument])
            )

        # self.to_file parameter set to False if self.from_file is True
        if self.from_file and self.to_file:
            log("Output of '{}' from and to file. "
                "The file will not be updated with its own data.", self.name)
            self.to_file = False

        # Import the association size metrics file and skip the header line
        metrics_file = join(path.dirname(__file__), 'resources/association_size_metrics.csv')
        metrics_dataframe = pd.read_csv(metrics_file, delimiter=';')

        # Association size metric configuration
        self.metrics = []
        for index, metric in metrics_dataframe.iterrows():
            self.Metric(self, metric)

    def configure_mode(self, mode=None):
        """ Checks if selected mode is valid, and if one and no more than one mode has been
            selected.
        """

        # Mode import
        if mode is not None:
            self.stop(
                type(mode) != str, 'TypeError', "'mode' must be a string ({} given).", type(mode)
            )
            if mode.lower().replace('_', '').replace('from', '') == 'data':
                self.from_data = True
                self.from_model = False
            elif mode.lower().replace('_', '').replace('from', '') == 'model':
                self.from_data = False
                self.from_model = True
            else:
                self.stop(True, 'ValueError', "Could not understand mode '{}'", mode)

        # Check if at least one mode has been selected
        self.stop(
            self.from_data == False and self.from_model == False, 'ValueError',
            "Either traceback '{}' from data or a model (None selected).", self.name
        )

        # Check if no more than one mode has been selected
        self.stop(
            self.from_data == True and self.from_model == True,
            'ValueError', "No more than one traceback mode ('from_data' or 'from_model') "
            "can be selected for '{}'", self.name
        )

    def configure_file(self, file_path=None):
        """
        Checks if the file directory and the file itself exist and creates them if needed.
        self.file_path is redefined as the absolute path to the file. Errors are handed
        based on whether the input or output from and to the file is required.
        """

        # file_path parameter
        self.file_path = (
            file_path if file_path is not None
            else self.config.file_path.values if self.config.file_path.values is not None
            else output(check=self.from_file, create=self.to_file) + '/'
        )

        # Check if file_path parameter is a string, which must be done before the directory call
        self.stop(
            type(self.file_path) != str, 'TypeError',
            "'file_path' must be a string ({} given).", type(self.file_path)
        )

        # self.file_path redefined as the absolute path, default name and directory creation
        self.file_path = path.join(
            directory(collection.base_dir, path.dirname(self.file_path),
                'file_path', check=self.from_file, create=self.to_file),
            '{}.series'.format(self.name) if path.basename(self.file_path) == ''
            else path.basename(self.file_path)
        )

        # Check if the series file exists, if loading from a file
        self.stop(
            self.from_file and not path.exists(self.file_path), 'FileNotFoundError',
            "No series file located at '{}'.", self.file_path
        )

        # Check if the file is a series file
        self.stop(
            path.splitext(self.file_path)[1].lower() != '.series', 'TypeError',
            "'{}' is not a series file (with a .series extension).", path.basename(self.file_path)
        )

    def configure_traceback(self):
        """Checks traceback configuration parameters for 'From data' and 'From model' modes."""

        # number_of_groups parameter
        self.number_of_groups = self.configure_integer(self.config.number_of_groups)

        # number_of_steps parameter with one more step to account for t = 0
        self.number_of_steps = self.configure_integer(self.config.number_of_steps) + 1

        # initial_time parameter
        self.initial_time = self.configure_quantity(self.config.initial_time)

        # final_time parameter
        self.final_time = self.configure_quantity(self.config.final_time)
        self.stop(
            not self.final_time > self.initial_time, 'ValueError',
            "'final_time' must be greater than initial_time ({} and {} given).",
            self.final_time, self.initial_time
        )

        # duration, timesteps and time parameters
        self.duration = self.final_time - self.initial_time
        self.timestep = self.duration / (self.number_of_steps - 1)
        self.time = np.linspace(
            self.initial_time.value, self.final_time.value, self.number_of_steps
        )

        # rv_shift paramater
        self.rv_shift = self.configure_quantity(self.config.rv_shift)

        # data_rv_shifts parameter
        self.data_rv_shifts = self.config.data_rv_shifts.values
        self.stop(
            self.data_rv_shifts is None, 'NameError',
            "Required traceback parameter 'data_rv_shifts' is missing in the configuration."
        )
        self.stop(
            type(self.data_rv_shifts) != bool, 'TypeError',
            "'data_rv_shifts' must be a boolean ({} given).", type(self.data_rv_shifts)
        )

        # data_errors parameter
        self.data_errors = self.config.data_errors.values
        self.stop(
            self.data_errors is None, 'NameError',
            "Required traceback parameter 'data_errors' is missing in the configuration."
        )
        self.stop(
            type(self.data_errors) != bool, 'TypeError',
            "'data_errors' must be a boolean ({} given).", type(self.data_errors)
        )

        # number_of_iterations parameter
        self.number_of_iterations = self.configure_integer(self.config.number_of_iterations)
        self.stop(
            self.number_of_iterations <= 0, 'ValueError',
            "'number_of_iterations' must be greater to 0 ({} given).", self.number_of_iterations
        )

        # iteration_fraction parameter
        self.iteration_fraction = self.configure_quantity(self.config.iteration_fraction).value
        self.stop(
            self.iteration_fraction <= 0 or self.iteration_fraction > 1, 'ValueError',
            "'iteration_fraction' must be between 0 and 1 ({} given).", self.iteration_fraction
        )

        # cutoff parameter
        if self.config.cutoff.values is not None:
            self.stop(
                type(self.config.cutoff.values) not in (int, float), 'TypeError',
                "'{}' must be an integer, float or None ({} given).",
                self.config.cutoff.label, type(self.config.cutoff.values)
            )
            self.stop(
                self.config.cutoff.values <= 0.0, 'ValueError',
                "'cutoff' must be greater to 0.0 ({} given).", self.config.cutoff.values
            )
        self.cutoff = self.config.cutoff.values

        # sample parameter
        self.sample = self.config.sample.values
        self.stop(
            self.sample is None, 'NameError',
            "Required traceback parameter 'sample' is missing in the configuration."
        )
        self.stop(
            type(self.sample) != str, 'TypeError',
            "'sample' must be a string ({} given).", type(self.sample)
        )
        self.sample = self.sample.lower()
        self.stop(
            self.sample not in ('outliers', 'sample', 'subsample'), 'ValueError',
            "'sample' must be outliers, sample or subsample ({} given).", self.sample
        )

        # potential parameter
        self.potential = self.config.potential.values
        if self.potential is not None:
            self.stop(
                type(self.potential) != str, 'TypeError',
                "'potential' must be a string or None ({} given).", type(self.potential)
            )

            # Converts the potential string to a galpy.potential object
            try:
                exec(
                    f'from galpy.potential.mwpotentials import {self.potential} as potential',
                    globals()
                )
            except:
                self.stop(
                    True, 'ValueError', "'potenial' is invalid ({} given).", self.potential
                )
            self.potential = potential

        # Data configuration
        if self.from_data:
            self.configure_data()

        # Model configuration
        if self.from_model:
            self.configure_model()

        # Logging
        log("'{}' series ready for traceback.", self.name)

    def configure_data(self):
        """
        Checks if traceback and output from data is possible and creates a Data object from
        a CSV file, or a Python dictionary, list, tuple or np.ndarray. Simulation parameters
        are also set to None or False if the traceback is from data. This allows to use data
        errors without resetting simulation parameters.
        """

        # Logging
        log("Initializing '{}' series from data.", self.name)

        # Check if data is present
        self.stop(
            self.config.data is None, 'NameError',
            "Required traceback parameter 'data' is missing in the configuration."
        )

        # Stars creation from data
        from .data import Data
        self.data = Data(self)

        # number_of_stars parameter
        if self.from_data:
            self.number_of_stars = len(self.data)

            # Simulation parameters set to None or False if stars are imported from data
            for parameter in (
                    'age', *self.config.position_parameters, *self.config.velocity_parameters
                ):
                vars(self)[parameter] = None

    def configure_model(self):
        """ Checks if traceback and output from a model is possible. """

        # Logging
        log("Initializing '{}' series from a model.", self.name)

        # Check if all the necessary parameters are present
        for parameter in (
                'number_of_stars', 'age',
                *self.config.position_parameters, *self.config.velocity_parameters
            ):
            self.stop(
                vars(self.config)[parameter].values is None, 'NameError',
                "Required simulation parameter '{}' is missing in the configuration.", parameter
            )

        # number_of_stars parameter
        self.number_of_stars = self.configure_integer(self.config.number_of_stars)

        # age parameter
        self.age = self.configure_quantity(self.config.age)
        self.stop(
            self.age.value < 0.0, 'ValueError',
            "'age' must be greater than or equal to 0.0 Myr ({} given).", self.age
        )

        # position parameter
        self.position = self.configure_coordinate(self.config.position)

        # position_error parameter
        self.position_error = self.configure_coordinate(self.config.position_error)

        # position_scatter parameter
        self.position_scatter = self.configure_coordinate(self.config.position_scatter)

        # velocity parameter
        self.velocity = self.configure_coordinate(self.config.velocity)

        # velocity_error parameter
        self.velocity_error = self.configure_coordinate(self.config.velocity_error)

        # velocity_scatter parameter
        self.velocity_scatter = self.configure_coordinate(self.config.velocity_scatter)

        # Data configured to use actual error measurements or radial velocity shift
        if self.data_errors or self.data_rv_shifts:
            self.configure_data()

            # Check if the data length matches the number of stars
            self.stop(
                len(self.data) == 0, 'ValueError', "If 'data_errors' or 'data_rv_shift' "
                "is True, the data length must be greater than 0."
                )

        # Data set to None since measurement errors and radial velocity shifts are simulated
        else:
            self.data = None

    def configure_integer(self, parameter):
        """Checks if an integer value is valid and converts it if needed."""

        # Check the presence and type of parameter.values
        self.stop(
            parameter.values is None, 'NameError',
            "Required traceback parameter '{}' is missing in the configuration.", parameter.label
            )
        self.stop(
            type(parameter.values) not in (int, float), 'TypeError',
            "'{}' must be an integer or a float ({} given).",
            parameter.label, type(parameter.values)
            )

        # Check if parameter.values is convertible to an integer and greater than or equal to 1
        self.stop(
            parameter.values % 1 != 0, 'ValueError',
            "'{}' must be convertible to an integer ({} given).",
            parameter.label, parameter.values
            )
        self.stop(
            parameter.values < 0, 'ValueError',
            "'{}' must be greater than or equal to 1 ({} given).",
            parameter.label, parameter.values
            )

        # Conversion to an integer
        return int(parameter.values)

    def configure_quantity(self, parameter):
        """
        Checks if a value is valid and converts it to default units if needed.
        """

        # Check the presence and type of parameter.values component
        self.stop(
            parameter.values is None, 'NameError',
            "Required traceback parameter '{}' is missing in the configuration.", parameter.label
            )
        self.stop(
            type(parameter.values) not in (int, float), 'TypeError',
            "'{}' must be an integer or a float ({} given).",
            parameter.label, type(parameter.values)
            )

        # Default units component
        if parameter.units is None:
            parameter.units = self.config.default_parameters[parameter.label].units

        # Check if parameter.units is a string
        self.stop(
            type(parameter.units) != str, 'TypeError',
            "'units' component of '{}' must be a string ({} given).",
            parameter.label, type(parameter.units)
            )

        # Check if parameter.units can be converted to Unit
        try:
            Unit(parameter.units)
        except:
            self.stop(
                True, 'ValueError',
                "'units' component of '{}' must represent a unit.", parameter.label
                )

        # Quantity object creation
        try:
            quantity = Quantity(**vars(parameter))
        except:
            self.stop(
                True, 'ValueError',
                "'{}' could not be converted to a Quantity object.", parameter.label
                )

        # Check if the physical type is valid
        default_physical_type = Unit(
            self.config.default_parameters[parameter.label].units
        ).physical_type
        self.stop(
            quantity.physical_types.flatten()[0] != default_physical_type, 'ValueError',
            "Unit of '{}' does not have the correct physical type ('{}' given, '{}' required).",
            parameter.label, quantity.physical_types.flatten()[0], default_physical_type
        )

        # Unit conversion to default units
        return quantity.to()

    def configure_coordinate(self, parameter):
        """
        Converts a Parameter into a Quantity object and raises an error if an exception
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
        self.stop(
            parameter.values is None, 'NameError',
            "Required simulation parameter '{}' is missing in the configuration.", parameter.label
        )
        self.stop(
            type(parameter.values) not in (tuple, list, np.ndarray), 'TypeError',
            "'values' component of '{}' must be a tuple, list or np.ndarray ({} given).'",
            parameter.label, type(parameter.values)
        )

        # Check if all elements in parameter.values are numerical
        try:
            np.vectorize(float)(parameter.values)
        except:
            self.stop(
                True, 'ValueError',
                "'values' component of '{}' contains non-numerical elements.", parameter.label
            )

        # Check the dimensions of parameter.values
        shape = np.array(parameter.values).shape
        ndim = len(shape)
        self.stop(
            ndim > 2, 'ValueError',
            "'{}' must have 1 or 2 dimensions ({} given).", parameter.label, ndim
        )
        self.stop(
            shape[-1] != 3, 'ValueError',
            "'{}' last dimension must have a size of 3 ({} given).", parameter.label, shape[-1]
        )
        self.stop(
            ndim == 2 and shape[0] not in (1, self.number_of_stars), 'ValueError',
            "'{}' first dimension ({} given) must have a size of 1 or equal to the "
            "number of stars ({} given).", parameter.label, shape[0], self.number_of_stars
        )

        # Default parameter.units component
        if parameter.units is None:
            parameter.units = [variable.unit.label for variable in variables]

        # Check if parameter.units is a string representing a coordinate system
        if type(parameter.units) == str:
            if parameter.units.lower() in systems.keys():
                if parameter.label in self.config.position_parameters:
                    parameter.units = [
                        variable.usual_unit.unit
                        for variable in systems[parameter.units.lower()].position
                    ]
                elif parameter.label in self.config.velocity_parameters:
                    parameter.units = [
                        variable.usual_unit.unit
                        for variable in systems[parameter.units.lower()].velocity
                    ]
            else:
                parameter.units = (parameter.units,)

        # Check the type of parameter.units component
        self.stop(
            type(parameter.units) not in (tuple, list, np.ndarray), 'TypeError',
            "'units' component of '{}' must be a string, tuple, list or np.ndarray ({} given).",
            parameter.label, type(parameter.values)
        )

        # Check if all elements in parameter.units component can be converted to Unit
        try:
            Unit(np.array(parameter.units, dtype=object))
        except:
            self.stop(
                True, 'ValueError',
                "'units' components of '{}' must all represent units.", parameter.label
            )

        # Quantity object creation
        try:
            quantity = Quantity(**vars(parameter))
        except:
            self.stop(
                True, 'ValueError',
                "'{}' could not be converted to a Quantity object.", parameter.label
            )

        # Check if physical types are valid based on parameter.system
        physical_types = np.array([variable.physical_type for variable in variables])
        self.stop(
            not (quantity.physical_types == physical_types).all(), 'ValueError',
            "Units in '{}' do not have the correct physical type "
            "({} given, {} required for '{}' system.)",
            parameter.label, quantity.physical_types.tolist(),
            physical_types.tolist(), quantity.system
        )

        # Units conversion to default units
        return quantity.to()

    class Metric():
        """
        Association size metric including its average values, age error, age at minimum and
        minimum.
        """

        def __init__(self, series, metric):
            """ Initializes an average association size metric. """

            # Initialization
            self.series = series
            self.label = eval(metric['label'])
            self.name = np.array(eval(metric['name']))
            self.latex_short = np.array(eval( metric['latex_short']))
            self.latex_long = np.array(eval( metric['latex_long']))
            self.valid = np.array(eval(metric['valid']))
            self.order = int(metric['order'])
            self.age_shift = np.array(eval(metric['age_shift']))
            self.status = False
            self.ndim = self.valid.size

            # Add the association size metric to the series
            vars(self.series)[self.label] = self
            self.series.metrics.append(self)

        def __call__(self):
            """
            Computes the values and errors of an association size metric. If the series has
            been initialized from data, the first group (index 0) is used for values, age and
            mininum. Other groups are used to compute uncertainty due to measurement errors.
            If only one group is present, the internal error is used as the total error.
            """

            # Average association size metric for stars from data
            if self.status and self.series.from_data:

                # Value and errors
                self.value = vars(self.series[0])[self.label].value
                self.value_int_error = vars(self.series[0])[self.label].value_int_error
                self.value_ext_error = (
                    np.std([vars(group)[self.label].value for group in self.series[1:]], axis=0)
                    if self.series.number_of_groups > 1 else np.zeros(self.value.shape)
                )
                self.values = (
                    np.array([vars(group)[self.label].values for group in self.series[1:]])
                    if self.series.number_of_groups > 1
                    else vars(self.series[0])[self.label].values[None]
                )
                self.value_error = np.std(self.values, axis=(0, 1))
                self.value_error_quad = (self.value_int_error**2 + self.value_ext_error**2)**0.5

                # Age and errors
                self.age = vars(self.series[0])[self.label].age
                self.age_int_error = vars(self.series[0])[self.label].age_int_error
                self.age_ext_error = (
                    np.std([vars(group)[self.label].age for group in self.series[1:]], axis=0)
                    if self.series.number_of_groups > 1 else np.zeros(self.age.shape)
                )
                self.ages = (
                    np.array([vars(group)[self.label].ages for group in self.series[1:]])
                    if self.series.number_of_groups > 1
                    else vars(self.series[0])[self.label].ages[None]
                )
                self.age_error = np.std(self.ages, axis=(0, 1))
                self.age_error_quad = (self.age_int_error**2 + self.age_ext_error**2)**0.5

                # Minimum and errors
                self.min = vars(self.series[0])[self.label].min
                self.min_int_error = vars(self.series[0])[self.label].min_int_error
                self.min_ext_error = (
                    np.std([vars(group)[self.label].min for group in self.series[1:]], axis=0)
                    if self.series.number_of_groups > 1 else np.zeros(self.min.shape)
                )
                self.minima = (
                    np.array([vars(group)[self.label].minima for group in self.series[1:]])
                    if self.series.number_of_groups > 1
                    else vars(self.series[0])[self.label].minima[None]
                )
                self.min_error = np.std(self.minima, axis=(0, 1))
                self.min_error_quad = (self.min_int_error**2 + self.min_ext_error**2)**0.5

                # Age shift based on simulation
                self.age_ajusted = self.age + self.age_shift

                # Minimum change
                self.min_change = (self.min / self.value[0] - 1.) * 100.

            # Average association size metric for stars from a model
            elif self.status and self.series.from_model:
                self.values = np.mean(
                    [vars(group)[self.label].values for group in self.series], axis=0
                )

                # Value and errors
                self.value = np.mean(
                    [vars(group)[self.label].value for group in self.series], axis=0
                )
                self.value_int_error = np.mean(
                    [vars(group)[self.label].value_int_error for group in self.series], axis=0
                )
                self.value_ext_error = np.std(
                    [vars(group)[self.label].value for group in self.series], axis=0
                )
                self.values = np.array([vars(group)[self.label].values for group in self.series])
                self.value_error = np.std(self.values, axis=(0, 1))
                self.value_error_quad = (self.value_int_error**2 + self.value_ext_error**2)**0.5

                # Age and errors
                self.age = np.mean(
                    [vars(group)[self.label].age for group in self.series], axis=0
                )
                self.age_int_error = np.mean(
                    [vars(group)[self.label].age_int_error for group in self.series], axis=0
                )
                self.age_ext_error = np.std(
                    [vars(group)[self.label].age for group in self.series], axis=0
                )
                self.ages = np.array([vars(group)[self.label].ages for group in self.series])
                self.age_error = np.std(self.ages, axis=(0, 1))
                self.age_error_quad = (self.age_int_error**2 + self.age_ext_error**2)**0.5

                # Minimum and errors
                self.min = np.mean(
                    [vars(group)[self.label].min for group in self.series], axis=0
                )
                self.min_int_error = np.mean(
                    [vars(group)[self.label].min_int_error for group in self.series], axis=0
                )
                self.min_ext_error = np.std(
                    [vars(group)[self.label].min for group in self.series], axis=0
                )
                self.minima = np.array([vars(group)[self.label].minima for group in self.series])
                self.min_error = np.std(self.minima, axis=(0, 1))
                self.min_error_quad = (self.min_int_error**2 + self.min_ext_error**2)**0.5

                # Age shift based on simulation
                self.age_shift = self.series.age.value - self.age
                self.age_ajusted = self.series.age.value

                # Minimum change
                self.min_change = (self.min / self.value[0] - 1.) * 100.

            # Null average association size metric
            else:
                null_1d = np.full((self.ndim,), np.nan)
                null_2d = np.full((self.series.number_of_steps, self.ndim), np.nan)
                null_3d = np.full(
                    (
                        self.series.number_of_groups - (1 if self.series.from_data else 0),
                        self.series.number_of_iterations,
                        self.ndim
                    ), np.nan
                )
                null_4d = np.full(
                    (
                        self.series.number_of_groups - (1 if self.series.from_data else 0),
                        self.series.number_of_iterations,
                        self.series.number_of_steps,
                        self.ndim
                    ), np.nan
                )

                # Value and errors
                self.value = null_2d
                self.value_int_error = null_2d
                self.value_ext_error = null_2d
                self.values = null_4d
                self.value_error = null_2d
                self.value_error_quad = null_2d

                # Age and errors
                self.age = null_1d
                self.age_int_error = null_1d
                self.age_ext_error = null_1d
                self.ages = null_3d
                self.age_error = null_1d
                self.age_error_quad = null_1d

                # Minimum and errors
                self.min = null_1d
                self.min_int_error = null_1d
                self.min_ext_error = null_1d
                self.minima = null_3d
                self.min_error = null_1d
                self.min_error_quad = null_1d

                # Age shift based on simulation
                self.age_shift = null_1d
                self.age_ajusted = null_1d

                # Minimum change
                self.min_change = null_1d

            # Box size (Myr) converted to the corresponding number of steps
            # box_size = 1. # Transform in to parameter
            # box_size = 0.01
            # box_size = int(
            #     box_size * self.series.number_of_steps /
            #     (self.series.final_time.value - self.series.initial_time.value)
            # )
            # if box_size > 1:
            #     box = np.squeeze(np.tile(
            #         np.ones(box_size) / box_size, (1 if self.values.ndim == 1 else 3, 1)
            #     )).T
            #     box = np.ones(box_size) / box_size

                # Smoothing with moving average
                # if self.values.ndim == 1:
                #     self.values = np.convolve(self.values, box, mode='same')
                # else:
                #     self.values = np.apply_along_axis(
                #         lambda x: np.convolve(x, box, mode='same'), axis=0, arr=self.values
                #     )

    def add(self, forced=False, default=False, cancel=False, logging=True):
        """
        Adds the series to the collection. If 'forced' is True, any existing series with the
        same name is overwritten, otherwise user input is required to proceed (overwrite, do
        nothing or default). If 'default' is True, then instead of asking for an input, a
        default name is used and the series added to the collection. If 'cancel' is True,
        instead of asking for an input, the series isn't added to the collection.
        """

        # Check if a series already exists
        if self.name in collection.series.keys():
            choice = None
            self.stop(
                type(forced) != bool, 'TypeError',
                "'forced' must be a boolean ({} given).", type(forced)
            )
            if not forced:
                self.stop(
                    type(default) != bool, 'TypeError',
                    "'default' must be a default ({} given).", type(default)
                )
                if not default:
                    self.stop(
                        type(cancel) != bool, 'TypeError',
                        "'cancel' must be a boolean ({} given).", type(cancel)
                    )
                    if not cancel:

                        # User input
                        while choice == None:
                            choice = input(
                                "The series '{}' already exists. Do you wish to overwrite (Y), "
                                "keep both (K) or cancel (N)? ".format(self.name)
                            ).lower()

                            # Loop over question if input could not be interpreted
                            if choice not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                                print("Could not understand '{}'.".format(choice))
                                choice = None

                    # Addition cancellation and series deletion
                    if cancel or choice in ('n', 'no'):

                        # Logging
                        log(
                            "'{}' series was not added to the collection because a series "
                            "with the same name already exists.", self.name, logging=logging
                        )
                        del self

                # Default name and addition to the collection
                if default or choice in ('k', 'keep'):
                    name = deepcopy(self.name)
                    self.name = collection.default_name(self.name)
                    self.config.name.values = self.name
                    collection.append(self)

                    # Logging
                    log(
                        "'{}' series renamed '{}' and added to the collection.",
                        name, self.name, logging=logging
                    )

            # Existing series deletion and addition to the collection
            if forced or choice in ('y', 'yes'):
                del collection[collection.series[self.name]]
                collection.append(self)

                # Logging
                log(
                    "Existing '{}' series deleted and new series added to the collection.",
                    self.name, logging=logging
                )

        # Addition to the collection
        else:
            collection.append(self)

            # Logging
            log("'{}' series added to the collection.", self.name, logging=logging)

        # Collection series re-initialization
        collection.initialize_series()

    def remove(self, logging=True):
        """ Removes the series from the collection. """

        # Check if the series is in the collection
        self.stop(
            self.name not in collection.series.keys(), 'NameError',
            "'{}' is not in the collection", self.name
        )

        # Series deletion from the collection
        del collection[collection.series[self.name]]

        # Collection series re-initialization
        collection.initialize_series()

        # Logging
        log("'{}' series removed from the collection.", self.name, logging=logging)

        # Series deletion
        del self

    def reset(self, logging=True):
        """
        Clears all groups, as well as all parameter and returns the series to its original
        configuration. This effectively undo a load or traceback operation.
        """

        # Initialization
        parent = self.config

        # Deletion of all parameters and groups
        vars(self).clear()
        self.clear()

        # Series re-initialization
        self.configure(parent=parent)

        # Logging
        log("'{}' series reset.", self.name, logging=logging)

    def update(
            self, parent=None, path=None, args=False, create=True,
            forced=False, default=False, cancel=False, logging=True, **parameters
        ):
        """
        Updates the series by modifying its self.config configuration, re-initializing itself
        and deleting existing groups. The groups are also recreated if they had been created
        before the update, unless 'create' is set to False. 'parent', 'path', 'args' and
        '**parameters' are the same as in Series.__init__. If 'name' parameter is the only
        one modified, existing groups are kept and also renamed.
        """

        # Initialization
        parent = self.config if parent is None else parent
        self.stop(
            type(create) != bool, 'TypeError',
            "'create' must be a boolean ({} given).", type(create)
        )
        create = False if len(self) == 0 else create

        # New configuration
        new_config = Config(parent, path, args, **parameters)

        # Check what parameters, if any, are modified
        new_parameters = [
            parameter for parameter in vars(self.config)
            if vars(vars(new_config)[parameter]) != vars(vars(self.config)[parameter])
        ]

        # No parameters are modified
        if len(new_parameters) == 0:

            # Logging
            log("'{}' series unchanged.", self.name, logging=logging)

        # Only 'name' parameter is modified
        elif len(new_parameters) == 1 and 'name' in new_parameters:

            # Check if name is a string
            if new_config.name.values is not None:
                self.stop(
                    type(new_config.name.values) != str, 'TypeError',
                    "'name' must be a string ('{}' given).", type(new_config.name.values)
                )

            # Series removal from the collection
            if self.name in collection.series.keys():
                self.remove(logging=False)

            # Change groups and series names
            name = deepcopy(self.name)
            self.name = (
                collection.default_name()
                if new_config.name.values is None else new_config.name.values
            )
            self.config.name = new_config.name
            for group in self:
                group.name = group.name.replace(name, self.name)

            # Series re-addition to the collection
            self.add(forced=forced, default=default, cancel=cancel, logging=False)

            # Logging
            log("'{}' series renamed '{}'.", name, self.name, logging=logging)

        # Any other parameter is modified
        else:

            # Check if a traceback has already been done
            self.stop(
                type(forced) != bool, 'TypeError',
                "'forced' must be a boolean ({} given).", type(forced)
            )
            if len(self) > 0:

                # User input
                if not forced:
                    forced = None
                    while forced is None:
                        choice = input(
                            "'{}' series has already been traced back. Do you wish "
                            "to delete existing groups? (Y/N) ".format(self.name)
                        ).lower()
                        forced = (
                            True if choice in ('y', 'yes')
                            else False if choice in ('n', 'no') else None
                        )
                        if forced is None:
                            print("Could not understand '{}'.".format(choice))

                # Updating cancellation
                if not forced:

                    # Logging
                    log(
                        "'{}' series was not updated because it has already been traced back.",
                        self.name, logging=logging
                    )
            else:
                forced = True

            # Updated series creation
            if forced:
                updated_series = Series(
                    parent=parent, path=path, args=args, addition=False, **parameters
                )

                # Groups creation, if needed
                if create:
                    updated_series.create()

                # Save current series name
                if 'name' in new_parameters:
                     name = deepcopy(self.name)

                # Current series removal from the collection
                if self.name in collection.series.keys():
                    self.remove(logging=False)

                # Current series re-definition with the updated series
                del self
                self = updated_series

                # Updated series addition to the collection
                self.add(forced=forced, default=default, cancel=cancel, logging=False)

                # Logging
                if 'name' in new_parameters:
                    log("'{}' series renamed '{}'.", name, self.name, logging=logging)
                log("'{}' series updated.", self.name, logging=logging)

        return self

    def copy(self, parent=None, path=None, args=False, logging=True, traceback=True, **parameters):
        """
        Copies the series under a new name. If 'parent', 'path', 'args' or '**parameters' are
        provided, the same as in Series.__init__, this new Series object is updated as well.
        If no new 'name' is provided, a default name is used instead.
        """

        # Self cloning
        clone = deepcopy(self)

        # Clone addition to the collection
        clone.add(default=True)

        # Clone update, if needed
        if parent is not None or path is not None or args == True or len(parameters) > 0:
            clone.update(
                parent=parent, path=path, args=args, logging=False,
                traceback=traceback, **parameters
            )

        # Logging
        log("'{}' series copied to '{}'.", self.name, clone.name, logging=logging)

        return clone

    def load(self, file_path=None, forced=False, logging=True):
        """
        Loads a series from a binary file. self.file_path is defined as the absolute path to
        the file. If forced, the existing groups are overwritten, otherwise user input is
        required to proceed if groups already exists.
        """

        # Check if a traceback has already been done
        self.stop(
            type(forced) != bool, 'TypeError',
            "'forced' must be a boolean ({} given).", type(forced)
        )
        if len(self) > 0:

            # User input
            if not forced:
                forced = None
                while forced is None:
                    choice = input(
                        "'{}' series has already been traced back. Do you wish "
                        "to overwrite existing groups? (Y/N) ".format(self.name)
                    ).lower()
                    forced = (
                        True if choice in ('y', 'yes')
                        else False if choice in ('n', 'no') else None
                    )
                    if forced is None:
                        print("Could not understand '{}'.".format(choice))

            # Groups deletion
            if forced:
                self.clear()

                # Logging
                log("Existing groups in series '{}' deleted.", self.name, logging=logging)

            # Loading cancellation
            else:

                # Logging
                log(
                    "'{}' series was not loaded because it has already been traced back.",
                    self.name, logging=logging
                )
        else:
            forced = True

        # Loading
        if forced:

            # File configuration
            self.from_file = True
            self.to_file = False
            self.configure_file(file_path=file_path)

            # Series unpickling
            from pickle import load
            file = open(self.file_path, 'rb')
            parameters, groups = load(file)
            file.close()

            # Parameters deletion and series removal from the collection
            self.remove(logging=False)
            for parameter in [
                    parameter for parameter in vars(self).keys()
                    if parameter not in ('file_path', 'from_file', 'to_file')
                ]:
                del vars(self)[parameter]

            # Parameters and groups import, and series re-addition to the collection
            vars(self).update(parameters)
            for group in groups:
                self.append(group)
            self.add(logging=False)

            # Logging
            log("'{}' series loaded from '{}'.", self.name, self.file_path, logging=logging)

    def traceback(self, forced=False, logging=True, mode=None):
        """
        Traces back Group and embeded Star objects using either imported data or by modeling a
        group based on simulation parameters.
        """

        # Check if a traceback has already been done
        self.stop(
            type(forced) != bool, 'TypeError',
            "'forced' must be a boolean ({} given).", type(forced)
        )
        if len(self) > 0:

            # User input
            if not forced:
                forced = None
                while forced is None:
                    choice = input(
                        "'{}' series has already been traced back. Do you wish "
                        "to overwrite existing groups? (Y/N) ".format(self.name)
                    ).lower()
                    forced = (
                        True if choice in ('y', 'yes')
                        else False if choice in ('n', 'no') else None
                    )
                    if forced is None:
                        print("Could not understand '{}'.".format(choice))

            # Groups deletion
            if forced:
                self.clear()
                log("Existing groups in series '{}' deleted.", self.name, logging=logging)

            # Traceback cancellation
            else:
                self.from_data = self.config.from_data.values
                self.from_model = self.config.from_model.values
                log(
                    "'{}' series was not loaded because it has already been traced back.",
                    self.name, logging=logging
                )
        else:
            forced = True

        # Groups creation
        if forced:

            # Traceback configuration
            from .group import Group
            self.configure_traceback()

            # Traceback
            for number in range(self.number_of_groups):

                # Group name
                name = f'{self.name}-{number}'

                # Logging
                log(
                    "Tracing back '{}' group from {}.", name,
                    'data' if self.from_data else 'a model',
                    display=True, logging=logging
                )

                # Group traceback
                self.append(Group(self, number, name))

            # Logging
            log("'{}' series succesfully traced back.", self.name, logging=logging)

            # Compute average association size metrics
            for metric in self.metrics:
                metric()

    def save(self, file_path=None, forced=False, default=False, cancel=False, logging=True):
        """
        Saves a series to a binary file. self.file_path is defined as the actual path to the
        file. If forced, the existing file is overwritten, otherwise user input is required
        to proceed if a file already exists.
        """

        def save(self):
            """Saves the series."""

            # Series pickling
            from pickle import dump
            file = open(self.file_path, 'wb')
            dump(
                (
                    {
                        parameter: vars(self)[parameter] for parameter in vars(self).keys()
                        if parameter not in ('file_path', 'to_file', 'from_file')
                    }, tuple(group for group in self)
                ), file
            )
            file.close()

        # File configuration
        self.from_file = False
        self.to_file = True
        self.configure_file(file_path=file_path)

        # Check if a file already exists
        if path.exists(self.file_path):
            choice = None
            self.stop(
                type(forced) != bool, 'TypeError',
                "'forced' must be a boolean ({} given).", type(forced)
            )
            if not forced:
                self.stop(
                    type(default) != bool, 'TypeError',
                    "'default' must be a default ({} given).", type(default)
                )
                if not default:
                    self.stop(
                        type(cancel) != bool, 'TypeError',
                        "'cancel' must be a boolean ({} given).", type(cancel)
                    )
                    if not cancel:

                        # User input
                        while choice == None:
                            choice = input(
                                "A file already exists at '{}'. Do you wish to overwrite (Y), "
                                "keep both (K) or cancel (N)? ".format(self.file_path)
                            ).lower()

                            # Loop over question if input could not be interpreted
                            if choice not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                                print("Could not understand '{}'.".format(choice))
                                choice = None

                    # Saving cancellation
                    if cancel or choice in ('n', 'no'):
                        self.from_file = self.config.from_file.values
                        self.to_file = self.config.to_file.values

                        # Logging
                        log(
                            "'{}' series was not saved because a file already exists at '{}'",
                            self.name, self.file_path, logging=logging
                        )
                        del vars(self)['file_path']

                # Default name and saving
                if default or choice in ('k', 'keep'):
                    from .tools import default_name
                    self.file_path = default_name(self.file_path)
                    save(self)

                    # Logging
                    log(
                        "File name changed and series saved at '{}'.", self.file_path,
                        logging=logging)

            # Existing file deletion and saving
            if forced or choice in ('y', 'yes'):
                from os import remove
                remove(self.file_path)
                save(self)

                # Logging
                log(
                    "Existing file located at '{}' deleted and replaced.", self.file_path,
                    logging=logging
                )

        # Saving
        else:
            save(self)

            # Logging
            log("'{}' series saved at '{}'.", self.name, self.file_path, logging=logging)

    def create(self, forced=False, default=False, cancel=False, logging=True, mode=None):
        """
        Either loads a series from a file, or traces a series back from data or a model. If
        needed, the series is also saved. If both self.from_file, and self.from_data or
        self.from_model are True, loading operation supercedes the traceback mode, which is
        ignored and replaced by the value loaded from the file.
        """

        # Mode configuration
        self.configure_mode(mode)

        # Load from a file
        if self.from_file:
            self.load(forced=forced, logging=logging)

        # Traceback from data or a model
        elif self.from_data or self.from_model:
            self.traceback(forced=forced, logging=logging)

        # Save to a file
        if self.to_file:
            self.save(forced=forced, default=default, cancel=cancel, logging=logging)

    def check_traceback(self):
        """Checks whether a traceback has been computed in the series."""

        self.stop(
            len(self) < 1, 'ValueError', "'{}' series hasn't been traceback. "
            "Impossible to create an output.", self.name
        )

    def stop(self, condition, error, message, *words):
        """Calls the stop function from collection with self.name, if it has been set."""

        # Addition of series name to stop function call
        stop(
            condition, error, message, *words,
            name=self.name if 'name' in vars(self) else None, extra=2
        )