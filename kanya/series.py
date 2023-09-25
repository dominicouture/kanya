# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
series.py: Defines the class Series which uses a Config object to create a series of tracebacks
either from a file, data or a model. First, it handles exceptions (name, type, value, shape)
of all parameters, then creates or imports the file, if needed, handles unit, conversions, and
checks for the presence of output and logs directories and creates them, if needed.
"""

from tqdm import tqdm
from .init import *
from .group import Group
from .output import Output_Series
from .tools import enumerate_strings

class Series(list, Output_Series):
    """
    Contains a series of Groups and their embedded Star objects and the methods required to
    initialize a series from a Config object. Values and data in the Config object are, copied,
    checked for errors and converted to usable types such as Quantity objects, and values are
    converted to default units. The number of groups defines how many objects are in a series
    in which all groups have the same Config object as their progenitor.
    """

    def __init__(
            self, parent=None, file_path=None, args=False, forced=False, default=False,
            cancel=False, logging=True, addition=True, **parameters
        ):
        """
        Initializes a Series object by first configuring it with 'parent', 'file_path', 'args'
        and '**parameter' and then adding it to the collection. 'forced', 'default', 'cancel'
        and 'logging' arguments are passed to self.add function.
        """

        # Configuration
        self.configure(parent, file_path, args, **parameters)

        # Series addition to the collection
        if addition:
            self.add(forced=forced, default=default, cancel=cancel, logging=logging)

    def configure(self, parent=None, file_path=None, args=False, **parameters):
        """
        Configures a Series objects from 'parent', an existing Config object, 'file_path', a string
        representing a path to a configuration file, 'args' a boolean value that sets whether
        command line arguments are used, and '**parameters', a dictionary of dictionaries or
        Config.Parameter objects, in that order. If no value are given the default parameter is
        used instead. Only values that match a key in self.default_parameters are used. Then,
        parameters are copied, and checked for errors and converted. Checks if self.name is a
        string and creates a default value if needed. Checks the type and values of the modes
        provided in the configuration. self.save is set to False if self.load is True. Moreover,
        self.date is defined and self.metrics is configured.
        """

        # Initialize configuration
        self.config = Config(parent, file_path, args, **parameters)

        # Set name parameter
        self.name = self.set_string(self.config.name, none=True)

        # Create default name, if needed
        if self.name is None:
            self.name = collection.get_default_name()

        # Set date parameter
        from time import strftime
        self.date = strftime('%Y-%m-%d %H:%M:%S')

        # Set load and save parameters
        self.load = self.set_boolean(self.config.load)
        self.save = self.set_boolean(self.config.save)

        # Set association size metric parameters
        for parameter in (
            'size_metrics', 'cov_metrics', 'cov_robust_metrics',
            'cov_sklearn_metrics', 'mad_metrics', 'mst_metrics'
        ):
            vars(self)[parameter] = self.set_boolean(vars(self.config)[parameter])

        # Import the association size metrics file and skip the header line
        metrics_file = path.join(path.dirname(__file__), 'resources/association_size_metrics.csv')
        metrics_dataframe = pd.read_csv(metrics_file, delimiter=';')

        # Configure association size metric
        self.metrics = []
        for index, metric in metrics_dataframe.iterrows():
            self.Metric(self, metric)

    def configure_mode(self, mode=None):
        """
        Checks if selected mode, either 'from_data' or 'from_model' is valid, and if one and no
        more than one mode has been selected. A 'mode' argument can be provided to override the
        'from_data' and 'from_model' parameters.
        """

        # Set mode from data or model
        if mode is None:
            self.from_data = self.set_boolean(self.config.from_data)
            self.from_model = self.set_boolean(self.config.from_model)
        else:
            self.stop(
                type(mode) != str, 'TypeError', "'mode' must be a string or None ({} given).",
                type(mode)
            )
            if mode.lower() == 'from_data':
                self.from_data = True
                self.from_model = False
            elif mode.lower() == 'from_model':
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

    def configure_traceback(self):
        """Checks traceback configuration parameters for 'From data' and 'From model' modes."""

        # Set number of groups parameter
        self.number_of_groups = self.set_integer(self.config.number_of_groups)

        # Set number of steps parameter with one more step to account for t = 0
        self.number_of_steps = self.set_integer(self.config.number_of_steps) + 1

        # Set number of iterations parameter
        self.number_of_iterations = self.set_integer(self.config.number_of_iterations)

        # Check if number of iterations is greater than 0
        self.stop(
            self.number_of_iterations <= 0, 'ValueError',
            "'number_of_iterations' must be greater to 0 ({} given).", self.number_of_iterations
        )

        # Set iteration fraction parameter
        self.iteration_fraction = self.set_quantity(self.config.iteration_fraction).value

        # Check if iteration fraction is between 0 and 1
        self.stop(
            self.iteration_fraction <= 0 or self.iteration_fraction > 1, 'ValueError',
            "'iteration_fraction' must be between 0 and 1 ({} given).", self.iteration_fraction
        )

        # Set initial time parameter
        self.initial_time = self.set_quantity(self.config.initial_time)

        # Set final time parameter
        self.final_time = self.set_quantity(self.config.final_time)

        # Check if initial and final times are equal
        if self.final_time.value == self.initial_time.value:
            self.log(
                "The initial and final times are equal ({} given).",
                str(self.initial_time), level='warning', display=True
            )

        # Set duration, timesteps and integration times parameters
        self.duration = abs(self.final_time - self.initial_time)
        self.timestep = self.duration / (self.number_of_steps - 1)
        self.time = np.linspace(
            self.initial_time.value, self.final_time.value, self.number_of_steps
        )

        # Set direction of the orbit integration parameters
        self.forward = self.final_time.value >= self.initial_time.value
        self.backward = not self.forward

        # Set data errors parameter
        self.data_errors = self.set_boolean(self.config.data_errors)

        # Set data radial velocity shifts parameter
        self.data_rv_shifts = self.set_boolean(self.config.data_rv_shifts)

        # Set radial velocity shift parameter
        self.rv_shift = self.set_quantity(self.config.rv_shift)

        # Set cutoff parameter
        self.cutoff = self.set_quantity(self.config.cutoff, none=True).value

        # Check if cutoff is greater than 0
        if self.config.cutoff.values is not None:
            self.stop(
                self.cutoff <= 0.0, 'ValueError',
                "'cutoff' must be greater than 0 ({} given).", self.cutoff
            )

        # Set sample parameter
        self.sample = self.set_string(self.config.sample, none=True)

        # Check if the sample is valid
        if self.sample is not None:
            self.sample = self.sample.lower()
            valid_samples = ('input', 'core', 'extended', 'rejected')
            self.stop(
                self.sample not in valid_samples, 'ValueError',
                "'sample' must be {} ({} given).", enumerate_strings(*valid_samples), self.sample
            )

        # Set potential parameter
        self.potential = self.set_string(self.config.potential, none=True)

        # Check if the potential is convertible to a galpy.potential object
        if self.potential is not None:
            try:
                exec(
                    f'from galpy.potential.mwpotentials import {self.potential} as potential',
                    globals()
                )
            except:
                self.stop(True, 'ValueError', "'potenial' is invalid ({} given).", self.potential)
            self.potential = potential

        # Set principal component analysis parameter
        self.pca = self.set_boolean(self.config.pca)

        # Set timer parameter
        self.timer = self.set_boolean(self.config.timer)

        # Data configuration
        if self.from_data:
            self.configure_data()

        # Model configuration
        if self.from_model:
            self.configure_model()

        # Logging
        self.log("'{}' series ready for traceback.", self.name)

    def configure_data(self):
        """
        Checks if traceback and output from data is possible and creates a Data object from a
        CSV file, or a Python dictionary, list, tuple or np.ndarray. Model parameters are also
        set to None or False if the traceback is from data. This allows to use data errors
        without resetting model parameters.
        """

        # Logging
        if self.from_data:
            self.log("Initializing '{}' series from data.", self.name)

        # Import data
        from .data import Data
        self.data = Data(self)

        # Set number_of_stars parameter
        if self.from_data:
            self.number_of_stars = len(self.data)

            # Set age parameter
            self.age = None

            # Set model position and velocity parameters
            for parameter in (*self.config.position_parameters, *self.config.velocity_parameters):
                vars(self)[parameter] = None

            # Set model position and velocity errors
            if not self.data_errors:
                self.position_error = self.set_coordinate(self.config.position_error)
                self.velocity_error = self.set_coordinate(self.config.velocity_error)

            # Set model stars integration times
            self.model_time = None

    def configure_model(self):
        """Checks if traceback and output from a model is possible."""

        # Logging
        self.log("Initializing '{}' series from a model.", self.name)

        # Set number of stars parameter
        self.number_of_stars = self.set_integer(self.config.number_of_stars, mode=True)

        # Set age parameter
        self.age = self.set_quantity(self.config.age, mode=True)

        # Check if age is equal to the initial time
        self.stop(
            self.age.value == self.initial_time.value, 'ValueError',
            "'age' cannot be equal to 'initial_time' ({} and {} given).",
            self.age, self.initial_time
        )

        # Set model position parameters
        self.position = self.set_coordinate(self.config.position, mode=True)
        self.position_error = self.set_coordinate(self.config.position_error, mode=True)
        self.position_scatter = self.set_coordinate(self.config.position_scatter, mode=True)

        # Set model velocity parameters
        self.velocity = self.set_coordinate(self.config.velocity, mode=True)
        self.velocity_error = self.set_coordinate(self.config.velocity_error, mode=True)
        self.velocity_scatter = self.set_coordinate(self.config.velocity_scatter, mode=True)

        # Configure data to use actual error measurements or radial velocity shift
        if self.data_errors or self.data_rv_shifts:
            self.configure_data()

            # Check if the data length matches the number of stars
            self.stop(
                len(self.data) == 0, 'ValueError', "If 'data_errors' or 'data_rv_shift' "
                "is True, the data length must be greater than 0 ({} given).", len(self.data)
            )

        # Set data to None since measurement errors and radial velocity shifts are modeled
        else:
            self.data = None

        # Set model stars integration times
        self.model_time = np.linspace(
            self.initial_time.value, self.age.value,
            int(abs(self.age.value - self.initial_time.value) / self.timestep.value) + 1
        )

    def set_parameter(self, parameter, *type_labels, none=False, mode=False, all=False):
        """
        Checks if the parameter is None or missing in the configuration, and if the type of the
        parameter is valid. The value of the parameter is returned if those checks are passed. If
        'none' is True, the parameter can be None of missing in the configuration. If 'mode' is
        True, the model is included in the error message, If 'all' is True, all the parameter and
        all of its components is return, not just its value.
        """

        # Check if none, mode and all are boolean
        self.stop(
            type(none) != bool, 'TypeError',
            "'none' must be a boolean ({} given).", type(none)
        )
        self.stop(
            type(mode) != bool, 'TypeError',
            "'mode' must be a boolean ({} given).", type(mode)
        )
        self.stop(
            type(all) != bool, 'TypeError',
            "'all' must be a boolean ({} given).", type(all)
        )

        # Check if parameter.values is None or missing
        if not none:
            self.stop(
                parameter.values is None, 'ValueError',
                "Required parameter '{}' cannot be None or missing in the configuration" + (
                    (
                        ' when tracing back from data.' if self.from_data else
                        ' when tracing back from a model.' if self.from_model else ''
                    ) if mode else '.'
                ), parameter.label
            )

        # Valid types and labels
        types = [self.config.Parameter.types[label] for label in type_labels]
        if none:
            types += (type(None),)
            type_labels += ('None',)

        # Check the type of parameter.values
        self.stop(
            type(parameter.values) not in types, 'TypeError',
            "'{}' must be a {} ({} given).",
            parameter.label, enumerate_strings(*type_labels), type(parameter.values)
        )

        return deepcopy(parameter if all else parameter.values)

    def set_boolean(self, parameter, none=False, mode=False):
        """
        Checks if the boolean is None or missing in the configuration, and if the type of the
        boolean is valid. The value of the boolean is returned if those checks are passed.
        """

        return self.set_parameter(parameter, 'boolean', none=none, mode=mode, all=False)

    def set_string(self, parameter, none=False, mode=False):
        """
        Checks if the string is None or missing in the configuration, and if the type of the
        string is valid. The value of the string is returned if those checks are passed.
        """

        return self.set_parameter(parameter, 'string', none=none, mode=mode, all=False)

    def set_path(
            self, file_path, parameter, name=None, extension=None, file_type=None,
            check=False, create=False, none=False, mode=False
    ):
        """
        Checks if the file path is None or missing in the configuration, and if the type of the
        path is valid, or uses the provided file path. Redefines the directory path as the
        absolute path. Checks if the directory exists or creates it if needed. Creates a name with,
        an extension if needed. Checks if the file exists and its extension, if needed.
        """

        # Check if parameter is a string
        self.stop(
            type(parameter) != str, 'TypeError',
            "'{}' must be a string ({} given).", parameter, type(parameter)
        )

        # Set file path parameter
        if type(file_path) == self.config.Parameter:
            file_path = self.set_string(file_path, none=none, mode=mode)
            file_path = file_path.values if type(file_path) == self.config.Parameter else file_path

        # Check if file path is a string
        else:
            self.stop(
                type(file_path) != str, 'TypeError',
                "'{}' must be a string ({} given).", parameter, type(file_path)
            )

        # Check if name, extension and file_type are a string or None
        self.stop(
            type(name) not in (str, type(None)), 'TypeError',
            "'name' must be a string or None ({} given).", parameter, type(name)
        )
        self.stop(
            type(extension) not in (str, type(None)), 'TypeError',
            "'extension' must be a string or None ({} given).", parameter, type(extension)
        )
        self.stop(
            type(file_type) not in (str, type(None)), 'TypeError',
            "'file_type' must be a string or None ({} given).", parameter, type(file_type)
        )

        # Redefine directory path as the absolute path
        dir_path = directory(
            collection.base_dir, path.dirname(file_path), parameter,
            check=check, create=create
        )

        # Add a default file name with an extension, if needed
        if path.basename(file_path) == '':
            if name is not None:
                if path.splitext(name)[-1].lower() == '' and extension is not None:
                    name = f'{name}.{extension}'
                file_path = path.join(dir_path, name)

        # Add the file name to the directory path
        else:
            file_path = path.join(dir_path, path.basename(file_path))

        # Check if there's an extension and add an extension, if needed
        if path.basename(file_path) != '':
            if path.splitext(file_path)[-1] != '':
                extension = path.splitext(file_path)[-1][1:]
            if path.splitext(file_path)[-1] == '' and extension is not None:
                file_path += f'.{extension}'

            # Set file type, if needed
            if file_type is None and extension is not None:
                file_type = extension.upper()

            # Check if the file exists
            if check:
                self.stop(
                    not path.exists(file_path), 'FileNotFoundError',
                    "No {} file located at '{}'.", file_type, file_path
                )

            # Check if the file has the right extension
            if check and extension is not None:
                self.stop(
                    path.splitext(file_path)[-1].lower() != f'.{extension}', 'ValueError',
                    "The file located at '{}' is not a {} file (with a .{} extension).",
                    file_path, file_type, extension
                )

        return file_path

    def set_integer(self, parameter, none=False, mode=False):
        """Checks if an integer value is valid and converts it if needed."""

        # Set integer parameter
        parameter = self.set_parameter(
            parameter, 'integer', 'float', none=none, mode=mode, all=True
        )

        # Check if the parameter is convertible to an integer and greater than or equal to 1
        self.stop(
            parameter.values % 1 != 0, 'ValueError',
            "'{}' must be convertible to an integer ({} given).",
            parameter.label, parameter.values
        )
        self.stop(
            parameter.values < 1, 'ValueError',
            "'{}' must be greater than or equal to 1 ({} given).",
            parameter.label, parameter.values
        )

        # Conversion to an integer
        return int(parameter.values)

    def set_quantity(self, parameter, none=False, mode=False):
        """Checks if a value is valid and converts it to default units if needed."""

        # Set quantity parameter
        parameter = self.set_parameter(
            parameter, 'integer', 'float', none=none, mode=mode, all=True
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

    def set_coordinate(self, parameter, none=False, mode=False):
        """
        Converts a Parameter into a Quantity object and raises an error if an exception
        occurs in the process. Returns a vector converted to default units for the physical
        type defined by a Variable object.
        """

        # Set coordinate parameter
        parameter = self.set_parameter(
            parameter, 'tuple', 'list', 'array', none=none, mode=mode, all=True
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

        # Variables from label and check for invalid label
        if parameter.label in self.config.position_parameters:
            variables = parameter.system.position
        elif parameter.label in self.config.velocity_parameters:
            variables = parameter.system.velocity
        else:
            self.stop(True, 'ValueError', "'{}' is not a supported label.", parameter.label)

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

                # Age shift based on models
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

                # Age shift based on models
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

                # Age shift based on models
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
                        self.log(
                            "'{}' series was not added to the collection because a series "
                            "with the same name already exists.", self.name, logging=logging
                        )
                        del self

                # Default name and addition to the collection
                if default or choice in ('k', 'keep'):
                    name = deepcopy(self.name)
                    self.name = collection.get_default_name(self.name)
                    self.config.name.values = self.name
                    collection.append(self)

                    # Logging
                    self.log(
                        "'{}' series renamed '{}' and added to the collection.",
                        name, self.name, logging=logging
                    )

            # Existing series deletion and addition to the collection
            if forced or choice in ('y', 'yes'):
                del collection[collection.series[self.name]]
                collection.append(self)

                # Logging
                self.log(
                    "Existing '{}' series deleted and new series added to the collection.",
                    self.name, logging=logging
                )

        # Addition to the collection
        else:
            collection.append(self)

            # Logging
            self.log("'{}' series added to the collection.", self.name, logging=logging)

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
        self.log("'{}' series removed from the collection.", self.name, logging=logging)

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
        self.log("'{}' series reset.", self.name, logging=logging)

    def update(
            self, parent=None, file_path=None, args=False, create=True,
            forced=False, default=False, cancel=False, logging=True, **parameters
    ):
        """
        Updates the series by modifying its self.config configuration, re-initializing itself
        and deleting existing groups. The groups are also recreated if they had been created
        before the update, unless 'create' is set to False. 'parent', 'file_path', 'args' and
        '**parameters' are the same as in Series.__init__. If 'name' parameter is the only
        one modified, existing groups are simply kept and renamed.
        """

        # Initialization
        parent = self.config if parent is None else parent
        self.stop(
            type(create) != bool, 'TypeError',
            "'create' must be a boolean ({} given).", type(create)
        )
        create = False if len(self) == 0 else create

        # New configuration
        new_config = Config(parent, file_path, args, **parameters)

        # Check what parameters, if any, are modified
        new_parameters = [
            parameter for parameter in vars(self.config)
            if vars(vars(new_config)[parameter]) != vars(vars(self.config)[parameter])
        ]

        # No parameters are modified
        if len(new_parameters) == 0:

            # Logging
            self.log("'{}' series unchanged.", self.name, logging=logging)

        # Only 'name' parameter is modified
        elif len(new_parameters) == 1 and 'name' in new_parameters:

            # Change series name
            old_name = deepcopy(self.name)
            self.name = self.set_string(self.new_config.name, none=True)

            # Create default name, if needed
            if self.name is None:
                self.name = collection.get_default_name()

            # Change configuration name
            self.config.name = new_config.name

            # Change groups names
            for group in self:
                group.name = group.name.replace(old_name, self.name)

            # Series removal from the collection
            if old_name in collection.series.keys():
                self.remove(logging=False)

            # Series re-addition to the collection
            self.add(forced=forced, default=default, cancel=cancel, logging=False)

            # Logging
            self.log("'{}' series renamed '{}'.", old_name, self.name, logging=logging)

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
                    self.log(
                        "'{}' series was not updated because it has already been traced back.",
                        self.name, logging=logging
                    )
            else:
                forced = True

            # Updated series creation
            if forced:
                updated_series = Series(
                    parent=parent, file_path=file_path, args=args, addition=False, **parameters
                )

                # Save current series name
                if 'name' in new_parameters:
                     old_name = deepcopy(self.name)

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
                    self.log("'{}' series renamed to '{}'.", old_name, self.name, logging=logging)
                self.log("'{}' series updated.", self.name, logging=logging)

                # Create series, if needed
                if create:
                    self.create()

        return self

    def copy(
            self, parent=None, file_path=None, args=False,
            logging=True, traceback=True, **parameters
    ):
        """
        Copies the series under a new name. If 'parent', 'file_path', 'args' or '**parameters'
        are provided, the same as in Series.__init__, this new Series object is updated as well.
        If no new 'name' is provided, a default name is used instead.
        """

        # Self cloning
        clone = deepcopy(self)

        # Clone addition to the collection
        clone.add(default=True)

        # Clone update, if needed
        if parent is not None or file_path is not None or args == True or len(parameters) > 0:
            clone.update(
                parent=parent, file_path=file_path, args=args, logging=False,
                traceback=traceback, **parameters
            )

        # Logging
        self.log("'{}' series copied to '{}'.", self.name, clone.name, logging=logging)

        return clone

    def load_from_file(self, load_path=None, forced=False, logging=True):
        """
        Loads a series from a binary file. self.load_path is defined as the absolute path to
        the file. If forced, the existing groups are overwritten, otherwise user input is
        required to proceed if groups already exists.
        """

        # Check if a traceback has already been done
        self.stop(
            type(forced) != bool, 'TypeError',
            "'forced' must be a boolean ({} given).", type(forced)
        )

        # User input
        if len(self) > 0:
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
                self.log("Existing groups in series '{}' deleted.", self.name, logging=logging)

            # Loading cancellation
            else:

                # Logging
                self.log(
                    "'{}' series was not loaded because it has already been traced back.",
                    self.name, logging=logging
                )
        else:
            forced = True

        # Loading
        if forced:

            # File configuration
            self.load = True

            # Set load path parameter
            self.load_path = self.set_path(
                self.config.load_path if load_path is None else load_path, 'load_path',
                name=self.name, extension='series', file_type='Series', check=True, create=False
            )

            # Series unpickling
            from pickle import load
            file = open(self.load_path, 'rb')
            parameters, groups = load(file)
            file.close()

            # Parameters deletion and series removal from the collection
            self.remove(logging=False)
            for parameter in [
                    parameter for parameter in vars(self).keys()
                    if parameter not in (
                        'load', 'save', 'load_path', 'save_path',
                        'data_dir', 'output_dir', 'logs_dir'
                    )
                ]:
                del vars(self)[parameter]

            # Parameters and groups import, and series re-addition to the collection
            vars(self).update(parameters)
            for group in groups:
                self.append(group)
            self.add(logging=False)

            # Logging
            self.log("'{}' series loaded from '{}'.", self.name, self.load_path, logging=logging)

    def traceback(self, forced=False, logging=True, mode=None):
        """
        Traces back Group and embeded Star objects using either imported data or by modeling a
        group based on model parameters.
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
                self.log("Existing groups in series '{}' deleted.", self.name, logging=logging)

            # Traceback cancellation
            else:
                self.from_data = self.config.from_data.values
                self.from_model = self.config.from_model.values
                self.log(
                    "'{}' series was not created because it has already been traced back.",
                    self.name, logging=logging
                )
        else:
            forced = True

        # Groups creation
        if forced:

            # Traceback configuration
            self.configure_mode(mode=mode)
            self.configure_traceback()

            # !!! Redinition of the number of stars when using the from data mode !!!
            if self.from_data:
                self.number_of_stars = len(self.data.core_sample)

            # Logging and progress bar
            self.log(
                "Tracing back '{}' series from {}.",
                self.name, 'data' if self.from_data else 'a model',
                display=True, logging=logging
            )
            self.progress_bar = tqdm(
                total=self.number_of_groups * self.number_of_stars, unit=' star',
                bar_format=(
                    '{desc}{percentage:3.0f}% |{bar}| '
                    '{n_fmt}/{total_fmt} {elapsed} {remaining} '
                )
            )

            # Traceback
            for number in range(self.number_of_groups):

                # Group name and message
                name = f'{self.name}-{number}'
                message = f"Tracing back '{name}' group"

                # Traceback logging and progress bar
                self.log(f'{message}.', display=False, logging=logging)
                self.progress_bar.set_description(desc=message, refresh=True)

                # Group traceback
                self.append(Group(self, number, name))

            # Traceback logging and progress bar
            message = f"'{self.name}' series succesfully traced back"
            self.log(f'{message}.', display=False, logging=logging)
            self.progress_bar.set_description(message, refresh=True)
            self.progress_bar.close()
            del self.progress_bar

            # Outlier logging
            if self.from_data:
                for message in self[0].outliers_messages:
                    self.log(message, display=True, logging=logging)

            # Show timer
            if self.timer:
                self[0].show_timer()

            # Compute average association size metrics
            for metric in self.metrics:
                metric()

    def save_to_file(self, save_path=None, forced=False, default=False, cancel=False, logging=True):
        """
        Saves a series to a binary file. self.save_path is defined as the actual path to the
        file. If forced, the existing file is overwritten, otherwise user input is required
        to proceed if a file already exists.
        """

        def save_to_file(self):
            """Saves the series."""

            # Series pickling
            from pickle import dump
            file = open(self.save_path, 'wb')
            dump(
                (
                    {
                        parameter: vars(self)[parameter] for parameter in vars(self).keys()
                        if parameter not in (
                            'load', 'save', 'load_path', 'save_path',
                            'data_dir', 'output_dir', 'logs_dir'
                        )
                    }, tuple(group for group in self)
                ), file
            )
            file.close()

        # Set save path parameter
        self.save = True
        self.save_path = self.set_path(
            self.config.save_path if save_path is None else save_path, 'save path',
            name=self.name, extension='series', file_type='Series', check=False, create=True
        )

        # Check if a file already exists
        if path.exists(self.save_path):
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
                                "keep both (K) or cancel (N)? ".format(self.save_path)
                            ).lower()

                            # Loop over question if input could not be interpreted
                            if choice not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                                print("Could not understand '{}'.".format(choice))
                                choice = None

                    # Saving cancellation
                    if cancel or choice in ('n', 'no'):
                        self.load = self.config.load.values
                        self.save = self.config.save.values

                        # Logging
                        self.log(
                            "'{}' series was not saved because a file already exists at '{}'",
                            self.name, self.save_path, logging=logging
                        )
                        del vars(self)['file_path']

                # Default name and saving
                if default or choice in ('k', 'keep'):
                    from .tools import default_name
                    self.save_path = default_name(self.save_path)
                    save_to_file(self)

                    # Logging
                    self.log(
                        "File name changed and series saved at '{}'.", self.save_path,
                        logging=logging)

            # Existing file deletion and saving
            if forced or choice in ('y', 'yes'):
                from os import remove
                remove(self.save_path)
                save_to_file(self)

                # Logging
                self.log(
                    "Existing file located at '{}' deleted and replaced.", self.save_path,
                    logging=logging
                )

        # Saving
        else:
            save_to_file(self)

            # Logging
            self.log("'{}' series saved at '{}'.", self.name, self.save_path, logging=logging)

    def create(self, forced=False, default=False, cancel=False, logging=True, mode=None):
        """
        Either loads a series from a file, or traces a series back from data or a model. If
        needed, the series is also saved. If both self.load, and self.from_data or self.from_model
        are True, loading operation supercedes the traceback mode, which is ignored and replaced
        by the value loaded from the file.
        """

        # Load from a file
        if self.load:
            self.load_from_file(forced=forced, logging=logging)

        # Traceback from data or a model
        else:
            self.traceback(forced=forced, logging=logging, mode=mode)

        # Save to a file
        if self.save:
            self.save_to_file(forced=forced, default=default, cancel=cancel, logging=logging)

    def check_traceback(self):
        """Checks whether a traceback has been computed in the series."""

        self.stop(
            len(self) < 1, 'ValueError', "'{}' series hasn't been traceback. "
            "Impossible to create an output.", self.name
        )

    def log(self, message, *words, logs_path=None, level='info', display=False, logging=True):
        """
        Logs the 'message' with the appropriate 'level', if logs have been configured. If logs
        have not been configured, logs are configured by checking if the 'logs_path' already
        exists, creating it if needed and configuring the logs file.  By default, logs files will
        created in the base directory with a default name. The file name must end with an '.log'
        extension. Futhermore, if 'display' is True, the message is also printed onscreen.
        """

        # Set logs path parameter
        self.logs_path = self.set_string(
            self.config.logs_path if logs_path is None else logs_path, none=True
        )

        # Create logging message
        log(
            message, *words, logs_path=None,
            level=level, display=display, logging=logging
        )

    def stop(self, condition, error, message, *words):
        """Calls the stop function from collection with self.name, if it has been set."""

        # Addition of series name to stop function call
        stop(
            condition, error, message, *words,
            name=self.name if 'name' in vars(self) else None, extra=2
        )