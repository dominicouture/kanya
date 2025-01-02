# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
group.py: Defines the Group class, which uses a Config object to create a group of stars either
from a file, data or a model. First, it handles exceptions (name, type, value, and shape) of all
parameters, then creates or imports the file, if needed, handles unit conversions.
"""

from time import time as get_time, strftime
from .init import *
from .traceback import Traceback
from .chrono import Chrono
from .output import Output

class Group(Traceback, Chrono, Output):
    """
    Group of stars. This class contains the data and methods necessary to traceback the
    trajectories of stars and find their age by minimizing the size of the group. It is
    initialized from a Config object. Values and data in the Config object are, copied,
    checked for errors and converted to usable types such as Quantity objects, and values
    are converted to default units. The number of groups defines how many copies of the group
    are used to compute errors with a Monte Carlo approach.
    """

    def __init__(
        self, parent=None, file_path=None, load_path=None, args=False, forced=None,
        default=None, cancel=None, logging=True, addition=True, **parameters
    ):
        """
        Initializes a Group object by first configuring it with 'parent', 'file_path', 'args'
        and '**parameter' and then adding it to the collection. 'forced', 'default', 'cancel'
        and 'logging' arguments are passed to self.add function.
        """

        # Configuration
        self.configure(parent, file_path, args,  **parameters)

        # Load group from a file
        if self.load:
            self.load_group(
                load_path=load_path, forced=forced, default=default, cancel=cancel,
                remove=False, logging=True
            )

        # Add group to the collection
        elif addition:
            self.add(forced=forced, default=default, cancel=cancel, logging=logging)

    def __repr__(self):
        """Creates a string with the name of the Group"""

        return self.name

    def configure(self, parent=None, file_path=None, args=False, **parameters):
        """
        Configures a Group objects from 'parent', an existing Config object, 'file_path', a string
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
        self.date = strftime('%Y-%m-%d %H:%M:%S')

        # Set load and save parameters
        self.load = self.set_boolean(self.config.load)
        self.save = self.set_boolean(self.config.save)

        # Set traceback_present and chrono_present parameters
        self.traceback_present = False
        self.chrono_present = False

        # Import the association size metrics file and skip the header line
        metrics_file = path.join(path.dirname(__file__), 'resources/association_size_metrics.csv')
        metrics_dataframe = pd.read_csv(metrics_file, delimiter=';', na_filter=False)

        # Initialize association size metrics
        self.metrics = []
        for index, metric in metrics_dataframe.iterrows():
            self.Metric(self, metric)

    def set_parameter(self, parameter, *type_labels, none=False, mode=False, all=False):
        """
        Checks if the parameter is None or missing in the configuration, and if the type of the
        parameter is valid. The value of the parameter is returned if those checks are passed. If
        'none' is True, the parameter can be None of missing in the configuration. If 'mode' is
        True, the model is included in the error message, If 'all' is True, all the parameter and
        all of its components is return, not just its value.
        """

        # Check the types of none, mode and all
        self.check_type(none, 'none', 'boolean')
        self.check_type(mode, 'mode', 'boolean')
        self.check_type(all, 'all', 'boolean')

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

        # Check the type of parameter.values
        else:
            type_labels += ('None',)
        self.check_type(parameter.values, parameter.label, type_labels)

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
        dir_only=False, full_path=False, check=False, create=False, none=False, mode=False
    ):
        """
        Checks if the file path is None or missing in the configuration, and if the type of the
        path is valid, or uses the provided file path. Redefines the directory path as the
        absolute path. Checks if the directory exists or creates it if needed. Creates a name with,
        an extension if needed. Checks if the file exists and its extension, if needed.
        """

        # Set file_path parameter
        if type(file_path) == self.config.Parameter:
            file_path = self.set_string(file_path, none=none, mode=mode)
        else:
            self.check_type(file_path, parameter, 'string')

        # Check the types of name, extension, file_type, dir_only and full_path
        self.check_type(name, 'name', ('string', 'None'))
        self.check_type(extension, 'extension', ('string', 'None'))
        self.check_type(file_type, 'file_type', ('string', 'None'))
        self.check_type(dir_only, 'dir_only', 'boolean')
        self.check_type(full_path, 'full_path', 'boolean')

        # Add a name and an extension to the file path, if needed
        if path.basename(file_path) == '' and name is not None:
            file_path = path.join(file_path, name)
        if path.splitext(file_path)[-1] == '' and extension is not None:
            file_path += f'.{extension}'

        # Redefine the file path as the absolute path
        file_path = get_abspath(
            collection.base_dir, file_path, parameter,
            check=check, create=create
        )

        # Check if the file path is a directory only or a full path, if needed
        self.stop(
            dir_only and path.basename(file_path) != '', 'ValueError',
            "'{}' must be a path to directory, not a file ({} given).", parameter, file_path
        )
        self.stop(
            full_path and path.basename(file_path) == '', 'ValueError',
            "'{}' must be a path to file, not a directory ({} given).", parameter, file_path
        )

        # Check if the file has the right extension
        if path.basename(file_path) != '' and extension is not None:
            self.stop(
                path.splitext(file_path)[-1].lower() != f'.{extension}', 'ValueError',
                "The file located at '{}' is not a {} file (with a .{} extension).",
                file_path, file_type if file_type is not None else extension.upper(), extension
            )

        return file_path

    def set_integer(self, parameter, minimum, none=False, mode=False):
        """Checks if an integer value is valid and converts it if needed."""

        # Set integer parameter
        parameter = self.set_parameter(
            parameter, 'integer', 'float', none=none, mode=mode, all=True
        )

        # Check if the parameter is convertible to an integer
        self.stop(
            parameter.values % 1 != 0, 'ValueError',
            "'{}' must be convertible to an integer ({} given).",
            parameter.label, parameter.values
        )

        # Check if the parameter is greater than or equal to 1
        self.stop(
            parameter.values < minimum, 'ValueError',
            "'{}' must be greater than or equal to {} ({} given).",
            parameter.label, minimum, parameter.values
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

        # Check the type of parameter.units
        self.check_type(parameter.units, f'{parameter.label}.units', 'string')

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

        # Check the type of parameter.units
        self.check_type(
            parameter.units, f'{parameter.label}.units', ('string', 'tuple', 'list', 'array')
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
        print(quantity)
        return quantity.to()

    def create(
        self, load_path=None, save_path=None, mode=None, size_metrics=None, cov_metrics=None,
        cov_robust_metrics=None, cov_sklearn_metrics=None, mad_metrics=None, tree_metrics=None,
        forced=None, default=None, cancel=None, logging=True
    ):
        """
        Either loads a group from a file, or traces a group back from data or a model. If needed,
        the group is also saved. If both self.load, and self.from_data or self.from_model are
        True, loading operation supercedes the traceback mode, which is ignored and replaced by
        the value loaded from the file.
        """

        # Load group from a file
        if self.load:
            self.load_group(
                load_path=load_path, forced=forced,
                default=default, cancel=cancel, logging=logging
            )

        # Traceback groups from data or a model
        else:
            self.traceback(mode=mode, forced=forced, logging=logging)
            self.chronologize(
                size_metrics=size_metrics, cov_metrics=cov_metrics,
                cov_robust_metrics=cov_robust_metrics, cov_sklearn_metrics=cov_sklearn_metrics,
                mad_metrics=mad_metrics, tree_metrics=tree_metrics, logging=logging
            )

            # Show timer
            self.show_timer()

        # Save group to a file
        if self.save:
            self.save_group(
                save_path=save_path, forced=forced,
                default=default, cancel=cancel, logging=logging
            )

    def check_traceback(self):
        """Checks whether a traceback has been computed for the group."""

        self.stop(
            not self.traceback_present, 'ValueError', "'{}' group hasn't been traceback. "
            "Impossible to create an output.", self.name
        )

    def check_type(self, value, label, type_labels):
        """Checks if the type of the value if valid."""

        # Set types
        types = {
            'None': type(None),
            'boolean': bool,
            'string': str,
            'integer': int,
            'float': float,
            'tuple': tuple,
            'list': list,
            'dictionary': dict,
            'array': np.ndarray
        }

        # Set type labels
        type_labels = type_labels if type(type_labels) == tuple else (type_labels,)

        # Check the type of value
        self.stop(
            type(value) not in [types[label] for label in type_labels],
            'TypeError', "'{}' must be a {} ({} given).",
            label, enumerate_strings(*type_labels), type(value)
        )

    def choose(self, problem, style, forced=None, default=None, cancel=None):
        """
        Checks the type of forced, default and cancel. If at least one is True, then its behaviour
        is selected. If more than one is True, forced is prioritized over default, and default is
        prioritized over cancel. If all three are False, then the user is asked for their input.
        """

        # Check the type of forced
        self.check_type(forced, 'forced', ('None', 'boolean'))

        # Ask for user input
        if style == 1:
            while forced is None:
                choice = input(f"{problem} Do you wish to overwrite (Y/N)?").lower()

                # Set forced
                forced = (
                    True if choice in ('y', 'yes') else
                    False if choice in ('n', 'no') else None
                )

                # Loop over the question if input could not be interpreted
                if forced is None:
                    print(f"Could not understand '{choice}'.")

            return forced

        # Check the type of default and cancel
        if style == 3:
            self.check_type(default, 'default', ('None', 'boolean'))
            self.check_type(cancel, 'cancel', ('None', 'boolean'))

            # Set forced, default and cancel, if one is True
            if forced and not default and not cancel:
                default = cancel = False
            elif not forced and default and not cancel:
                forced = cancel = False
            elif not forced and not default and cancel:
                forced = default = False

            # Set forced, default and cancel, if two are False
            elif forced is None and default is False and cancel is False:
                forced = True
            elif forced is False and default is None and cancel is False:
                default = True
            elif forced is False and default is False and cancel is None:
                cancel = True

            # Ask for user input in other cases
            else:
                choice = None
                while choice is None:
                    choice = input(
                        f"{problem} Do you wish to overwrite (Y), keep both (K) or cancel (N)?"
                    ).lower()

                    # Loop over the question if input could not be interpreted
                    if choice not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                        print(f"Could not understand '{choice}'.")
                        choice = None

                # Choose behaviour
                forced = True if choice in ('y', 'yes') else False
                default = True if choice in ('k', 'keep') else False
                cancel = True if choice in ('n', 'no') else False

            return forced, default, cancel

    def add(self, forced=None, default=None, cancel=None, logging=True):
        """
        Adds the group to the collection. If 'forced' is True, any existing group with the
        same name is overwritten, otherwise user input is required to proceed (overwrite, do
        nothing or default). If 'default' is True, then instead of asking for an input, a
        default name is used and the group added to the collection. If 'cancel' is True,
        instead of asking for an input, the group isn't added to the collection.
        """

        # Choose behaviour if the group already exists in the collection
        if self.name in collection.groups.keys():
            forced, default, cancel = self.choose(
                f"'{self.name}' group already exists in the collection.", 3,
                forced, default, cancel
            )

            # Delete existing group from the collection and add group to the collection
            if forced:
                del collection[collection.groups[self.name]]
                collection.append(self)

                # Logging
                self.log(
                    "Existing '{}' group in the collection deleted and replaced.",
                    self.name, logging=logging
                )

            # Set default name and add group to the collection
            if default:
                old_name = deepcopy(self.name)
                self.name = self.config.name.values = collection.get_default_name(self.name)
                collection.append(self)

                # Logging
                self.log(
                    "Group renamed from '{}' to '{}', and added to the collection.",
                    old_name, self.name, logging=logging
                )

            # Cancel save
            if cancel:
                self.log(
                    "'{}' group was not added to the collection because the group "
                    "already exists in the collection.", self.name, logging=logging
                )
                del self

        # Add Group to the collection
        else:
            collection.append(self)

            # Logging
            self.log("'{}' group added to the collection.", self.name, logging=logging)

        # Re-initialize groups in the collection
        collection.initialize_groups()

    def remove(self, logging=True):
        """Removes the group from the collection, if it is in the collection."""

        # Delete group from the collection
        if self.name in collection.groups.keys():
            del collection[collection.groups[self.name]]

            # Re-initialize groups in the collection
            collection.initialize_groups()

            # Logging
            self.log("'{}' group removed from the collection.", self.name, logging=logging)

            # Delete group
            del self

    def reset(self, logging=True):
        """
        Clears all tracebacks, as well as all parameter and returns the group to its original
        configuration. This effectively undoes a load or traceback operation.
        """

        # Initialization
        parent = self.config

        # Delete all parameters and groups
        vars(self).clear()
        self.clear()

        # Re-initialize group
        self.configure(parent=parent)

        # Logging
        self.log("'{}' group reset.", self.name, logging=logging)

    def update(
            self, parent=None, file_path=None, args=False, create=True,
            forced=None, default=None, cancel=None, logging=True, **parameters
    ):
        """
        Updates the group by modifying its self.config configuration, re-initializing itself
        and deleting existing groups. The groups are also recreated if they had been created
        before the update, unless 'create' is set to False. 'parent', 'file_path', 'args' and
        '**parameters' are the same as in Group.__init__. If 'name' parameter is the only
        one modified, existing groups are simply kept and renamed.
        """

        # Set parent parameter
        parent = self.config if parent is None else parent

        # Set create parameter
        self.check_type(create, 'create', 'boolean')
        create = False if len(self) == 0 else create

        # Create new configuration
        new_config = Config(parent, file_path, args, **parameters)

        # Check what parameters, if any, are modified
        new_parameters = [
            parameter for parameter in vars(self.config)
            if vars(vars(new_config)[parameter]) != vars(vars(self.config)[parameter])
        ]

        # No parameters are modified
        if len(new_parameters) == 0:

            # Logging
            self.log("'{}' group unchanged.", self.name, logging=logging)

        # Only 'name' parameter is modified
        elif len(new_parameters) == 1 and 'name' in new_parameters:

            # Change group name
            old_name = deepcopy(self.name)
            self.name = self.set_string(self.new_config.name, none=True)

            # Create a default name, if needed
            if self.name is None:
                self.name = collection.get_default_name()

            # Change configuration name
            self.config.name = new_config.name

            # Change groups names
            for group in self:
                group.name = group.name.replace(old_name, self.name)

            # Group removal from the collection
            if old_name in collection.groups.keys():
                self.remove(logging=False)

            # Group re-addition to the collection
            self.add(forced=forced, default=default, cancel=cancel, logging=False)

            # Logging
            self.log("Group renamed from '{}' to '{}'.", old_name, self.name, logging=logging)

        # Any other parameter is modified
        else:

            # Choose behaviour if groups have already been traced back
            if len(self) > 0:
                forced = self.choose(
                    f"'{self.name}' group has already been traced back.", 1, forced
                )

                # Delete groups
                if forced:
                    self.clear()
                    self.log("Existing groups in '{}' group deleted.", self.name, logging=logging)

                # Cancel update
                if not forced:
                    self.log(
                        "'{}' group was not updated because it has already been traced back.",
                        self.name, logging=logging
                    )

            # Create updated group
            if len(self) == 0:
                updated_group = Group(
                    parent=parent, file_path=file_path, args=args, addition=False, **parameters
                )

                # Save current group name
                if 'name' in new_parameters:
                     old_name = deepcopy(self.name)

                # Remove current group from the collection
                if self.name in collection.groups.keys():
                    self.remove(logging=False)

                # Delete current group and redefine it as updated group
                del self
                self = updated_group

                # Add updated group to the collection
                self.add(forced=forced, default=default, cancel=cancel, logging=False)

                # Logging
                if 'name' in new_parameters:
                    self.log(
                        "Group renamed from '{}' to '{}'.", old_name, self.name, logging=logging
                    )
                self.log("'{}' group updated.", self.name, logging=logging)

                # Create groups, if needed
                if create:
                    self.create()

        return self

    def copy(
            self, parent=None, file_path=None, args=False,
            logging=True, traceback=True, **parameters
    ):
        """
        Copies the group under a new name. If 'parent', 'file_path', 'args' or '**parameters'
        are provided, the same as in Group.__init__, this new Group object is updated as well.
        If no new 'name' is provided, a default name is used instead.
        """

        # Clone self
        clone = deepcopy(self)

        # Add clone to the collection
        clone.add(default=True)

        # Update clone, if needed
        if parent is not None or file_path is not None or args == True or len(parameters) > 0:
            clone.update(
                parent=parent, file_path=file_path, args=args, logging=False,
                traceback=traceback, **parameters
            )

        # Logging
        self.log("'{}' group copied to '{}'.", self.name, clone.name, logging=logging)

        return clone

    def load_group(
        self, load_path=None, remove=True, forced=None, default=None, cancel=None, logging=True
    ):
        """
        Loads a group from a binary file. self.load_path is defined as the absolute path to
        the file. If forced, the existing groups are overwritten, otherwise user input is
        required to proceed if groups already exists.
        """

        # Choose behaviour if groups have already been traced back
        if len(self) > 0:
            forced = self.choose(f"'{self.name}' group has already been traced back.", 1, forced)

            # Delete groups
            if forced:
                self.clear()
                self.log("Existing groups in '{}' group deleted.", self.name, logging=logging)

            # Cancel loading
            else:
                self.log(
                    "'{}' group was not loaded because it has already been traced back.",
                    self.name, logging=logging
                )

        # Set load path parameter
        if len(self) == 0:
            self.load = True
            self.load_path = self.set_path(
                self.config.load_path if load_path is None else load_path, 'load_path',
                name=self.name, extension='group', file_type='Group',
                full_path=True, check=True, create=False
            )

            # Group unpickling
            from pickle import load
            file = open(self.load_path, 'rb')
            parameters, groups = load(file)
            file.close()

            # Remove group from the collection and delete parameters
            if remove:
                self.remove(logging=False)
            for parameter in [
                    parameter for parameter in vars(self).keys()
                    if parameter not in (
                        'load', 'save', 'load_path', 'save_path',
                        'data_dir', 'output_dir', 'logs_path'
                    )
                ]:
                del vars(self)[parameter]

            # Parameters and groups import, and group re-addition to the collection
            vars(self).update(parameters)
            for group in groups:
                self.append(group)
            self.add(forced=forced, default=default, cancel=cancel, logging=False)

            # Logging
            self.log("'{}' group loaded from '{}'.", self.name, self.load_path, logging=logging)

    def save_group(self, save_path=None, forced=None, default=None, cancel=None, logging=True):
        """
        Saves a gorup to a binary file. self.save_path is defined as the actual path to the
        file. If forced, the existing file is overwritten, otherwise user input is required
        to proceed if a file already exists.
        """

        def save(self):
            """Saves the group to a file."""

            # Group pickling
            from pickle import dump
            file = open(self.save_path, 'wb')
            dump(
                (
                    {
                        parameter: vars(self)[parameter] for parameter in vars(self).keys()
                        if parameter not in (
                            'load', 'save', 'load_path', 'save_path',
                            'data_dir', 'output_dir', 'logs_path'
                        )
                    }, tuple(group for group in self)
                ), file
            )
            file.close()

        # Set save path parameter
        self.save = True
        self.save_path = self.set_path(
            self.config.save_path if save_path is None else save_path, 'save path',
            name=self.name, extension='group', file_type='Group',
            full_path=True, check=False, create=True
        )

        # Choose behaviour if a file already exists
        if path.exists(self.save_path):
            forced, default, cancel = self.choose(
                f"A '{self.name}' group file already exists at '{save_path}'.", 3,
                forced, default, cancel
            )

            # Delete existing file and save group
            if forced:
                remove(self.save_path)
                save(self)

                # Logging
                self.log(
                    "Existing '{}' group file located at '{}' deleted and replaced.",
                    self.name, self.save_path, logging=logging
                )

            # Set default file name and save group
            if default:
                old_filename = path.basename(self.save_path)
                self.save_path = get_default_filename(self.save_path)
                save(self)

                # Logging
                self.log(
                    "Group file renamed from '{}' to '{}', and '{}' group saved at '{}'.",
                    old_filename, path.basename(self.save_path), self.name, self.save_path,
                    logging=logging
                )

            # Cancel save
            if cancel:
                self.log(
                    "'{}' group was not saved because a group file already exists at '{}'.",
                    self.name, self.save_path, logging=logging
                )
                self.save = self.set_boolean(self.config.save)
                del self.save_path

        # Save group
        else:
            save(self)

            # Logging
            self.log("'{}' group saved at '{}'.", self.name, self.save_path, logging=logging)

    def log(self, message, *words, logs_path=None, level='info', display=False, logging=True):
        """
        Logs the 'message' with the appropriate 'level', if logs have been configured. If logs
        have not been configured, logs are configured by checking if the 'logs_path' already
        exists, creating it if needed and configuring the logs file.  By default, logs files will
        created in the base directory with a default name. The file name must end with an '.log'
        extension. Futhermore, if 'display' is True, the message is also printed onscreen.
        """

        # Set logs path parameter
        if 'logs_path' not in vars(self).keys():
            self.logs_path = self.set_path(
                self.config.logs_path, 'logs_path', create=True, none=True
            )

        # Create logging message
        log(
            message, *words, logs_path=self.logs_path if logs_path is None else logs_path,
            level=level, display=display, logging=logging
        )

    def stop(self, condition, error, message, *words):
        """Calls the stop function from Collection with self.name, if it has been set."""

        # Addition of group name to stop function call
        stop(
            condition, error, message, *words,
            name=self.name if 'name' in vars(self) else None, extra=2
        )

    def set_timer(self, operation=None):
        """Records the operation time."""

        # Add the timer of the previous operation
        if self.timer:
            if 'previous_operation' in vars(self):
                if self.previous_operation is not None:

                    # Compute operation time
                    operation_time = get_time() - self.previous_time
                    if self.previous_operation in self.timers.keys():
                        self.timers[self.previous_operation] += operation_time
                    else:
                        self.timers[self.previous_operation] = operation_time

            # Save preivous operation and time
            self.previous_operation = operation
            self.previous_time = get_time()

    def show_timer(self):
        """Displays the time required to perform various steps in the group's creation."""

        # Compute the total time
        if self.timer:
            total_time = sum([time for operation, time in self.timers.items()])

            # Create the time string
            def create_time_str(name, delay):
                print('{}: {:.3f} s, {:.2f}%'.format(name, delay, delay / total_time * 100))

            # Show time for galactic orbit integration
            create_time_str('Galactic orbits', self.timers['orbits'])

            # Show times for association size metrics computation
            if self.size_metrics:
                if self.cov_metrics:
                    create_time_str(
                        'Covariance (empirical)',
                        self.timers['cov_metrics']
                    )
                if self.cov_robust_metrics:
                    create_time_str(
                        'Covariance (robust)',
                        self.timers['cov_robust_metrics']
                    )
                if self.cov_sklearn_metrics:
                    create_time_str(
                        'Covariance (sklearn)',
                        self.timers['cov_sklearn_metrics']
                    )
                if self.mad_metrics:
                    create_time_str(
                        'Median absolute deviation',
                        self.timers['mad_metrics']
                    )
                if self.tree_metrics:
                    create_time_str(
                        'Tree branches',
                        self.timers['tree_metrics']
                    )
