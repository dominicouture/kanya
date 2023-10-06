# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
init.py: Imports information from the configuration file, command line arguments and parameters
into a Config. This script must be run first to configure a Series.
"""

import pandas as pd
from copy import deepcopy
from .collection import *
from .coordinate import *

class Config():
    """
    Contains the parameters imported from a configuration file (which must be a Python file),
    command line arguments, parameters in the __init__ function call or another Config object,
    as well as related methods, a Parameter class and a dictionary of default values. A Config
    object can then be used as the input of a Series object.
    """

    class Parameter():
        """Components of a configuration parameter."""

        # Default components
        default_components = {
            component: None for component in (
                'label', 'name', 'values', 'units', 'system', 'type'
            )
        }

        def __init__(self, **components):
            """Initializes a Parameter object with the given 'components'."""

            # Initialization
            vars(self).update(self.default_components)

            # Update
            self.update(components.copy())

        def update(self, parameter):
            """
            Updates the components of 'self' with those of another 'parameter' or a dictionary
            of components, only if those new components are part of the default components
            tuple or singular forms of default components.
            """

            # Parameter conversion into a dictionary and copy
            if type(parameter) == type(self):
                parameter = vars(parameter)
            parameter = deepcopy(parameter)
            if type(parameter) == dict:

                # Component conversion from singular to plural form in parameter
                for component in ('value', 'unit'):
                    components = component + 's'
                    if component in parameter.keys():
                        parameter[components] = parameter[component]
                        parameter.pop(component)

                # Update self with parameter, only if the component already exists in the default
                # component and is not None
                if type(parameter) == dict:
                    vars(self).update(
                        {
                            key: component for key, component in parameter.items()
                            if key in self.default_components and component is not None
                        }
                    )

        def __repr__(self):
            """Returns a string with all the components of the parameter."""

            return '({})'.format(
                ', '.join(['{}: {}'.format(key, value) for key, value in vars(self).items()])
            )

    # Default units
    unitless = ''
    time_unit = System.default_units['time'].label
    length_unit = System.default_units['length'].label
    length_units = tuple(variable.unit.label for variable in systems['cartesian'].position)
    speed_unit = System.default_units['speed'].label
    speed_units = tuple(variable.unit.label for variable in systems['cartesian'].velocity)

    # Default parameters
    default_parameters_file = path.join(path.dirname(__file__), 'resources/default_parameters.csv')
    parameter_dataframe = pd.read_csv(default_parameters_file, delimiter=';')
    default_parameters = {}
    for index, row in parameter_dataframe.iterrows():
        components = {}
        for column in row.index:
            components[column] = eval(row[column])
        parameter = Parameter(**components)
        default_parameters[parameter.label] = parameter

    # Position and velocity paramaters
    position_parameters = ('position', 'position_error', 'position_scatter')
    velocity_parameters = ('velocity', 'velocity_error', 'velocity_scatter')

    def __init__(self, parent=None, file_path=None, args=False, **parameters):
        """
        Configures a Config objects from, in order, 'parent', an existing Config object,
        'file_path', a string representing a path to a configuration file, 'args' a boolean value
        that sets whether command line arguments are used, and '**parameters', a dictionary
        of dictionaries, where keys must match values in Parameter.default_components, or
        Config.Parameter objects. Only values that match a key in self.default_parameters are
        used. If no value are given the default parameter is used instead.
        """

        # Check if parent is a Config or None
        stop(
            type(parent) not in (Config, type(None)), 'TypeError',
            "'parent' can either be a Config object or None ({} given).", type(parent)
        )

        # Import default or parent's parameters
        if parent is None:
            self.initialize_from_parameters(self.default_parameters)
        if type(parent) == Config:
            self.initialize_from_parameters(vars(parent))

        # Import parameters from file path, arguments and parameters dictionary
        if file_path is not None:
            self.initialize_from_path(file_path)
        if args:
            self.initialize_from_arguments(args)
        if len(parameters) > 0:
            self.initialize_from_parameters(parameters)

        # Configure parameters
        self.configure_parameters()

    def initialize_from_path(self, config_path):
        """
        Initializes a Config object from a configuration file located at 'config_path', and
        checks for errors. 'config_path' can be an absolute path or relative to the current
        working directory.
        """

        # Check the type of config_path
        check_type(config_path, 'config_path', 'string')

        # Redefine the configuration path as the absolute path
        config_path = get_abspath(collection.base_dir, config_path, 'config_path', check=True)

        # Check if the configuration path is a full path
        stop(
            path.basename(config_path) == '', 'ValueError',
            "'config_path' must be a path to file, not a directory ({} given).", config_path
        )

        # Check if the configuration file is a Python file
        stop(
            path.splitext(config_path)[-1].lower() != '.py', 'ValueError',
            "The file located at '{}' is not a Python file. (with a .py extension)", config_path
        )

        # Configuration file import
        from importlib.util import spec_from_file_location, module_from_spec
        try:
            spec = spec_from_file_location(
                path.splitext(path.basename(config_path))[0], config_path
            )
            parameters = module_from_spec(spec)
            vars(parameters).update(deepcopy(vars(self)))
            spec.loader.exec_module(parameters)

        # Check if all names are valid
        except NameError as error:
            stop(
                True, 'NameError',
                "{}, only values in 'default_parameters' are configurable.", error.args[0]
            )

        # Parameters import
        self.initialize_from_parameters(vars(parameters))

    def initialize_from_arguments(self, args):
        """
        Parses arguments from the commmand line, creates an arguments object and adds these
        new values to the Config object. Also checks if 'args' is a boolean value. Overwrites
        values given in a configuration file.
        """

        # Check the type of args
        check_type(args, 'args', 'boolean')

        # Parse arguments
        from argparse import ArgumentParser
        parser = ArgumentParser(
            prog='kanya',
            description='traces given or simulated moving groups of stars back to their origin.'
        )
        parser.add_argument(
            '-n', '--name', action='store', type=str,
            help='name of the series of tracebacks.'
        )
        parser.add_argument(
            '-d', '--data', action='store_true',
            help='use the data parameter in the configuration file as input.'
        )
        parser.add_argument(
            '-m', '--model', action='store_true',
            help='model an input based on simulation parameters in the configuration file.'
        )
        parser.add_argument(
            '-l', '--load', action='store_true',
            help='load the input data from a file.'
        )
        parser.add_argument(
            '-s', '--save', action='store_true',
            help='save the output data to a file.'
        )
        args = parser.parse_args()

        # Series name import if not None
        if args.name is not None:
            self.name.values = args.name

        # Mode import, overwrites any value imported from a path
        self.from_data.values = args.data
        self.from_model.values = args.model
        self.load.values = args.load
        self.save.values = args.save

    def initialize_from_parameters(self, parameters):
        """
        Initializes a Config object from a parameters dictionary. Overwrites values given in
        a configuration file or as arguments in command line. Values in the dictionary can
        either be Parameter objects or dictionaries of components.
        """

        # Filter parameters that don't match a default parameter
        for key, parameter in filter(
                lambda item: item[0] in self.default_parameters.keys(), parameters.items()
        ):

            # Copy parameter
            parameter = deepcopy(parameter)

            # Check if the parameter is a dictionary or a Parameter object
            stop(
                type(parameter) not in (dict, self.Parameter), 'TypeError',
                "A parameter must be a Config.Parameter object or a dictionary ({} given).",
                type(parameter)
            )

            # Parameter update from a Parameter object or dictionary
            if key in vars(self).keys() and type(vars(self)[key]) == Config.Parameter:
                vars(self)[key].update(parameter)

            # Parameter object import from default parameters or parent configuration
            elif type(parameter) == Config.Parameter:
                vars(self)[key] = parameter

    def configure_parameters(self):
        """
        Checks if all parameters are present and are Parameter objects with all their components,
        and checks for invalid parameters and components. The type of 'label', 'name' and 'system'
        components is checked (str), and the 'system' component is converted to its corresponding
        class if it is not None.
        """

        # Check if all parameters are present and are Config.Parameter objects
        for parameter_label, parameter in self.default_parameters.items():
            stop(
                parameter_label not in vars(self), 'ValueError',
                "Required parameter '{}' is missing in the configuration.", parameter_label
            )
            stop(
                type(parameter) != self.Parameter, 'TypeError',
                "'{}' must be a Config.Parameter object ({} given).", parameter_label,
                type(parameter)
            )

            # Check if all components are present
            for component_label, component in self.Parameter.default_components.items():
                stop(
                    component_label not in vars(parameter).keys(),
                    'ValueError',
                    "Required component '{}' is missing in '{}' parameter in the configuration.",
                    component_label, parameter_label
                )

        # Check for invalid parameters and components
        for parameter_label, parameter in vars(self).items():
            stop(
                parameter_label not in self.default_parameters.keys(), 'ValueError',
                "Parameter '{}' is invalid in the configuration.", parameter_label
            )
            for component_label, component in vars(parameter).items():
                stop(
                    component_label not in self.Parameter.default_components.keys(),
                    'ValueError',
                    "Component '{}' is invalid in '{}' parameter in the configuration.",
                    component_label, parameter_label
                )

                # Check whether all components, but values, units, are strings or None
                if component_label not in ('values', 'units'):
                    stop(
                        component is not None and type(component) not in (str, System), 'TypeError',
                        "'{}' component in '{}' parameter must be a string or None ('{}' given.)",
                        component_label, parameter_label, type(component)
                    )

            # Check if label and name were changed
            default_parameter = self.default_parameters[parameter_label]
            if parameter.label != parameter_label:
                parameter.label = parameter_label
            if parameter.name != default_parameter.name:
                parameter.name = default_parameter.name

            # Check if system is valid and converts it to a System object
            if type(parameter.system) == System:
                pass
            elif parameter.system is not None:
                stop(
                    parameter.system.lower() not in systems.keys(), 'ValueError',
                    "'system' component of '{}' is invalid ({} required, {} given).",
                    parameter.label, list(systems.keys()), parameter.system
                )
                parameter.system = systems[parameter.system.lower()]
            elif default_parameter.system is not None:
                parameter.system = systems[default_parameter.system]

    def __repr__(self):
        """Returns a string of name of the configuration."""

        return '{} Configuration'.format(self.name.values)
