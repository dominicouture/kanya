# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
collection.py: Defines the Collection class and initializes global parameters. Global functions
used by the Series class and more, such as directory, output, log and stop are also defined.
"""

import numpy as np
from os import path, makedirs, getcwd, chdir, listdir, remove
from .tools import enumerate_strings

class Collection(list):
    """
    Contains all series objects created from data, a model or a file. Functions to add,
    remove, update, copy, load, traceback, save, create and select series are defined, as
    well as a function to generate default series names from the existing series names in
    self.series.
    """

    def __init__(self, base_dir=None, data_dir=None, output_dir=None, logs_path=None):
        """
        Initializes a collection by creating a self.series dictionary, as well as default output
        directories and logs path. A base directory is also created, from which other relative
        directory paths originate.
        """

        # Initialize self.series dictionary
        self.initialize_series()

        # Set the base directory
        self.set_base(base_dir)

        # Set default values for data and output directories, and logs path
        self.set_default_data(data_dir)
        self.set_default_output(output_dir)
        self.set_default_logs(logs_path)

    def initialize_series(self):
        """
        Creates a self.series dictionary of the index of Series object using their Series.name
        value as dictionary keys.
        """

        self.series = {self[i].name: i for i in range(len(self))}

    def set_base(self, base_dir=None):
        """
        Sets the default base directory ('base_dir'), from which other relative directory paths
        originate. This directory is not created if it doesn't already exist. By default, the data
        directory is set to the working directory.
        """

        # Check if base directory is a string or None
        check_type(base_dir, 'base_dir', ('string', 'None'))

        # Set base directory
        # self.base_dir = path.abspath(path.join(path.dirname(path.realpath(__file__)), '..'))
        self.base_dir = path.abspath(getcwd() if base_dir is None else base_dir)

    def set_default_data(self, data_dir=None):
        """
        Sets the default data directory ('data_dir'). This directory is not created if it doesn't
        already exist. By default, the data directory is set to 'Data' in the base directory.
        """

        # Check the type of data_dir
        check_type(data_dir, 'data_dir', ('string', 'None'))

        # Set default data directory as the absolute directory
        self.data_dir = get_abspath(
            self.base_dir, '' if data_dir is None else data_dir, 'data_dir'
        )

    def set_default_output(self, output_dir=None):
        """
        Sets the default output directory ('output_dir'). The directory is only created if
        something is actually saved in it. By default, the output directory is set to 'Output' in
        the base directory.
        """

        # Check the type of output_dir
        check_type(output_dir, 'output_dir', ('string', 'None'))

        # Set default output directory as the absolute directory
        self.output_dir = get_abspath(
            self.base_dir, '' if output_dir is None else output_dir, 'output_dir'
        )

    def set_default_logs(self, logs_path=None):
        """
        Sets the default logs path ('logs_path'). The log file and directory are only created if a
        message is actually logged. The file name must end with an '.log' extension. A default logs
        name, based on the current date and time is created.
        """

        # Check the type of logs_path
        check_type(logs_path, 'logs_path', ('string', 'None'))

        # Set the logs path
        self.logs_path = '' if logs_path is None else logs_path
        print(self.logs_path)

        # Add a name and an extension to the logs path, if needed
        if path.basename(self.logs_path) == '':
            from time import strftime
            self.logs_path = path.join(
                self.logs_path, 'kanya_{}.log'.format(strftime('%Y-%m-%d_%H-%M-%S'))
            )
        if path.splitext(self.logs_path)[-1] == '':
            self.logs_path += '.logs'

        # Redefine the logs path as the absolute path
        self.logs_path = get_abspath(self.base_dir, self.logs_path, 'logs_path')

        # Check if the file is a logs file
        stop(
            path.splitext(self.logs_path)[-1].lower() != '.log', 'ValueError',
            "'{}' is not a log file (with a .log extension).", path.basename(self.logs_path)
        )

        # Set the logs name
        self.logs_name = path.basename(self.logs_path)

        # Set logs configuration as False
        self.logs_configured = False

    def new(
        self, parent=None, file_path=None, args=False, forced=None,
        default=None, cancel=None, logging=True, **parameters
    ):
        """ Creates a new Series in the collection. Arguments are the same as those of
            Series.__init__.
        """

        from .series import Series

        Series(
            parent=parent, file_path=file_path, args=args, forced=forced,
            default=default, cancel=cancel, logging=logging, **parameters
        )

    def add(self, *series, forced=None, default=None, cancel=None, logging=True):
        """
        Adds one or multiple new series to the collection. If forced is True, any existing
        series with the same name is overwritten, otherwise user input is required to proceed
        (overwrite, do nothing or default). If default is True, then instead of asking for an
        input, a default name is used and the series added to the collection.
        """

        # Series addition and replacement, if needed
        for series in series:
            stop(
                str(type(series)) != "<class 'series.Series'>", 'TypeError',
                "'series' must be Series object ({} given).", type(series)
            )
            series.add(forced=forced, default=default, cancel=cancel, logging=logging)

    def remove(self, *series, logging=True):
        """ Removes one or multiple series from the collection from series.name. """

        # Series deletion from collection
        for series in self.select(*series):
            series.remove(logging=logging)

    def reset(self, *series, logging=True):
        """ Resets one or multiple series from the collection from series.name. """

        # Series deletion from collection
        for series in self.select(*series):
            series.reset(logging=logging)

    def update(
        self, *series, parent=None, file_path=None, args=False,
        logging=True, traceback=True, **parameters
    ):
        """
        Updates the series by modifying its self.config configuration, re-initializing itself
        and deleting existing groups. The groups are also traced back again if they had been
        traced back before the update.
        """

        for series in self.select(*series):
            series.update(
                parent=parent, file_path=file_path, args=args,
                logging=logging, traceback=traceback, **parameters
            )

    def copy(
        self, *series, parent=None, file_path=None, args=False,
        logging=True, traceback=True, **parameters
    ):
        """
        Copies the series under a new name. If 'parameters' are provided, this new Series
        object is updated as well with those new parameters. If no new 'name' is provided
        as a parameter, a default name is used instead.
        """

        for series in self.select(*series):
            series.copy(
                parent=parent, file_path=file_path, args=args,
                logging=logging, traceback=traceback, **parameters
            )

    def load_series(self, *series, load_path=None, forced=None, logging=True):
        """ Loads one or multiple series from the binary file. If forced, existing groups are
            overwritten.
        """

        for series in self.select(*series):
            series.load_series(load_path=load_path, forced=forced, logging=logging)

    def traceback(self, *series, mode=None, forced=None, logging=True):
        """
        Traces back all series in self if no series name is given or selected series given
        given in argument by name. If forced, existing groups are overwritten.
        """

        for series in self.select(*series):
            series.traceback(mode=mode, forced=forced, logging=logging)

    def chronologize(
            self, *series, size_metrics=None, cov_metrics=None, cov_robust_metrics=None,
            cov_sklearn_metrics=None, mad_metrics=None, mst_metrics=None, logging=True
        ):
        """
        Computes the kinematic age of all series in self if no series name is given or selected
        series given given in argument by name.
        """

        for series in self.select(*series):
            series.chronologize(
                size_metrics=size_metrics, cov_metrics=cov_metrics,
                cov_robust_metrics=cov_robust_metrics, cov_sklearn_metrics=cov_sklearn_metrics,
                mad_metrics=mad_metrics, mst_metrics=mst_metrics, logging=logging
            )

    def save_series(
        self, *series, save_path=None, forced=None, default=None, cancel=None, logging=True
    ):
        """
        Saves one or multiple series to a binary file. If forced, existing files are
        overwritten.
        """

        for series in self.select(*series):
            series.save_series(
                save_path=save_path, forced=forced, default=default, cancel=cancel, logging=logging
            )

    def create(
        self, *series, load_path=None, save_path=None, mode=None, size_metrics=None,
        cov_metrics=None, cov_robust_metrics=None, cov_sklearn_metrics=None, mad_metrics=None,
        mst_metrics=None, forced=None, default=None, cancel=None, logging=True
    ):
        """
        Either load one or multiple series from a file, or traces back one or multiple
        series from data or a model. If needed, the series is also saved.
        """

        for series in self.select(*series):
            series.create(
                load_path=load_path, save_path=save_path, mode=mode, size_metrics=size_metrics,
                cov_metrics=cov_metrics, cov_robust_metrics=cov_robust_metrics,
                cov_sklearn_metrics=cov_sklearn_metrics, mad_metrics=mad_metrics,
                mst_metrics=mst_metrics, forced=forced, default=default, cancel=cancel,
                logging=logging
            )

    def select(self, *series):
        """
        Selects one or multiple series from a collection. If no series name are specified,
        all series are selected.
        """

        # All series
        if len(series) == 0:
            return [series for series in self]

        # Selected series only
        else:

            # Selected series from an Series object
            selected_series = []
            for name in series:
                if str(type(name)) != "<class 'series.Series'>":
                    selected_series.append(name)

                # Selected series from a name, check if it exists
                else:
                    check_type(name, 'name', 'string')
                    stop(
                        name not in self.series.keys(), 'NameError',
                        "Series '{}' is not in the collection.", name
                    )

            return [self[self.series[name]] for name in series]

    def get_default_name(self, name=None):
        """
        Loops over all Series in self and returns a default 'name-i' name where i is the
        lowest possible number for an series titled 'name'.
        """

        # Initialization
        name = name if name is not None else 'untitled'
        check_type(name, 'name', 'string')

        # Identify whether 'name' ends with a digit
        i = name.split('-')[-1]
        if i.isdigit():
            name = name[:-(len(i) + 1)]

        # Loops over Series in self
        i = 1
        while '{}-{}'.format(name, i) in [series.name for series in self]:
            i += 1

        return '{}-{}'.format(name, i)

def get_abspath(base_path, file_path, parameter, check=False, create=False):
    """
    Returns the absolute path to the directory or file. 'file_path' can either be relative path
    to 'base_path' or an absolute path. Checks if 'file_path' already exists, creates the
    directory if needed.
    """

    # Check the type of paremeter, base_path, file_path, check and create
    check_type(parameter, 'parameter', 'string')
    check_type(base_path, 'base_path', 'string')
    check_type(file_path, 'file_path', 'string')
    check_type(check, 'check', 'boolean')
    check_type(create, 'create', 'boolean')

    # Check if base_path is a directory
    stop(
        not path.isdir(base_path), 'ValueError',
        "'base_path' must be a directory ({} given).", base_path
    )

    # Directory or file path formatting
    working_dir = getcwd()
    chdir(path.abspath(base_path))
    file_path = path.join(
        path.abspath(path.dirname(file_path)),
        path.basename(file_path)
    )
    chdir(working_dir)

    # Check if the directory or file exists
    if check:
        stop(
            not path.exists(file_path), 'FileNotFoundError',
            "'{}' located at '{}' does not exist.", parameter, file_path
        )

    # Directory creation, if needed
    if create and not path.exists(path.dirname(file_path)):
        makedirs(path.dirname(file_path))

    return file_path

def get_default_filename(file_path):
    """
    Loops over all files and sub-directories in the directory of 'file_path' and returns a
    default 'path/name-i.extension' path where i is the lowest possible number for a file
    titled 'name'.
    """

    # Initialization
    directory = path.dirname(file_path)
    name = path.splitext(path.basename(file_path))[0]
    name = name if name != '' else 'untitled'
    extension = path.splitext(file_path)[-1]

    # Identify whether 'name' ends with a digit
    i = name.split('-')[-1]
    if i.isdigit():
        name = name[:-(len(i) + 1)]

    # Loops over Series in self
    i = 1
    while '{}-{}{}'.format(name, i, extension) in listdir(directory):
        i += 1

    return path.join(directory, '{}-{}{}'.format(name, i, extension))

def log(message, *words, logs_path=None, level='info', display=False, logging=True):
    """
    Logs the 'message' with the appropriate 'level', if logs have been configured. If logs have
    not been configured, logs are configured by checking if the 'logs_path' already exists,
    creating it if needed and configuring the logs file.  If blank (i.e. ''), logs files will
    created in the base directory with a default name. The file name must end with an '.log'
    extension. Futhermore, if 'display' is True, the message is also printed onscreen.
    """

    # Check the type of logging
    check_type(logging, 'logging', 'boolean')

    # Log the message, if needed
    if logging:

        # Check the type of logs_path
        check_type(logs_path, 'logs_path', ('string', 'None'))

        # Set the logs path
        logs_path = collection.logs_path if logs_path is None else logs_path

        # Add a name and an extension to the logs path, if needed
        if path.basename(logs_path) == '':
            logs_path = path.join(logs_path, collection.logs_name)
        if path.splitext(logs_path)[-1] == '':
            logs_path += '.logs'

        # Redefine the logs path as the absolute path
        logs_path = get_abspath(collection.base_dir, logs_path, 'logs_path', create=True)

        # Check if the file is a logs file
        stop(
            path.splitext(logs_path)[-1].lower() != '.log', 'ValueError',
            "'{}' is not a log file (with a .log extension).", path.basename(logs_path)
        )

        # Logs configuration, if needed
        from logging import basicConfig, root, info, warning, INFO
        if not collection.logs_configured or collection.logs_path != logs_path:

            # Logs configuration, if logs_path matches collection.logs_path
            if logs_path == collection.logs_path:
                if not collection.logs_configured:
                    for handler in root.handlers[:]:
                        root.removeHandler(handler)
                    collection.logs_configured = True
                    basicConfig(
                        filename=collection.logs_path, format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S -', level=INFO
                    )

            # Logs configuration, if logs_path doesn't match collection.logs_path
            else:
                if collection.logs_configured:
                    for handler in root.handlers[:]:
                        root.removeHandler(handler)
                    collection.logs_configured = False
                basicConfig(
                    filename=logs_path, format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S -', level=INFO
                )

        # Check the type of message
        check_type(message, 'message', 'string')

        # Check the type of words, and format message
        if len(words) > 0:
            for word in words:
                stop(
                    type(word) != str, 'TypeError',
                    "'words' must all be strings ({} given).", type(word)
                )
            message = message.format(*words)

        # Check the type and value of level
        check_type(level, 'level', 'string')
        level = level.lower()
        stop(
            level not in ('info', 'warning'), 'ValueError',
            "'level' must either be 'info' or 'warning' ({} given).", level
        )

        # Log message
        if level == 'info':
            info(message)
        if level == 'warning':
            warning(message)

        # Show message
        check_type(display, 'display', 'boolean')
        if display:
            print(message)

def check_type(value, label, type_labels):
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
    stop(
        type(value) not in [types[label] for label in type_labels],
        'TypeError', "'{}' must be a {} ({} given).",
        label, enumerate_strings(*type_labels), type(value)
    )

def stop(condition, error, message, *words, name=None, extra=1):
    """
    Raises an exception if 'condition' is True, logs it into the log file if logs have
    been configured and stops the execution. 'error' is a string representing an exception
    class, 'message' is the error message to be displayed, 'words' is a list of words to
    be inserted into 'message' and 'extra' is the number of superfluous lines in the
    traceback stack. If an exception isn't being handled, the function is recursively
    called within an except statement.
    """

    # Check if condition is True
    if condition:

        # Exception information extraction
        from sys import exc_info, exit
        exc_type, exc_value = exc_info()[:2]

        # If no exception is being handled, an exception is raised
        if exc_type is None and exc_value is None:
            try:
                exec(f'raise {error}')
            except:
                stop(True, error, message, *words, name=name, extra=extra + 1)

        # If an exception is being handled, its traceback is formatted and execution is terminated
        else:

            # Traceback message creation
            if len(words) > 0:
                message = message.format(*words)
            tb_message = (
                f"{error} in '{name}' series: {message}" if name is not None
                else f'{error}: {message}'
            )

            # Traceback stack formatting
            from traceback import format_stack
            tb_stack = ''.join(
                ['An exception has been raised: \n'] + format_stack()[:-extra] + [tb_message]
            )

            # Exception logging, if logs have been configured, and code termination
            if 'collection' in globals().keys() and collection.logs_configured:
                log(tb_stack, level='warning')
            print(tb_stack)
            exit()

# Collection initialization
collection = Collection()
