# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
collection.py: Defines the Collection class and initializes global parameters. Global functions
used by the Series class and more, such as directory, output, log and stop are also defined.
"""

from os import path, makedirs, getcwd, chdir

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
        stop(
            base_dir is not None and type(base_dir) != str, 'TypeError',
            "'base_dir' must be a string or None ({} given).", type(base_dir)
        )

        # Set base directory
        # self.base_dir = path.abspath(path.join(path.dirname(path.realpath(__file__)), '..'))
        self.base_dir = path.abspath(getcwd() if base_dir is None else base_dir)

    def set_default_data(self, data_dir=None):
        """
        Sets the default data directory ('data_dir'). This directory is not created if it doesn't
        already exist. By default, the data directory is set to 'Data' in the base directory.
        """

        # Check if data directory parameter is a string or None
        stop(
            data_dir is not None and type(data_dir) != str, 'TypeError',
            "'data_dir' must be a string or None ({} given).", type(data_dir)
        )

        # Set default data directory as the absolute directory
        self.data_dir = directory(
            self.base_dir, 'Data' if data_dir is None else data_dir, 'data_dir'
        )

    def set_default_output(self, output_dir=None):
        """
        Sets the default output directory ('output_dir'). The directory is only created if
        something is actually saved in it. By default, the output directory is set to 'Output' in
        the base directory.
        """

        # Check if output directory parameter is a string or None
        stop(
            output_dir is not None and type(output_dir) != str, 'TypeError',
            "'output_dir' must be a string or None ({} given).", type(output_dir)
        )

        # Set default output directory as the absolute directory
        self.output_dir = directory(
            self.base_dir, 'Output' if output_dir is None else output_dir, 'output_dir'
        )

    def set_default_logs(self, logs_path=None):
        """
        Sets the default logs path ('logs_path'). The log file and directory are only created if a
        message is actually logged. The file name must end with an '.log' extension. By default,
        a 'Logs' directory is created in the output directory and the name is based on the current
        date and time.
        """

        from time import strftime

        # Check if logs path parameter is a string or None
        stop(
            logs_path is not None and type(logs_path) != str, 'TypeError',
            "'logs_path' must be a string or None ({} given).", type(logs_path)
        )

        # Set default logs path
        self.logs_path = (
            path.join(self.output_dir, 'Logs') + '/' if logs_path is None else logs_path
        )

        # Redefine logs path as the absolute directory
        # Create a default file name, if no file name is provided
        self.logs_path = path.join(
            directory(self.base_dir, path.dirname(self.logs_path), 'logs_path'),
            'kanya_{}.log'.format(strftime('%Y-%m-%d_%H-%M-%S'))
            if path.basename(self.logs_path) == '' else path.basename(self.logs_path)
        )

        # Check if the file is a logs file
        stop(
            path.splitext(self.logs_path)[1].lower() != '.log', 'TypeError',
            "'{}' is not a log file (with a .log extension).", path.basename(self.logs_path)
        )

        # Set logs configuration as False
        self.logs_configured = False

    def new(
        self, parent=None, file_path=None, args=False, forced=False, default=False,
        cancel=False, logging=True, **parameters
    ):
        """ Creates a new Series in the collection. Arguments are the same as those of
            Series.__init__.
        """

        from .series import Series
        Series(parent, file_path, args, forced, default, cancel, logging, **parameters)

    def add(self, *series, forced=False, default=False, cancel=False, logging=True):
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
            series.add(forced, default, cancel, logging)

    def remove(self, *series, logging=True):
        """ Removes one or multiple series from the collection from series.name. """

        # Series deletion from collection
        for series in self.select(*series):
            series.remove(logging)

    def reset(self, *series, logging=True):
        """ Resets one or multiple series from the collection from series.name. """

        # Series deletion from collection
        for series in self.select(*series):
            series.reset(logging)

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
            series.update(parent, file_path, args, logging, traceback, **parameters)

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
            series.copy(parent, file_path, args, logging, traceback, **parameters)

    def load_from_file(self, *series, load_path=None, forced=False):
        """ Loads one or multiple series from the binary file. If forced, existing groups are
            overwritten.
        """

        for series in self.select(*series):
            series.load_from_file(load_path, forced)

    def traceback(self, *series, forced=False, mode=None):
        """
        Traces back all series in self if no series name is given or selected series given
        given in argument by name. If forced, existing groups are overwritten.
        """

        for series in self.select(*series):
            series.traceback(forced, mode)

    def save_to_file(self, *series, save_path=None, forced=False, default=False, cancel=False):
        """
        Saves one or multiple series to a binary file. If forced, existing files are
        overwritten.
        """

        for series in self.select(*series):
            series.save_to_file(save_path, forced, default, cancel)

    def create(self, *series, forced=False, default=False, cancel=False):
        """
        Either load one or multiple series from a file, or traces back one or multiple
        series from data or a model. If needed, the series is also saved.
        """

        for series in self.select(*series):
            series.create(forced, default, cancel)

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
                    stop(
                        type(name) != str, 'TypeError',
                        "'series' must be a string ({} given).", type(name)
                    )
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
        stop(type(name) != str, 'TypeError', "'name' must be a string ({} given).", type(name))

        # Identify whether 'name' ends with a digit
        i = name.split('-')[-1]
        if i.isdigit():
            name = name[:-(len(i) + 1)]

        # Loops over Series in self
        i = 1
        while '{}-{}'.format(name, i) in [series.name for series in self]:
            i += 1

        return '{}-{}'.format(name, i)

def directory(base, directory, parameter, check=False, create=False):
    """
    Checks for type errors, checks if 'directory' already exists, creates it if needed and
    returns the absolute directory path. The directory can either be relative path to 'base'
    or an absolute path.
    """

    # Check the type of base, directory and parameter
    stop(
        type(base) != str, 'TypeError',
        "The base of '{}' must be a string ({} given).", parameter, type(base)
    )
    stop(
        type(directory) != str, 'TypeError',
        "The base '{}' must be a string ({} given).", parameter, type(directory)
    )
    stop(
        type(parameter) != str, 'TypeError',
        "'parameter' must be a string ({} given).", type(parameter)
    )

    # Directory formatting
    working_dir = getcwd()
    chdir(path.abspath(base))
    directory = path.abspath(directory)
    chdir(working_dir)

    # Check if the directory exists
    stop(
        type(check) != bool, 'TypeError',
        "'check' must be a boolean ({} given).", type(check)
    )
    if check:
        stop(
            not path.exists(directory), 'FileNotFoundError',
            "'{}' at '{}' does not exist.", parameter, directory
        )

    # Directory creation, if needed
    stop(
        type(create) != bool, 'TypeError',
        "'create' must be a boolean ({} given).", type(create)
    )
    if create and not path.exists(directory):
        makedirs(directory)

    return directory

def get_default_filename(file_path):
    """
    Loops over all files and sub-directories in the directory of 'file_path' and returns a
    default 'path/name-i.extension' path where i is the lowest possible number for a file
    titled 'name'.
    """

    from os import path, listdir

    # Initialization
    directory = path.dirname(file_path)
    name = path.splitext(path.basename(file_path))[0]
    name = name if name != '' else 'untitled'
    extension = path.splitext(file_path)[1]

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

    from logging import basicConfig, root, info, warning, INFO

    # Check if logging is a boolean
    stop(
        type(logging) != bool, 'TypeError',
        "'logging' must be a boolean ({} given).", type(logging)
    )

    # logs path parameter
    if logging:
        logs_path = collection.logs_path if logs_path is None else logs_path

        # Checks and configuration skipped if logs have been configured already
        if not collection.logs_configured or collection.logs_path != logs_path:

            # Check if logs path is a string
            stop(
                type(logs_path) != str, 'TypeError',
                "'logs_path' must be a string ({} given).", type(logs_path)
            )

            # Redine logs path as the absolute path
            logs_path = path.join(
                directory(collection.base_dir, path.dirname(logs_path), 'logs_path', create=True),
                path.basename(collection.logs_path) if path.basename(logs_path) == ''
                else path.basename(logs_path)
            )

            # Check if the file is a logs file
            stop(
                path.splitext(logs_path)[1].lower() != '.log', 'TypeError',
                "'{}' is not a log file (with a .log extension).", path.basename(logs_path)
            )

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

        # Check if message is a string
        stop(
            type(message) != str, 'TypeError',
            "'message' must be a string ({} given).", type(message)
        )

        # Check if words are strings and message formatting
        if len(words) > 0:
            for word in words:
                stop(
                    type(word) != str, 'TypeError',
                    "'words' must all be strings ({} given).", type(word)
                )
            message = message.format(*words)

        # Check if level is valid
        stop(
            type(level) != str, 'TypeError',
            "'level' must be a string ({} given).", type(level)
        )
        level = level.lower()
        stop(
            level not in ('info', 'warning'), 'ValueError',
            "'level' must either be 'info' or 'warning' ({} given).", level
        )

        # Message logging
        if level == 'info':
            info(message)
        if level == 'warning':
            warning(message)

        # Message display
        stop(
            type(display) != bool, 'TypeError',
            "'display' must be a boolean ({} given).", type(display)
        )
        if display:
            print(message)

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
