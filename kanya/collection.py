# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" collection.py: Defines the Collection class and initializes global parameters. Global
    functions used by Series class and more, such as directory, output, log and stop are also
    defined.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

from os import path

class Collection(list):
    """ Contains all series objects created from data, a model or a file. Functions to add,
        remove, update, copy, load, traceback, save, create and select series are defined, as
        well as a function to generate default series names from the existing series names in
        self.series.
    """

    def __init__(self):
        """ Initializes a collection by creating a self.series dictionary, as well as default
            output directories and log path. A base directory is also created and represents the
            directory where the kanya package is located, from which other relative directory
            paths originate. By default, an 'Output' directory is created and the logs are saved
            in a 'Logs' directory inside the 'Output' directory.
        """

        # Initialize self.series dictionary
        self.initialize_series()

        # Initialize default values for the base and output directories, and logs path
        self.base_dir = path.abspath(path.join(path.dirname(path.realpath(__file__)), '..'))
        self.set_output()
        self.set_logs()

    def initialize_series(self):
        """ Creates a self.series dictionary of the index of Series object using their Series.name
            value as dictionary keys.
        """

        self.series = {self[i].name: i for i in range(len(self))}

    def set_output(self, output_dir=None):
        """ Sets the collection output directory ('output_dir'). The directory is only created if
            something is actually saved in it. By default, the output directory is set to 'Output'.
        """

        # Check if output_dir parameter is a string, which must be done before the directory call
        stop(
            output_dir is not None and type(output_dir) != str, 'TypeError',
            "'output_dir' must be a string ({} given).", type(output_dir))

        # self.output_dir parameter
        self.output_dir = directory(
            self.base_dir, '../Output' if output_dir is None else output_dir, 'output_dir')

    def set_logs(self, logs_path=None):
        """ Sets the global log path. The log file and directory are only created if a message
            is actually logged. The file name must end with an '.log' extension. By default, a
            'Logs' directory is created in the output directory.
        """

        from time import strftime

        # self.logs_path parameter
        self.logs_path = (
            path.join(self.output_dir, 'Logs') + '/' if logs_path is None else logs_path)

        # Check if logs_path parameter is a string, which must be done before the directory call
        stop(
            type(self.logs_path) != str, 'TypeError',
            "'logs_path' must be a string ({} given).", type(self.logs_path))

        # logs_path redefined as the absolute path
        self.logs_path = path.join(
            directory(self.base_dir, path.dirname(self.logs_path), 'logs_path'),
            'kanya_{}.log'.format(strftime('%Y-%m-%d_%H-%M-%S'))
            if path.basename(self.logs_path) == '' else path.basename(self.logs_path))
        self.logs_configured = False

        # Check if the file is a logs file
        stop(
            path.splitext(self.logs_path)[1].lower() != '.log', 'TypeError',
            "'{}' is not a log file (with a .log extension).", path.basename(self.logs_path))

    def new(
            self, parent=None, path=None, args=False, forced=False, default=False,
            cancel=False, logging=True, **parameters):
        """ Creates a new Series in the collection. Arguments are the same as those of
            Series.__init__.
        """

        from .series import Series
        Series(parent, path, args, forced, default, cancel, logging, **parameters)

    def add(self, *series, forced=False, default=False, cancel=False, logging=True):
        """ Adds one or multiple new series to the collection. If forced is True, any existing
            series with the same name is overwritten, otherwise user input is required to proceed
            (overwrite, do nothing or default). If default is True, then instead of asking for an
            input, a default name is used and the series added to the collection.
        """

        # Series addition and replacement, if needed
        for series in series:
            stop(
                str(type(series)) != "<class 'series.Series'>", 'TypeError',
                "'series' must be Series object ({} given).", type(series))
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

    def update(self, *series, parent=None, path=None, args=False,
            logging=True, traceback=True, **parameters):
        """ Updates the series by modifying its self.config configuration, re-initializing itself
            and deleting existing groups. The groups are also traced back again if they had been
            traced back before the update.
        """

        for series in self.select(*series):
            series.update(parent, path, args, logging, traceback, **parameters)

    def copy(self, *series, **parameters):
        """ Copies the series under a new name. If 'parameters' are provided, this new Series
            object is updated as well with those new parameters. If no new 'name' is provided
            as a parameter, a default name is used instead.
        """

        for series in self.select(*series):
            series.copy(parent, path, args, logging, traceback, **parameters)

    def load(self, *series, file_path=None, forced=False):
        """ Loads one or multiple series from the binary file. If forced, existing groups are
            overwritten.
        """

        for series in self.select(*series):
            series.load(file_path, forced)

    def traceback(self, *series, forced=False, mode=None):
        """ Traces back all series in self if no series name is given or selected series given
            given in argument by name. If forced, existing groups are overwritten.
        """

        for series in self.select(*series):
            series.traceback(forced, mode)

    def save(self, *series, file_path=None, forced=False, default=False, cancel=False):
        """ Saves one or multiple series to a binary file. If forced, existing files are
            overwritten.
        """

        for series in self.select(*series):
            series.save(file_path, forced, default, cancel)

    def create(self, *series, forced=False, default=False, cancel=False):
        """ Either load one or multiple series from a file, or traces back one or multiple
            series from data or a model. If needed, the series is also saved.
        """

        for series in self.select(*series):
            series.create(forced, default, cancel)

    def select(self, *series):
        """ Selects one or multiple series from a collection. If no series name are specified,
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
                        type(name) != str, 'TypeError', "'series' must be a string ({} given).",
                        type(name))
                    stop(
                        name not in self.series.keys(), 'NameError',
                        "Series '{}' is not in the collection.", name)


            return [self[self.series[name]] for name in series]

    def default_name(self, name=None):
        """ Loops over all Series in self and returns a default 'name-i' name where i is the
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

def directory(base, directory, name, check=False, create=False):
    """ Checks for type errors, checks if 'directory' already exists, creates it if needed and
        returns the absolute directory path. The directory can either be relative path to 'base'
        or an absolute path.
    """

    from os import makedirs, getcwd, chdir

    # Check the type of name, base and directory
    stop(type(name) != str, 'TypeError', "'name' must be a string ({} given).", type(name))
    stop(
        type(base) != str, 'TypeError',
        "The base of '{}' must be a string ({} given).", name, type(base))
    stop(
        type(directory) != str, 'TypeError',
        "The base '{}' must be a string ({} given).", name, type(directory))

    # Output directory formatting
    working_dir = getcwd()
    chdir(path.abspath(base))
    directory = path.abspath(directory)
    chdir(working_dir)

    # Check if the directory exists
    stop(
        type(check) != bool, 'TypeError',
        "'check' must be a boolean ({} given).", type(check))
    if check:
        stop(
            not path.exists(directory), 'NameError',
            "No existing directory located at '{}'.", directory)

    # Directory creation, if needed
    stop(
        type(create) != bool, 'TypeError',
        "'create' must be a boolean ({} given).", type(create))
    if create and not path.exists(directory):
        makedirs(directory)

    return directory

def output(output_dir=None, check=False, create=True):
    """ Returns the absolute path of the output directory and creates it if needed. 'output_dir'
        is either relative to collection.base_dir or absolute (str). If blank (i.e. ''), output
        files will be created in the base directory. By default, if 'output_dir' is None, an
        'Output' directory is created.
    """

    # Collection output directory if None is provided
    output_dir = collection.output_dir if output_dir is None else output_dir

    # Check if output_dir is a string
    stop(
        type(output_dir) != str, 'TypeError',
        "'output_dir' must be a string ({} given).", type(output_dir))

    # Absolute directory, check and creation, if needed
    return directory(collection.base_dir, output_dir, 'output_dir', check=check, create=create)

def log(message, *words, logs_path=None, level='info', display=False, logging=True):
    """ Logs the 'message' with the appropriate 'level', if logs have been configured. If logs have
        not been configured, logs are configured by checking if the 'logs_path' already exists,
        creating it if needed and configuring the logs file.  If blank (i.e. ''), logs files will
        created in the base directory with a default name. The file name must end with an '.log'
        extension. Futhermore, if 'display' is True, the message is also printed onscreen.
    """

    from time import strftime
    from logging import basicConfig, root, info, warning, INFO

    # Check if logging is True
    stop(
        type(logging) != bool, 'TypeError',
        "'logging' must be a boolean ({} given).", type(logging))
    if logging:

        # logs_path parameter
        logs_path = collection.logs_path if logs_path is None else logs_path

        # Checks and configuration skipped if logs have been configured already
        if not collection.logs_configured or collection.logs_path != logs_path:

            # Check if logs_path parameter is a string, which must be done before the directory call
            stop(
                type(logs_path) != str, 'TypeError',
                "'logs_path' must be a string ({} given).", type(logs_path))

            # logs_path redefined as the absolute path and directory creation
            logs_path = path.join(
                directory(collection.base_dir, path.dirname(logs_path), 'logs_path', create=True),
                path.basename(collection.logs_path) if path.basename(logs_path) == ''
                else path.basename(logs_path))

            # Check if the file is a logs file
            stop(
                path.splitext(logs_path)[1].lower() != '.log', 'TypeError',
                "'{}' is not a log file (with a .log extension).", path.basename(logs_path))

            # Logs configuration, if logs_path matches collection.logs_path
            if logs_path == collection.logs_path:
                if not collection.logs_configured:
                    for handler in root.handlers[:]:
                        root.removeHandler(handler)
                    collection.logs_configured = True
                    basicConfig(
                        filename=collection.logs_path, format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

            # Logs configuration, if logs_path doesn't match collection.logs_path
            else:
                if collection.logs_configured:
                    for handler in root.handlers[:]:
                        root.removeHandler(handler)
                    collection.logs_configured = False
                basicConfig(
                    filename=logs_path, format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

        # Check if message is a string
        stop(
            type(message) != str, 'TypeError',
            "'message' must be a string ({} given).",
            type(message), name='marmalade')

        # Check if words are strings and message formatting
        if len(words) > 0:
            for word in words:
                stop(
                    type(word) != str, 'TypeError',
                    "'words' must all be strings ({} given).", type(word))
            message = message.format(*words)

        # Check if level is valid
        stop(
            type(level) != str, 'TypeError',
            "'level' must be a string ({} given).", type(level))
        level = level.lower()
        stop(
            level not in ('info', 'warning'), 'ValueError',
            "'level' must either be 'info' or 'warning' ({} given).", level)

        # Message logging
        if level == 'info':
            info(message)
        if level == 'warning':
            warning(message)

        # Message display
        stop(
            type(display) != bool, 'TypeError',
            "'display' must be a boolean ({} given).", type(display))
        if display:
            print(message)

def stop(condition, error, message, *words, name=None, extra=1):
    """ Raises an exception if 'condition' is True, logs it into the log file if logs have
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
                exec("raise {}".format(error))
            except:
                stop(True, error, message, *words, name=name, extra=extra + 1)

        # If an exception is being handled, its traceback is formatted and execution is terminated
        else:

            # Traceback message creation
            if len(words) > 0:
                message = message.format(*words)
            tb_message = (
                "{} in '{}': {}".format(error, name, message) if name is not None
                else "{}: {}".format(error, message))

            # Traceback stack formatting
            from traceback import format_stack
            tb_stack = ''.join(
                ['An exception has been raised: \n'] + format_stack()[:-extra] + [tb_message])

            # Exception logging, if logs have been configured, and code termination
            if 'collection' in globals().keys() and collection.logs_configured:
                log(tb_stack, level='warning')
            print(tb_stack)
            exit()

# Collection initialization
collection = Collection()
