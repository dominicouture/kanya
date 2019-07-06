# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" collection.py: Defines the Collection class and initializes global parameters. Global
    functions, such directory, output, log and stop are also defined.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

from os import path, makedirs, getcwd, chdir

class Collection(list):
    """ Contains all series objects created from data, a model or a file. Functions to add,
        remove, update, copy, load, traceback, save, create and select series are defined, as
        well as a function to generate default series name from the names in self.series.
    """

    def __init__(self):
        """ Initializes a collection by creating a self.series dictionary, as well as default
            output directories and log path. A base directory is also created and represents the
            directory where the Traceback package is located, from which other relative directory
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
        stop(output_dir is not None and type(output_dir) != str, 'TypeError',
            "'output_dir' must be a string ({} given).", type(output_dir))

        # self.output_dir parameter
        self.output_dir = directory(
            self.base_dir, 'Output' if output_dir is None else output_dir, 'output_dir')

    def set_logs(self, logs_path=None):
        """ Sets the global log path. The log file and directory are only created if a message
            is actually logged. The file name must end with an '.log' extension. By default, a
            'Logs' directory is created in the output directory.
        """

        from time import strftime

        # self.logs_path parameter
        self.logs_path = path.join(self.output_dir, 'Logs') + '/' if logs_path is None \
            else logs_path

        # Check if logs_path parameter is a string, which must be done before the directory call
        stop(type(self.logs_path) != str, 'TypeError', "'logs_path' must be a string ({} given).",
            type(self.logs_path))

        # logs_path redefined as the absolute path
        self.logs_path = path.join(
            directory(self.base_dir, path.dirname(self.logs_path), 'logs_path'),
            'Traceback_{}.log'.format(strftime('%Y-%m-%d_%H-%M-%S')) \
                if path.basename(self.logs_path) == '' else path.basename(self.logs_path))
        self.logs_configured = False

        # Check if the file is a logs file
        stop(path.splitext(self.logs_path)[1].lower() != '.log', 'TypeError',
            "'{}' is not a log file (with a .log extension).", path.basename(self.logs_path))

    def add(self, *series, forced=False, default=False, cancel=False):
        """ Adds one or multiple new series to the collection. If forced is True, any existing
            series with the same name is overwritten, otherwise user input is required to proceed
            (overwrite, do nothing or default). If default is True, then instead of asking for an
            input, a default name is used and the series added to the collection.
        """

        # Series addition and replacement, if needed
        for series in series:
            series.add(forced, default, cancel)

    def remove(self, *series):
        """ Removes one or multiple series from the collection from series.name. """

        # Series deletion from collection
        for series in self.select(*series):
            series.remove()

    def reset(self, *series):
        """ Resets one or multiple series from the collection from series.name. """

        # Series deletion from collection
        for series in self.select(*series):
            series.reset()

    def update(self, *series, **parameters):
        """ Updates the series by modifying its self.config configuration, re-initializing itself
            and deleting existing groups. The groups are also traced back again if they had been
            traced back before the update.
        """

        for series in self.select(*series):
            series.update(**parameters)

    def copy(self, *series, **parameters):
        """ Copies the series under a new name. If 'parameters' are provided, this new Series
            object is updated as well with those new parameters. If no new 'name' is provided
            as a parameter, a default name is used instead.
        """

        for series in self.select(*series):
            series.copy(**parameters)

    def load(self, *series, forced=False):
        """ Loads one or multiple series from the binary file. If forced, existing groups are
            overwritten.
        """

        for series in self.select(*series):
            series.load(forced=forced)

    def traceback(self, *series, forced=False):
        """ Traces back all series in self if no series name is given or selected series given
            given in argument by name. If forced, existing groups are overwritten.
        """

        for series in self.select(*series):
            series.traceback(forced=forced)

    def save(self, *series, forced=False):
        """ Saves one or multiple series to a binary file. If forced, existing files are
            overwritten.
        """

        for series in self.select(*series):
            series.save(forced=forced)

    def create(self, *series):
        """ Either load one or multiple series from a file, or traces back one or multiple
            series from data or a model. If needed, the series is also saved.
        """

        for series in self.select(*series):
            series.create()

    def select(self, *series):
        """ Selects one or multiple series from a collection. If no series name are specified,
            all series are selected. """

        # All series
        if len(series) == 0:
            return [series for series in self]

        # Selected series only
        else:

            # Check if all series exists
            from series import Series
            for name in series:
                stop(name not in self.series.keys(), 'NameError',
                    "Series '{}' is not in the collection.", name)

            return [self[self.series[name]] for name in series]

    def default_name(self, name=None):
        """ Loops over all Series in self and returns a default 'name-i' name where i is the
            lowest possible number for an series titled 'name'.
        """

        # Initialization
        name = name if name is not None else 'untitled'

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
        returns the absolute directory path. The directory can either be relative path to the
        'base' or an absolute path.
    """

    # Check the type of directory
    stop(type(directory) is not str, 'TypeError', "'{}' must be a string ({} given).",
        name, type(directory))

    # Output directory formatting
    working_dir = getcwd()
    chdir(path.abspath(base))
    directory = path.abspath(directory)
    chdir(working_dir)

    # Check if the directory exists
    if check:
        stop(not path.exists(directory), 'NameError',
            "No existing directory located at '{}'.", directory)

    # Directory creation
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

    # Absolute directory, check and creation, if needed
    return directory(collection.base_dir, output_dir, 'output_dir', check=check, create=create)

def log(message, *words, logs_path=None, level='info', display=False):
    """ Logs the 'message' with the appropriate 'level', if logs have been configured. If logs have
        not been configured, logs are configured by checking if the 'logs_path' already exists,
        creating it if needed and configuring the logs file.  If blank (i.e. ''), logs files will
        created in the base directory with a default name. The file name must end with an '.log'
        extension. Futhermore, if 'display' is True, the message is also printed on screen.
    """

    from time import strftime
    from logging import basicConfig, root, info, warning, INFO

    # logs_path parameter
    logs_path = collection.logs_path if logs_path is None else logs_path

    # Checks and configuration skipped if logs have been configured already
    if not collection.logs_configured or collection.logs_path != logs_path:

        # Check if logs_path parameter is a string, which must be done before the directory call
        stop(type(logs_path) != str, 'TypeError', "'logs_path' must be a string ({} given).",
            type(logs_path))

        # logs_path redefined as the absolute path and directory creation
        logs_path = path.join(
            directory(collection.base_dir, path.dirname(logs_path), 'logs_path', create=True),
            path.basename(collection.logs_path) if path.basename(logs_path) == '' \
                else path.basename(logs_path))

        # Check if the file is a logs file
        stop(path.splitext(logs_path)[1].lower() != '.log', 'TypeError',
            "'{}' is not a log file (with a .log extension).", path.basename(logs_path))

        # Logs configuration, if logs_path matches collection.logs_path
        if logs_path == collection.logs_path:
            if not collection.logs_configured:
                for handler in root.handlers[:]:
                    root.removeHandler(handler)
                collection.logs_configured = True
                basicConfig(filename=collection.logs_path, format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

        # Logs configuration, if logs_path doesn't match collection.logs_path
        else:
            if collection.logs_configured:
                for handler in root.handlers[:]:
                    root.removeHandler(handler)
                collection.logs_configured = False
            basicConfig(filename=logs_path, format='%(asctime)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

    # Message logging
    if len(words) > 0:
        message = message.format(*words)
    if level=='info':
        info(message)
    if level=='warning':
        warning(message)
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

    # Exception information extraction
    from sys import exc_info, exit
    exc_type, exc_value = exc_info()[:2]

    # If no exception is being handled, one is raised if 'conidition' is True
    if exc_type is None and exc_value is None:
        try:
            if condition:
                exec("raise {}".format(error))
        except:
            stop(True, error, message, *words, name=name, extra=extra+1)

    # If an exception is being handled, its traceback is formatted and execution is terminated
    else:
        # Traceback message creation
        if len(words) > 0:
            message = message.format(*words)
        tb_message = "{} in '{}': {}".format(error, name, message) if 'name' is not None \
            else "{}: {}".format(error, message)

        # Traceback stack formatting
        from traceback import format_stack
        tb_stack = ''.join(
            ['An exception has been raised: \n'] + format_stack()[:-extra] + [tb_message])

        # Exception logging only if logs have been configured and code termination
        if 'collection' in globals().keys() and collection.logs_configured:
            log(tb_stack, level='warning')
        print(tb_stack)
        exit()

# Collection initialization
collection = Collection()
