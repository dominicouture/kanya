# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __init__.py: Initializes the Traceback package. A class Collection is defined to contain
    all Series objects.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

from series import info
from tools import stop

class Collection(list):
    """ Contains all series objects created from data, a model or a database. Functions to add,
        remove, update, copy, load, traceback, save, create and select series are defined, as
        well as a function to generate default series name from the names in self.series.
    """

    def __init__(self):
        """ Initializes a collection by updating its self.series dictionary. """

        self.series = {self[i].name: i for i in range(len(self))}

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

    def load(self, *series):
        """ Loads one or multiple series from the database binary file. """

        for series in self.select(*series):
            series.load()

    def traceback(self, *series):
        """ Traces back all series in self if no series name is given or selected series given
            given in argument by name.
        """

        for series in self.select(*series):
            series.traceback()

    def save(self, *series, forced=False):
        """ Saves one or multiple series to a database binary file. If forced, the existing
            files are overwritten.
        """

        for series in self.select(*series):
            series.save(forced=forced)

    def create(self, *series):
        """ Either load one or multiple series from a database file, or traces back one or multiple
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
                Series.stop(Series, name not in self.series.keys(),
                    'NameError', "Series '{}' does not exist.", name)

            return [self[self.series[name]] for name in series]

    def default(self, name):
        """ Loops over all Series in self and returns a default 'name-i' name where i is the
            lowest possible number for an series titled 'name'.
        """

        # Identify whether 'name' ends with a digit
        i = name.split('-')[-1]
        if i.isdigit():
            name = name[:-(len(i) + 1)]

        # Loops over Series in self
        i = 1
        while '{}-{}'.format(name, i) in [series.name for series in self]:
            i += 1

        return '{}-{}'.format(name, i)

# Collection initialization
collection = Collection()
