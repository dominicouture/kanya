# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __init__.py: Initializes the Traceback package. A class Collection is defined to contain
    all Series objects.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

from series import info

class Collection(list):
    """ Contains all series objects created from data, a model or a database. """

    def __init__(self):
        """ Initializes a collection. """

        self.series = {}

    def add(self, *series):
        """ Adds one or multiple new series to the collection. """

        # Previous entry deletion if it already exists
        for series in series:
            if series.name in self.series.keys():
                info("Existing series '{}' deleted and replaced.".format(series.name))
                del self[self.series[series.name]]

            # If it does, aks user whether the old series must be replaced.
            self.append(series)

        # Series dictionary update
        self.series = {self[i].name: i for i in range(len(self))}

    def remove(self, *series):
        """ Removes one or multiple series from the collection from series.name or an index. """

        pass

    def select(self, *series):
        """ Selects one or multiple series from a collection. If no series name are specified,
            all series are selected. """

        # All series
        if len(series) == 0:
            return tuple(self)

        # Selected series only
        else:
            self.series = {self[i].name: i for i in range(len(self))}

            # Check if all series exists
            for name in filter(lambda name: name not in self.series.keys(), series):
                from series import Series
                Series.stop(Series, True, 'NameError', "Series '{}' does not exist.", name)

            return tuple(self[self.series[name]] for name in series)

    def default(self, name):
        """ Loops over all Series in self and returns a default 'untitled-i' name where i is the
            lowest possible number for an untittled series.
        """

        # Loops over Series in self
        i = 1
        while '{}-{}'.format(name, i) in [series.name for series in self]:
            i += 1

        return '{}-{}'.format(name, i)

    def update(self, *series, **parameters):
        """ Updates the series by modifying its self.config configuration, re-initializing itself
            and deleting existing groups. The groups are also traced back again if they had been
            traced back before the update.
        """

        for series in self.select(*series):
            series.update(**parameters)

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

    def save(self, *series):
        """ Saves one or multiple series to a database binary file. If needed, the existing
            files are overwritten.
        """

        for series in self.select(*series):
            series.save()

    def create(self, *series):
        """ Either load one or multiple series from a database file, or traces back one or multiple
            series from data or a model. If needed, the series is also saved.
        """

        for series in self.select(*series):
            series.create()

# Collection initialization
collection = Collection()
