# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" data.py: Defines the Data class, which contains all the information from CSV file or a Python
    dictionary that can then be used as an input for a Group object and all related methods to
    check and convert this data.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

import numpy as np
from astropy import units as un
from time import strftime
from os import path, getcwd, chdir
from csv import reader, Sniffer
from tools import *
from init import Config, info

class Data:
    """ Contains the data imported from a CSV file or a Python dictionary and related methods.
        Data is converted into a Quantity object and units are converted to default units.
    """

    # Position and velocity labels per representation
    representation_labels = {
        'observables': (('p', 'δ', 'α'), ('rv', 'μδ', 'μα_cos_δ')),
        'spherical':   (('r', 'δ', 'α'), ('rv', 'μδ', 'μα')),
        'cartesian':   (('x', 'y', 'z'), ('u', 'v', 'w'))
    }

    # Label names
    names = {
        'r': 'distance',
        'p': 'paralax',
        'δ': 'declination',
        'α': 'right ascension',
        'rv': 'radial velocity',
        'μδ': 'declination proper motion',
        'μα': 'right ascension proper motion',
        'μα_cos_δ': 'right ascension proper motion * cos(declination)',
        'x': 'x position',
        'y': 'y position',
        'z': 'z position',
        'u': 'u velocity',
        'v': 'v velocity',
        'w': 'w velocity'
    }

    # Label physical types
    physical_types = {
        'r': 'length',
        'p': 'angle',
        'δ': 'angle',
        'α': 'angle',
        'rv': 'speed',
        'μδ': 'angular speed',
        'μα': 'angular speed',
        'μα_cos_δ': 'angular speed',
        'x': 'length',
        'y': 'length',
        'z': 'length',
        'u': 'speed',
        'v': 'speed',
        'w': 'speed'
    }

    # Label permutations
    permutations = {
        '-': '',
        '_': '',
        ' ': '',
        'error': 'Δ',
        'err': 'Δ',
        'series': 'group',
        'kinematic_group': 'group',
        'association': 'group',
        'spt': 'type',
        'spectraltype': 'type',
        'fixed': '',
        'distance': 'r',
        'plx': 'p',
        'paralax': 'p',
        'declination': 'δ',
        'dec': 'δ',
        'rightascension': 'α',
        'ra': 'α',
        'radialvelocity': 'rv',
        'propermotion': 'μ',
        'pm': 'μ'
    }

    def __init__(self, data, series):
        """ Initizalizes a Data object from a CSV file or a Python dictionary.
        """
        # Series name
        self.series = series
        # !!! Add check for type and value of self.representation here !!!
        self.representation = 'observables'
        # Initialize from a CSV file
        if type(data) == str:
            self.initialize_from_CSV(data)
        elif type(data) == dict:
            self.initialize_from_dictionary(data)
        else:
            Config.stop(Config, "'{}' parameter must be a path to CSV file or a Python dictionary"
                "({} given).", 'TypeError', type(data))

    def initialize_from_dictionary(self, data):
        """ Initializes a Data object from the data 'self.series' in a Python dictionary. If the
            dictionary is not split into series or group, all data is used.
        """
        # !!! Add import from a Python dictionary here !!!
        # !!! Selects only the data with the correct series or use all data !!!
        # !!! Add check for the correct units physical types based on the representation !!!
        pass


    def initialize_from_CSV(self, data):
        """ Initializes a Data object from the data 'self.series' in a CSV file. If the the CSV
            file doesn't specify a series or group column, all data is used.
        """
        # CSV file absolute path
        working_dir = getcwd()
        chdir(path.abspath(path.join(path.dirname(path.realpath(__file__)), '..')))
        self.data_path = path.abspath(data)
        chdir(working_dir)
        # Check if the path exists
        if not path.exists(self.data_path):
            Config.stop(Config, "'{}' does not exist.", 'NameError', self.data_path)
        # Check if the path links to a CSV file.
        if path.splitext(self.data_path)[1] != '.csv':
            Config.stop(Config, "'{}' is not a CSV file.", 'TypeError', self.data_path)

        # Reading of CSV file.
        data_csv = open(self.data_path, 'r')
        dialect = Sniffer().sniff(data_csv.read(2048))
        data_csv.seek(0)
        # Check if the file has a header
        if not Sniffer().has_header(data_csv.read(2048)):
            Config.stop(Config, "The CSV data file located at '{}' doesn't have a header.",
                'ValueError', self.data_path)
        data_csv.seek(0)
        self.data_array = np.array([row for row in reader(data_csv, dialect)], dtype=object)
        data_csv.close()

        # Header identification
        self.header = np.array([label.lower() for label in self.data_array[0]], dtype=object)

        # Columns identification
        self.columns = {}
        for i in range(len(self.header)):
            label = self.header[i]
            for j, k in self.permutations.items():
                label = label.replace(j, k)
            if label[-1] in ('e', 'Δ') and label not in ('name', 'type'):
                label = 'Δ' + label[:-1]
            if label[0] == 'e':
                label = 'Δ' + label[1:]
            if self.representation == 'observables' and label[-2:] == 'μα':
                label = label + '_cos_δ'
            self.columns[label] = i

        # Value labels identification
        self.position_labels, self.velocity_labels = self.representation_labels[
            self.representation.lower()]
        self.value_labels = {
            label: self.names[label] for label in self.position_labels + self.velocity_labels}
        for label in self.value_labels.keys():
            if label not in self.columns.keys():
                Config.stop(Config, "The column '{}' ('{}') is missing from the CSV data file "
                    "located at '{}'.", 'NameError', self.value_labels[label], label, self.data_path)

        # Error labels identification
        self.position_error_labels = tuple('Δ' + label for label in self.position_labels)
        self.velocity_error_labels = tuple('Δ' + label for label in self.velocity_labels)
        self.error_labels = {'Δ' + label: name + ' error' for label, name in self.value_labels.items()}
        for label, name in self.error_labels.items():
            self.physical_types[label] = self.physical_types[label[1:]]
        self.labels = {**self.value_labels, **self.error_labels}

        # Units header identification
        self.units_header = np.array([label.lower() for label in self.data_array[1]], dtype=object)
        # Check if there is a units header below the labels header
        for label in self.value_labels.keys():
            if self.units_header[self.columns[label]].replace('.', '').replace(',', '').isdigit():
                self.units_header = None
                break

        # Units identification
        self.units = {}
        for label in self.labels.keys():
            if label in self.columns and self.units_header is not None and \
                    self.units_header[self.columns[label]] != '':
                try:
                    self.units[label] = un.Unit(self.units_header[self.columns[label]])
                except ValueError:
                    Config.stop(Config, "Unit '{}' used for column '{}' is not a invalid.",
                        'ValueError',
                        self.units_header[self.columns[label]],
                        self.header(self.columns[label]))
            else:
                self.units[label] = Config.default_units[self.physical_types[label]]
        # !!! Add check for the correct units physical types based on the representation !!!

        # Lines identification and group filtering
        if 'group' in self.columns:
            self.lines = []
            for line in range(1 if self.units_header is None else 2, self.data_array.shape[0]):
                if self.data_array[self.columns['group'], line] == self.series:
                    self.lines.append(i)
        else:
            self.lines = list(
                range(1 if self.units_header is None else 2, self.data_array.shape[0]))

        # Star objects creation
        i = 1
        self.stars = []
        for line in self.lines:
            self.stars.append(self.Star(self, i,
                {label: self.data_array[line, column] for label, column in self.columns.items()}))
            i += 1

    class Star:
        """ Contains the data for an individual star.
        """

        def __init__(self, data, i, line):
            """ Initializes a Star object from a 'data' object, a number 'i' and a line with a
                dictionary containing the information found on one line of a dictionary or CSV
                file. Default values are given if
            """
            # Name column
            self.name = line['name'] if 'name' in data.columns and line['name'] != '' \
                else 'Star_{}'.format(str(i))

            # Type column
            self.type = line['type'] if 'type' in data.columns and line['type'] != '' else None

            # ID column
            self.id = line['id'] if 'id' in data.columns and line['id'] != '' else str(i)

            # Default errors if they're not present
            for error_label in data.error_labels:
                if error_label not in line:
                    line[error_label] = ''

            # Float conversion
            for label in data.labels:
                if line[label] == '':
                    line[label] = 0.0
                else:
                    try:
                        line[label] = float(line[label].replace(',', '.'))
                    except ValueError:
                        Config.stop(Config, "'{}' value could not be converted to float in '{}'"
                            "column.", 'ValueError', line[label], label)

            # Position columns
            self.position = (
                tuple(line[label] for label in data.position_labels),
                tuple(data.units[label] for label in data.position_labels),
                tuple(line[label] for label in data.position_error_labels)
            )

            # Velocity columns
            self.velocity = (
                tuple(line[label] for label in data.velocity_labels),
                tuple(data.units[label] for label in data.velocity_labels),
                tuple(line[label] for label in data.velocity_error_labels)
            )

            print(self.name, ':', self.type, self.id, self.position, self.velocity)
