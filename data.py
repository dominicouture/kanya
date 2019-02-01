# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" data.py: Defines the Data and embeded Star classes, which contains all the information from
    CSV file or a Python dictionary, list, tuple or np.ndarray that can then be used as an input
    for a Group object and all related methods to check and convert this data.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

import numpy as np
from astropy import units as un
from os import path, getcwd, chdir
from csv import reader, Sniffer
from init import Config, info
from tools import *

class Data(list):
    """ Contains the data imported from a CSV file or a Python dictionary and related methods.
        Data is converted into a Quantity object and units are converted to default units.
    """

    # Position and velocity labels per representation
    # ??? Move to init.py ???
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

    def __init__(self, series, representation, data):
        """ Initizalizes a Data object from a CSV file or a Python dictionary, list, tuple or
            np.ndarray. The first row must be a header and the second row may be a units row.
            If no units row is speficied, default units are used.
        """
        # Series name
        self.series = series
        # Check representation parameter
        # ??? Move to init.py ???
        if representation is None:
            raise NameError("Required argument 'representation' is missing.")
        if type(representation) != str:
            Config.stop(Config, "'representation' must be a string ({} given).", 'TypeError',
                type(representation))
        self.representation = representation.lower()
        if self.representation not in self.representation_labels:
            Config.stop(Config, "'{}' is not a valid representation value.", 'ValueError',
                self.representation)

        # Initialize from a CSV file
        if type(data) == str:
            self.initialize_from_CSV(data)
        # Initialize from data
        elif type(data) in (dict, list, tuple, np.ndarray):
            self.initialize_from_data(data)
        else:
            Config.stop(Config, "'{}' parameter must be a path to CSV file, or a Python dictionary,"
                "list, tuple or np.ndarray. ({} given).", 'TypeError', type(data))

        # Data configuration
        self.configure_data()

    def initialize_from_data(self, data):
        """ Initializes a Data object from a Python dictionary, list, tuple or np.ndarray (2D). If
            a dictionary is used, only the value with a key that matches 'self.series' is used and
            its type must be one of the other 3 possible types. If a list, tuple or np.ndarray is
            used, all data is imported.
        """
        self.from_data = True
        self.from_CSV = False
        # Data import and group filtering
        if type(data) == dict:
            if self.series in data.keys():
                if type(data[self.series]) in (list, tuple, np.ndarray):
                    self.data = np.array(data[self.series], dtype=object)
                else:
                    Config.stop(Config, "Data '{}' in the Python dictionary must be a list, tuple"
                        "or np.ndarray. ('{}' given).", 'TypeError',
                        self.series, type(data[self.series]))
            else:
                Config.stop(Config, "Group '{}' is not in the data dictionary.",
                    'ValueError', self.series)
        if type(data) in (list, tuple, np.ndarray):
            self.data = np.array(data, dtype=object)

        # Check if self.data is a 2D array
        if self.data.ndim != 2:
            Config.stop(Config, "'data' must be a 2D array ({} dimensions in the given data).",
                'ValueError', self.data.ndim)

        # String conversion
        self.data = np.vectorize(lambda x: str(x))(self.data)

    def initialize_from_CSV(self, data):
        """ Initializes a Data object from the data 'self.series' in a CSV file. If the the CSV
            file doesn't specify a series or group column, all data is used. The file name must
            have a '.csv' extension.
        """
        self.from_CSV = True
        self.from_data = False
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
            Config.stop(Config, "'{}' is not a CSV data file.", 'TypeError', self.data_path)

        # Reading of CSV file.
        data_csv = open(self.data_path, 'r')
        dialect = Sniffer().sniff(data_csv.read(2048))
        data_csv.seek(0)
        # Check if the file has a header
        if not Sniffer().has_header(data_csv.read(2048)):
            Config.stop(Config, "The CSV data file located at '{}' doesn't have a header.",
                'ValueError', self.data_path)
        data_csv.seek(0)
        # Data import
        self.data = np.array([row for row in reader(data_csv, dialect)], dtype=object)
        data_csv.close()

    def configure_data(self):
        """ Configures data in a np.ndarray. Headers, columns, errors units and lines
            identification. If no errors are specified, errors are set to 0.0 and if no units are
            specified, default units are used. The first row must be a header and the second row
            is an can be used as a units row. A star object is then created from each line.
        """

        # Header identification
        self.header = np.array([label.lower() for label in self.data[0]], dtype=object)

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
                if self.from_CSV:
                    Config.stop(Config, "The column '{}' ('{}') is missing "
                        "from the CSV data file located at '{}'.", 'NameError',
                        self.value_labels[label], label, self.data_path)
                if self.from_data:
                    Config.stop(Config, "The column '{}' ('{}') is missing from the data.",
                        'NameError', self.value_labels[label], label)

        # Error labels identification
        self.position_error_labels = tuple('Δ' + label for label in self.position_labels)
        self.velocity_error_labels = tuple('Δ' + label for label in self.velocity_labels)
        self.error_labels = {
            'Δ' + label: name + ' error' for label, name in self.value_labels.items()}
        for label, name in self.error_labels.items():
            self.physical_types[label] = self.physical_types[label[1:]]
        self.labels = {**self.value_labels, **self.error_labels}

        # Units header identification
        self.units_header = np.array([label.lower() for label in self.data[1]], dtype=object)
        print(self.units_header)
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
                    Config.stop(Config, "Unit '{}' used for column '{}' is not valid.",
                        'ValueError', self.units_header[self.columns[label]],
                        self.header(self.columns[label]))
            else:
                self.units[label] = Config.default_units[self.physical_types[label]]
        # Check for units physical types
        for label in self.columns.keys():
            if label in self.labels and self.units[label].physical_type != self.physical_types[label]:
                    Config.stop(Config, "The unit used for the '{}' ('{}') column, '{}', has an "
                        "incorrect physical type ('{}' required, '{}' given)", 'ValueError',
                        self.names[label], label, str(self.units[label]),
                        self.physical_types[label], self.units[label].physical_type)

        # Lines identification and group filtering
        if 'group' in self.columns:
            self.lines = []
            for line in range(1 if self.units_header is None else 2, self.data.shape[0]):
                if self.data[self.columns['group'], line] == self.series:
                    self.lines.append(i)
            if len(self.lines) == 0:
                if self.from_CSV:
                    Config.stop(Config, "No data for the group '{}' in the CSV data file located"
                        "at '{}'.", 'ValueError', self.series, self.data_path)
                if self.from_data:
                    Config.stop(Config, "No information for the group '{}' in the data.",
                        'ValueError', self.series, self.data_path)
        else:
            self.lines = list(
                range(1 if self.units_header is None else 2, self.data.shape[0]))

        # Star objects creation
        i = 1
        for line in self.lines:
            self.append(self.Star(self, i,
                {label: self.data[line, column] for label, column in self.columns.items()}))
            i += 1

    class Star:
        """ Contains the data for an individual star, including name, type, id, position and
            velocity.
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
