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
from init import Config, System
from series import info, Series
from tools import *

class Data(list):
    """ Contains the data imported from a CSV file or a Python dictionary and related methods.
        Data is converted into a Quantity object and units are converted to default units.
    """

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

    def __init__(self, series):
        """ Initizalizes a Data object from a CSV file or a Python dictionary, list, tuple or
            np.ndarray. The first row must be a header and the second row may be a units row.
            Units specified by series.config.data.units will override the units row. If no units,
            are given default units are used, based on the value of series.config. data.system.
        """

        # Initialization
        self.series = series

        # Initialize Data from a CSV file
        if type(self.series.config.data.values) == str:
            self.initialize_from_CSV()
        # Initialize Data from Python object
        elif type(self.series.config.data.values) in (dict, list, tuple, np.ndarray):
            self.initialize_from_data()
        else:
            self.series.stop(True, 'TypeError', "'data.values' component must be a path to "
                "a CSV file or a Python dictionary, list, tuple or np.ndarray. ({} given).",
                type(self.series.config.data.values))

        # Data configuration
        self.configure_data()

    def initialize_from_data(self):
        """ Initializes a Data object from a Python dictionary, list, tuple or np.ndarray (2D). If
            a dictionary is used, only the value with a key that matches 'series.name' is used and
            its type must be one of the other 3 possible types. If a list, tuple or np.ndarray is
            used, all data is imported.
        """

        # Data origin
        self.from_data = True
        self.from_CSV = False

        # Data import and group filtering
        if type(self.series.config.data.values) == dict:
            if self.series.name in self.series.config.data.values.keys():
                if type(self.series.config.data.values[self.series.name]) in (list, tuple, np.ndarray):
                    self.data = np.array(self.series.config.data.values[self.series.name], dtype=object)
                else:
                    self.series.stop(True, 'TypeError', "Data '{}' in the Python dictionary "
                        "must be a list, tuple or np.ndarray. ('{}' given).",
                        self.series.name, type(self.series.config.data.values[self.series.name]))
            else:
                self.series.stop(True, 'ValueError' "Group '{}' is not in the data dictionary.",
                    self.series.name)
        if type(self.series.config.data.values) in (list, tuple, np.ndarray):
            self.data = np.array(self.series.config.data.values, dtype=object)

        # Check if self.data has been converted a 2D array
        self.series.stop(self.data.ndim != 2, 'ValueError',
            "'data' parameter must represent a 2D array ({} dimensions in the given data).",
            self.data.ndim)

        # String conversion
        try:
            self.data = np.vectorize(lambda value: str(value))(self.data)
        except ValueError:
            self.series.stop(True, 'ValueError', "Data could not be converted to string.")

    def initialize_from_CSV(self):
        """ Initializes a Data object from the data 'series.name' in a CSV file. If the the CSV
            file doesn't specify a series or group column, all data is used. The file name must
            have a '.csv' extension.
        """

        # Data origin
        self.from_CSV = True
        self.from_data = False

        # CSV file absolute path
        working_dir = getcwd()
        chdir(path.abspath(path.join(path.dirname(path.realpath(__file__)), '..')))
        self.data_path = path.abspath(self.series.config.data.values)
        chdir(working_dir)
        # Check if the path exists
        self.series.stop(not path.exists(self.data_path), 'NameError',
            "'{}' does not exist.", self.data_path)
        # Check if the path links to a CSV file.
        self.series.stop(path.splitext(self.data_path)[1] != '.csv', 'TypeError',
            "'{}' is not a CSV data file.", self.data_path)

        # Reading of CSV file.
        data_csv = open(self.data_path, 'r')
        dialect = Sniffer().sniff(data_csv.read(2048))
        data_csv.seek(0)
        # Check if the file has a header
        self.series.stop(not Sniffer().has_header(data_csv.read(2048)), 'ValueError',
            "The CSV data file located at '{}' doesn't have a header.", self.data_path)
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
            if self.series.config.data.system.name == 'observables' and label[-2:] == 'μα':
                label = label + 'cosδ'
            self.columns[label] = i

        # Value labels identification
        self.position_labels = [variable.label for variable in self.series.config.data.system.position]
        self.velocity_labels = [variable.label for variable in self.series.config.data.system.velocity]
        self.value_labels = {label: self.series.config.data.system.variables[label].name \
            for label in self.position_labels + self.velocity_labels}
        for label in self.value_labels.keys():
            if label not in self.columns.keys():
                self.series.stop(self.from_CSV, 'NameError',
                    "The column '{}' ('{}') is missing from the CSV data file located at '{}'.",
                    self.value_labels[label], label, self.data_path)
                self.series.stop(self.from_data, 'NameError',
                    "The column '{}' ('{}') is missing from the data.",
                    self.value_labels[label], label)

        # Error labels identification
        self.position_error_labels = tuple('Δ' + label for label in self.position_labels)
        self.velocity_error_labels = tuple('Δ' + label for label in self.velocity_labels)
        self.error_labels = {
            'Δ' + label: name + ' error' for label, name in self.value_labels.items()}
        self.labels = {**self.value_labels, **self.error_labels}

        # Units header identification
        self.units_header = np.array([label.lower() for label in self.data[1]], dtype=object)
        # Check if there is a units header below the labels header
        for label in self.value_labels.keys():
            if self.units_header[self.columns[label]].replace('.', '').replace(',', '').isdigit():
                self.units_header = None
                break

        # Units identification
        self.units = {}
        for label in self.labels.keys():

            # !!! Units specified in data.units !!!
            # Traduire le numéro de colonne (self.columns) selon l'ordre 0, 1, 3, 6, 9 devient 0, 1, 2, 3, 4, 5
            if False:
                pass

            # Units specified in CSV
            elif label in self.columns and self.units_header is not None and \
                    self.units_header[self.columns[label]] != '':
                try:
                    self.units[label] = self.series.config.data.system.Unit(
                        self.units_header[self.columns[label]])
                except ValueError:
                    self.series.stop(True, 'ValueError',
                        "Unit '{}' used for column '{}' is not valid.",
                        self.units_header[self.columns[label]], self.header(self.columns[label]))


            # Default units based on physical type
            # S'il n'y pas d'unité pour l'erreur, mais qu'il y en a une pour la valeur, on assume que l'erreur a la même unité que la valeur
            else:
                self.units[label] = self.series.config.data.system.default_units[
                    self.series.config.data.system.variables[label].physical_type].unit

        # Check for units physical types
        for label in self.columns.keys():
            if label in self.labels:
                self.series.stop(self.units[label].physical_type \
                    != self.series.config.data.system.variables[label].physical_type,
                    'ValueError', "The unit used for the '{}' ('{}') column, '{}', has "
                    "an incorrect physical type ('{}' required, '{}' given)",
                    self.labels[label], label, str(self.units[label]),
                    self.series.config.data.system.variables[label].physical_type,
                    self.units[label].physical_type)

        # Lines identification and group filtering
        if 'group' in self.columns:
            self.lines = []
            for line in range(1 if self.units_header is None else 2, self.data.shape[0]):
                if self.data[self.columns['group'], line] == self.series.name:
                    self.lines.append(i)
            if len(self.lines) == 0:
                self.series.stop(self.from_CSV, 'ValueError',
                    "No data for the group '{}' in the CSV data file located at '{}'.",
                    self.series.name, self.data_path)
                self.series.stop(self.from_data, 'ValueError',
                    "No information for the group '{}' in the data.",
                    self.series.name, self.data_path)
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
                        self.series.stop(True, 'ValueError',
                            "'{}' value could not be converted to float in '{}' column.",
                            line[label], label)

            # !!! Conversion into Quantity objects and unit conversion here !!!
            # Position columns
            self.position = (
                np.array([line[label] for label in data.position_labels]),
                np.array([data.units[label] for label in data.position_labels]),
                np.array([line[label] for label in data.position_error_labels]))

            # Velocity columns
            self.velocity = (
                np.array([line[label] for label in data.velocity_labels]),
                np.array([data.units[label] for label in data.velocity_labels]),
                np.array([line[label] for label in data.velocity_error_labels]))
