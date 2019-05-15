# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" data.py: Defines the Data and embeded Row classes, which contains all the information from
    CSV file or a Python dictionary, list, tuple or np.ndarray that can then be used as an input
    for a Group object and all related methods to check and convert this data.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

import numpy as np
from astropy import units as un
from time import strftime
from os import path, getcwd, chdir, remove
from csv import reader, writer, Sniffer
from init import Config, System
from series import info, Series
from tools import *

class Data(list):
    """ Contains the data imported from a CSV file or a Python dictionary and related methods.
        Data is converted into a Quantity object and units are converted to default units.
    """

    def __init__(self, series):
        """ Initizalizes a Data object from a CSV file or a Python dictionary, list, tuple or
            np.ndarray. The first row must be a header and the second row may be a units row.
            Units specified by series.config.data.units will override the units row. If no units,
            are given default units are used, based on the value of series.config. data.system.
        """

        # Initialization
        self.series = series
        self.data = self.series.config.data

        # Initialize self from a CSV file
        if type(self.data.values) == str:
            self.initialize_from_CSV()

        # Initialize self from a Python object
        elif type(self.data.values) in (dict, list, tuple, np.ndarray):
            self.initialize_from_data()
        else:
            self.series.stop(True, 'TypeError', "'data.values' component must be a string, "
                "dictionary, list, tuple or np.ndarray. ({} given).",
                type(self.data.values))

        # Data configuration
        self.configure_data()

        # Columns and rows creation
        self.create_columns()
        self.create_rows()

    def initialize_from_CSV(self):
        """ Initializes a Data object from the data 'series.name' in a CSV file. If the the CSV
            file doesn't specify a series or group column, all data is used. The file name must
            have a '.csv' extension.
        """

        self.from_data = False
        self.from_CSV = True

        # CSV file absolute path
        working_dir = getcwd()
        chdir(path.abspath(path.join(path.dirname(path.realpath(__file__)), '..')))
        self.data_path = path.abspath(self.data.values)
        chdir(working_dir)

        # Check if the path exists
        self.series.stop(not path.exists(self.data_path), 'NameError',
            "'{}' does not exist.", self.data_path)
        # Check if the path links to a CSV file.
        self.series.stop(path.splitext(self.data_path)[1].lower() != '.csv', 'TypeError',
            "'{}' is not a CSV data file (with a .csv extension).", self.data_path)

        # Reading of CSV file
        data_csv = open(self.data_path, 'r', encoding='utf-8')
        data_csv_reader = data_csv.read(2048)
        data_csv.seek(0)

        # Data import into a table and CSV file closing
        self.table = np.array(
            [row for row in reader(data_csv, Sniffer().sniff(data_csv_reader))], dtype=object)
        data_csv.close()

    def initialize_from_data(self):
        """ Initializes a Data object from a Python dictionary, list, tuple or np.ndarray (2D). If
            a dictionary is used, only the value with a key that matches 'series.name' is used and
            its type must be one of the other 3 possible types. If a list, tuple or np.ndarray is
            used, all data is imported. A temporary CSV file is created to check whether the array
            has a header and then deleted.
        """

        self.from_data = True
        self.from_CSV = False

        # Data import and group filtering of a dictionary
        if type(self.data.values) == dict:
            if self.series.name in self.data.values.keys():
                if type(self.data.values[self.series.name]) in (list, tuple, np.ndarray):
                    self.table = np.array(self.data.values[self.series.name], dtype=object)
                else:
                    self.series.stop(True, 'TypeError', "Data '{}' in the Python dictionary "
                        "must be a list, tuple or np.ndarray. ('{}' given).",
                        self.series.name, type(self.data.values[self.series.name]))
            else:
                self.series.stop(True, 'ValueError' "Group '{}' is not in the data dictionary.",
                    self.series.name)

        # Data import of a list, tuple or np.ndarray
        if type(self.data.values) in (list, tuple, np.ndarray):
            self.table = np.array(self.data.values, dtype=object)

        # Check if self.table has been converted a 2D array
        self.series.stop(self.table.ndim != 2, 'ValueError',
            "'data' parameter must represent a 2D array ({} dimensions in the given data).",
            self.table.ndim)

        # Conversion of data into strings
        try:
            self.table = np.vectorize(str)(self.table)
        except ValueError:
            self.series.stop(True, 'ValueError', "Data could not be converted to string.")

    def configure_data(self):
        """ Configures data in a np.ndarray. Variables, units, errors and error units and
            identification. If no errors are specified, errors are set to 0.0 and if no units are
            specified, default units are used. The first row must be a header and the second row
            is an can be used as a units row.
        """

        # Value variables
        self.position_variables = {variable.label: variable for variable in self.data.system.position}
        self.velocity_variables = {variable.label: variable for variable in self.data.system.velocity}
        self.value_variables = {**self.position_variables, **self.velocity_variables}

        # Error variables
        self.position_error_variables = {
            variable.label: variable for variable in self.data.system.position_error}
        self.velocity_error_variables = {
            variable.label: variable for variable in self.data.system.velocity_error}
        self.error_variables = {**self.position_error_variables, **self.velocity_error_variables}
        self.variables = {**self.value_variables, **self.error_variables}

        # Checks for the presence of a header in self.table
        self.header = np.vectorize(
            lambda label: label.replace('.', '').replace(',', '').isdigit())(self.table[0]).any()
        if self.from_CSV:
            self.series.stop(self.header, 'ValueError',
                "The CSV data file located at '{}' doesn't have a header.", self.data_path)
        elif self.from_data:
            self.series.stop(self.header, 'ValueError', "The data doesn't have a header.")

        # Checks for the presence of a unit header in self.table
        self.unit_header = True if True not in [
            unit.replace('.', '').replace(',', '').isdigit() for unit in self.table[1]] else False

        # Checks data.units component
        if self.data.units is not None:
            pass
            # !!! Check if units component is valid (TypeError and Value Error) !!!
            # !!! Units header creation from data.units component (either dict or tuple). !!!

    def create_columns(self):
        """ Creates a self.columns dictionary along with self.Column objects for every column
            in self.table. Also checks for the presence of every required columns based on
            self.variables.
        """

        # Columns creation
        self.columns = {column.label: column for column in [
            self.Column(self, column) for column in range(self.table.shape[1])]}

        # Check if all required columns are present
        for label in self.value_variables.keys():
            if label not in self.columns.keys():
                self.series.stop(self.from_CSV, 'NameError',
                    "The column '{}' ('{}') is missing from the CSV data file located at '{}'.",
                    self.value_variables[label].name, label, self.data_path)
                self.series.stop(self.from_data, 'NameError',
                    "The column '{}' ('{}') is missing from the data.",
                    self.value_variables[label].name, label)

        # Missing error columns creation and unit set as None changed to the matching value unit
        for label in self.error_variables.keys():
            if label not in self.columns.keys():
                self.columns[label] = self.Column(self, self.columns[label[1:]].column, True)
            if self.columns[label].unit == None:
                self.columns[label].unit = self.columns[label[1:]].unit

    class Column:
        """ Contains information on a column (parameter), including its index, label, name and
            units.
        """

        # Label permutations
        permutations = {
            '-': '',
            '_': '',
            ' ': '',
            'none': '',
            'fixed': '',
            'error': 'Δ',
            'err': 'Δ',
            'series': 'group',
            'movinggroup': 'group',
            'kinematicgroup': 'group',
            'association': 'group',
            'spt': 'type',
            'spectraltype': 'type',
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

        def __init__(self, data, column, missing=False):
            """ Initializes a Column object from a 'data' object and a number 'column' matching the
                position of the corresponding column in 'data'. If 'error' is set as true, a 'Δ' is
                added to self.label.
            """

            # Initialization
            self.data = data
            self.column = column
            self.missing = missing

            # Label identification and permutations
            self.data_label = self.data.table[0, self.column]
            self.label = self.data_label.lower()
            for old, new in self.permutations.items():
                self.label = self.label.replace(old, new)

            # Error label permutations
            if self.label[-1] in ('e', 'Δ') and self.label not in ('name', 'type'):
                self.label = 'Δ' + self.label[:-1]
            if self.label[0] == 'e':
                self.label = 'Δ' + self.label[1:]

            # Assumption that μα stands for μαcosδ when importing observables system
            if self.data.data.system.name == 'observables' and self.label[-2:] == 'μα':
                self.label = self.label + 'cosδ'

            # Error label addition
            if self.missing:
                self.label = 'Δ' + self.label

            # Variable identification
            if self.label in self.data.variables.keys():
                self.variable = self.data.variables[self.label]

                # Unit specified in data.units component
                if self.data.data.units is not None:
                    pass

                # Unit specified in CSV
                elif self.data.unit_header and self.data.table[1, self.column] != '':
                    try:
                        self.unit = self.data.data.system.Unit(self.data.table[1, self.column])
                    except ValueError:
                        self.series.stop(True, 'ValueError',
                            "Unit '{}' used for column '{}' is not valid.",
                            self.data.table[1, self.column], self.data_label)

                # Error unit temporaraly set as None
                elif self.label[0] == 'Δ':
                    self.unit = None

                # Default unit per physical type
                else:
                    self.unit = self.data.data.system.default_units[self.variable.physical_type]

                # Check unit physical type
                if self.unit is not None:
                    self.data.series.stop(self.unit.physical_type != self.variable.physical_type,
                        'ValueError', "The unit used for '{}' ('{}') in column ('{}'), '{}', has "
                            "an incorrect physical type ('{}' required, '{}' given)",
                        self.variable.name, self.label, self.data_label, self.unit.label,
                        self.variable.physical_type, self.unit.physical_type)

            else:
                self.variable = None
                self.unit = None

    def create_rows(self):
        """ Filters what rows in self.table are part of the selected group definied by
            self.series.name and adds a row object to self for every valid row.
        """

        # Rows identification
        self.rows = list(range(2 if self.unit_header else 1, self.table.shape[0]))

        # Check if there is at least one row
        if len(self.rows) < 1:
            self.series.stop(self.from_CSV, 'ValueError',
                "No entry in the CSV data file located at '{}'.", self.data_path)
            self.series.stop(self.from_data, 'ValueError',
                "No entry for the group in the data.")

        # Group filtering
        if 'group' in self.columns.keys():
            for row in range(2 if self.unit_header else 1, self.table.shape[0]):
                if self.table[row, self.columns['group'].column] != self.series.name:
                    del self.rows[row]

            # Check if there is at least one row remaining
            if len(self.rows) < 1:
                self.series.stop(self.from_CSV, 'ValueError',
                    "No entry for the group '{}' in the CSV data file located at '{}'.",
                    self.series.name, self.data_path)
                self.series.stop(self.from_data, 'ValueError',
                    "No entry for the group '{}' in the data.", self.series.name)

        # Rows creation
        for row in self.rows:
            self.append(self.Row(self, row))

    class Row:
        """ Contains the data for an individual row (star), including name, type, id, position
            and velocity.
        """

        def __init__(self, data, row):
            """ Initializes a Row object from a 'data' object, a 'row' number. Default values are
                given if no value is given. Errors are set to 0.0 if no error column is present.
            """

            # Initialization
            self.data = data
            self.row = row

            # Values dictionary creation
            self.values = {}
            for label in self.data.columns.keys():

                # Values present in self.table
                if not self.data.columns[label].missing:
                    self.values[label] = self.data.table[
                        self.row, self.data.columns[label].column]

                # Error values missing from self.table
                else:
                    self.values[label] = ''

            # Float conversion
            for label in self.data.variables.keys():

                # Empty '' values
                if self.values[label] == '':
                    self.values[label] = 0.0

                # Numerical values
                else:
                    try:
                        self.values[label] = float(self.values[label].replace(',', '.'))
                    except ValueError:
                        self.data.series.stop(True, 'ValueError',
                            "'{}' value could not be converted to float in '{}' column.",
                            self.values[label], label)

            # Name column
            self.name = self.values['name'] if 'name' in self.data.columns \
                and self.values['name'] != '' else 'Star_{}'.format(str(self.row))

            # Type column
            self.type = self.values['type'] if 'type' in self.data.columns \
                and self.values['type'] != '' else None

            # ID column
            self.id = self.values['id'] if 'id' in self.data.columns \
                and self.values['id'] != '' else str(self.row)

            # Position columns and unit conversion
            self.position = Quantity(
                [self.values[label] for label in self.data.position_variables.keys()],
                [self.data.columns[label].unit for label in self.data.position_variables.keys()],
                [self.values[label] for label in self.data.position_error_variables.keys()],
                [self.data.columns[label].unit for label in self.data.position_error_variables.keys()]).to()

            # Velocity columns and unit conversion
            self.velocity = Quantity(
                [self.values[label] for label in self.data.velocity_variables.keys()],
                [self.data.columns[label].unit for label in self.data.velocity_variables.keys()],
                [self.values[label] for label in self.data.velocity_error_variables.keys()],
                [self.data.columns[label].unit for label in self.data.velocity_error_variables.keys()]).to()
