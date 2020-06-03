# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" data.py: Defines Data, and embeded Column and Row classes, which contains all the information
    from a CSV file or a Python dictionary, list, tuple or np.ndarray that can then be used as
    an input for a Group object, and all related methods to check and convert this data. Units are
    identified and converted to default units.
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

import numpy as np
from csv import reader, writer, Sniffer
from Traceback.collection import *
from Traceback.coordinate import *

class Data(list):
    """ Contains the data imported from a CSV file or a Python dictionary and related methods.
        Data is converted into a Quantity object and units are converted to default units.
    """

    def __init__(self, series):
        """ Initializes a Data object from a CSV file or a Python dictionary, list, tuple or
            np.ndarray. The first row must be a header and the second row may be a units row.
            Units specified by series.config.data.units will override the units row. If no units,
            are given default units are used, based on the value of series.config and data.system.
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

        # Non-valid 'data.values' types
        else:
            self.series.stop(True, 'TypeError', "'data.values' component must be a string, "
                "dictionary, list, tuple or np.ndarray. ({} given).", type(self.data.values))

        # Data configuration
        self.configure_data()

        # Columns and rows creation
        self.create_columns()
        self.create_rows()

    def initialize_from_CSV(self):
        """ Initializes a Data object from the data self.series.name in a CSV file. If the the
            CSV file doesn't specify a series or group column, all data is used. The file name
            must have a '.csv' extension.
        """

        # Initialization
        self.from_data = False
        self.from_CSV = True

        # CSV file absolute path
        self.data_path = directory(collection.base_dir, self.data.values, 'data_path')

        # Check if the data file exists
        self.series.stop(not path.exists(self.data_path), 'FileNotFoundError',
            "No data file located at '{}'.", self.data_path)

        # Check if the path links to a CSV file.
        self.series.stop(path.splitext(self.data_path)[1].lower() != '.csv', 'TypeError',
            "'{}' is not a CSV data file (with a .csv extension).", path.basename(self.data_path))

        # Reading of CSV file
        data_csv = open(self.data_path, 'r', encoding='utf-8')
        data_csv_reader = data_csv.read(2048)
        data_csv.seek(0)

        # Data import into a table and CSV file closing
        self.table = np.array(
            [row for row in reader(data_csv, Sniffer().sniff(data_csv_reader))], dtype=object)
        data_csv.close()

        # Check if self.table is a 2D array
        self.series.stop(self.table.ndim != 2, 'ValueError',
            "'data' parameter must represent a 2D array ({} dimensions in the given CSV). "
            "Make sure all lines have an equal number of columns in the CSV file.", self.table.ndim)

    def initialize_from_data(self):
        """ Initializes a Data object from a Python dictionary, list, tuple or np.ndarray (2D). If
            a dictionary is used, only the value with a key that matches self.series.name is used
            and its type must be one of the other 3 possible types. If a list, tuple or np.ndarray
            is used, all data is imported.
        """

        # Initialization
        self.from_data = True
        self.from_CSV = False
        self.data_path = None

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

        # Check if self.table is a 2D array
        self.series.stop(self.table.ndim != 2, 'ValueError',
            "'data' parameter must represent a 2D array ({} dimensions in the given data). "
            "Make sure all lines have an equal number of columns.", self.table.ndim)

        # Conversion of data into strings
        try:
            self.table = np.vectorize(str)(self.table)
        except ValueError:
            self.series.stop(True, 'ValueError', "Data could not be converted to string.")

    def configure_data(self):
        """ Configures data in a np.ndarray. Variables, units, errors and error units are
            identified. If no errors are specified, errors are set to 0.0 and if no units are
            specified, default units are used. The first row must be a header and the second
            row can be used as a units row.
        """

        # Value variables
        self.position_variables = {variable.label:
            variable for variable in self.data.system.position}
        self.velocity_variables = {variable.label:
            variable for variable in self.data.system.velocity}
        self.value_variables = {**self.position_variables, **self.velocity_variables}

        # Error variables
        self.position_error_variables = {
            variable.label: variable for variable in self.data.system.position_error}
        self.velocity_error_variables = {
            variable.label: variable for variable in self.data.system.velocity_error}
        self.error_variables = {**self.position_error_variables, **self.velocity_error_variables}

        # All variables
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
        self.unit_header = np.vectorize(
            lambda unit: unit.replace('.', '').replace(',', '').isdigit())(self.table[1]).any()

        # Units from a unit header, if present
        if self.data.units is None:
            self.data.units = self.table[1] if self.unit_header else None

        # Units from a string representing a coordinate system to create a dictionary
        elif type(self.data.units) == str:
            self.series.stop(self.data.units.lower() not in systems.keys(), 'ValueError',
                "'{}' is not a valid coordinates system (use {} instead).",
                self.data.units, list(systems.keys()))

            # Usual units of system retrieval
            # !!! Should also include stellar mass and radius units !!!
            self.data.units = {
                **{variable.label: variable.usual_unit.label \
                    for variable in systems[self.data.units.lower()].position},
                **{variable.label: variable.usual_unit.label \
                    for variable in systems[self.data.units.lower()].velocity}}

        # Units from a dictionary
        elif type(self.data.units) == dict:

            # Check if all labels can be matched to a data.system variable
            # !!! The match should also include stellar mass and radius units !!!
            self.data.units = {self.Column.identify(self.Column, self, label):
                self.data.units[label] for label in self.data.units.keys()}
            self.series.stop(not np.vectorize(lambda label: label in self.variables.keys()) \
                (tuple(self.data.units.keys())).all(), 'ValueError',
                "All labels in 'data.units' couldn't be matched to a variable of '{}' system.",
                self.data.system.name)

            # Check if all values in self.data.units are strings
            self.series.stop(not np.vectorize(
                    lambda label: type(self.data.units[label]) == str)(tuple(
                        self.data.units.keys())).all(),
                'TypeError', "All elements in 'data.units' component must be strings.")

        # Units from an array mimicking a unit header
        elif type(self.data.units) in (tuple, list, np.ndarray):
            self.data.units = np.squeeze(np.array(self.data.units, dtype=object))

            # Check if all values in self.data.units array are strings
            self.series.stop(not np.vectorize(lambda unit: type(unit) == str) \
                (self.data.units).all(), 'TypeError',
                "All elements in 'data.units' component must be strings.")

            # Check if self.data.units array has valid dimensions
            self.series.stop(self.data.units.ndim != 1, 'ValueError', "'data.units' component "
                "must have 1 dimension ({} given).", self.data.units.ndim)
            self.series.stop(len(self.data.units) != self.table.shape[1], 'ValueError',
                "'data.units' component must be as long as the number of columns in "
                    "'data.values' component. ({} required, {} given).",
                self.table[1], len(self.data.units))

        # Non-valid 'data.units' types
        else:
            self.series.stop(True, 'TypeError', "'data.units' component must be a string, "
                "dictionary, list, tuple or np.ndarray. ({} given).", type(self.data.units))

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
            units and a permutations dictionary to identify labels.
        """

        def __init__(self, data, column, missing=False):
            """ Initializes a Column object from a 'data' object and a number 'column' matching the
                position of the corresponding column in 'data'.
            """

            # Initialization
            self.data = data
            self.column = column
            self.missing = missing

            # Label identification
            self.data_label = self.data.table[0, self.column]
            self.label = self.identify(self.data, self.data_label, self.missing)

            # Variables and units identification
            # !!! S'arranger pour éviter d'avoir à gérer un dictionnaire et une liste ici !!!
            # Préconvertir les dictionnaires en liste en mettant à jour le unit header ou header par défaut.
            # Les permutations des labels de colonnes se feraient avant et ça permettrait d'avoir des dictionaires
            if self.label in self.data.variables.keys():
                self.variable = self.data.variables[self.label]
                self.units = self.data.data.units

                # Unit specified in an array (CSV unit header or data.units component)
                if type(self.units) == np.ndarray and self.units[self.column] != '':
                    self.unit = self.units[self.column]

                # Unit specified in a dictionary (data.units component)
                elif type(self.units) == dict and self.label in self.units.keys():
                    self.unit = self.units[self.label]

                # Error unit set temporarily as None, if units aren't specified
                elif self.label[0] == 'Δ':
                    self.unit = None

                # Value unit set as the default unit per physical type (if self.data.data.units is None)
                else:
                    self.unit = self.data.data.system.default_units[self.variable.physical_type]

                # Conversion into Unit object
                if self.unit is not None:
                    try:
                        self.unit = Unit(self.unit)
                    except ValueError:
                        self.series.stop(True, 'ValueError', "Unit '{}' used for column '{}' "
                            "is not valid.", self.unit, self.data_label)

                    # Check unit physical type
                    self.data.series.stop(self.unit.physical_type != self.variable.physical_type,
                        'ValueError', "The unit used for '{}' ('{}') in column ('{}'), '{}', has "
                            "an incorrect physical type ('{}' required, '{}' given)",
                        self.variable.name, self.label, self.data_label, self.unit.label,
                        self.variable.physical_type, self.unit.physical_type)

            # Non-variable columns
            else:
                self.variable = None
                self.unit = None

        # Basic permutations including uppercase and lowercase latin letters
        from string import ascii_uppercase
        basic_permutations = {
            **{letter: letter.lower() for letter in ascii_uppercase}, '-': ' ', '_': ' '}

        # Advanced label permutations
        advanced_permutations = {
            ' ': '',
            'none': '',
            'fixed': '',
            'name': 'n',
            'series': 'g',
            'movinggroup': 'g',
            'kinematicgroup': 'g',
            'association': 'g',
            'spt': 't',
            'spectraltype': 't',
            'mass': 'm',
            'radius': 'r',
            'star': '',
            'stellar': '',
            'distance': 'd',
            'dist': 'd',
            'parallax': 'p',
            'plx': 'p',
            'radialvelocity': 'rv',
            'declination': 'δ',
            'dec': 'δ',
            'delta': 'δ',
            'rightascension': 'α',
            'ra': 'α',
            'alpha': 'α',
            'rho': 'ρ',
            'theta': 'θ',
            'phi': 'θ',
            'φ': 'θ',
            'propermotion': 'μ',
            'pm': 'μ',
            'cosine': 'cos',
            '(δ)': 'δ',
            'error': 'Δ',
            'err': 'Δ'}

        # Label matches
        matches = {'n': 'name', 'g': 'group', 't': 'type', 'm': 'mass', 'r': 'radius'}

        def identify(self, data, label, missing=False):
            """ Identifies a 'label' and returns a matching wanted label or, including coordinates
                from 'data.system' or a reformatted label, if no match is found. If 'missing' is
                True, an 'Δ' character is added.
            """

            # Basic label permutation and strip
            for old, new in self.basic_permutations.items():
                label = label.replace(old, new)
            label = label.strip()
            old_label = label

            # Wanted labels
            wanted = ('name', 'group', 'id', 'type', 'mass', 'radius',
                *data.value_variables.keys(), *data.error_variables.keys())

            # Advanced permutations to match a wanted label
            for old, new in self.advanced_permutations.items():
                label = label.replace(old, new)

            # Error label permutations
            if label[-1] in ('e', 'Δ'):
                label = 'Δ' + label[:-1]
            if label[0] == 'e':
                label = 'Δ' + label[1:]

            # Proper motion label permutation
            if label[-1] == 'μ':
                label = 'Δμ' + label[1:-1] if label[0] == 'Δ' else 'μ' + label[:-1]

            # Assumption that 'μα' stands for 'μαcosδ' when importing from observables
            if data.data.system == 'observables' and label[-2:] == 'μα':
                label = label + 'cosδ'

            # If missing, error label addition
            if missing:
                label = 'Δ' + label

            # Change label to a matching label
            if label in self.matches.keys():
                label = self.matches[label]

            # Return old label if label is not a wanted label
            return label if label in wanted else old_label

    def create_rows(self):
        """ Filters what rows in self.table are part of the selected group definied by
            self.series.name and adds a self.Row object to self for every valid row.
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
            """ Initializes a Row object from a 'data' object and a 'row' number. Default values
                are used if no value is given. Errors are set to 0.0 if no error column is present.
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
            for label in list(self.data.variables.keys()) + [
                label for label in ('mass', 'radius') if label in self.data.columns]:
                    # ['mass'] if 'mass' in self.data.columns else []]:

                # Empty '' values
                if self.values[label] == '':
                    self.values[label] = 0.

                # Numerical values
                else:
                    try:
                        self.values[label] = float(self.values[label].replace(',', '.').strip())
                    except ValueError:
                        self.data.series.stop(True, 'ValueError',
                            "'{}' value could not be converted to float in '{}' column.",
                            self.values[label], label)

            # Name column
            self.name = self.values['name'].strip() if 'name' in self.data.columns \
                and self.values['name'].strip()  != '' else 'Star_{}'.format(str(self.row))

            # Spectral type column
            self.type = self.values['type'].strip() if 'type' in self.data.columns \
                and self.values['type'].strip() != '' else None

            # ID column
            self.id = self.values['id'].strip() if 'id' in self.data.columns \
                and self.values['id'].strip() != '' else str(self.row)

            # Mass column
            self.mass = Quantity(self.values['mass'], self.data.columns['mass'].unit) \
                if 'mass' in self.data.columns and self.values['mass'] != 0. else None

            # Radius column
            self.radius = Quantity(self.values['radius'], self.data.columns['radius'].unit) \
                if 'radius' in self.data.columns and self.values['radius'] != 0. else None

            if self.mass is not None and self.radius is not None:
                self.rv_offset = c.value * ((1 - 2 * G.value * self.mass.value * M_sun.value / (
                    c.value**2 * self.radius.value * R_sun.value))**0.5 - 1) / 1000
            else:
                self.rv_offset = None

            # Position columns and unit conversion
            self.position = Quantity(
                [self.values[label] for label in self.data.position_variables.keys()],
                [self.data.columns[label].unit for label in self.data.position_variables.keys()],
                [self.values[label] for label in self.data.position_error_variables.keys()],
                [self.data.columns[label].unit
                    for label in self.data.position_error_variables.keys()]).to()

            # Velocity columns and unit conversion
            self.velocity = Quantity(
                [self.values[label] for label in self.data.velocity_variables.keys()],
                [self.data.columns[label].unit for label in self.data.velocity_variables.keys()],
                [self.values[label] for label in self.data.velocity_error_variables.keys()],
                [self.data.columns[label].unit
                    for label in self.data.velocity_error_variables.keys()]).to()
