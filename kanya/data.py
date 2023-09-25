# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data.py: Defines Data, and embeded Column and Row classes, which contains all the information
from a CSV file or a Python dictionary, list, tuple or np.ndarray that can then be used as
an input for a Group object, and all related methods to check and convert this data. Units are
identified and converted to default units.
"""

import pandas as pd
from csv import reader, Sniffer
from re import split
from .collection import *
from .coordinate import *

class Data(list):
    """
    Contains the data imported from a CSV file or a Python dictionary and related methods.
    Data is converted into a Quantity object and units are converted to default units.
    """

    def __init__(self, series, data=None, data_dir=None):
        """
        Initializes a Data object from a CSV file or a Python dictionary, list, tuple or
        np.ndarray. The first row must be a header and the second row may be a units row.
        Units specified by series.config.data.units will override the units row. If no units,
        are given default units are used, based on the value of series.config and data.system.
        """

        # Initialization
        self.series = series

        # Set data parameter
        self.data = self.series.set_parameter(
            self.series.config.data if data is None else data,
            'string', 'dictionary', 'tuple', 'list', 'array',
            mode=True if self.series.from_data else False, all=True
        )

        # Initialize self from a CSV file
        if type(self.data.values) == str:
            self.initialize_from_CSV(data_dir=data_dir)

        # Initialize self from a Python object
        if type(self.data.values) in (dict, list, tuple, np.ndarray):
            self.initialize_from_data()

        # Data configuration
        self.configure_data()

        # Stellar spectral parameters
        self.get_stellar_spectal_parameters()

        # Columns and rows creation
        self.create_columns()
        self.create_rows()

    def initialize_from_CSV(self, data_dir=None):
        """
        Initializes a Data object from the data self.series.name in a CSV file. If the the
        CSV file doesn't specify a series or group column, all data is used. The file name
        must have a '.csv' extension.
        """

        # Initialization
        self.from_data = False
        self.from_CSV = True

        # Set data directory parameter
        self.data_dir = self.series.set_string(
            self.series.config.data_dir if data_dir is None else data_dir,
            mode=True if self.series.from_data else False
        )

        # Set data path parameter
        self.data_path = self.series.set_path(
            path.join(self.data_dir, self.data.values), 'data_path',
            extension='csv', file_type='CSV', check=True, create=False,
            mode=True if self.series.from_data else False
        )

        # Read CSV file
        data_csv = open(self.data_path, 'r', encoding='utf-8')
        data_csv_reader = data_csv.read()
        data_csv.seek(0)

        # Import data into a table and close CSV file
        self.table = np.array(
            [row for row in reader(data_csv, Sniffer().sniff(data_csv_reader))], dtype=object
        )
        data_csv.close()

        # Check if the table is a 2D array
        self.series.stop(
            self.table.ndim != 2, 'ValueError',
            "'data' parameter must represent a 2D array ({} dimensions in the given CSV). "
            "Make sure all lines have an equal number of columns in the CSV file.", self.table.ndim
        )

    def initialize_from_data(self):
        """
        Initializes a Data object from a Python dictionary, list, tuple or np.ndarray (2D). If
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
                    self.series.stop(
                        True, 'TypeError',
                        "Data '{}' in the Python dictionary "
                        "must be a list, tuple or np.ndarray. ('{}' given).",
                        self.series.name, type(self.data.values[self.series.name])
                    )
            else:
                self.series.stop(
                    True, 'ValueError',
                    "Group '{}' is not in the data dictionary.", self.series.name)

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
        """
        Configures data in a np.ndarray. Variables, units, errors and error units are
        identified. If no errors are specified, errors are set to 0.0 and if no units are
        specified, default units are used. The first row must be a header and the second
        row can be used as a units row.
        """

        # Value variables
        self.position_variables = {
            variable.label: variable for variable in self.data.system.position
        }
        self.velocity_variables = {
            variable.label: variable for variable in self.data.system.velocity
        }
        self.value_variables = {**self.position_variables, **self.velocity_variables}

        # Error variables
        self.position_error_variables = {
            variable.label: variable for variable in self.data.system.position_error
        }
        self.velocity_error_variables = {
            variable.label: variable for variable in self.data.system.velocity_error
        }
        self.error_variables = {**self.position_error_variables, **self.velocity_error_variables}

        # All variables
        self.variables = {**self.value_variables, **self.error_variables}

        # Valid labels (strings, numerial, all)
        self.valid_labels_str = ('name', 'group', 'id', 'sample', 'spectral_type')
        self.valid_labels_num = tuple(self.variables.keys()) + (
            'mass', 'mass_error',
            'radius', 'radius_error',
            'rv_shift', 'rv_shift_error',
            'grav_shift', 'grav_shift_error',
            'conv_shift', 'conv_shift_error'
        )
        self.valid_labels = self.valid_labels_str + self.valid_labels_num

        # Checks for the presence of a header in self.table
        self.header = np.vectorize(
            lambda label: label.replace('.', '').replace(',', '').replace('-', '').isdigit()
        )(self.table[0]).any()
        if self.from_CSV:
            self.series.stop(
                self.header, 'ValueError',
                "The CSV data file located at '{}' doesn't have a header.", self.data_path
            )
        elif self.from_data:
            self.series.stop(self.header, 'ValueError', "The data doesn't have a header.")

        # Checks for the presence of a unit header in self.table
        self.unit_header = not np.vectorize(
            lambda unit: unit.replace('.', '').replace(',', '').replace('-', '').isdigit()
        )(self.table[1]).any()

        # !!! Additional check to see nonzero values can be transformed in units !!!

        # Units from a unit header, if present
        if self.data.units is None:
            self.data.units = self.table[1] if self.unit_header else None

        # Units from a string representing a coordinate system to create a dictionary
        elif type(self.data.units) == str:
            self.series.stop(
                self.data.units.lower() not in systems.keys(), 'ValueError',
                "'{}' is not a valid coordinates system (use {} instead).",
                self.data.units, list(systems.keys())
            )

            # Usual units of system retrieval
            # !!! Should also include units for mass and radius (and errors) units !!!
            self.data.units = {
                **{
                    variable.label: variable.usual_unit.label
                    for variable in systems[self.data.units.lower()].position},
                **{
                    variable.label: variable.usual_unit.label
                    for variable in systems[self.data.units.lower()].velocity}
                }

        # Units from a dictionary
        elif type(self.data.units) == dict:

            # Check if all labels can be matched to a data.system variable
            # !!! The match should also include units for mass, radius, rv_shift,
            # grav_shift and conv_shift, and their error units !!!
            self.data.units = {
                self.Column.identify(self.Column, self, label):
                    self.data.units[label] for label in self.data.units.keys()
                }
            self.series.stop(
                not np.vectorize(lambda label: label in self.variables.keys())(
                    tuple(self.data.units.keys())
                ).all(), 'ValueError',
                "All labels in 'data.units' couldn't be matched to a variable of '{}' system.",
                self.data.system.name
                )

            # Check if all values in self.data.units are strings
            self.series.stop(
                not np.vectorize(lambda label: type(self.data.units[label]) == str)(
                    tuple(self.data.units.keys())).all(), 'TypeError',
                "All elements in 'data.units' component must be strings."
            )

        # Units from an array mimicking a unit header
        elif type(self.data.units) in (tuple, list, np.ndarray):
            self.data.units = np.squeeze(np.array(self.data.units, dtype=object))

            # Check if all values in self.data.units array are strings
            self.series.stop(
                not np.vectorize(lambda unit: type(unit) == str)(self.data.units).all(),
                'TypeError', "All elements in 'data.units' component must be strings."
            )

            # Check if self.data.units array has valid dimensions
            self.series.stop(
                self.data.units.ndim != 1, 'ValueError',
                "'data.units' component must have 1 dimension ({} given).", self.data.units.ndim
            )
            self.series.stop(
                len(self.data.units) != self.table.shape[1], 'ValueError',
                "'data.units' component must be as long as the number of columns in "
                "'data.values' component. ({} required, {} given).",
                len(self.table[1]), len(self.data.units)
            )

        # Non-valid 'data.units' types
        else:
            self.series.stop(
                True, 'TypeError',
                "'data.units' component must be a string, "
                "dictionary, list, tuple or np.ndarray. ({} given).", type(self.data.units)
            )

    def get_stellar_spectal_parameters(self):
        """
        Import stellar mass and radius, and gravitational, convective and total radial velocity
        shift sequences based on spectral type.
        """

        # Stellar spectral parameters data
        stellar_spectral_parameters_file = path.join(
            path.dirname(__file__),
            'resources/stellar_spectral_parameters.csv'
        )
        dataframe = pd.read_csv(stellar_spectral_parameters_file, delimiter=',')

        # Convert values to float and delete spectral types with NaN
        data = dataframe.to_numpy()[2:]
        data[:,1:] = np.array(data[:,1:], dtype=float)
        rows_with_NaN = [
            row for row in range(data.shape[0])
            if np.isnan(np.array(data[row,1:], dtype=float)).any()
        ]
        data = np.delete(data, np.array(rows_with_NaN, dtype=int), axis=0)
        dataframe = pd.DataFrame(data=data, columns=list(dataframe.columns))

        # Uncomment if the file uses Jonathan's spectral type numbers definition
        # dataframe['sptn'] = dataframe['sptn'] + 60.0

        # Spectral types
        self.spectral_types_str = dataframe['spt']
        self.spectral_types_num = np.array(
            [self.convert_spt(spt) for spt in self.spectral_types_str], dtype=float
        )

        # Stellar masses
        self.mass = dataframe['mass_ms']
        self.mass_error = dataframe['mass_ms_error']

        # Stellar radii
        self.radius = dataframe['radius_ms']
        self.radius_error = dataframe['radius_ms_error']

        # Gravitational radial velocity shifts
        self.grav_shift = dataframe['grav_redshift_ms']
        self.grav_shift_error = dataframe['grav_redshift_ms_error']

        # Convective radial velocity shifts
        self.conv_shift = dataframe['conv_blueshift']
        self.conv_shift_error = dataframe['conv_blueshift_error']

        # Total radial velocity shifts
        self.total_shift = dataframe['total_shift_ms']
        self.total_shift_error = dataframe['total_shift_ms_error']

    def convert_spt(self, spt):
        """Converts spectral type letter into a number."""

        # Permutation dictionary
        permutations = {
            'Y': 90.0, 'T': 80.0, 'L': 70.0, 'M': 60.0, 'K': 50.0,
            'G': 40.0, 'F': 30.0, 'A': 20.0, 'B': 10.0, 'O':  0.0
        }

        # Match spectral type components
        new_spt = spt.replace(' ', '').replace('(', '').replace(')', '')
        try:
            letter, number, other = split('(\d+\.?\d*)', new_spt)
        except ValueError as e:
            raise ValueError(f"{spt} is an invalid spectral type.") from e

        # Uppercase letter and match
        letter = letter.upper()
        if letter not in permutations.keys():
            raise ValueError(f"{spt} is an invalid spectral type.")
        letter = permutations[letter]

        # Convert letter
        try:
            number = float(number)
        except ValueError as e:
            raise ValueError(f"{spt} is an invalid spectral type.") from e
        if number < 0. or number >= 10.:
            raise ValueError(f"{spt} is an invalid spectral type.")

        # New spectral type
        return letter + number

    def convert_sptn(self, sptn):
        """Converts spectral type number into a letter form."""

        # Permutation dictionary
        permutations = {
            90.0: 'Y', 80.0: 'T', 70.0: 'L', 60.0: 'M', 50.0: 'K',
            40.0: 'G', 30.0: 'F', 20.0: 'A', 10.0: 'B',  0.0: 'O'
        }

        # Check if 'sptn' is a valid number
        try:
            sptn = float(sptn)
        except ValueError as e:
            raise ValueError(f'{sptn} must be a number ({type(sptn)} given).') from e
        if sptn < 0.0 or sptn >= 100.0:
            raise ValueError(f'{sptn} is an invalid spectral type number.')

        # Convert number
        letter = permutations[float((sptn // 10) * 10)]
        number = sptn % 10

        return f'{letter}{number:.1f}'

    def find_rv_shift(self, spt):
        """
        Finds the closest total radial velocity shift, including the effects of gravitational
        redshift and convective blueshift, based on the spectral type or spectral type number
        of a star or brown dwarf.
        """

        # Convert a spectral type into a number if needed
        if np.char.isnumeric(spt):
            sptn = float(spt)
            # sptn = float(spt) + 60.0
        else:
            sptn = self.convert_spt(spt)

        # Find index and matching spectral type
        index = (np.abs(self.spectral_types_num - sptn)).argmin()
        spectral_type = self.spectral_types_str[index]

        # Stellar mass
        mass = self.mass[index]
        mass_error = self.mass_error[index]

        # Stellar radius
        radius = self.radius[index]
        radius_error = self.radius_error[index]

        # Gravitational radial velocity shift
        grav_shift = self.grav_shift[index]
        grav_shift_error = self.grav_shift_error[index]

        # Convective radial velocity shift
        conv_shift = self.conv_shift[index]
        conv_shift_error = self.conv_shift_error[index]

        # Total radial velocity shift
        rv_shift = self.total_shift[index]
        rv_shift_error = self.total_shift_error[index]

        return spectral_type, rv_shift, rv_shift_error

    def create_columns(self):
        """
        Creates a self.columns dictionary along with self.Column objects for every column
        in self.table. Also checks for the presence of every required columns based on
        self.variables.
        """

        # Columns creation
        self.columns = {
            column.label: column for column in [
                self.Column(self, column) for column in range(self.table.shape[1])
            ]
        }

        # Check if all required columns are present
        for label in self.value_variables.keys():
            if label not in self.columns.keys():
                self.series.stop(
                    self.from_CSV, 'NameError',
                    "The column '{}' ('{}') is missing from the CSV data file located at '{}'.",
                    self.value_variables[label].name, label, self.data_path
                )
                self.series.stop(
                    self.from_data, 'NameError',
                    "The column '{}' ('{}') is missing from the data.",
                    self.value_variables[label].name, label
                )

        # Missing error columns creation and unit set as None changed to the matching value unit
        for label in self.error_variables.keys():
            if label not in self.columns.keys():
                self.columns[label] = self.Column(
                    self, self.columns[label[1:]].column, True
            )
            if self.columns[label].unit is None:
                self.columns[label].unit = self.columns[label[1:]].unit

    class Column:
        """
        Contains information on a column (parameter), including its index, label, name and
        units and a permutations dictionary to identify labels.
        """

        def __init__(self, data, column, missing=False):
            """
            Initializes a Column object from a 'data' object and a number 'column' matching the
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
            # !!! S'arranger pour éviter d'avoir à gérer un dictionnaire et une liste ici
            # Préconvertir les dictionnaires en liste en mettant à jour le unit header ou header
            # par défaut. Les permutations des labels de colonnes se feraient avant et ça
            # permettrait d'avoir des dictionaires !!!
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
                        self.series.stop(
                            True, 'ValueError', "Unit '{}' used for column '{}' "
                            "is not valid.", self.unit, self.data_label)

                    # Check unit physical type
                    self.data.series.stop(
                        self.unit.physical_type != self.variable.physical_type,
                        'ValueError', "The unit used for '{}' ('{}') in column ('{}'), '{}', has "
                        "an incorrect physical type ('{}' required, '{}' given)",
                        self.variable.name, self.label, self.data_label, self.unit.label,
                        self.variable.physical_type, self.unit.physical_type
                    )

            # Non-variable columns
            else:
                self.variable = None
                self.unit = None

        # Basic permutations including uppercase and lowercase latin letters
        from string import ascii_uppercase
        basic_permutations = {
            **{letter: letter.lower() for letter in ascii_uppercase}, '-': '_', ' ': '_'
        }

        # Advanced label permutations
        advanced_permutations = {
            '_': '',
            '(s)': '',
            'none': '',
            'fixed': '',
            'source': '',
            'new': '',
            'final': '',
            'name': 'n',
            'designation': 'n',
            'series': 'g',
            'movinggroup': 'g',
            'kinematicgroup': 'g',
            'association': 'g',
            'sample': 's',
            'sptn': 't',
            'spt': 't',
            'spectraltype': 't',
            'mass': 'm',
            'radius': 'r',
            'star': '',
            'stellar': '',
            'distance': 'ρ',
            'dist': 'ρ',
            'parallax': 'π',
            'plx': 'π',
            'radialvelocity': 'rv',
            'dvr': 'drv',
            'rvshift': 'drv',
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
            'err': 'Δ'
        }

        # Label matches
        # !!! Add spectral type errors !!!
        matches = {
            'n': 'name',
            'g': 'group',
            's': 'sample',
            't': 'spectral_type',
            'm': 'mass',
            'Δm': 'mass_error',
            'r': 'radius',
            'Δr': 'radius_error',
            'drv': 'rv_shift',
            'Δdrv': 'rv_shift_error'
        }

        def identify(self, data, label, missing=False):
            """
            Identifies a 'label' and returns a matching label or, including coordinates from
            'data.system' or a reformatted label, if no match is found. If 'missing' is True,
            an 'Δ' character is added.
            """

            # Check for blank labels
            if len(label) == 0:
                return ''

            # Basic label permutation and strip
            for old, new in self.basic_permutations.items():
                label = label.replace(old, new)
            label = label.strip()
            old_label = label

            # Advanced permutations to matching label
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

            # Return old label if label is not a valid label
            return label if label in self.data.valid_labels else old_label

    def create_rows(self):
        """
        Filters what rows in self.table are part of the selected group definied by
        self.series.name and adds a self.Row object to self for every valid row.
        """

        # Rows identification
        self.rows = list(range(2 if self.unit_header else 1, self.table.shape[0]))

        # Check if there is at least one row
        if len(self.rows) < 1:
            self.series.stop(
                self.from_CSV, 'ValueError',
                "No entry in the CSV data file located at '{}'.", self.data_path
            )
            self.series.stop(
                self.from_data, 'ValueError',
                "No entry for the group in the data."
            )

        # Group filtering
        if 'group' in self.columns.keys():
            for row in range(2 if self.unit_header else 1, self.table.shape[0]):
                if self.table[row, self.columns['group'].column] != self.series.name:
                    del self.rows[row]

            # Check if there is at least one row remaining
            if len(self.rows) < 1:
                self.series.stop(
                    self.from_CSV, 'ValueError',
                    "No entry for the group '{}' in the CSV data file located at '{}'.",
                    self.series.name, self.data_path
                )
                self.series.stop(
                    self.from_data, 'ValueError',
                    "No entry for the group '{}' in the data.", self.series.name
                )

        # Rows creation
        for row in self.rows:
            self.append(self.Row(self, row))

        # Create input sample
        self.input_sample = list(
            filter(
                lambda row: row.valid and row.sample in ('core_sample', 'extended_sample'), self
            )
        )

        # Create rejected sample
        self.rejected_sample = list(
            filter(
                lambda row: not row.valid or row.sample not in ('core_sample', 'extended_sample'),
                self
            )
        )

        # Create core sample
        self.core_sample = list(
            filter(lambda row: row.sample == 'core_sample', self.input_sample)
        )

        # Create extended sample
        self.extended_sample = list(
            filter(lambda row: row.sample == 'extended_sample', self.input_sample)
        )

        # Uncomment this line to use the input sample
        # self.core_sample = self.input_sample

        # Uncomment this line to limit radial velocity precision
        # self.core_sample = list(
        #     filter(
        #         lambda row: row.sample == 'core_sample' and row.velocity.errors[0] < 2.5,
        #         self.input_sample
        #     )
        # )

    class Row:
        """
        Contains the data for an individual row (star), including name, type, id, position
        and velocity.
        """

        def __init__(self, data, row):
            """
            Initializes a Row object from a 'data' object and a 'row' number. Default values
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
                        self.row, self.data.columns[label].column
                    ]

                # Error values missing from self.table
                else:
                    self.values[label] = ''

            # Convert numerical values
            for label in self.data.valid_labels_num:
                if label in self.data.columns:
                    # !!! ['mass'] if 'mass' in self.data.columns else []]: !!!

                    # Convert empty '' errors to 0.0 and others to None
                    if self.values[label] in ('', '...'):
                        if label in self.data.error_variables.keys():
                            self.values[label] = 0.0
                        else:
                            self.values[label] = None

                    # Convert numerical values to float
                    else:
                        try:
                            self.values[label] = float(self.values[label].replace(',', '.').strip())
                        except ValueError:
                            self.data.series.stop(
                                True, 'ValueError',
                                "'{}' value could not be converted to float in '{}' column.",
                                self.values[label], label
                            )

            # Name column
            self.name = (
                self.values['name'].strip()
                if 'name' in self.data.columns and self.values['name'].strip() != ''
                else 'Star_{}'.format(
                        str(self.row - 1) if self.data.unit_header else str(self.row)
                    )
                )

            # Spectral type column
            self.spectral_type = (
                self.values['spectral_type'].strip()
                if 'spectral_type' in self.data.columns and
                self.values['spectral_type'].strip() != '' else None
            )

            # ID column
            self.id = (
                self.values['id'].strip()
                if 'id' in self.data.columns and self.values['id'].strip() != '' else str(self.row)
            )

            # Final sample column
            self.sample = (
                self.values['sample'].strip()
                if 'sample' in self.data.columns and self.values['sample'].strip() != ''
                else 'core_sample'
            )

            # Mass column
            self.mass = (
                Quantity(self.values['mass'], self.data.columns['mass'].unit)
                if 'mass' in self.data.columns and self.values['mass'] is not None else None
            )

            # Radius column
            self.radius = (
                Quantity(self.values['radius'], self.data.columns['radius'].unit)
                if 'radius' in self.data.columns and self.values['radius'] is not None else None
            )

            # Radial velocity shift column
            if 'rv_shift' in self.data.columns and self.values['rv_shift'] is not None:
                default_unit = Unit('km/s')
                unit = self.data.columns['rv_shift'].unit

                # Radial velocity shift error
                if 'rv_shift_error' in self.data.columns:
                    error = self.values['rv_shift_error']
                    error_unit = self.data.columns['rv_shift_error'].unit
                else:
                    error = None
                    error_unit = None

                # Unit conversion
                if self.values['rv_shift'] is not None:
                    self.rv_shift = Quantity(
                        self.values['rv_shift'],
                        default_unit if unit is None else unit,
                        0.0 if error is None else error,
                        default_unit if error_unit is None else error_unit
                    ).to()

            # Radial velocity shift based on spectral type
            if self.spectral_type is not None:
                self.spectral_type, rv_shift, rv_shift_error = self.data.find_rv_shift(
                    self.spectral_type
                )
                self.rv_shift = Quantity(rv_shift, 'km/s', rv_shift_error).to()

            # Radial velocity shift from mass and radius columns
            elif self.mass is not None and self.radius is not None:
                rv_shift = c.value / 1000 * (
                    (
                        1 - 2 * G.value * self.mass.value * M_sun.value / (
                            c.value**2 * self.radius.value * R_sun.value
                        )
                    )**(-0.5) - 1
                )
                self.rv_shift = Quantity(rv_shift, Unit('km/s'), 0.0).to()

            # Configuration radial velocity shift otherwise
            else:
                 self.rv_shift = self.data.series.rv_shift

            # Position columns and unit conversion
            self.position = Quantity(
                [self.values[label] for label in self.data.position_variables.keys()],
                [self.data.columns[label].unit for label in self.data.position_variables.keys()],
                [self.values[label] for label in self.data.position_error_variables.keys()],
                [
                    self.data.columns[label].unit
                    for label in self.data.position_error_variables.keys()
                ]
            ).to()

            # Velocity columns and unit conversion
            self.velocity = Quantity(
                [self.values[label] for label in self.data.velocity_variables.keys()],
                [self.data.columns[label].unit for label in self.data.velocity_variables.keys()],
                [self.values[label] for label in self.data.velocity_error_variables.keys()],
                [
                    self.data.columns[label].unit
                    for label in self.data.velocity_error_variables.keys()
                ]
            ).to()

            # Check for non-numerical values in position and velocity
            self.valid = (~np.isnan(self.position.values) & ~np.isnan(self.velocity.values)).all()
