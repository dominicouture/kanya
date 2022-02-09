# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" quantity.py: Defines Quantity and Unit classes to handle n dimension values with unit
    conversions and error handling,
"""

import numpy as np
from decimal import Decimal
from astropy import units as u
from astropy.constants import c, G, M_sun, R_sun
from Traceback.tools import squeeze, broadcast, full

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Quantity:
    """ Contains a value and unit, its associated error and unit if needed. Error is converted
        to value's unit if error unit don't match with value unit.
    """

    def __init__(
            self, values, units=None, errors=None, error_units=None,
            parent=None, index=None, **optional):
        """ Initializes a Quantity object with its values, errors and their respective units.
            Errors are converted into the units of values.
        """

        # Values import
        if type(values) in (int, float, np.float64):
            values = [values]
        if type(values) in (tuple, list, np.ndarray):
            self.values = squeeze(np.array(values, dtype=float))
        else:
            raise TypeError("{} is not a supported type "
                "('int, float, tuple, list or ndarray') for values.".format(type(values)))
        self.shape = self.values.shape
        self.ndim = self.values.ndim

        # Units import
        self.units = Unit(units, shape=self.shape)
        self.physical_types = self.units.physical_types

        # Errors import
        if errors is None:
            errors = [0.0]
        elif type(errors) in (int, float, np.float64):
            errors = [errors]
        if type(errors) in (tuple, list, np.ndarray):
            self.errors = np.array(full('Errors', self.shape, squeeze(errors)), dtype=float)
        else:
            raise TypeError("{} is not a supported type "
                "(int, float, tuple, list or ndarray) for errors.".format(type(errors)))

        # Error units import
        self.error_units = self.units if error_units is None else Unit(error_units, shape=self.shape)
        self.error_physical_types = self.error_units.physical_types

        # Conversion of errors into value units
        if not np.equal(self.units.units, self.error_units.units).all():
            factors = self.error_units.compare(self.units)
            self.errors *= factors
            self.error_units = self.units

        # Singular values if only one value is present
        if self.shape == (1,):
            self.value = float(self.values[0])
            self.unit = self.units.units[0]
            self.physical_type = str(self.physical_types[0])
            self.error = float(self.values[0])
            self.error_unit = self.unit
            self.error_physical_type = self.physical_type

        # Optional parameters
        vars(self).update(optional.copy())

        # Indexing
        if parent is not None:
            self.parent = parent
            self.index = index
        else:
            self.parent = None
            self.index = None

    def find_precisions(self):
        """ Finds the precision of a value, that is, the exponent of its last significant digit.
            If no error is provided, the last significant digit of the value is used. Otherwise,
            the first significant digit of the error is used.
        """

        # Find indexes with and without errors
        zeros = self.errors == 0.0
        nonzeros = ~zeros

        # Compute precisions and errors for values without errors
        self.precisions = np.zeros_like(self.errors, dtype=int)
        self.precisions[zeros] = -np.vectorize(
            lambda x: Decimal(str(x)).as_tuple().exponent, otypes=[np.int])(self.values[zeros])
        self.errors[zeros] = np.float_power(10., -self.precisions[zeros])

        # Compute precisions for values with errors
        self.precisions[nonzeros] = -np.array(np.floor(np.log10(self.errors[nonzeros])), dtype=int)

    def apply_precision(self):
        """ Reduces the number of significant numbers based on the precison. """

        # Find precisions, round errors, update precisions and errors, round values
        self.find_precisions()
        self.errors = np.vectorize(np.round)(self.errors, self.precisions)
        self.find_precisions()
        self.errors = np.vectorize(np.round)(self.errors, self.precisions)
        self.values = np.vectorize(np.round)(self.values, self.precisions)

        if self.shape == (1,):
            self.value = float(self.values[0])
            self.error = float(self.errors[0])
            self.precision = int(self.precisions[0])

    def get_LaTeX(self):
        """ Creates a LaTeX compatible string. """

        self.apply_precision()

        return f'${self.value:.{self.precision}f} \\pm {self.error:.{self.precision}f}$'

    def convert_deg(self, ra):
        """ Converts a right ascension or declination value from degree to HH:MM:SS.SS or
            DD:MM:SS.SS.
        """

        hh = int(self.value / (15 if ra else 1))
        mm = int(self.value * 60 % 60)
        ss = float(self.value * 3600 % 60)

        return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>5.2f}'


    def __repr__(self):
        """ Creates a string with the values, errors and units of the Quantity object. """

        def reduce(array):
            """ Removes all but one value in an 'array' for all dimension if all values in a given
                dimension are the same.
            """

            for i in range(array.ndim):
                swapped_array = np.swapaxes(array, 0, i) if i != 0 else array
                if len(np.unique(swapped_array, axis=0)) == 1:
                    swapped_array = np.array([swapped_array[0,...]])
                array = np.swapaxes(swapped_array, 0, i) if i != 0 else swapped_array
            return array

        def flatten(array):
            """ Returns the value of single-value 'array' or a list version. """

            from json import dumps
            return array.flatten()[0] if np.equal(np.array(array.shape), 1).all() \
                else dumps(array.tolist()).replace('"', '').replace(' ', '').replace(',', ', ')

        return '({} ± {}) {}'.format(
            flatten(self.values),
            flatten(reduce(self.errors)),
            flatten(reduce(np.vectorize(
                lambda unit: unit.to_string().replace(' ', ''))(self.units.units))))

    def __bool__(self):
        if len(self) == 1:
            return False if self.values.flatten()[0] == 0.0 and self.errors.flatten()[0] == 0.0 \
                else True
        else:
            raise ValueError("The truth value of an Quantity with more than one value "
                "is ambiguous. Use Quantity.any() or Quantity.all()")

    def all(self):
        """ Returns True if all values are non zero, False otherwise. """

        return np.vectorize(
            lambda value, error: True if value or error else False
        )(self.values, self.errors).all()

    def any(self):
        """ Returns True if any value is non zero, False otherwise. """

        return np.vectorize(
            lambda value, error: True if value or error else False
        )(self.values, self.errors).any()

    def __pos__(self):
        """ Computes the positve Quantity. """

        return self

    def __neg__(self):
        """ Computes the negative Quantity. """

        return Quantity(-1 * self.values, self.units, self.errors)

    def __abs__(self):
        """ Computes the absolute Quantity. """

        return Quantity(np.absolute(self.values), self.units, self.errors)

    def __round__(self, n):
        """ Computes the rounded Quantity to the nth decimal. """

        rounded_values = np.round(self.values, n)

        return Quantity(
            rounded_values, self.units, np.round(self.errors * rounded_values / self.values, n))

    def __floor__(self):
        """ Computes the floor of a Quantity. """

        return Quantity(np.floor(self.values), self.units, self.errors)

    def __ceil__(self):
        """ Computes the ceiling of a Quantity. """

        return Quantity(np.ceil(self.values), self.units, self.errors)

    def __reversed__(self):
        """ Computes the reserved of flip Quantity. """

        return Quantity(np.flip(self.values), np.flip(self.units), np.flip(self.errors))

    def __len__(self):
        """ Computes how many values are in Quantity. """

        return np.prod(self.shape)

    def __lt__(self, other):
        """ Tests whether values in self are lower than values in 'other'. """

        return self.values < other.values * self.compare(other)[1]

    def __le__(self, other):
        """ Tests whether values in self are lower than or equal to values in 'other'. """

        return self.values <= other.values * self.compare(other)[1]

    def __eq__(self, other):
        """ Tests whether values in self are equal to values in 'other' (values and errors). """

        shape, factors = self.compare(other)

        return np.vectorize(
            lambda value, error: True if value and error else False
        )(self.values == other.values * factors, self.errors == other.errors * factors)

    def __ne__(self, other):
        """ Tests whether values in self are not equal to values in 'other' (values and errors). """

        return ~(self == other)

    def __ge__(self, other):
        """ Tests whether values in self are greater than or equal to values in 'other'. """

        return self.values >= other.values * self.compare(other)[1]

    def __gt__(self, other):
        """ Tests whether values in self are greater than or equal to values in 'other'. """

        return self.values > other.values * self.compare(other)[1]

    def __add__(self, other):
        """ Computes the addition for a Quantity. Both 'self' and 'other' have to be Quantities. """

        shape, factors = self.compare(other)

        return Quantity(
            self.values + other.values * factors, self.units,
            np.vectorize(lambda x, y: np.linalg.norm((x, y)))(self.errors, other.errors * factors))

    def __sub__(self, other):
        """ Computes the substraction for a Quantity. Both 'self' and 'other' have to be
            Quantities.
        """

        return self + -other

    def __mul__(self, other):
        """ Computes the product for a Quantity. 'other' can be an int or a float, or a
            np.ndarray or a Quantity of a shape that can be broadcast to 'self'.
        """

        # Check if 'other' is a Quantity object
        if type(other) != type(self):
            other = Quantity(other)

        # Check the shape of 'self' and 'other' can be broadcast together
        shape = broadcast('Quantities', self.values, other.values)

        # Conversion factors between 'self' and 'other'
        factors = other.units.compare(self.units, shape, False)

        # Values computation
        values = self.values * other.values * factors

        # Units and errors computation
        return Quantity(
            values,
            self.units.units * np.vectorize(lambda self_unit, other_unit: self_unit \
                if self_unit.physical_type == other_unit.physical_type else other_unit)(
                    self.units.units, other.units.units),
            np.vectorize(lambda x, y: np.linalg.norm((x, y)))(
                self.errors / self.values, (other.errors / other.values)) * values)

    def __truediv__(self, other):
        """ Computes the division for a Quantity. 'other' can be an int or a float, or np.ndarray
            or a Quantity of a shape that can be broadcast to 'self'.
        """

        # Check if 'other' is a Quantity object
        if type(other) != type(self):
            other = Quantity(other)

        # Check the shape of 'self' and 'other' can be broadcast together
        shape = broadcast('Quantities', self.values, other.values)

        # Conversion factors between 'self' and 'other'
        factors = other.units.compare(self.units, shape, False)

        # Values computation
        values = self.values / (other.values * factors)

        # Units and errors computation
        return Quantity(
            values,
            self.units.units / np.vectorize(lambda self_unit, other_unit: self_unit \
                if self_unit.physical_type == other_unit.physical_type else other_unit)(
                    self.units.units, other.units.units),
            np.vectorize(lambda x, y: np.linalg.norm((x, y)))(
                self.errors / self.values, (other.errors / other.values)) * values)

    def __floordiv__(self, other):
        """ Computes the floor division of a Quantity. 'other' can be an int or a float, or
            np.ndarray or a Quantity of a shape that can be broadcast to 'self'.
        """

        truediv = self / other

        return Quantity(np.floor(truediv.values), truediv.units, truediv.errors)

    def __mod__(self, other):
        """ Computes the remain of the floor division. 'other' can be an int or a float, or
            np.ndarray or a Quantity of a shape that can be broadcast to 'self'.
        """

        truediv = self / other
        floordiv = self // other

        return Quantity(truediv.values - floordiv.values, truediv.units, truediv.errors)

    def __pow__(self, other):
        """ Computes the raise to the power for a Quantity object. 'other' can be an integer
            or a float, or np.ndarray or a Quantity object of a shape that can be broadcast
            to 'self'.
        """

        # Check if other is a Quantity object
        if type(other) != type(self):
            other = Quantity(other)

        # Check if 'other' exponant is dimensionless
        elif not np.equal(other.units.units, u.Unit('')).all():
            raise ValueError("Exponant must be dimensionless.")

        # Check the shape of 'self' and 'other' can be broadcast together
        shape = broadcast('Quantities', self.values, other.values)

        # Units and errors computation
        return Quantity(
            self.values**other.values,
            self.units.units**other.values,
            np.vectorize(lambda x, y: np.linalg.norm((x, y)))(
                self.values**(other.values - 1) * other.values * self.errors,
                self.values**other.values * np.log(self.values) * other.errors))

    def __matmul__(self, other):
        """ Computes the scalar of matrix product of 'self' and 'other'. """

        return self

    def __contains__(self, other):
        """ Determines whether 'other' is in 'self'. """

        return True

    def count(self, other):
        """ Counts the number of occurrences of 'other' in 'self'. """

        return 0

    def where(self, other):
        """ Determines the index of the occurrences of 'other' in 'self'. """

        return 0

    def concatenate(self, other):
        """ Concatenates 'self' and 'other' together. """

        return self

    def remove(self, other):
        """ Removes 'other' from 'self'. """

        return self

    def __iter__(self):
        """ Initializes the iterator. """

        self.index = -1
        return self

    def __next__(self):
        """ Returns a Quantity object with one fewer dimension. """

        if self.index < self.index - 1:
            self.index += 1
            return Quantity(
                self.values[self.index], self.units[self.index], self.errors[self.index])
        else:
            raise StopIteration

    def __getitem__(self, index):
        """ Returns a Quantity object with the specified index. !!! Add slicing support !!! """

        if type(index) != int:
            raise TypeError('Can only index with integer, not {}.'.format(type(index)))
        try:
            return Quantity(
                self.values[index], self.units.units[index],
                self.errors[index], parent=self, index=index)
        except IndexError:
            raise IndexError(
                'Index {} is out of range of axis of size {}.'.format(index, len(self.values)))

    def __setitem__(self, index, item):
        """ Modify the value located at 'index' in a Quantity object with the 'item', which can
            be a Quantity object or a np.ndarray.
            !!! Add slicing support !!!
        """

        # Check if the types of 'index' and 'item' are valid
        if type(index) != int:
            raise TypeError("Can only index with integer, not {}.".format(type(index)))
        if type(item) != type(self):
            item = Quantity(item, self.units.units[index], self.errors[index])

        # Modify the values, units and errors at the specified 'index'
        try:
            self.values[index] = item.values
            self.units.units[index] = item.units
            self.errors[index] = item.errors
        except IndexError:
            raise IndexError("Index {} is out of range of axis of size {}.".format(
                index, len(self.values)))

        # Set self.parent as self at the specified 'index'
        if self.parent is not None:
            self.parent[self.index] = self

    def compare(self, other):
        """ Computes the shape of the broadcast array, assesses whether the physical types are
            compatible and calculates conversion factors to compare Quantities.
        """

        # Check if 'other' is a Quantity object
        if type(other) != type(self):
            raise TypeError("Cannot compare {} to {}.".format(type(other), type(self)))

        # Check the shape of 'self' and 'other' can be broadcast together
        shape = broadcast('Quantities', self.values, other.values)

        # Shape and conversion factors
        return shape, self.units.compare(other.units, shape)

    def to(self, units=None):
        """ Converts a Quantity object to new units or default units if none are given. """

        # Default units per physical types if 'units' is None.
        if units is None:
            from Traceback.coordinate import System
            units = Unit(np.vectorize(lambda unit: System.default_units[unit.physical_type].unit \
                if unit.physical_type in System.default_units.keys() else unit)(self.units.units))

        # Units import
        else:
            units = Unit(units, shape=self.shape)

        # Conversion factors between 'self' and 'other'
        factors = self.units.compare(units, self.shape)

        # Quantity object initialization
        return self.new(Quantity(self.values * factors, units, self.errors * factors))

    def new(self, new):
        """ Transferts all optional parameters from the old 'self' and returns a 'new' value. """

        optional = {key: vars(self)[key] for key in filter(
            lambda key: key not in vars(new), vars(self).keys())}
        vars(new).update(optional)

        return new

class Unit():
    """ Defines a unit or array of units matching one or several Astropy units.Unit objects.
        A Unit also has unique labels and names, and physical types.
    """

    def __init__(self, units, names=None, shape=None):
        """ Initializes a Unit from a string, Unit, NoneType, or astropy units.core.PrefixUnit,
            units.core.CompositeUnit, units.core.IrreducibleUnit, units.core units.core.Unit into
            a Unit object. 'names' can also be used to define the names of units in the array and
            'shape' to define the final shape of the array.
        """

        # Units import
        if type(units) in (str, type(None), u.core.PrefixUnit, u.core.CompositeUnit,
                u.core.IrreducibleUnit, u.core.NamedUnit, u.core.Unit, np.str_, np.unicode_):
            units = [units]
        if type(units) == Unit:
            vars(self).update(vars(units))
        elif type(units) in (tuple, list, np.ndarray):
            self.units = squeeze(np.array(np.vectorize(self.to)(units), dtype=object))
        else:
            raise TypeError("{} is not a supported type (str, NoneType, astropy units, "
                "tuple, list, np.ndarray or Unit) for units.".format(type(units)))

        # Shape broadcast
        if shape is not None:
            self.units = full('Units', shape, self.units)
        self.shape = self.units.shape
        self.ndim = self.units.ndim

        # Physical types
        self.physical_types = np.vectorize(lambda unit: unit.physical_type)(self.units)

        # Labels
        self.labels = np.vectorize(lambda unit: unit.to_string().replace(' ', ''))(self.units)

        # Names
        if type(names) == str:
            names = [names]
        self.names = full('Names', self.shape, squeeze(np.array(names))) if names is not None \
            else self.labels

        # Singular values if only one unit is present
        if self.shape == (1,):
            self.unit = self.units[0]
            self.physical_type = str(self.physical_types[0])
            self.label = str(self.labels[0])
            self.name = str(self.names[0])

    def __repr__(self):
        """ Returns the label of a unit. """

        return self.label if self.shape == (1,) else str(self.labels)

    def __eq__(self, other):
        """ Tests whether self is not the equal to other. """

        return vars(self) == vars(other)

    def compare(self, other, shape=None, types=True):
        """ Compare two Unit objects for compatible shapes and physical types and returns
            conversion factors.
        """

        # Check if the shape of 'self' and 'other' can be broadcast together
        if shape is None:
            shape = broadcast('Units', self.units, other.units)

        # Check if physical types of 'self' and 'other' match, if needed
        if types:
            if not (self.physical_types == other.physical_types).all():
                raise ValueError("Units have incompatible physical types: {} and {}.".format(
                    self.physical_types, other.physical_types))

        # Conversion factors from 'self' to 'other' with matching physical types
            return np.vectorize(lambda self_unit, other_unit: self_unit.to(other_unit))(
                self.units, other.units)

        # Conversion factors from 'self' to 'other' without matching physical types
        else:
            return np.vectorize(lambda self_unit, other_unit: self_unit.to(other_unit) \
                if self_unit.physical_type == other_unit.physical_type else 1.0)(
                    self.units, other.units)

    def to(self, unit):
        """ Uses the astropy units.Unit function to convert 'unit' into an astropy.units object. """

        if type(unit) == Unit:
            unit = unit.unit
        try:
            return u.Unit('' if unit is None else unit)
        except Exception as error:
            error.args = ("'{}' could not be converted to an astropy.units object.".format(unit),)
            raise
