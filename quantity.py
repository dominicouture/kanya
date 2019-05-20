# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" quantity.py: Defines Quantity class to handle n dimension values with unit conversions and error
    handling, and Coordinate class to handle coordinates transformation, and a montecarlo function.
"""

import numpy as np
from astropy import units as un

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Quantity:
    """ Contains a value and unit, its associated error and unit if needed. Error is converted
        to value's unit is error unit don't match with value unit.
    """

    def __init__(
            self, values, units=None, errors=None, error_units=None,
            parent=None, index=None, **optional):
        """ Initializes a Quantity object with its values, errors and their units. """

        # Import of values
        if type(values) in (int, float, np.float64):
            values = [values]
        if type(values) in (tuple, list, np.ndarray):
            self.values = np.squeeze(np.array(values, dtype=float))
        else:
            raise TypeError("{} is not a supported type "
                "('int, float, tuple, list or ndarray') for values.".format(type(values)))
        self.shape = self.values.shape
        self.ndim = len(self.shape)

        # Import of units
        if units is None:
            units = ['']
        elif type(units) in (
                str, un.core.PrefixUnit, un.core.CompositeUnit,
                un.core.IrreducibleUnit, un.core.Unit, Unit):
            units = [units]
        if type(units) in (tuple, list, np.ndarray):
            try:
                self.units = np.full(self.shape, np.squeeze(np.vectorize(Unit.get)(units)))
            except ValueError:
                raise ValueError(
                    "Units ({}) cannot be broadcast to the shape of values ({}).".format(
                        units, self.values.shape))
        else:
            raise TypeError("{} is not a supported type "
                "(str, *Unit, tuple, list or ndarray) for units.".format(type(units)))
        self.physical_types = np.vectorize(lambda unit: unit.physical_type)(self.units)

        # Import of errors
        if errors is None:
            errors = [0.0]
        elif type(errors) in (int, float, np.float64):
            errors = [errors]
        if type(errors) in (tuple, list, np.ndarray):
            try:
                self.errors = np.full(self.shape, np.squeeze(errors), dtype=float)
            except ValueError:
                raise ValueError("Errors ({}) cannot be broadcast "
                    "to the shape of values ({}).".format(errors, self.values.shape))
        else:
            raise TypeError("{} is not a supported type "
                "(int, float, tuple, list or ndarray) for errors.".format(type(errors)))

        # Import of error units
        if error_units is None:
            error_units = units
        elif type(error_units) in (
                str, un.core.PrefixUnit, un.core.CompositeUnit,
                un.core.IrreducibleUnit, un.core.Unit, Unit):
            error_units = [error_units]
        if type(error_units) in (tuple, list, np.ndarray):
            try:
                self.error_units = np.full(self.shape, np.squeeze(np.vectorize(Unit.get)(error_units)))
            except ValueError:
                raise ValueError(
                    "Error units ({}) cannot be broadcast to the shape of values ({}).".format(
                        units, self.values.shape))
        else:
            raise TypeError("{} is not a supported type "
                "(str, *Unit, tuple, list or ndarray) for error units.".format(type(error_units)))
        self.error_physical_types = np.vectorize(lambda unit: unit.physical_type)(self.error_units)

        # Conversion of errors into value units
        # !!! Check for physical type first, then convert, like in self.to())
        if not np.equal(self.units, self.error_units).all():
            try:
                self.errors = np.vectorize(
                    lambda errors, units, error_units: errors * error_units.to(units)
                )(self.errors, self.units, self.error_units)
                self.error_units = self.units
            except un.core.UnitConversionError:
                raise un.core.UnitConversionError(
                    "Value units and error units have incompatible physical types: "
                    " {} and {}.".format(self.physical_types, self.error_physical_types))

        # Optional parameters
        vars(self).update(optional.copy())

        # Indexing
        if parent is not None:
            self.parent = parent
            self.index = index
        else:
            self.parent = None
            self.index = None

    def __repr__(self):
        """ Create a string with the value, error and unit of the Quantity object. """

        def reduce(array):
            """ Remove all but one value of all dimension if all values a given dimension are
                the same.
            """

            for i in range(array.ndim):
                swapped_array = np.swapaxes(array, 0, i) if i != 0 else array
                if len(np.unique(swapped_array, axis=0)) == 1:
                    swapped_array = np.array([swapped_array[0,...]])
                array = np.swapaxes(swapped_array, 0, i) if i != 0 else swapped_array
            return array

        def flatten(array):
            """ Returns the value of single-value arrays or a list version. """

            from json import dumps
            return array.flatten()[0] if np.equal(np.array(array.shape), 1).all() \
                else dumps(array.tolist()).replace('"', '').replace(' ', '').replace(',', ', ')

        return '({} ± {}) {}'.format(
            flatten(self.values),
            flatten(reduce(self.errors)),
            flatten(reduce(np.vectorize(lambda unit: unit.to_string())(self.units))))

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
        """ Tests whether values in self are lower than values in other. """

        return self.values < other.values * self.compare(other)[1]

    def __le__(self, other):
        """ Tests whether values in self are lower than or equal to values in other. """

        return self.values <= other.values * self.compare(other)[1]

    def __eq__(self, other):
        """ Tests whether values in self are equal to values in other (values and errors). """

        shape, factors = self.compare(other)

        return np.vectorize(
            lambda value, error: True if value and error else False
        )(self.values == other.values * factors, self.errors == other.errors * factors)

    def __ne__(self, other):
        """ Tests whether values in self are not equal to values in other (values and errors). """

        return ~(self == other)

    def __ge__(self, other):
        """ Tests whether values in self are greater than or equal to values in other. """

        return self.values >= other.values * self.compare(other)[1]

    def __gt__(self, other):
        """ Tests whether values in self are greater than or equal to values in other. """

        return self.values > other.values * self.compare(other)[1]

    def __add__(self, other):
        """ Defines the addition for a Quantity. Both arguments have to be Quantities. """

        shape, factors = self.compare(other)

        return Quantity(
            self.values + other.values * factors, np.full(shape, self.units),
            np.vectorize(lambda x, y: np.linalg.norm((x, y)))(self.errors, other.errors * factors))

    def __sub__(self, other):
        """ Computes the substraction for a Quantity. Both arguments have to be Quantities. """

        return self + -other

    def __mul__(self, other):
        """ Computes the product for a Quantity. The second argument can be an int or
            a float, or a nd.array or a Quantity of a shape that can be broadcast to self.
        """

        # Check if other is a Quantity object
        if type(other) != type(self):
            other = Quantity(other)

        # Check the shape of self and other can be broadcast together
        try:
            shape = np.broadcast(self.values, other.values).shape
        except ValueError:
            raise ValueError("Quantities with shapes {} and {} cannot be "
                "broadcast together.".format(self.shape, other.shape))

        # Conversion factors between self and other
        self_units = np.full(shape, self.units)
        other_units = np.full(shape, other.units)
        factors = np.vectorize(
            lambda self_unit, other_unit: other_unit.to(self_unit) \
                if self_unit.physical_type == other_unit.physical_type else 1.0)(
                self_units, other_units)

        # Calculation of multiplication values
        mul_values = self.values * (other.values * factors)

        return Quantity(
            mul_values,
            self_units * np.vectorize(
                lambda self_unit, other_unit: self_unit \
                    if self_unit.physical_type == other_unit.physical_type else other_unit
            )(self_units, other_units),
            np.vectorize(
                lambda x, y: np.linalg.norm((x, y))
            )(self.errors / self.values, (other.errors / other.values)) * mul_values)

    def __truediv__(self, other):
        """ Computes the division for a Quantity. The second argument can be an int or a
            float, or nd.array or a Quantity of a shape that can be broadcast to self.
        """

        # Check if other is a Quantity object
        if type(other) != type(self):
            other = Quantity(other)

        # Check the shape of self and other can be broadcast together
        try:
            shape = np.broadcast(self.values, other.values).shape
        except ValueError:
            raise ValueError("Quantities with shapes {} and {} "
                "cannot be broadcast together.".format(self.shape, other.shape))

        # Conversion factors between self and other
        self_units = np.full(shape, self.units)
        other_units = np.full(shape, other.units)
        factors = np.vectorize(
            lambda self_unit, other_unit: other_unit.to(self_unit) \
                if self_unit.physical_type == other_unit.physical_type else 1.0)(
                self_units, other_units)

        # Calculation of division values
        div_values = self.values / (other.values * factors)

        return Quantity(
            div_values,
            self_units / np.vectorize(
                lambda self_unit, other_unit: self_unit \
                    if self_unit.physical_type == other_unit.physical_type else other_unit
            )(self_units, other_units),
            np.vectorize(
                lambda x, y: np.linalg.norm((x, y))
            )(self.errors / self.values, (other.errors / other.values)) * div_values)

    def __floordiv__(self, other):
        """ Computes the floor division of a Quantity. """

        truediv = self / other

        return Quantity(np.floor(truediv.values), truediv.units, truediv.errors)

    def __mod__(self, other):
        """ Computes the remain of the floor division. """

        truediv = self / other
        floordiv = self // other

        return Quantity(truediv.values - floordiv.values, truediv.units, truediv.errors)

    def __pow__(self, other):
        """ Computes the raise to the power for a Quantity object. The second argument can be an
            integer or a float, or nd.array or a Quantity object of a shape that can be broadcast
            to self.
        """

        # Check if other is a Quantity object
        if type(other) != type(self):
            other = Quantity(other)

        # Check if exponant is dimensionless
        elif not np.equal(other.units, u.Unit('')).all():
            raise ValueError("Exponant must be dimensionless.")

        # Check the shape of self and other can be broadcast together
        try:
            shape = np.broadcast(self.values, other.values).shape
        except ValueError:
            raise ValueError("Terms with shapes {} and {} cannot be broadcast together.".format(
                self.shape, other.shape))

        # Fix error calculation !!!
        return Quantity(
            self.values**other.values,
            self.units**other.values,
            np.vectorize(lambda x, y: np.linalg.norm((x, y)))(
                self.values * other.values * self.errors,
                (self.values**other.values) * np.log(self.values)))

    def __matmul__(self, other):
        """ Computes the scalar of matrix product of self and other. """

        return self

    def __contains__(self, other):
        """ Determines whether other is in self. """

        return True

    def count(self, other):
        """ Counts the number of occurrences of other in a. """

        return 0.0

    def where(self, other):
        """ Determines the index of the occurrences of other in self. """

        return 0

    def concatenate(self, other):
        """ Concatenates two Quantites together. """

        return self

    def remove(self, other):
        """ Removes other from self. """

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
                self.values[index], self.units[index], self.errors[index], parent=self, index=index)
        except IndexError:
            raise IndexError(
                'Index {} is out of range of axis of size {}.'.format(index, len(self.values)))

    def __setitem__(self, index, item):
        """ Modify the specified value in a Quantity object with the item.
            !!! Add slicing support !!!
        """

        if type(index) != int:
            raise TypeError('Can only index with integer, not {}.'.format(type(index)))
        if type(item) != type(self):
            item = Quantity(item, self.units[index], self.errors[index])

        try:
            self.values[index] = item.values
            self.units[index] = item.units
            self.errors[index] = item.errors
        except IndexError:
            raise IndexError(
                'Index {} is out of range of axis of size {}.'.format(index, len(self.values)))

        if self.parent is not None:
            self.parent[self.index] = self

    def compare(self, other):
        """ Determines the shape of the broadcast array and conversion factors to compare
            Quantities.
        """

        # Check if other is a Quantity object
        if type(other) != type(self):
            raise TypeError("Cannot compare {} to {}.".format(type(other), type(self)))

        # Check the shape of self and other can be broadcast together
        try:
            shape = np.broadcast(self.values, other.values).shape
        except ValueError:
            raise ValueError(
                "Quantities with shapes {} and {} cannot be broadcast together.".format(
                    self.shape, other.shape))

        # Check if physical types of self and other match
        if not (self.physical_types == other.physical_types).all():
            raise ValueError(
                "Quantities have incompatible physical types: {} and {}.".format(
                    self.physical_types, other.physical_types))

        # Conversion factors between self and other
        factors = np.vectorize(
            lambda self_unit, other_unit: other_unit.to(self_unit))(
                np.full(shape, self.units), np.full(shape, other.units))

        return shape, factors

    def to(self, units=None):
        """ Converts Quantity object into new units or default units if none are given. """

        # Default units for physical types if no units are given.
        if units is None:
            from coordinate import System
            units = np.vectorize(lambda unit: System.default_units[unit.physical_type].unit \
                if unit.physical_type in System.default_units.keys() else unit)(self.units)
            factors = np.vectorize(lambda self_unit, unit: self_unit.to(unit))(self.units, units)

        # Import of units
        else:
            if type(units) in (
                    str, un.core.PrefixUnit, un.core.CompositeUnit,
                    un.core.IrreducibleUnit, un.core.Unit, Unit):
                units = [units]
            if type(units) in (tuple, list, np.ndarray):
                try:
                    units = np.full(self.units.shape, np.vectorize(Unit.get)(units))
                except ValueError:
                    raise ValueError(
                        "Units with shapes {} and {} cannot be broadcast together.".format(
                            self.values.shape, units.shape))
            else:
                raise TypeError("{} is not a supported type "
                    "(str, *Unit, tuple, list or ndarray) for units.".format(type(units)))

            # Check if physical types of self and units match
            units_physical_types = np.vectorize(lambda unit: unit.physical_type)(units)
            if not (self.physical_types == units_physical_types).all():
                raise ValueError(
                    "Units have incompatible physical types: {} and {}.".format(
                        self.physical_types, units_physical_types))

            # Conversion factors between self and other
            factors = np.vectorize(lambda self_unit, unit: self_unit.to(unit))(
                np.full(units.shape, self.units), units)

        return self.new(Quantity(self.values * factors, units, self.errors * factors))

    def new(self, new):
        """ Transferts all optional parameters from the old self and returns a new value. """

        optional = {key: vars(self)[key] for key in filter(
            lambda key: key not in vars(new), vars(self).keys())}
        vars(new).update(optional)

        return new

class Unit():
    """ Defines a unit or array of units matching one or several Astropy units.Unit objects.
        A Unit also has a unique label and name, and physical type.
    """

    def __init__(self, unit, name=None):
        """ Initializes a Unit from a NoneType, string, tuple, list, np.ndarray, astropy
            un.core.PrefixUnit, un.core.CompositeUnit, un.core.IrreducibleUnit, un.core.Unit or
            Unit.
        """

        # Initialization from a string
        if type(unit) == str:
            self.label = unit

        # Initialization from a Unit object
        elif type(unit) == self:
            self.label = unit.label

        # un.Unit object and physical type
        self.unit = un.Unit(self.label)
        self.physical_type = self.unit.physical_type

        # Unit name
        self.name = name if name is not None and type(name) == str else self.unit.to_string()

    def __repr__(self):

        return self.label

    def get(self):
        """ Allow for handling of Unit type. """

        if type(self) == Unit:
            return self.unit
        else:
            return un.Unit(self)

def montecarlo(function, values, errors, n=200):
    """ Wraps a function to output both its value and errors, calculated with Monte Carlo
        algorithm with n iterations. The inputs and outputs are Quantity objects with values,
         units and errors.
    """

    values, errors = [i if type(i) in (tuple, list, np.ndarray) else [i] for i in (values, errors)]
    outputs = function(*values)
    output_errors = np.std(
        np.array([function(*arguments) for arguments in np.random.normal(
            values, errors, (n, len(values)))]),
        axis=0)

    return (outputs, output_errors)

def montecarlo2(function, values, errors, n=10000):
    """ Wraps a function to output both its value and errors, calculated with Monte Carlo
        algorithm with n iterations. The inputs and outputs are Quantity objects with values,
        units and errors.
    """

    outputs = function(values)
    output_errors = np.std(
        np.array([function(arguments) for arguments in np.random.normal(
            values, errors, (n, len(values)))]),
        axis=0)

    return (outputs, output_errors)
