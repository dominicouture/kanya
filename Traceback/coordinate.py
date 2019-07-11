# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" coordinate.py: Defines System class to handle a coordinate system initialization with a set
    of position and velocity variables, default and usual units, axis, and origins. Variable, Axis
    and Origin subclasses are definied as well. Individual coordinates are defined by a
    Coordinate class which includes an initialization method and tranformation methods.
"""

import numpy as np
from math import cos, sin, asin, atan2, pi as π, degrees, radians
from Traceback.quantity import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class System():
    """ Defines a coordinate system object with variables, default units, usual units, axes,
        origins, and Variable, Axis and Origin classes.
    """

    def __init__(self, name: str, position: tuple, velocity: tuple):
        """ Initializes a System from position and velocity tuples with 3 Variables objects. """

        # Name and index
        self.name = name

        # Values
        self.position = [self.variables[label] for label in position]
        self.velocity = [self.variables[label] for label in velocity]

        # Errors
        self.position_error = [self.variables['Δ' + label] for label in position]
        self.velocity_error = [self.variables['Δ' + label] for label in velocity]

    # Default units per physical type
    default_units = {
        'time': Unit('Myr', 'megayear'),
        'length': Unit('pc', 'parsec'),
        'speed': Unit('pc/Myr', 'parsec per megayear'),
        'angle': Unit('rad', 'radian'),
        'angular speed': Unit('rad/Myr', 'radian per megayear')}

    # Usual units used for observalbes per physical type and magnitude
    usual_units = {
        'speed': Unit('km/s', 'kilometer per second'),
        'angle': Unit('deg', 'degree'),
        'small angle': Unit('mas', 'milliarcsecond'),
        'angular speed': Unit('mas/yr', 'milliarcsecond per year')}

    class Axis():
        """ Defines a Coordinate system axis. """

        def __init__(self, name):
            """ Initializes an Axis object. """

            # Initialization
            self.name = name

    # Coordinate system axes
    axes = {axis.name: axis for axis in (Axis('galactic'), Axis('equatorial'))}

    class Origin(Axis):
        """ Defines a Coordinate system origin. """

        pass

    # Coordinate system origins
    origins = {origin.name: origin for origin in (Origin('sun'), Axis('galaxy'))}

    class Variable():
        """ Defines a Variable object and required variables from all systems. """

        def __init__(self, label, name, unit, usual_unit=None):
            """ Initializes a Variable from a name, label and Unit object. """

            # Initialization
            self.label = label
            self.name = name
            self.unit = unit
            self.physical_type = unit.physical_type
            self.usual_unit = self.unit if usual_unit is None else usual_unit

    # Variables
    variables = {variable.label: variable for variable in (

        # Castesian coordinate system variables
        Variable('x', 'y position', default_units['length']),
        Variable('y', 'y position', default_units['length']),
        Variable('z', 'z position', default_units['length']),
        Variable('u', 'u velocity', default_units['speed'], usual_units['speed']),
        Variable('v', 'v velocity', default_units['speed'], usual_units['speed']),
        Variable('w', 'w velocity', default_units['speed'], usual_units['speed']),

        # Cylindrical coordinate system variable
        Variable('θ', 'galactic angle', default_units['angle'], usual_units['angle']),
        Variable('μθ', 'galactic angle motion', default_units['angular speed'],
            usual_units['angular speed']),

        # Spherical coordinate system variables
        Variable('r', 'distance', default_units['length']),
        Variable('δ', 'declination', default_units['angle'], usual_units['angle']),
        Variable('α', 'right ascension', default_units['angle'], usual_units['angle']),
        Variable('rv', 'radial velocity', default_units['speed'], usual_units['speed']),
        Variable('μδ', 'declination proper motion', default_units['angular speed'],
            usual_units['angular speed']),
        Variable('μα', 'right ascension proper motion', default_units['angular speed'],
            usual_units['angular speed']),

        # Observables coordinate system variables
        Variable('p', 'parallax', default_units['angle'], usual_units['small angle']),
        Variable('μαcosδ', 'right ascension proper motion * cos(declination)',
            default_units['angular speed'], usual_units['angular speed']))}

    # Error variables
    for label, variable in variables.copy().items():
        variables['Δ' + label] = Variable(
            'Δ' + label, variable.name + ' error', variable.unit, variable.usual_unit)

# Coordinate systems
systems = {system.name: system for system in (
    System('cartesian', ('x', 'y', 'z'), ('u', 'v', 'w')),
    System('cylindrical', ('r', 'θ', 'z'), ('rv', 'μθ', 'w')),
    System('spherical', ('r', 'δ', 'α'), ('rv', 'μδ', 'μα')),
    System('observables', ('p', 'δ', 'α'), ('rv', 'μδ', 'μαcosδ')))}

class Coordinate:
    """ Contains the values and related methods of a coordinate, including its position, velocity,
        their errors and units, and methods to transform a coordinate from one system to another.
    """

    # J2000.0 Galactic-equatorial rotation matrix from Liu et al. (2018) 1010.3773
    # !!! This part should be moved to Axis !!!
    germ = np.array([
        [-0.054875657707, -0.873437051953, -0.483835073621],
        [ 0.494109437203, -0.444829721222,  0.746982183981],
        [-0.867666137554, -0.198076337284,  0.455983813693]])

    # J2000.0 Equatorial right ascension (α) and declination (δ) of the Galactic North (b = 90°)
    # from Liu et al. (2018) 1010.3773
    α_north = radians(192.859477875)
    δ_north = radians(27.1282524167)

    # Cosine and sine of δ_north
    cos_δ_north = cos(δ_north)
    sin_δ_north = sin(δ_north)

    # J2000.0 Galactic longitude (l) of the Celestial North pole (δ = 90°)
    # from Liu et al. (2018) 1010.3773
    l_north = radians(122.931925267)

    # Parallax conversion from radians to parsecs constant
    k = π / 648000

    def __init__(self, position, velocity=None):
        """ Initializes a Coordinate object from a Quantity objects representing n positions
            (shape = (n, 3)) and optionnally n corresponding velocities (shape = (n, 3)).
            Position and velocity must be broadcast together and can either be observables, or
            cartesian, spherical or cylindrical coordinate.
        """

        # Import of position
        if type(position) != Quantity:
            raise TypeError(
                "Position must be a Quantity of shape (n, 3)), not {}".format(type(position)))
        elif position.ndim in (1, 2) and position.shape[-1] == 3:
            position = position.to()
        else: # !!! changer pour n'importe ndim si shape[-1] == 3 !!!
            raise ValueError("Position must have a shape of (n, 3), not {}".format(position.shape))

        # Import of velocity
        if velocity is None:
            velocity = Quantity((0.0, 0.0, 0.0), 'pc/Myr')
        elif type(velocity) != Quantity:
            raise TypeError("Velocity must be a Quantity of shape (n, 3)) or None, not {}".format(
                type(velocity)))
        elif velocity.ndim in (1, 2) and velocity.shape[-1] == 3:
            velocity = velocity.to()
        else:
            raise ValueError("Velocity must have a shape of (n, 3), not {}".format(position.shape))

        # Check if the shape of the position and velocity can be broadcast together
        try:
            shape = np.broadcast(position.values, velocity.values).shape
        except ValueError:
            raise ValueError(
                "Position and velocity with shapes {} and {} cannot be broadcast together.".format(
                    position.shape, velocity.shape))
        if position.shape != shape: # !!! np.full() Doesn't work on quantity object !!!
            position = np.full(shape, position)
        if velocity.shape != shape:
            velocity = np.full(shape, velocity)

        # Conversion of position in cartesian or spherical coordinate system
        # Slice the input arrays in three arrays for each arguent. Array for x values, y values, z values.
        if (position.physical_types == np.array(['length', 'length', 'length'])).all():
            self.position_xyz = position
            self.position_rδα = None
        elif (velocity.physical_types == np.array(['length', 'angle', 'angle'])).all():
            self.position_rδα = position
            self.position_xyz = None
        else:
            raise ValueError("Position physical types ({}) don't fit a coordinate "
                "in a cartesian of spherical system.".format(position.physical_types))

        # Conversion of velocity in cartesian or spherical coordinate system
        if (velocity.physical_types == np.array(['speed', 'speed', 'speed'])).all():
            self.velocity_uvw = velocity
            self.velocity_rvμδμα = None
        elif (velocity.physical_types == np.array(['speed', 'angular speed', 'angular speed'])).all():
            self.velocity_rvμδμα = velocity
            self.velocity_uvw = None
        else:
            raise ValueError("Velocity physical types ({}) don't fit a coordinate "
                "in a cartesian of spherical system.".format(velocity.physical_types))

        # System and axis
        self.system = None
        self.axis = None

    def to(self, system=None, axis=None):
        """ Converts a Coordinate object from its original coordinate system, axis and origin to
            a new coordinate system, axis and origin by wrapping conversion functions together.
        """

        pass

def xyz_to_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ cartesian coordinate position vector (pc) to a rδα, distance (r; pc),
        declination (δ, DEC; rad) and right ascension (α, RA; rad), spherical coordinate position
        vector (observables), along with measurement errors. 'x', 'y' and 'z' can't all be null.
    """

    # Norm calculation
    norm_2, norm_xy_2 = (x**2 + y**2 + z**2), (x**2 + y**2)
    norm, norm_xy = norm_2**0.5, norm_xy_2**0.5

    # Distance and angles calculation
    values = np.array((norm, asin(z / norm), atan2(y, x) + (2 * π if y < 0.0 else 0.0)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of r: dr/dx, dr/dy and dr/dz
            (x / norm, y / norm, z / norm),

            # Partial derivatives of δ: dδ/dx, dδ/dy and dδ/dz
            (-x * z / (norm_2 * norm_xy),  -y * z / (norm_2 * norm_xy),  norm_xy / norm_2),

            # Partial derivatives of α: dα/dx, dα/dy and dα/dz
            (-y / norm_xy_2, x / norm_xy_2, 0.0))
        )**2, np.array((Δx, Δy, Δz))**2)**0.5
        return values, errors

def rδα_to_xyz(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Converts a rδα , distance (r; pc), declination (δ, DEC; rad) and right ascension
        (α, RA; rad), spherical coordinate position vector (observables) to a XYZ cartesian
        coordinate position vector (pc), along with measurement errors.
    """

    # Cosine and sine calculation
    cos_δ, sin_δ, cos_α, sin_α = cos(δ), sin(δ), cos(α), sin(α)

    # Position calculation
    values = np.array((r * cos_δ * cos_α, r * cos_δ * sin_α, r * sin_δ))

    # Errors calculation
    if not np.array((Δr, Δδ, Δα)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of x: dx/dr, dr/dδ and dx/dα
            (cos_δ * cos_α, r * sin_δ * cos_α, -r * cos_δ * sin_α),

            # Partial derivatives of y: dy/dr, dy/dδ and dy/dα
            (cos_δ * sin_α, r * sin_δ * sin_α, r * cos_δ * cos_α),

            # Partial derivatives of z: dz/dr, dz/dδ and dz/dα
            (sin_δ, -r * cos_δ, 0.0))
        )**2, np.array((Δr, Δδ, Δα))**2)**0.5
        return values, errors

def uvw_to_rvμδμα(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts a UVW cartesian coordinate velocity vector (pc/Myr) to a rvµδµα, radial velocity
        (rv; pc/Myr), declination proper motion (μδ; rad/Myr) and right ascension proper motion
        (µα; rad/Myr), spherical coordinate velocity vector (observables), along with measurement
        errors. 'x', 'y' and 'z' (pc) can't all be null.
    """

    # Norms calculation
    norm_2 = x**2 + y**2 + z**2
    norm_xy_2 = x**2 + y**2
    norm = norm_2**0.5
    norm_xy = norm_xy_2**0.5

    # Radial velocity and proper motion calculation
    values = np.array((
        ((u * x) + (v * y) + (z * w)) / norm,
        (w * norm_xy - ((u * x * z) + (v * y * z)) / norm_xy) / norm_2,
        ((v * x) - (u * y)) / norm_xy_2))

    # Errors calculation
    if not np.array((Δx, Δy, Δz, Δu, Δv, Δw)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of rv:
            # d(rv)/dx, d(rv)/dy, d(rv)/dz, d(rv)/du, d(rv)/dv and d(rv)/dw)
            ((u * (y**2 + z**2) - x * (v * y + w * z)) / norm**3,
            (v * (x**2 + z**2) - y * (u * x + w * z)) / norm**3,
            (w *   norm_xy_2   - z * (u * x + v * y)) / norm**3,
            x / norm, y / norm, z / norm),

            # Partial derivatives of μδ:
            # (d(μδ)/dx, d(μδ)/dy, d(μδ)/dz, d(μδ)/du, d(μδ)/dv and d(μδ)/dw)
            ((u * z * (2 * x**4 + x**2 * y**2 - y**2 * (y**2 + z**2))
                 + v * x * y * z * (3 * norm_xy_2 + z**2)
                 - w * x * norm_xy_2 * (norm_xy_2 - z**2)) / (norm_xy**3 * norm_2**2),
            (u * x * y * z * (3 * norm_xy_2 + z**2)
                - v * z * (x**4 + x**2 * (z**2 - y**2) - 2 * y**4)
                - w * y * norm_xy_2 * (norm_xy_2 - z**2)) / (norm_xy**3 * norm_2**2),
            (-u * x * (norm_xy_2 - z**2) - v * y * (norm_xy_2 - z**2)
                - 2 * w * z * norm_xy_2) / (norm_xy * norm_2**2),
            -(x * z) / (norm_xy * norm_2), -(y * z) / (norm_xy * norm_2), norm_xy / norm_2),

            # Partial derivatives of μα:
            # (d(μα)/dx, d(μα)/dy d(μα)/dz, d(μα)/du, d(μα)/dv and d(μα)/dw)
            ((v * (y**2 - x**2) + 2 * u * x * y) / norm_xy_2**2,
            (u * (y**2 - x**2) - 2 * v * x * y) / norm_xy_2**2,
            0.0, -y / norm_xy_2, x / norm_xy_2, 0.0))
        )**2, np.array((Δx, Δy, Δz, Δu, Δv, Δw))**2)**0.5
        return values, errors

def rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts a rvµδµα, radial velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr)
        and right ascension proper motion (µα; rad/Myr), spherical coordinate velocity vector
        (observables) to an UVW cartesian coordinate velocity vector (pc/Myr), along with
        measurement errors.
    """

    # Cosine and sine calculation
    cos_δ, sin_δ, cos_α, sin_α = cos(δ), sin(δ), cos(α), sin(α)

    # Velocity calculation
    values = np.array((
        rv * (cos_δ * cos_α) - μδ * (r * sin_δ * cos_α) - μα * (r * cos_δ * sin_α),
        rv * (cos_δ * sin_α) - μδ * (r * sin_δ * sin_α) + μα * (r * cos_δ * cos_α),
        rv * sin_δ + μδ * (r * cos_δ)))

    # Errors calculation
    if not np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of u:
            # du/dr, du/dδ, du/dα, du/d(rv), du/d(µδ) and du/d(µα)
            (-μδ * (sin_δ * cos_α) - μα * (cos_δ * sin_α),
            -rv * (sin_δ * cos_α) - μδ * (r * cos_δ * cos_α) + μα * (r * sin_δ * sin_α),
            -rv * (cos_δ * sin_α) + μδ * (r * sin_δ * sin_α) - μα * (r * cos_δ * cos_α),
            cos_δ * cos_α, -r * sin_δ * cos_α, -r * cos_δ * sin_α),

            # Partial derivatives of v:
            # dv/dr, dv/dδ, dv/dα, dv/d(rv), dv/d(µδ) and dv/d(µα)
            (-μδ * (sin_δ * sin_α) + μα * (cos_δ * cos_α),
            -rv * (sin_δ * sin_α) - μδ * (r * cos_δ * sin_α) - μα * (r * sin_δ * cos_α),
            rv * (cos_δ * cos_α) - μδ * (r * sin_δ * cos_α) - μα * (r * cos_δ * sin_α),
            cos_δ * sin_α, -r * sin_δ * sin_α, r * cos_δ * cos_α),

            # Partial derivatives of w:
            # (dw/dr, dw/dδ, dw/dα, dw/d(rv), dw/d(µδ) and dw/d(µα)
            (μδ * cos_δ, rv * cos_δ - μδ * r * sin_δ, 0.0, sin_δ, r * cos_δ, 0.0))
        )**2, np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα))**2)**0.5
        return values, errors

def equatorial_galactic_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian XYZ position or UVW velocity from a galactic to an equatorial plane.
        All arguments must have the same units.
    """

    # Coordinate calculation
    values = np.dot(Coordinate.germ, np.array((x, y, z)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(Coordinate.germ**2, np.array((Δx, Δy, Δz))**2)**0.5
        return values, errors

def galactic_equatorial_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian XYZ position or UVW velocity from an equatorial to a galactic plane.
        All arguments must have the same units.
    """

    # Coordinate calculation
    values = np.dot(Coordinate.germ.T, np.array((x, y, z)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(Coordinate.germ.T**2, np.array((Δx, Δy, Δz))**2)**0.5
        return values, errors

def galactic_xyz_equatorial_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ (pc) galactic cartesian coordinate position vector to a rδα, distance
        (r; pc), declination (δ, DEC; rad) and right ascension (α, RA; rad), equatorial spherical
        coordinate position vector (observables), along with measurement errors. 'x', 'y' and 'z'
        can't all be null.
    """

    values, errors = galactic_equatorial_xyz(x, y, z, Δx, Δy, Δz)
    return xyz_to_rδα(*values, *errors)

def equatorial_rδα_galactic_xyz(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Converts a rδα, distance (r; pc), declination (δ, DEC; rad) and right ascension
        (α, RA; rad), equatorial spherical coordinate position vector to a XYZ (pc) galactic
        cartesian coordinate position vector, along with measurement errors.
    """

    values, errors = rδα_to_xyz(r, δ, α, Δr, Δδ, Δα)
    return equatorial_galactic_xyz(*values, *errors)

def galactic_uvw_equatorial_rvμδμα(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts an UVW (pc/Myr) galactic cartesian coordinate velocity vector to a rvµδµα, radial
        velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr) and right ascension proper
        motion (μδ; rad/Myr), equatorial spherical coordinate velocity vector, along with
        measurement errors.
    """

    position_values, position_errors = galactic_equatorial_xyz(x, y, z, Δx, Δy, Δz)
    velocity_values, velocity_errors = galactic_equatorial_xyz(u, v, w, Δu, Δv, Δw)
    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def equatorial_rvμδμα_galactic_uvw(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts a rvµδµα, radial velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr)
        and right ascension proper motion (µα; rad/Myr), equatorial spherical coordinate velocity
        vector to an UVW (pc/Myr) galactic cartesian coordinate velocity vector, along with
        measurement errors.
    """

    values, errors = rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)
    return equatorial_galactic_xyz(*values, *errors)

def equatorial_xyz_galactic_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ (pc) equatorial cartesian coordinate position vector to a rδα, distance
        (r; pc), declination (δ, DEC; rad) and right ascension (α, RA; rad), galactic spherical
        coordinate position vector, along with measurement errors. 'x', 'y' and 'z' can't all be
        null.
    """

    values, errors = equatorial_galactic_xyz(x, y, z, Δx, Δy, Δz)
    return xyz_to_rδα(*values, *errors)

def galactic_rδα_equatorial_xyz(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Converts a rδα, distance (r; pc), declination (δ, DEC; rad) and right ascension
        (α, RA; rad), galactic spherical coordinate position vector to a XYZ (pc) equatorial
        cartesian coordinate position vector, along with measurement errors.
    """

    values, errors = rδα_to_xyz(r, δ, α, Δr, Δδ, Δα)
    return galactic_equatorial_xyz(*values, *errors)

def equatorial_uvw_galactic_rvµδµα(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts an UVW (pc/Myr) equatorial cartesian coordinate velocity vector to a rvµδµα,
        radial velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr) and right ascension
        proper motion (μδ; rad/Myr), galactic spherical coordinate velocity vector, along with
        measurement errors.
    """

    position_values, position_errors = equatorial_galactic_xyz(x, y, z, Δx, Δy, Δz)
    velocity_values, velocity_errors = equatorial_galactic_xyz(u, v, w, Δu, Δv, Δw)
    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def galactic_rvµδµα_equatorial_uvw(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts a rvµδµα, radial velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr)
        and right ascension proper motion (μα; rad/Myr), equatorial spherical coordinate velocity
        vector to an UVW (pc/Myr) galactic cartesian coordinate velocity vector, along with
        measurement errors.
    """

    values, errors = rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)
    return galactic_equatorial_xyz(*values, *errors)

def equatorial_galactic_rδα(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Rotates an equatorial coordinate rδα vector to a galactic coordinate rδα vector. """

    values, errors = equatorial_rδα_galactic_xyz(r, δ, α, Δr, Δδ, Δα)
    return xyz_to_rδα(*values, *errors)

def galactic_equatorial_rδα(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Rotates a galactic coordinate rδα vector to an equatorial coordinate rδα vector. """

    values, errors = galactic_rδα_equatorial_xyz(r, δ, α, Δr, Δδ, Δα)
    return xyz_to_rδα(*values, *errors)

def equatorial_galactic_rvμδμα(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Rotates an equatorial velocity rvμδμα vector to a galactic velocity rvμδμα vector. """

    position_values, position_errors = equatorial_rδα_galactic_xyz(r, δ, α, Δr, Δδ, Δα)
    velocity_values, velocity_errors = equatorial_rvµδµα_galactic_uvw(
        r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)

    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def galactic_equatorial_rvμδμα(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Rotates a galactic velocity rvμδμα vector to an equatorial velocity rvμδμα) vector. """

    position_values, position_errors = galactic_rδα_equatorial_xyz(r, δ, α, Δr, Δδ, Δα)
    velocity_values, velocity_errors = galactic_rvµδµα_equatorial_uvw(
        r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)

    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def observables_spherical(p, δ, α, rv, μδ, μα_cos_δ, Δp=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα_cos_δ=0):
    """ Converts observables, paralax (p; rad), declination (δ, DEC; rad), right ascension
        (α, RA; rad), radial velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr) and
        and right ascension proper motion * cos(δ) (μα_cos_δ; rad/Myr), into an equatorial
        spherical coordinate, distance (r; pc), declination (δ, DEC; rad), right ascension
        (α, RA; rad), radial velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr) and
        right ascension proper motion (μα_cos_δ; rad/Myr), along with measurement errors.
    """

    # Cosine calculation
    cos_δ = cos(δ)

    # Values calculation
    position = np.array((Coordinate.k / p, δ, α))
    velocity = np.array((rv, μδ, μα_cos_δ / cos_δ))

    # Errors calculation
    if not np.array((Δp, Δδ, Δα, Δrv, Δμδ, Δμα_cos_δ)).any():
        return position, velocity, np.zeros(3), np.zeros(3)
    else:
        return position, velocity, np.array((Δp * Coordinate.k / p**2, Δδ, Δα)), \
            np.array((Δrv, Δμδ, ((Δμα_cos_δ / μα_cos_δ)**2 + (Δδ / δ)**2)**0.5 * μα_cos_δ / cos_δ))

def spherical_observables(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts an equatorial spherical coordinate, distance (r; pc), declination (δ, DEC; rad)
        and right ascension (α, RA; rad), radial velocity (rv; pc/Myr), declination proper motion
        (µδ; rad/Myr) and right ascension proper motion (μα_cos_δ; rad/Myr)) into observables,
        paralax (p; rad), declination (δ, DEC; rad), right ascension (α, RA; rad), radial velocity
        (rv; pc/Myr), declination proper motion (µδ; rad/Myr) and and right ascension proper motion
        * cos(δ) (μα_cos_δ; rad/Myr), along with measurement errors.
    """

    # Cosine calculation
    cos_δ = cos(δ)

    # Values calculation
    position = np.array((Coordinate.k / r, δ, α))
    velocity = np.array((rv, μδ, μα * cos_δ))

    # Errors calculation
    if not np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα)).any():
        return position, velocity, np.zeros(3), np.zeros(3)
    else:
        return position, velocity, np.array((Δr * Coordinate.k / r**2, Δδ, Δα)), \
            np.array((Δrv, Δμδ, ((Δμα / μα)**2 + (Δδ / δ)**2)**0.5 * μα * cos_δ))
