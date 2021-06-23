# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" coordinate.py: Defines System class to handle a coordinate system initialization with a set
    of position and velocity variables, default and usual units, axis, and origins. Variable, Axis
    and Origin subclasses are definied as well. Individual coordinates are defined by a
    Coordinate class which includes an initialization method and tranformation methods.
"""

import numpy as np
from math import cos, sin, asin, atan2, pi, degrees, radians
from Traceback.quantity import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class System():
    """ Defines a coordinate system object with variables, default units, usual units, axes,
        origins, and Variable, Axis and Origin classes.
    """

    def __init__(self, name: str, position: tuple, velocity: tuple):
        """ Initializes a System from position and velocity tuples with 3 Variables objects. """

        # Name
        self.name = name

        # Values
        self.position = [self.variables[label] for label in position]
        self.velocity = [self.variables[label] for label in velocity]

        # Errors
        self.position_error = [self.variables['Δ' + label] for label in position]
        self.velocity_error = [self.variables['Δ' + label] for label in velocity]

    def __eq__(self, other):
        """ Tests whether self is not the equal to other. """

        return self.name == self.name

    def __repr__(self):
        """ Returns a string of name of the system. """

        return self.name

    # Default units per physical type
    default_units = {
        'time': Unit('Myr', 'megayear'),
        'length': Unit('pc', 'parsec'),
        'speed': Unit('pc/Myr', 'parsec per megayear'),
        'angle': Unit('rad', 'radian'),
        'angular speed': Unit('rad/Myr', 'radian per megayear'),
        'mass': Unit('solMass', 'solar mass')}

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

        def __eq__(self, other):
            """ Tests whether self is not the equal to other. """

            return self.name == other.name

        def __repr__(self):
            """ Returns a string of name of the axis. """

            return self.name

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

        def __eq__(self, other):
            """ Tests whether self is not the equal to other. """

            return vars(self) == vars(other)

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
        Variable('r', 'galactic radius', default_units['length']),
        Variable('μρ', 'galactic radial velocity', default_units['speed']),
        Variable('θ', 'galactic angle', default_units['angle'], usual_units['angle']),
        Variable('μθ', 'galactic angular velocity', default_units['angular speed'],
            usual_units['angular speed']),

        # Spherical coordinate system variables
        Variable('ρ', 'distance', default_units['length']),
        Variable('δ', 'declination', default_units['angle'], usual_units['angle']),
        Variable('α', 'right ascension', default_units['angle'], usual_units['angle']),
        Variable('rv', 'radial velocity', default_units['speed'], usual_units['speed']),
        Variable('μδ', 'declination proper motion', default_units['angular speed'],
            usual_units['angular speed']),
        Variable('μα', 'right ascension proper motion', default_units['angular speed'],
            usual_units['angular speed']),

        # Observables coordinate system variables
        Variable('π', 'parallax', default_units['angle'], usual_units['small angle']),
        Variable('μαcosδ', 'right ascension proper motion * cos(declination)',
            default_units['angular speed'], usual_units['angular speed']))}

    # Error variables
    for label, variable in variables.copy().items():
        variables['Δ' + label] = Variable(
            'Δ' + label, variable.name + ' error', variable.unit, variable.usual_unit)

# Coordinate systems
systems = {system.name: system for system in (
    System('cartesian', ('x', 'y', 'z'), ('u', 'v', 'w')),
    System('cylindrical', ('r', 'θ', 'z'), ('μρ', 'μθ', 'w')),
    System('spherical', ('ρ', 'δ', 'α'), ('rv', 'μδ', 'μα')),
    System('observables', ('π', 'δ', 'α'), ('rv', 'μδ', 'μαcosδ')))}

class Coordinate:
    """ Contains the values and related methods of a coordinate, including its position, velocity,
        their errors and units, and methods to transform a coordinate from one system to another.
    """

    # Sun position (pc) and velocity (pc/Myr) in the Galaxy at the current epoch, along with current
    # errors from Quillen et al. (2020) in a left-handed galactocentric frame of reference (X points
    # away from the galactic center
    # sun_position = np.array([8122, 0., 20.8]) # Bennett & Bovy 2019
    sun_position = np.array([8400, 0., 0.])
    sun_position_error = np.array([0., 0., 0.])
    # sun_position_error = np.array([31, 0., 0.3])

    # Sun peculiar velocity
    sun_velocity_peculiar_kms = np.array([11.1, 12.24, 7.25]) # Galpy assumes right-handed coordinates
    # ??? Should be ajusted with matrix rotation to account for zo != 0.0 ???
    sun_velocity_peculiar = np.array([-11.352, 12.518, 7.41]) # Schöenrich et al. 2010

    # Local Standard of Rest (LSR) velocity
    # sun_angular_frequency = 0.029464 # 1/Myr
    sun_angular_frequency = 0.029443681828643653 # 1/Myr
    lsr_velocity = np.array([0., sun_position[0] * sun_angular_frequency, 0.]) # Irrgang et al. 2013

    # Sun total velocity
    sun_velocity = lsr_velocity + sun_velocity_peculiar
    sun_velocity_error = np.array([0., 0., 0.])

    # Galpy's ro (kpc), vo (km/s) and zo (kpc) values used for orbit integration
    ro = sun_position[0] / 1000
    vo = lsr_velocity[1] * 0.9777922216731283
    zo = sun_position[2] / 1000

    # Distance to the galactic center and theta angle trigonometric ratios
    distance_galactic_center = (sun_position[0]**2 + sun_position[2]**2)**0.5
    costheta = sun_position[0] / distance_galactic_center
    sintheta = sun_position[2] / distance_galactic_center

    # !!! This part should be moved to Axis !!!
    # Galactic-galactocentric rotation matrix
    # Rotates a vector from a galactic to a galactocentric right-handed reference frame
    ggrm = np.array([
        [  costheta,   0.  , sintheta ],
        [     0.   ,   1.  ,     0.    ],
        [ -sintheta,   0.  , costheta ]])

    # Sun position and velocity in a galactic coordinate system
    galactic_center = np.array([-distance_galactic_center, 0., 0.])
    galactic_center_error = np.array([0., 0., 0.])
    galactic_velocity = np.dot(ggrm, sun_velocity) * np.array([-1, 1, 1])
    galactic_velocity_error = np.array([0., 0., 0.])

    # J2000.0 Galactic-equatorial rotation matrix from Liu et al. (2018) 1010.3773
    # Rotates a vector from an equatorial to a galactic right-handed reference frame
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

    # J2000.0 position angle (l) of the galactic center from the equatorial pole (δ = 90°)
    # from Liu et al. (2018) 1010.3773
    l_galaxy = radians(122.931925267)

    # Parallax conversion from radians to parsecs constant
    k = pi / 648000

    # Gaia bias correction
    gaia_bias = -2.618e-10
    # gaia_bias = 0.0
    # sun_velocity = np.array([-11.370952123669653, -12.518 + 238.292, 7.385903925804384])
    # sun_velocity = np.array([-11.370952123669653, -12.518 + 247.50, 7.385903925804384])

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


# Vector translation functions

def translate(x, y, z, Δx=0, Δy=0, Δz=0, origin_vector=np.zeros(3), origin_vector_error=np.zeros(3)):
    """ Translates a cartesian XYZ position (pc) or UVW velocity (pc/Myr) vector to a new origin
        using the 'origin_vector' (pc or pc/Myr).
    """

    # Coordinate calculation
    values = np.array((x, y , z)) + origin_vector

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(values.shape)
    else:
        errors = (np.array((Δx, Δy, Δz))**2 + origin_vector_error**2)**0.5
        return values, errors

def heliocentric_galactocentric_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Translates a cartesian XYZ position vector (pc) from a heliocentric to a galactocentric
        origin.
    """

    return translate(x, y, z, Δx, Δy, Δz, Coordinate.galactic_center, Coordinate.galactic_center_error)

def heliocentric_galactocentric_uvw(u, v, w, Δu=0, Δv=0, Δw=0):
    """ Translates a cartesian UVW velocity vector (pc) from a heliocentric to a galactocentric
        reference frame.
    """

    return translate(u, v, w, Δu, Δv, Δw, Coordinate.galactic_velocity, Coordinate.galactic_velocity_error)

def galactocentric_heliocentric_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Translates a cartesian XYZ position vector (pc) from a galactocentric to a heliocentric
        origin.
    """

    return translate(x, y, z, Δx, Δy, Δz, -Coordinate.galactic_center, Coordinate.galactic_center_error)

def galactocentric_heliocentric_uvw(u, v, w, Δu=0, Δv=0, Δw=0):
    """ Translates a cartesian UVW velocity vector (pc) from a heliocentric to a galactocentric
        reference frame.
    """

    return translate(u, v, w, Δu, Δv, Δw, -Coordinate.galactic_velocity, Coordinate.galactic_velocity_error)


# Vector rotation functions

def rotate(x, y, z, Δx=0, Δy=0, Δz=0, rotation_matrix=np.identity(3)):
    """ Rotates a cartesian XYZ position (pc) or UVW velocity (pc/Myr) vector to a new reference
        plane using the 'rotation_matrix'.
    """

    # Coordinate calculation
    values = np.dot(rotation_matrix, np.array((x, y, z)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(values.shape)
    else:
        errors = np.dot(rotation_matrix**2, np.array((Δx, Δy, Δz))**2)**0.5
        return values, errors

def equatorial_galactic_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian XYZ position or UVW velocity vector from an equatorial to a galactic
        plane. All arguments must have the same units.
    """

    return rotate(x, y, z, Δx, Δy, Δz, Coordinate.germ)

def galactic_equatorial_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian XYZ position or UVW velocity vector from a galactic to an equatorial
        plane. All arguments must have the same units.
    """

    return rotate(x, y, z, Δx, Δy, Δz, Coordinate.germ.T)

def galactic_galactocentric_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian XYZ position or UVW velocity vector from a galactic to a galactocentric
        plane. All arguments must have the same units.
    """

    return rotate(x, y, z, Δx, Δy, Δz, Coordinate.ggrm)

def galactocentric_galactic_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian XYZ position or UVW velocity vector from a galactocentric to a galactic
        plane. All arguments must have the same units.
    """

    return rotate(x, y, z, Δx, Δy, Δz, Coordinate.ggrm.T)


# Cartesian and spherical coordinates transformation functions

def xyz_to_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ cartesian coordinate position vector (pc) to a rδα, distance (r; pc),
        declination (δ, DEC; rad) and right ascension (α, RA; rad), spherical coordinate position
        vector (observables), along with measurement errors. 'x', 'y' and 'z' can't all be null.
    """

    # Norm calculation
    norm_2, norm_xy_2 = (x**2 + y**2 + z**2), (x**2 + y**2)
    norm, norm_xy = norm_2**0.5, norm_xy_2**0.5

    # Distance and angles calculation
    values = np.array((norm, asin(z / norm), atan2(y, x) + (2 * pi * (y < 0))))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(values.shape)
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
    """ Converts a rδα, distance (r; pc), declination (δ, DEC; rad) and right ascension
        (α, RA; rad), spherical coordinate position vector (observables) to a XYZ cartesian
        coordinate position vector (pc), along with measurement errors.
    """

    # Cosine and sine calculation
    cos_δ, sin_δ, cos_α, sin_α = cos(δ), sin(δ), cos(α), sin(α)

    # Position calculation
    values = np.array((r * cos_δ * cos_α, r * cos_δ * sin_α, r * sin_δ))

    # Errors calculation
    if not np.array((Δr, Δδ, Δα)).any():
        return values, np.zeros(values.shape)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of x: dx/dr, dx/dδ and dx/dα
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
        return values, np.zeros(values.shape)
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
        return values, np.zeros(values.shape)
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


# Cartesian and cylindrical coordinates transformation functions

def xyz_to_ρθz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ cartesian coordinate position vector (pc) to a ρθz, radius (ρ; pc), angle
        (θ; rad) and heigth (z; pc), cylindrical coordinate position vector (galactocentric),
        along with measurement errors. 'x', 'y' and 'z' can't all be null.
    """

    # Position calculation
    values = np.array(((x**2 + y**2)**0.5, atan2(y, x) + (2 * pi * (y < 0)), z))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(values.shape)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of ρ: dρ/dx, dρ/dy and dρ/dz
            (0., 0., 0.),

            # Partial derivatives of θ: dθ/dx, dθ/dy and dθ/dz
            (0., 0., 0.),

            # Partial derivatives of z: dz/dx, dz/dy and dz/dz
            (0., 0., 0.))
        )**2, np.array((Δx, Δy, Δz))**2)**0.5
        return values, errors

def ρθz_to_xyz(ρ, θ, z, Δρ=0, Δθ=0, Δz=0):
    """ Converts a ρθz, radius (ρ; pc), angle (θ; rad) and heigth (z; pc), cylindrical coordinate
        position vector (galactocentric) to a XYZ cartesian coordinate position vector (pc), along
        with measurement errors.
    """

    # Cosine and sine calculation
    cos_θ, sin_θ = cos(θ), sin(θ)

    # Position calculation
    values = np.array((ρ * cos_θ, ρ * sin_θ, z))

    # Errors calculation
    if not np.array((Δρ, Δθ, Δz)).any():
        return values, np.zeros(values.shape)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of x: dx/dρ, dx/dθ and dx/dz
            (0., 0., 0.),

            # Partial derivatives of y: dy/dρ, dy/dθ and dy/dz
            (0., 0., 0.),

            # Partial derivatives of z: dz/dρ, dz/dθ and dz/dz
            (0., 0., 0.))
        )**2, np.array((Δρ, Δθ, Δz))**2)**0.5
        return values, errors

def uvw_to_μρμθw(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts a UVW cartesian coordinate velocity vector (pc/Myr) to a μρμθw, radial velocity
        (μρ; pc/Myr), angular velocity (μθ; rad/Myr) and W velocity (w; pc/Myr), cylindrical
        coordinate velocity vector (galactocentric), along with measurement errors. 'x', 'y' and
        'z' (pc) can't all be null.
    """

    # Norms calculation
    norm_xy_2 = x**2 + y**2
    norm_xy = norm_xy_2**0.5

    # Velocity calculation
    # values = np.array((((u * x) + (v * y)) / norm_xy, ((v * x) - (u * y)) / norm_xy_2, w))
    # Tangantial velocity instead
    values = np.array((((u * x) + (v * y)) / norm_xy, ((v * x) - (u * y)) / norm_xy, w))

    # Errors calculation
    if not np.array((Δx, Δy, Δz, Δu, Δv, Δw)).any():
        return values, np.zeros(values.shape)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of ρ: dρ/dx, dρ/dy, dρ/dz, dρ/du, dρ/dv, dρ/dw
            (0., 0., 0., 0., 0., 0.),

            # Partial derivatives of θ: dθ/dx, dθ/dy, dθ/dz, dθ/du, dθ/dv, dθ/dw
            (0., 0., 0., 0., 0., 0.),

            # Partial derivatives of z: dz/dx, dz/dy, dz/dz, dz/du, dz/dv, dz/dz
            (0., 0., 0., 0., 0., 0.))
        )**2, np.array((Δx, Δy, Δz, Δu, Δv, Δw))**2)**0.5
        return values, errors

def μρμθw_to_uvw(ρ, θ, z, μρ, μθ, w, Δρ=0, Δθ=0, Δz=0, Δμρ=0, Δμθ=0, Δw=0):
    """ Converts a μρμθw, radial velocity (μρ; pc/Myr), angular velocity (μθ; rad/Myr) and W
        velocity (w; pc/Myr), cylindrical coordinate velocity vector (galactocentric) to an UVW
        cartesian coordinate velocity vector (pc/Myr), along with measurement errors.
    """

    # Cosine and sine calculation
    cos_θ, sin_θ = cos(θ), sin(θ)

    # Velocity calculation
    values = np.array((μρ * cos_θ - μθ * (ρ * sin_θ), μρ * sin_θ + μθ * (ρ * cos_θ), w))

    # Errors calculation
    if not np.array((Δρ, Δθ, Δz, Δμρ, Δμθ, Δw)).any():
        return values, np.zeros(values.shape)
    else:
        errors = np.dot(np.array((

            # Partial derivatives of ρ: dρ/dx, dρ/dy, dρ/dz, dρ/du, dρ/dv, dρ/dw
            (0., 0., 0., 0., 0., 0.),

            # Partial derivatives of θ: dθ/dx, dθ/dy, dθ/dz, dθ/du, dθ/dv, dθ/dw
            (0., 0., 0., 0., 0., 0.),

            # Partial derivatives of z: dz/dx, dz/dy, dz/dz, dz/du, dz/dv, dz/dz
            (0., 0., 0., 0., 0., 0.))
        )**2, np.array((Δρ, Δθ, Δz, Δμρ, Δμθ, Δw))**2)**0.5
        return values, errors


# Cartesian galactic and spherical equatorial coordinates transformation decorator functions

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


# Cartesian equatorial and spherical galactic coordinates transformation decorator functions

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


# Spherical equatorial and galactic coordinates rotation decorator functions

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
    """ Rotates a galactic velocity rvμδμα vector to an equatorial velocity rvμδμα vector. """

    position_values, position_errors = galactic_rδα_equatorial_xyz(r, δ, α, Δr, Δδ, Δα)
    velocity_values, velocity_errors = galactic_rvµδµα_equatorial_uvw(
        r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)

    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)


# Cartesian galactic and cylindrical galactocentric transformation decorator functions

def galactic_xyz_galactocentric_ρθz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ (pc) galactic cartesian coordinate position vector to a ρθz, radius (ρ; pc),
        angle (θ; rad) and heigth (z; pc), cylindrical coordinate position vector (galactocentric),
        along with measurement errors. 'x', 'y' and 'z' can't all be null.
    """

    # Translation, rotation and left-handed cylindrical coordinates system
    values, errors = heliocentric_galactocentric_xyz(x, y, z, Δx, Δy, Δz)
    values, errors = galactic_galactocentric_xyz(*values, *errors)
    values[0] = -values[0]

    return xyz_to_ρθz(*values, *errors)

def galactocentric_ρθz_galactic_xyz(ρ, θ, z, Δρ=0, Δθ=0, Δz=0):
    """ Converts a ρθz, radius (ρ; pc), angle (θ; rad) and heigth (z; pc), cylindrical coordinate
        position vector (galactocentric) to a XYZ (pc) galactic cartesian coordinate position
        vector, along with measurement errors.
    """

    # Cartesian right-handed coordinates system, rotation and translation
    values, errors = ρθz_to_xyz(ρ, θ, z, Δρ, Δθ, Δz)
    values[0] = -values[0]
    values, errors = galactocentric_galactic_xyz(*values, *errors)
    return galactocentric_heliocentric_xyz(*values, *errors)

def galactic_uvw_galactocentric_ρθz(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts an UVW (pc/Myr) galactic cartesian coordinate velocity vector to a μρμθw, radial
        velocity (μρ; pc/Myr), angular velocity (μθ; rad/Myr) and W velocity (w; pc/Myr),
        cylindrical coordinate velocity vector (galactocentric), along with measurement errors.
    """

    # Position translation, rotation and left-handed system
    position_values, position_errors = heliocentric_galactocentric_xyz(x, y, z, Δx, Δy, Δz)
    position_values, position_errors = galactic_galactocentric_xyz(*position_values, *position_errors)
    position_values[0] = -position_values[0]

    # Velocity rotation, translation and left-handed system
    velocity_values, velocity_errors = galactic_galactocentric_xyz(u, v, w, Δu, Δw, Δz)
    velocity_values, velocity_errors = heliocentric_galactocentric_uvw(*velocity_values, *velocity_errors)
    velocity_values[0] = -velocity_values[0]

    # Cylindrical coordinates system
    return uvw_to_μρμθw(*position_values, *velocity_values, *position_errors, *velocity_errors)

def galactocentric_ρθz_galactic_uvw(ρ, θ, z, μρ, μθ, w, Δρ=0, Δθ=0, Δz=0, Δμρ=0, Δμθ=0, Δw=0):
    """ Converts a μρμθw, radial velocity (μρ; pc/Myr), angular velocity (μθ; rad/Myr) and W
        velocity (w; pc/Myr), cylindrical coordinate velocity vector (galactocentric) to an UVW
        (pc/Myr) galactic cartesian coordinate velocity vector, along with measurement errors.
    """

    # Cartesian right-handed coordinates system, rotation and translation
    values, errors = μρμθw_to_uvw(ρ, θ, z, μρ, μθ, w, Δρ, Δθ, Δz, Δμρ, Δμθ, Δw)
    values[0] = -values[0]
    values, errors = galactocentric_galactic_xyz(*values, *errors)
    return galactocentric_heliocentric_uvw(*values, *errors)

# Observatables and spherical coordinates transformation functions

def position_obs_rδα(p, δ, α, rv, μδ, μα_cos_δ, Δp=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα_cos_δ=0):
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
    position = np.array((Coordinate.k / (p + Coordinate.gaia_bias), δ, α))
    velocity = np.array((rv, μδ, μα_cos_δ / cos_δ))

    # Errors calculation
    if not np.array((Δp, Δδ, Δα, Δrv, Δμδ, Δμα_cos_δ)).any():
        return position, velocity, np.zeros(position.shape), np.zeros(velocity.shape)
    else:
        return position, velocity, np.array((Δp * Coordinate.k / (p + Coordinate.gaia_bias)**2, Δδ, Δα)), \
            np.array((Δrv, Δμδ, ((Δμα_cos_δ / μα_cos_δ)**2 + (Δδ / δ)**2)**0.5 * μα_cos_δ / cos_δ))

def position_rδα_obs(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
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
    position = np.array((Coordinate.k / r - Coordinate.gaia_bias, δ, α))
    velocity = np.array((rv, μδ, μα * cos_δ))

    # Errors calculation
    if not np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα)).any():
        return position, velocity, np.zeros(position.shape), np.zeros(velocity.shape)
    else:
        return position, velocity, np.array((Δr * (Coordinate.k / r**2), Δδ, Δα)), \
            np.array((Δrv, Δμδ, ((Δμα / μα)**2 + (Δδ / δ)**2)**0.5 * μα * cos_δ))


# Curvilinear and cylindrical galactocentric transformation functions

def position_rθz_ξηζ(r, θ, z, t):
    """ Converts a rθz, radius (r; pc), angle (θ; rad) and heigth (z; pc), galactocentric
        cylindrical coordinate position vector to an ξηζ, radius (ξ; pc), arc (η; pc) and height
        (z; pc) galactocentric curvilinear coordinate position vector, at a given epoch t (Myr).
    """

    return np.dot(Coordinate.ggrm.T, np.array([
        r - Coordinate.sun_position[0],
        Coordinate.sun_position[0] * (θ - Coordinate.sun_angular_frequency * t),
        z - Coordinate.sun_position[2]])).T

def position_ξηζ_rθz(ξ, η, ζ, t):
    """ Converts a ξηζ, radius (ξ; pc), arc (η; pc) and height (z; pc) galactocentric curvilinear
        coordinate position vector to a rθz, radius (r; pc), angle (θ; rad) and heigth (z; pc),
        galactocentric cylindrical coordinate position vector, at a given epoch t (Myr).
    """

    return np.dot(Coordinate.ggrm, np.array([
        ξ + Coordinate.sun_position[0],
        η / Coordinate.sun_position[0] + Coordinate.sun_angular_frequency * t,
        ζ + Coordinate.sun_position[2]])).T

def velocity_rθz_ξηζ(vr, vt, vz, t):
    """ Converts a rθz, radial velocity (vr; pc/Myr), tangantial velocity (vt; pc/Myr) and heigth
        velocity (vz; pc/Myr), galactocentric cylindrical coordinate velocity vector to an ξηζ,
        radial velocity (vξ; pc/Myr), arc velocity (vη; pc/Myr) and height velocity (vz; pc)
        galactocentric curvilinear coordinate velocity vector, at a given epoch t (Myr).
    """

    vr, vt, vz = np.dot(Coordinate.ggrm.T, np.array([vr, vt - Coordinate.sun_velocity[1], vz]))

    return np.array([-vr, vt, vz]).T

def velocity_ξηζ_rθz(vξ, vη, vζ, t):
    """ Converts a ξηζ, radial velocity (vξ; pc/Myr), arc velocity (vη; pc/Myr) and height velocity
        (vz; pc) galactocentric curvilinear coordinate velocity vector to a rθz, radial velocity
        (vr; pc/Myr), tangantial velocity (vt; pc/Myr) and heigth velocity (vz; pc/Myr),
        galactocentric cylindrical coordinate velocity vector, at a given epoch t (Myr).
    """

    return np.array([-vξ, vη + Coordinate.sun_velocity[1], vζ]).T
