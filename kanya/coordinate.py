# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
coordinate.py: Defines System class to handle a coordinate system initialization with a set
of position and velocity variables, default and usual units, axis, and origins. Variable,
Axis and Origin classes are definied as well. Individual coordinates are defined by a
Coordinate class which includes an initialization method and tranformation methods.
"""

import pandas as pd
from os import path
from copy import deepcopy
from .quantity import *

class Variable():
    """
    Defines a Variable object and required variables from all systems. 'physical_type', 'unit',
    'usual_unit', 'unit_error' and 'usual_unit_error' must be keys in default_units and usual_units
    or a Unit object.
    """

    def __init__(
        self, label, name, latex, physical_type, unit=None,
        usual_unit=None, unit_error=None, usual_unit_error=None
    ):
        """Initializes a Variable from a label, name, physical type and Unit objects or strings."""

        # Initialization
        self.label = label
        self.name = name
        self.latex = latex

        # Set units
        self.unit = (
            unit if type(unit) == Unit
            else default_units[physical_type if unit in (None, '') else unit]
        )
        self.usual_unit = (
            usual_unit if type(usual_unit) == Unit
            else usual_units[physical_type if usual_unit in (None, '') else usual_unit]
        )

        # Set error units
        self.unit_error = (
            unit_error if type(unit_error) == Unit
            else self.unit if unit_error in (None, '')
            else default_units[unit_error]
        )
        self.usual_unit_error = (
            usual_unit_error if type(usual_unit_error) == Unit
            else self.usual_unit if usual_unit_error in (None, '')
            else usual_units[usual_unit_error]
        )

        # Set physical type
        self.physical_type = self.unit.physical_type

    def __eq__(self, other):
        """Tests whether self is not the equal to other."""

        return vars(self) == vars(other)

    def __repr__(self):
        """Returns a string of name of the variable."""

        return f'{self.name} variable'

class Axis():
    """Defines an Axis system axis."""

    def __init__(self, name):
        """Initializes an Axis object."""

        # Initialization
        self.name = name

    def __eq__(self, other):
        """Tests whether self is not the equal to other."""

        return self.name == other.name

    def __repr__(self):
        """Returns a string of name of the axis."""

        return f'{self.name.capitalize()} axis'

class Origin():
    """Defines a Origin system origin."""

    def __init__(self, name):
        """Initializes an Origin object."""

        # Initialization
        self.name = name

    def __eq__(self, other):
        """Tests whether self is not the equal to other."""

        return self.name == other.name

    def __repr__(self):
        """Returns a string of name of the axis."""

        return f'{self.name.capitalize()} origin'

class System():
    """
    Defines a coordinate system object with variables, default units, usual units, axes,
    origins, and Variable, Axis and Origin classes.
    """

    def __init__(self, name, position, velocity, axis, origin):
        """Initializes a System from position and velocity tuples with 3 Variables objects."""

        # Name
        self.name = name

        # Values
        self.position = [variables[label] for label in position]
        self.velocity = [variables[label] for label in velocity]

        # Errors
        self.position_error = [variables['Δ' + label] for label in position]
        self.velocity_error = [variables['Δ' + label] for label in velocity]

        # Axis and origin
        self.axis = axes[axis]
        self.origin = origins[origin]

        # Set latex labels
        self.label = ''.join([variable.label for variable in self.position])
        self.labels = {
            'position': ''.join([variable.label for variable in self.position]),
            'velocity': ''.join([variable.label for variable in self.velocity])
        }
        self.latex = {
            'position': ''.join([variable.latex for variable in self.position]),
            'velocity': ''.join([variable.latex for variable in self.velocity])
        }

    def __eq__(self, other):
        """Tests whether self is not the equal to other."""

        return self.name == self.name

    def __repr__(self):
        """Returns a string of name of the system."""

        return f'{self.name.capitalize()} coordinate system'

# Default and usual units
units_file = path.join(path.dirname(__file__), 'resources/units.csv')
units_dataframe = pd.read_csv(units_file, delimiter=';', na_filter=False)
default_units = {}
usual_units = {}
for index, row in units_dataframe.iterrows():
    default_units[row['physical_type']] = Unit(row['default_unit_label'], row['default_unit_name'])
    usual_units[row['physical_type']] = Unit(row['usual_unit_label'], row['usual_unit_name'])
del units_file, units_dataframe

# Variables
variables_file = path.join(path.dirname(__file__), 'resources/variables.csv')
variables_dataframe = pd.read_csv(variables_file, delimiter=';', na_filter=False)
variables = {}
for index, row in variables_dataframe.iterrows():
    variable = Variable(*[row[column] for column in row.index])
    variables[variable.label] = variable
del variables_file, variables_dataframe

# Error variables
for label, variable in variables.copy().items():
    variables['Δ' + label] = Variable(
        'Δ' + label, variable.name + ' error', 'Δ' + variable.latex,
        variable.physical_type, unit=variable.unit, usual_unit=variable.usual_unit
    )

# Coordinate system axes
axes = {axis.name: axis for axis in (Axis('equatorial'), Axis('galactic'), Axis('galactic_sun'))}

# Coordinate system origins
origins = {origin.name: origin for origin in (Origin('sun'), Axis('galaxy'))}

# Coordinate systems
systems_file = path.join(path.dirname(__file__), 'resources/systems.csv')
systems_dataframe = pd.read_csv(systems_file, delimiter=';', na_filter=False)
systems = {}
for index, row in systems_dataframe.iterrows():
    system = System(
        row['name'], eval(row['position']), eval(row['velocity']), row['axis'], row['origin']
    )
    systems[system.name] = system
del systems_file, systems_dataframe

# Coordinate systems with shorter labels
for label, system in deepcopy(systems).items():
    systems[system.label] = system

class Coordinate:
    """
    Contains the values and related methods of a coordinate, including its position, velocity,
    their errors and units, and methods to transform a coordinate from one system to another.
    """

    # Sun position (pc) in the Galaxy at the current epoch, along with current errors from
    # Quillen et al. (2020) in a left-handed galactocentric frame of reference (X points away
    # from the galactic center
    sun_position = np.array([8400.0, 0.0, 20.8])
    # sun_position = np.array([8122.0, 0.0, 20.8]) # Bennett & Bovy 2019
    # sun_position = np.array([8400.0, 0.0, 0.0])
    sun_position_error = np.array([0.0, 0.0, 0.0])
    # sun_position_error = np.array([31.0, 0.0, 0.3])

    # Distance to the Galactic Center
    galactic_center_distance = (sun_position[0]**2 + sun_position[2]**2)**0.5
    costheta = sun_position[0] / galactic_center_distance
    sintheta = sun_position[2] / galactic_center_distance

    # Heliogalactic-galactic rotation matrix
    # Rotates a vector from a heliogalactic to a galactic left-handed reference frame
    ggrm = np.array(
        [
            [costheta, 0.0, -sintheta],
            [  0.0   , 1.0,    0.0   ],
            [sintheta, 0.0,  costheta]
        ]
    )

    # Sun peculiar velocity relative to the local standard of rest from Schöenrich et al. (2010)
    # in km/s converted to pc/Myr. Errors include both random and systematic components
    sun_velocity_peculiar = np.array([11.1, 12.24, 7.25]) * u.Unit('km/s').to('pc/Myr')
    sun_velocity_peculiar_error = (
        np.array([0.75, 0.47, 0.37]) + np.array([1.0, 2.0, 0.5])
    ) * u.Unit('km/s').to('pc/Myr')

    # Sun angular frequency from Irrgang et al. (2013) and Local Standard of Rest (LSR) velocity
    # relative to the Galactic center
    sun_angular_frequency = 0.029443681828643653 # 1/Myr
    # sun_angular_frequency = 0.029464 # 1/Myr
    lsr_velocity = np.array([0.0, sun_position[0] * sun_angular_frequency, 0.0])

    # Sun total velocity in a left-handed galactocentric frame of reference
    sun_velocity = np.dot(ggrm, lsr_velocity + sun_velocity_peculiar * np.array((-1, 1, 1)))
    # sun_velocity = np.array([0.0, 0.0, 0.0])
    sun_velocity_error = np.array([0.0, 0.0, 0.0])
    # sun_velocity_error = sun_velocity_peculiar_error

    # J2000.0 Equatorial right ascension (α) and declination (δ) of the Galactic North (b = 90°)
    # from Liu et al. (2018) 1010.3773
    α_north = np.radians(192.8594812065348)
    δ_north = np.radians(27.12825118085622)

    # J2000.0 position angle (l) of the galactic center from the equatorial pole (δ = 90°)
    # from Liu et al. (2018) 1010.3773
    l_galaxy = np.radians(122.9319185680026)

    # J2000.0 Galactic-equatorial rotation matrix from Liu et al. (2018) 1010.3773
    # Rotates a vector from an equatorial to a galactic right-handed reference frame
    germ = np.dot(
        np.array(
            [
                [np.cos(l_galaxy),  np.sin(l_galaxy), 0.0],
                [np.sin(l_galaxy), -np.cos(l_galaxy), 0.0],
                [      0.0       ,        0.0       , 1.0]
            ]
        ),
        np.dot(
            np.array(
                [
                    [-np.sin(δ_north), 0.0, np.cos(δ_north)],
                    [       0.0      , 1.0,       0.0      ],
                    [ np.cos(δ_north), 0.0, np.sin(δ_north)]
                ]
            ),
            np.array(
                [
                    [ np.cos(α_north), np.sin(α_north), 0.0],
                    [-np.sin(α_north), np.cos(α_north), 0.0],
                    [       0.0      ,       0.0      , 1.0]
                ]
            )
        )
    )

    # Parallax conversion from radians to parsecs constant
    k = np.pi / 648000

    # Gaia bias correction
    gaia_bias = -2.618e-10
    # gaia_bias = 0.0

    def __init__(self, position, velocity=None):
        """
        Initializes a Coordinate object from a Quantity objects representing n positions
        (shape = (n, 3)) and optionnally n corresponding velocities (shape = (n, 3)).
        Position and velocity must be broadcast together and can either be spherical, cartesian,
        cylindrical or curvilinear coordinates.
        """

        # Import of position
        if type(position) != Quantity:
            raise TypeError(
                "Position must be a Quantity of shape (n, 3)), not {}".format(type(position))
            )
        elif position.ndim in (1, 2) and position.shape[-1] == 3:
            position = position.to()
        else: # !!! changer pour n'importe ndim si shape[-1] == 3 !!!
            raise ValueError("Position must have a shape of (n, 3), not {}".format(position.shape))

        # Import of velocity
        if velocity is None:
            velocity = Quantity((0.0, 0.0, 0.0), 'pc/Myr')
        elif type(velocity) != Quantity:
            raise TypeError(
                "Velocity must be a Quantity of shape (n, 3)) or None, not {}".format(
                    type(velocity)
                )
            )
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
                    position.shape, velocity.shape
                )
            )
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
            raise ValueError(
                "Position physical types ({}) don't fit a coordinate "
                "in a cartesian of spherical system.".format(position.physical_types)
            )

        # Conversion of velocity in cartesian or spherical coordinate system
        if (
            velocity.physical_types == np.array(
                ['speed', 'speed', 'speed']
            )
        ).all():
            self.velocity_uvw = velocity
            self.velocity_rvμδμα = None
        elif (
            velocity.physical_types == np.array(
                ['speed', 'angular speed', 'angular speed']
            )
        ).all():
            self.velocity_rvμδμα = velocity
            self.velocity_uvw = None
        else:
            raise ValueError(
                "Velocity physical types ({}) don't fit a coordinate "
                "in a cartesian of spherical system.".format(velocity.physical_types)
            )

        # System and axis
        self.system = None
        self.axis = None

    def to(self, system=None):
        """
        Converts a Coordinate object from its original coordinate system, to a new coordinate
        system.
        """

        pass

# Spherical equatorial coordinates transforms

def position_πδα_to_xyz(π, δ, α):
    """
    Converts πδα, paralax (π; rad), declination (δ, DEC; rad), and right ascension (α, RA; rad),
    spherical equatorial coordinates to xyz, x position (x, pc), y position (y, pc), and z position
    (z, pc), cartesian galactic coordinates.
    """

    # Compute norm, cosines, and sines
    norm = Coordinate.k / (π + Coordinate.gaia_bias)
    cos_δ, sin_δ, cos_α, sin_α = np.cos(δ), np.sin(δ), np.cos(α), np.sin(α)

    # Convert coordinates
    x, y, z = norm * np.array((cos_δ * cos_α, cos_δ * sin_α, sin_δ))

    # Rotate axes
    return Coordinate.germ @ np.array((x, y, z))

def velocity_πδα_to_xyz(π, δ, α, rv, μδ, μαcosδ):
    """
    Converts rvµδμαcosδ, radial velocity (rv; pc/Myr), declination proper motion (µδ; rad/Myr),
    and right ascension proper motion * cos(δ) (μαcosδ; rad/Myr), spherical equatorial coordinates
    to uvw, u velocity (u, pc/Myr), v velocity (v, pc/Myr), and w velocity (w, pc/Myr), cartesian
    galactic coordinates.
    """

    # Compute norm, cosines and sines
    norm = Coordinate.k / (π + Coordinate.gaia_bias)
    cos_δ, sin_δ, cos_α, sin_α = np.cos(δ), np.sin(δ), np.cos(α), np.sin(α)

    # Convert coordinates
    u, v, w = np.array(
        (
            rv * cos_δ * cos_α - norm * (μδ * sin_δ * cos_α + μαcosδ * sin_α),
            rv * cos_δ * sin_α - norm * (μδ * sin_δ * sin_α - μαcosδ * cos_α),
            rv * sin_δ + μδ * norm * cos_δ
        )
    )

    # Rotate axes
    return Coordinate.germ @ np.array((u, v, w))


# Cartesian galactic coordinates transforms

def position_xyz_to_πδα(x, y, z):
    """
    Converts xyz, x position (x, pc), y position (y, pc), and z position (z, pc), cartesian
    galactic coordinates to πδα, paralax (π; rad), declination (δ, DEC; rad), and right ascension
    (α, RA; rad), spherical equatorial coordinates. x, y and z can't all be null.
    """

    # Rotate axes
    x, y, z = Coordinate.germ.T @ np.array((x, y, z))

    # Compute norm
    norm = (x**2 + y**2 + z**2)**0.5

    # Convert coordinates
    shift = 2 * np.pi * (y < 0) if False else 0.0
    return np.array(
        (Coordinate.k / norm - Coordinate.gaia_bias, np.arcsin(z / norm), np.arctan2(y, x) + shift)
    )

def velocity_xyz_to_πδα(x, y, z, u, v, w):
    """
    Converts uvw, u velocity (u, pc/Myr), v velocity (v, pc/Myr), and w velocity (w, pc/Myr),
    cartesian galactic coordinates to rvµδµα, radial velocity (rv; pc/Myr), declination proper
    motion (μδ; rad/Myr), and right ascension proper motion * cos(δ) (μα_cos_δ; rad/Myr), spherical
    equatorial coordinates. x, y and z can't all be null.
    """

    # Rotate axes
    x, y, z = Coordinate.germ.T @ np.array((x, y, z))
    u, v, w = Coordinate.germ.T @ np.array((u, v, w))

    # Compute norms, and cosine
    norm_xy_2 = x**2 + y**2
    norm_xy = norm_xy_2**0.5
    norm_2 = norm_xy_2 + z**2
    norm = norm_2**0.5
    cos_δ = norm_xy / norm

    # Convert coordinates
    return np.array(
        (
            ((u * x) + (v * y) + (z * w)) / norm,
            (w * norm_xy - ((u * x * z) + (v * y * z)) / norm_xy) / norm_2,
            ((v * x) - (u * y)) / norm_xy_2 * cos_δ
        )
    )

def position_xyz_to_rθh(x, y, z):
    """
    Converts xyz, x position (x; pc), y position (y; pc), and z position (z; pc), cartesian
    galactic coordinates to rθh, radius (r; pc), angle (θ; rad), and height (h; pc), cylindrical
    galactocentric left-handed coordinates.
    """

    # Rotate axes, and move origin
    x, y, z = ((Coordinate.ggrm @ np.array((-x, y, z))).T + Coordinate.sun_position).T

    # Convert coordinates
    shift = 2 * np.pi * (y < 0) if False else 0.0
    return np.array(((x**2 + y**2)**0.5, np.arctan2(y, x) + shift, z))

def velocity_xyz_to_rθh(x, y, z, u, v, w):

    """
    Converts uvw, u velocity (u; pc/Myr), v velocity (v; pc/Myr), and w velocity (w; pc/Myr),
    cartesian galactic coordinates to vrvtvh, radial velocity (vr; pc/Myr), tangential velocity
    (vt; pc/Myr), and height velocity (vh; pc/Myr), cylindrical galactocentric left-handed
    coordinates.
    """

    # Rotate axes, and move origin
    x, y, z = ((Coordinate.ggrm @ np.array((-x, y, z))).T + Coordinate.sun_position).T
    u, v, w = ((Coordinate.ggrm @ np.array((-u, v, w))).T + Coordinate.sun_velocity).T

    # Compute norm, cosine, and sine
    norm_xy = (x**2 + y**2)**0.5
    cos_θ, sin_θ = x / norm_xy, y / norm_xy

    # Convert coordinates
    return np.array((u * cos_θ + v * sin_θ, v * cos_θ - u * sin_θ, w))


# Cylindrical galactocentric transforms

def position_rθh_to_xyz(r, θ, h):
    """
    Converts rθh, radius (r; pc), angle (θ; rad), and height (h; pc), cylindrical galactocentric
    left-handed coordinates to xyz, x position (x; pc), y position (y; pc), and z position (z; pc),
    cartesian galactic coordinates.
    """

    # Convert coordinates, and move origin
    x, y, z = (np.array((r * np.cos(θ), r * np.sin(θ), h)).T - Coordinate.sun_position).T

    # Rotate axes
    return ((Coordinate.ggrm.T @ np.array((x, y, z))).T * np.array((-1, 1, 1))).T

    # Convert coordinates
    # x, y, z = np.array((r * np.cos(θ), r * np.sin(θ), h))

    # Rotate axes, and move origin
    # return (
    #     (Coordinate.ggrm.T @ np.array((x, y, z))).T * np.array((-1, 1, 1)) +
    #     np.array((Coordinate.galactic_center_distance, 0.0, 0.0))
    # ).T

def velocity_rθh_to_xyz(r, θ, h, vr, vt, vh):
    """
    Converts vrvtvh, radial velocity (vr; pc/Myr), tangential velocity (vt; pc/Myr), and height
    velocity (vh; pc/Myr), cylindrical galactocentric left-handed coordinates to uvw, u velocity
    (u; pc/Myr), v velocity (v; pc/Myr), and w velocity (w; pc/Myr), cartesian galactic coordinates.
    """

    # Compute cosine, and sine
    cos_θ, sin_θ = np.cos(θ), np.sin(θ)

    # Convert coordinates, and move origin
    u, v, w = (
        np.array((vr * cos_θ - vt * sin_θ, vr * sin_θ + vt * cos_θ, vh)).T - Coordinate.sun_velocity
    ).T

    # Rotate axes
    return ((Coordinate.ggrm.T @ np.array((u, v, w))).T * np.array((-1, 1, 1))).T

def position_rθh_to_ξηζ(r, θ, h, t):
    """
    Converts rθh, radius (r; pc), angle (θ; rad), and height (h; pc), cylindrical galactocentric
    coordinates to ξ'η'ζ', ξ' radius (ξ'; pc), η' length (η'; pc), and ζ' height (ζ'; pc),
    curvilinear comoving coordinates, at a given epoch t (Myr).
    """

    # Convert coordinates, and move origin
    ξ, η, ζ = (
        r - Coordinate.sun_position[0],
        Coordinate.sun_position[0] * (θ - Coordinate.sun_angular_frequency * t),
        h - Coordinate.sun_position[2]
    )

    # Rotate axes
    return Coordinate.ggrm.T @ np.array((ξ, η, ζ))

def velocity_rθh_to_ξηζ(vr, vt, vh):
    """
    Converts vrvtvh, radial velocity (vr; pc/Myr), tangential velocity (vt; pc/Myr), and height
    velocity (vh; pc/Myr), cylindrical galactocentric left-handed coordinates to vξ'vη'vζ', ξ'
    velocity (vξ'; pc/Myr), η' velocity (vη'; pc/Myr), and ζ' velocity (vζ'; pc/Myr), curvilinear
    comoving coordinates.
    """

    # Convert coordinates, rotate axes, and move origin
    return Coordinate.ggrm.T @ (np.array((vr, vt, vh)).T - Coordinate.lsr_velocity).T


# Curvilinear comoving transforms

def position_ξηζ_to_rθh(ξ, η, ζ, t):
    """
    Converts ξ'η'ζ', ξ' radius (ξ'; pc), η' length (η'; pc), and ζ' height (ζ'; pc), curvilinear
    comoving coordinates to rθh, radius (r; pc), angle (θ; rad), and height (z; pc), cylindrical
    galactocentric coordinates, at a given epoch t (Myr).
    """

    # Rotate axes
    ξ, η, ζ = Coordinate.ggrm @ np.array((ξ, η, ζ))

    # Convert coordinates, and move origin
    return np.array(
        (
            ξ + Coordinate.sun_position[0],
            η / Coordinate.sun_position[0] + Coordinate.sun_angular_frequency * t,
            ζ + Coordinate.sun_position[2]
        )
    )

def velocity_ξηζ_to_rθh(vξ, vη, vζ):
    """
    Converts vξ'vη'vζ', ξ' velocity (vξ'; pc/Myr), η' velocity (vη'; pc/Myr), and ζ' velocity (vζ';
    pc/Myr), curvilinear comoving coordinates to vρvtvz, radial velocity (vr; pc/Myr), tangential
    velocity (vt; pc/Myr), and height velocity (vh; pc/Myr), cylindrical galactocentric coordinates.
    """

    # Convert coordinates, rotate axes, and move origin
    return ((Coordinate.ggrm @ np.array((vξ, vη, vζ))).T + Coordinate.lsr_velocity).T

"""
Four systems (origin and orientation) :

- Equatorial (Heliocentric, equatorial)
- Heliogalactic (Heliocentric, Galactic Center to Sun Plane)
- Galactic (Galactocentric, Galactic Plane)
- Comoving (Local Standard of Rest, Galactic Center to Sun Plane)

Four coordinates :

- Spherical
- Cartesian
- Cylindrical
- Curvilinear

One can only move up or down in either systems or coordinates. The functions are:

- Equatorial to Galactic (2)
- Galactic to Heliogalactic (2)
- Heliogalactic to Comoving (2)
- πδα_to_xyz (2)
- xyz_to_rθh (2)
- rθh_to_ξηζ (4)

For the comoving system, only the cartesian and curvilinear coordinates are defined.
"""
