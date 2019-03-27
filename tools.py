# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" tools.py: Defines Quantity class to handle n dimension values with unit conversions and error
    handling, and Coordinates class to handle coordinates transformation, and a montecarlo function.
"""

import numpy as np
from astropy import units as un
from math import cos, sin, asin, atan2, pi as π, degrees, radians
from series import info

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Coordinates:
    """ Contains the values and related methods of coordinates, including its position, velocity,
        their errors and units, in both cartesian and spherical coordinates systems, as well as
        method to convert between them.
    """

    # J2000.0 Galactic-equatorial rotation matrix from Liu et al. (2018) 1010.3773
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

    # J2000.0 Galactic longitude (l) of the Celestial North pole (δ = 90°) from Liu et al. (2018) 1010.3773
    l_north = radians(122.931925267)

    def __init__(self, position, velocity=None):
        """ Initializes a Coordinates object from a Quantity objects representing n positions
            (shape = (n, 3)) and optionnally n corresponding velocities (shape = (n, 3)). Position
            and velocity must be broadcast together and can either be observables, cartesian
            coordiantes or spherical coordinates.
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
        if position.shape != shape: # np.full() Doesn't work on quantity object !!!
            position = np.full(shape, position)
        if velocity.shape != shape:
            velocity = np.full(shape, velocity)

        # Conversion of position in cartesian or spherical coordinates system
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

        # Conversion of velocity in cartesian or spherical coordinates system
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
        """ Converts a Coordinate object from its original coordinate system and axis to a new
            coordinate system and axis, by wrapping conversion functions together.
        """
        pass

def xyz_to_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ cartesian coordinates position vector (pc) to a rδα (distance (r; pc),
        declination (δ, DEC; deg) and right ascension (α, RA; deg)) spherical coordinates position
        vector (observables), along with measurement errors. x, y and z can't all be null.
    """

    # Norm calculation
    norm_2, norm_xy_2 = (x**2 + y**2 + z**2), (x**2 + y**2)
    norm, norm_xy = norm_2**0.5, norm_xy_2**0.5

    # Distance and angles calculation
    values = np.array(
        (norm, asin(z / norm), atan2(y, x) + (2 * π if y < 0.0 else 0.0))
    ) * np.array((1.0, un.rad.to(un.deg), un.rad.to(un.deg))) # Angle conversions from rad to deg

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(
            np.array(
                (   # Partial derivatives of r: dr/dx, dr/dy and dr/dz
                    (x / norm, y / norm, z / norm),
                    # Partial derivatives of δ: dδ/dx, dδ/dy and dδ/dz
                    (-x * z / (norm_2 * norm_xy),  -y * z / (norm_2 * norm_xy),  norm_xy / norm_2),
                    # Partial derivatives of α: dα/dx, dα/dy and dα/dz
                    (-y / norm_xy_2, x / norm_xy_2, 0.0)
                )
            )**2,
            np.array((Δx, Δy, Δz))**2 # Angle conversions from rad to deg
        )**0.5 * np.array((1.0, un.rad.to(un.deg), un.rad.to(un.deg)))
        return values, errors

def rδα_to_xyz(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Converts a rδα (distance (r; pc), declination (δ, DEC; deg) and right ascension (α, RA; deg))
        spherical coordinates position vector (observables) to a XYZ cartesian coordinates position
        vector (pc), along with measurement errors.
    """

    # Angle conversions from deg to rad
    δ, α, Δδ, Δα = np.array((δ, α, Δδ, Δα)) * un.deg.to(un.rad)
    # Cosine and sine calculation
    cos_δ, sin_δ, cos_α, sin_α = cos(δ), sin(δ), cos(α), sin(α)

    # Position calculation
    values = np.array((r * cos_δ * cos_α, r * cos_δ * sin_α, r * sin_δ))

    # Errors calculation
    if not np.array((Δr, Δδ, Δα)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(
            np.array(
                (   # Partial derivatives of x: dx/dr, dr/dδ and dx/dα
                    (cos_δ * cos_α, r * sin_δ * cos_α, -r * cos_δ * sin_α),
                    # Partial derivatives of y: dy/dr, dy/dδ and dy/dα
                    (cos_δ * sin_α, r * sin_δ * sin_α, r * cos_δ * cos_α),
                    # Partial derivatives of z: dz/dr, dz/dδ and dz/dα
                    (sin_δ, -r * cos_δ, 0.0)
                )
            )**2,
            np.array((Δr, Δδ, Δα))**2
        )**0.5
        return values, errors

def uvw_to_rvμδμα(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts a UVW cartesian coordinates velocity vector (km/s) to a rvµδµα (radial velocity
        (rv; km/s), declination proper motion (μδ; mas/yr) and right ascension proper motion
        (µα; mas/yr)) spherical coordinates velocity vector (observables), along with measurement
        errors. x, y and z (pc) can't all be null.
    """

    # Distance conversions from pc to km
    x, y, z, Δx, Δy, Δz = np.array((x, y, z, Δx, Δy, Δz)) * un.pc.to(un.km)
    # Norm calculation
    norm_2, norm_xy_2 = (x**2 + y**2 + z**2), (x**2 + y**2)
    norm, norm_xy = norm_2**0.5, norm_xy_2**0.5

    # Radial velocity and proper motion calculation
    values = np.array(
        (
            ((u * x) + (v * y) + (z * w)) / norm,
            (w * norm_xy - ((u * x * z) + (v * y * z)) / norm_xy) / norm_2,
            ((v * x) - (u * y)) / norm_xy_2
        ) # Angular velocity conversions from rad/s to mas/yr
    ) * np.array((1.0, (un.rad/un.s).to(un.mas/un.yr), (un.rad/un.s).to(un.mas/un.yr)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz, Δu, Δv, Δw)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(
            np.array(
                (
                    (   # Partial derivatives of rv: d(rv)/dx, d(rv)/dy, d(rv)/dz, d(rv)/du, d(rv)/dv and d(rv)/dw)
                        (u * (y**2 + z**2) - v * x * y - w * x * z) / norm**3,
                        (v * (x**2 + z**2) - y * (u*x + w*z)) / norm**3,
                        (w * norm_xy_2 - z * (u*x + v*y)) / norm**3,
                        x / norm, y / norm, z / norm
                    ), (# Partial derivatives of μδ: (d(μδ)/dx, d(μδ)/dy, d(μδ)/dz, d(μδ)/du, d(μδ)/dv and d(μδ)/dw)
                        (u * z * (2 * x**4 + x**2 * y**2 - y**2 * (y**2 + z**2)) + v * x * y * z * (3 * norm_xy_2 + z**2) - w * x * norm_2 * (norm_xy_2 - z**2)) / (norm_xy**3 * norm_2**2),
                        (u * x * y * z * (3 * norm_xy_2 + z**2) - v * z * (x**4 + x**2 * (z**2 - y**2) - 2 * y**4) - w * y * norm_xy_2 * norm_2) / (norm_xy**3 * norm_2**2),
                        (-u * x * (norm_xy_2 - z**2) - v * y * (norm_xy_2 - z**2) - 2 * w * z * norm_2) / (norm_xy * norm_2**2),
                        -(x * z) / (norm_xy * norm_2), -(y * z) / (norm_xy * norm_2), norm_xy / norm_2
                    ), (# Partial derivatives of μα: (d(μα)/dx, d(μα)/dy d(μα)/dz, d(μα)/du, d(μα)/dv and d(μα)/dw)
                        (v * (x**2 - y**2) - 2 * u * x * y) / norm_xy,
                        (u * (x**2 - y**2) + 2 * v * x * y) / norm_xy,
                        0.0, y / norm_xy_2, -z / norm_xy_2, 0.0
                    )
                )
            )**2,
            np.array((Δx, Δy, Δz, Δu, Δv, Δw))**2 # Angular velocity conversions from rad/s to mas/yr
        )**0.5 * np.array((1.0, (un.rad/un.s).to(un.mas/un.yr), (un.rad/un.s).to(un.mas/un.yr)))
        return values, errors

def rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts a rvµδµα (radial velocity (rv; km/s), declination proper motion (µδ; mas/yr)
        and right ascension proper motion (µα; mas/yr)) spherical coordinates velocity vector
        (observables) to a UVW cartesian coordinates velocity vector (km/s), along with measurement
        errors.
    """

    # Distance conversion from pc to km
    r, Δr = np.array((r, Δr)) * un.pc.to(un.km)
    # Angle conversions from deg to rad
    δ, α, Δδ, Δα = np.array((δ, α, Δδ, Δα)) * un.deg.to(un.rad)
    # Angular velocity conversion from mas/yr to rad/s
    μδ, μα, Δμδ, Δμα = np.array((μδ, μα, Δμδ, Δμα)) * (un.mas/un.yr).to(un.rad/un.s)
    # Cosine and sine calculation
    cos_δ, sin_δ, cos_α, sin_α = cos(δ), sin(δ), cos(α), sin(α)

    # Velocity calculation
    values = np.array(
        (
            rv * (cos_δ * cos_α) - μδ * (r * sin_δ * cos_α) - μα * (r * cos_δ * sin_α),
            rv * (cos_δ * sin_α) - μδ * (r * sin_δ * sin_α) + μα * (r * cos_δ * cos_α),
            rv * sin_δ + μδ * (r * cos_δ)
        )
    )

    # Errors calculation
    if not np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(
            np.array(
                (
                    (   # Partial derivatives of u: du/dr, du/dδ, du/dα, du/d(rv), du/d(µδ) and du/d(µα)
                        -μδ * (sin_δ * cos_α) - μα * (cos_δ * sin_α),
                        -rv * (sin_δ * cos_α) - μδ * (r * cos_δ * cos_α) + μα * (r * sin_δ * sin_α),
                        -rv * (cos_δ * sin_α) + μδ * (r * sin_δ * sin_α) - μα * (r * cos_δ * cos_α),
                        cos_δ * cos_α, -r * sin_δ * cos_α, -r * cos_δ * sin_α
                    ), (# Partial derivatives of v: dv/dr, dv/dδ, dv/dα, dv/d(rv), dv/d(µδ) and dv/d(µα)
                        -μδ * (sin_δ * sin_α) + μα * (cos_δ * cos_α),
                        -rv * (sin_δ * sin_α) - μδ * (r * cos_δ * sin_α) - μα * (r * sin_δ * cos_α),
                        rv * (cos_δ * cos_α) - μδ * (r * sin_δ * cos_α) - μα * (r * cos_δ * sin_α),
                        cos_δ * sin_α, -r * sin_δ * sin_α, r * cos_δ * cos_α
                    ), (# Partial derivatives of w: (dw/dr, dw/dδ, dw/dα, dw/d(rv), dw/d(µδ) and dw/d(µα)
                        μδ * cos_δ, rv * cos_δ - μδ * r * sin_δ, 0.0, sin_δ, r * cos_δ, 0.0
                    )
                )
            )**2,
            np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα))**2
        )**0.5
        return values, errors

def equatorial_galactic_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian position (x, y, z) or velocity (u, v, w) from a galactic to an
        equatorial plane. All arguments must have the same units.
    """

    # Coordinates calculation
    values = np.dot(Coordinates.germ, np.array((x, y, z)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(Coordinates.germ**2, np.array((Δx, Δy, Δz))**2)**0.5
        return values, errors

def galactic_equatorial_xyz(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Rotates a cartesian position (x, y, z) or velocity (u, v, w) from an equatorial to a
        galactic plane. All arguments must have the same units.
    """

    # Coordinates calculation
    values = np.dot(Coordinates.germ.T, np.array((x, y, z)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return values, np.zeros(3)
    else:
        errors = np.dot(Coordinates.germ.T**2, np.array((Δx, Δy, Δz))**2)**0.5
        return values, errors

def galactic_xyz_equatorial_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ (pc) galactic cartesian coordinates position vector to a rδα (distance
        (r; pc), declination (δ, DEC; deg) and right ascension (α, RA; deg)) equatorial spherical
        coordinates position vector (observables), along with measurement errors. x, y and z can't
        all be null.
    """

    values, errors = galactic_equatorial_xyz(x, y, z, Δx, Δy, Δz)
    return xyz_to_rδα(*values, *errors)

def equatorial_rδα_galactic_xyz(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Converts a rδα (distance (r; pc), declination (δ, DEC; deg) and right ascension (α, RA; deg))
        equatorial spherical coordinates position vector to a XYZ (pc) galactic cartesian coordinates
        position vector, along with measurement errors.
    """

    values, errors = rδα_to_xyz(r, δ, α, Δr, Δδ, Δα)
    return equatorial_galactic_xyz(*values, *errors)

def galactic_uvw_equatorial_rvμδμα(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts a UVW (km/s) galactic cartesian coordinates velocity vector to a rvµδµα (radial
        velocity (rv; km/s), declination proper motion (µδ; mas/yr) and right ascension proper
        motion (μδ; mas/yr)) equatorial spherical coordinates velocity vector, along with
        measurement errors.
    """

    position_values, position_errors = galactic_equatorial_xyz(x, y, z, Δx, Δy, Δz)
    velocity_values, velocity_errors = galactic_equatorial_xyz(u, v, w, Δu, Δv, Δw)
    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def equatorial_rvμδμα_galactic_uvw(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts a rvµδµα (radial velocity (rv; km/s), declination proper motion (µδ; mas/yr)
        and right ascension proper motion (µα; mas/yr)) equatorial spherical coordinates velocity
        vector to a UVW (km/s) galactic cartesian coordinates velocity vector, along with
        measurement errors.
    """

    values, errors = rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)
    return equatorial_galactic_xyz(*values, *errors)

def equatorial_xyz_galactic_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ (pc) equatorial cartesian coordinates position vector to a rδα (distance
        (r; pc), declination (δ, DEC; deg) and right ascension (α, RA; deg)) galactic spherical
        coordinates position vector, along with measurement errors. x, y and z can't all be null.
    """

    values, errors = equatorial_galactic_xyz(x, y, z, Δx, Δy, Δz)
    return xyz_to_rδα(*values, *errors)

def galactic_rδα_equatorial_xyz(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Converts a rδα (distance (r; pc), declination (δ, DEC; deg) and right ascension (α, RA; deg))
        galactic spherical coordinates position vector to a XYZ (pc) equatorial cartesian coordinates
        position vector, along with measurement errors.
    """

    values, errors = rδα_to_xyz(r, δ, α, Δr, Δδ, Δα)
    return galactic_equatorial_xyz(*values, *errors)

def equatorial_uvw_galactic_rvµδµα(x, y, z, u, v, w, Δx=0, Δy=0, Δz=0, Δu=0, Δv=0, Δw=0):
    """ Converts a UVW (km/s) equatorial cartesian coordinates velocity vector to a rvµδµα (radial
        velocity (rv; km/s), declination proper motion (µδ; mas/yr) and right ascension proper
        motion (μδ; mas/yr)) galactic spherical coordinates velocity vector, along with measurement
        errors.
    """

    position_values, position_errors = equatorial_galactic_xyz(x, y, z, Δx, Δy, Δz)
    velocity_values, velocity_errors = equatorial_galactic_xyz(u, v, w, Δu, Δv, Δw)
    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def galactic_rvµδµα_equatorial_uvw(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts a rvµδµα (radial velocity (rv; km/s), declination proper motion (µδ; mas/yr)
        and right ascension proper motion (μα; mas/yr)) equatorial spherical coordinates velocity
        vector to a UVW (km/s) galactic cartesian coordinates velocity vector, along with
        measurement errors.
    """

    values, errors = rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)
    return galactic_equatorial_xyz(*values, *errors)

def equatorial_galactic_rδα(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Rotates an equatorial coordinates (r, δ, α) vector to a galactic coordinates (r, δ, α)
        vector.
    """

    values, errors = equatorial_rδα_galactic_xyz(r, δ, α, Δr, Δδ, Δα)
    return xyz_to_rδα(*values, *errors)

def galactic_equatorial_rδα(r, δ, α, Δr=0, Δδ=0, Δα=0):
    """ Rotates a galactic coordinates (r, δ, α) vector to an equatorial coordinates (r, δ, α)
        vector.
    """

    values, errors = galactic_rδα_equatorial_xyz(r, δ, α, Δr, Δδ, Δα)
    return xyz_to_rδα(*values, *errors)

def equatorial_galactic_rvμδμα(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Rotates an equatorial velocity (rv, μδ, μα) vector to a galactic velocity (rv, μδ, μα)
        vector.
    """

    position_values, position_errors = equatorial_rδα_galactic_xyz(r, δ, α, Δr, Δδ, Δα)
    velocity_values, velocity_errors = equatorial_rvµδµα_galactic_uvw(
        r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)
    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def galactic_equatorial_rvμδμα(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Rotates a galactic velocity (rv, μδ, μα) vector to an equatorial velocity (rv, μδ, μα)
        vector.
    """

    position_values, position_errors = galactic_rδα_equatorial_xyz(r, δ, α, Δr, Δδ, Δα)
    velocity_values, velocity_errors = galactic_rvµδµα_equatorial_uvw(
        r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα)
    return uvw_to_rvμδμα(*position_values, *velocity_values, *position_errors, *velocity_errors)

def observables_spherical(p, δ, α, rv, μδ, μα_cos_δ, Δp=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα_cos_δ=0):
    """ Converts observations (paralax (p; mas), declination (δ, DEC; deg) and right ascension
        (α, RA; deg)), radial velocity (rv; km/s), declination proper motion (µδ; mas/yr) and
        and right ascension proper motion * cos(δ) (μα_cos_δ; mas/yr)) into equatorial spherical
        coordinates (distance (r; pc), declination (δ, DEC; deg) and right ascension (α, RA; deg)),
        radial velocity (rv; km/s), declination proper motion (µδ; mas/yr) and right ascension
        proper motion (μα_cos_δ; mas/yr)), along with measurement errors.
    """

    # Cosine calculation
    cos_δ = cos(radians(δ))

    # Values calculation
    position = np.array((un.arcsec.to(un.mas) / p, δ, α))
    velocity = np.array((rv, μδ, μα_cos_δ / cos_δ))

    # Errors calculation
    if not np.array((Δp, Δδ, Δα, Δrv, Δμδ, Δμα_cos_δ)).any():
        return position, velocity, np.zeros(3), np.zeros(3)
    else:
        return position, velocity, np.array((Δp * un.arcsec.to(un.mas) / p**2, Δδ, Δα)), \
            np.array((Δrv, Δμδ, ((Δμα_cos_δ / μα_cos_δ)**2 + (Δδ / δ)**2)**0.5 * μα_cos_δ / cos_δ))

def spherical_observables(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts equatorial spherical coordinates (distance (r; pc), declination (δ, DEC; deg)
        and right ascension (α, RA; deg)), radial velocity (rv; km/s), declination proper motion
        (µδ; mas/yr) and right ascension proper motion (μα_cos_δ; mas/yr)) into observations
        (paralax (p; mas), declination (δ, DEC; deg) and right ascension (α, RA; deg)), radial
        velocity (rv; km/s), declination proper motion (µδ; mas/yr) and and right ascension proper
        motion * cos(δ) (μα_cos_δ; mas/yr)), along with measurement errors.
    """

    # Cosine calculation
    cos_δ = cos(radians(δ))

    # Values calculation
    position = np.array((un.arcsec.to(un.mas) / r, δ, α))
    velocity = np.array((rv, μδ, μα * cos_δ))

    # Errors calculation
    if not np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα)).any():
        return position, velocity, np.zeros(3), np.zeros(3)
    else:
        return position, velocity, np.array((Δr * un.arcsec.to(un.mas) / r**2, Δδ, Δα)), \
            np.array((Δrv, Δμδ, ((Δμα / μα)**2 + (Δδ / δ)**2)**0.5 * μα * cos_δ))

class Quantity:
    """ Contains a value and unit, its associated error and unit if necessary. Error is converted
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
                un.core.IrreducibleUnit, un.core.Unit):
            units = [units]
        if type(units) in (tuple, list, np.ndarray):
            try:
                self.units = np.full(self.shape, np.squeeze(np.vectorize(un.Unit)(units)))
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
                un.core.IrreducibleUnit, un.core.Unit):
            error_units = [error_units]
        if type(error_units) in (tuple, list, np.ndarray):
            try:
                self.error_units = np.full(
                    self.shape, np.squeeze(np.vectorize(un.Unit)(error_units)))
            except ValueError:
                raise ValueError(
                    "Error units ({}) cannot be broadcast to the shape of values ({}).".format(
                        units, self.values.shape))
        else:
            raise TypeError("{} is not a supported type "
                "(str, *Unit, tuple, list or ndarray) for error units.".format(type(error_units)))
        self.error_physical_types = np.vectorize(lambda unit: unit.physical_type)(self.error_units)

        # Conversion of errors into value units
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

            return array.flatten()[0] if np.equal(np.array(array.shape), 1).all() \
                else array.tolist()

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
            default_units = {
                'time': un.Myr,
                'length': un.pc,
                'speed': (un.pc/un.Myr),
                'angle': un.rad,
                'angular speed': (un.rad/un.Myr)}

            units = np.vectorize(lambda unit: default_units[unit.physical_type] \
                if unit.physical_type in default_units.keys() else unit)(self.units)
            factors = np.vectorize(lambda self_unit, unit: self_unit.to(unit))(self.units, units)

        else:
            # Import of units
            if type(units) in (
                    str, un.core.PrefixUnit, un.core.CompositeUnit,
                    un.core.IrreducibleUnit, un.core.Unit):
                units = [units]
            if type(units) in (tuple, list, np.ndarray):
                try:
                    units = np.full(self.units.shape, np.vectorize(un.Unit)(units))
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
            factors = np.vectorize(
                lambda self_unit, unit: self_unit.to(unit)
            )(np.full(units.shape, self.units), units)

        return self.new(Quantity(self.values * factors, units, self.errors * factors))

    def new(self, new):
        """ Transferts all optional parameters from the old self and returns a new value. """

        optional = {key: vars(self)[key] for key in filter(
            lambda key: key not in vars(new), vars(self).keys())}
        vars(new).update(optional)

        return new

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
