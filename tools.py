# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" tools.py: Defines coordinates transforms.
"""

import numpy as np
from astropy import units as un
from logging import basicConfig, info, warning, INFO
from time import strftime
from os.path import join
from math import cos, sin, asin, atan2, pi as π
from init import *

# Configuration of the log file
basicConfig(
    filename=logs_path, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

def xyz_to_rδα(x, y, z, Δx=0, Δy=0, Δz=0):
    """ Converts a XYZ cartesian coordinates position vector (pc) to a rδα (distance (r; pc),
        declination (δ, DEC; deg) and right ascension (α, RA; deg)) spherical coordinates position
        vector (observables), along with measurement errors. X, Y and Z can't all be null.
    """
    # Norm calculation
    norm_2, norm_xy_2 = (x**2 + y**2 + z**2), (x**2 + y**2)
    norm, norm_xy = norm_2**0.5, norm_xy_2**0.5

    # Distance and coordinates calculation
    output_values = np.array(
        (norm, asin(z / norm), atan2(y, x) + (2 * π if y < 0.0 else 0.0))
    ) * np.array((1.0, un.rad.to(un.deg), un.rad.to(un.deg))) # Angle conversions from rad to deg

    # Errors calculation
    if not np.array((Δx, Δy, Δz)).any():
        return output_values
    else:
        output_errors = np.dot(
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
        return (output_values, output_errors)

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
    output_values = np.array(
        (r * cos_δ * cos_α, r * cos_δ * sin_α, r * sin_δ)
    )

    # Errors calculation
    if not np.array((Δr, Δδ, Δα)).any():
        return output_values
    else:
        output_errors = np.dot(
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
        return (output_values, output_errors)

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
    output_values = np.array(
        (
            ((u * x) + (v * y) + (z * w)) / norm,
            (w * norm_xy - ((u * x * z) + (v * y * z)) / norm_xy) / norm_2,
            ((v * x) - (u * y)) / norm_xy_2
        ) # Angular velocity conversions from rad/s to mas/yr
    ) * np.array((1.0, (un.rad/un.s).to(un.mas/un.yr), (un.rad/un.s).to(un.mas/un.yr)))

    # Errors calculation
    if not np.array((Δx, Δy, Δz, Δu, Δv, Δw)).any():
        return output_values
    else:
        output_errors = np.dot(
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
        return (output_values, output_errors)

def rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr=0, Δδ=0, Δα=0, Δrv=0, Δμδ=0, Δμα=0):
    """ Converts a rvµδµα (radial velocity (rv; km/s), declination proper motion (µα; mas/yr)
        and right ascension proper motion (μδ; mas/yr)) spherical coordinates velocity vector
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
    output_values = np.array(
        (
            rv * (cos_δ * cos_α) - μδ * (r * sin_δ * cos_α) - μα * (r * cos_δ * sin_α),
            rv * (cos_δ * sin_α) - μδ * (r * sin_δ * sin_α) + μα * (r * cos_δ * cos_α),
            rv * sin_δ + μδ * (r * cos_δ)
        )
    )

    # Errors calculation
    if not np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα)).any():
        return output_values
    else:
        output_errors = np.dot(
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
        return (output_values, output_errors)
