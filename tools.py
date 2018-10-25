# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Provide necessary tools to transform coordinates

import numpy as np
from astropy import coordinates as astropy_coordinates, units as un
from traceback import format_exc
from logging import basicConfig, info, warning, INFO
from time import strftime
from os.path import join
from config import *

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

def xyz_to_rδα(x, y, z, Δx, Δy, Δz):
    """ Converts a XYZ cartesian coordinates position vector (pc) to a rδα (distance (r; pc),
        declination (δ, DEC; deg) and right ascension (α, RA; deg)) spherical coordinates position
        vector (observables), along with measurement errors.
    """
    return np.array((
        np.array(( # Coordinates calculation
            np.linalg.norm((x, y, z)),
            np.arcsin(z/np.linalg.norm((x, y, z))),
            np.arccos(x/np.linalg.norm((x, y))) + (
                3*np.pi/2 if y < 0 and x > 0 else np.pi/2 if y < 0 else 0.0
            )
        )),
        np.sqrt( # Errors calculation
            np.dot(
                np.array((
                    ( # Partial derivatives of r : dr/dx, dr/dy and dr/dz
                        x/np.linalg.norm((x, y, z)),
                        y/np.linalg.norm((x, y, z)),
                        z/np.linalg.norm((x, y, z))
                    ),
                    ( # Partial derivatives of δ : dδ/dx, dδ/dy and dδ/dz
                        -x*z / ((x**2 + y**2 + z**2) * np.linalg.norm((x, y))),
                        -y*z / ((x**2 + y**2 + z**2) * np.linalg.norm((x, y))),
                        np.linalg.norm((x, y)) / (x**2 + y**2 + z**2)
                    ),
                    ( # Partial derivatives of α : dα/dx, dα/dy and dα/dz
                        -y/(x**2 + y**2), x/(x**2 + y**2), 0
                    )
                ))**2,
                np.array((Δx, Δy, Δz))**2
            )
        ) # Angle conversions from rad to deg
    )) * np.array((1.0, un.rad.to(un.deg), un.rad.to(un.deg)))

def rδα_to_xyz(r, δ, α, Δr, Δδ, Δα):
    """ Converts a rδα (distance (r; pc), declination (δ, DEC; deg) and right ascension (α, RA; deg))
        spherical coordinates position vector (observables) to a XYZ cartesian coordinates position
        vector (pc), along with measurement errors.
    """
    # Angle conversions from deg to rad
    δ, α, Δδ, Δα = np.array((δ, α, Δδ, Δα)) * un.deg.to(un.rad)

    return np.array((
        np.array(( # Coordinates calculation
            r*np.cos(δ)*np.cos(α),
            r*np.cos(δ)*np.sin(α),
            r*np.sin(δ)
        )),
        np.sqrt( # Errors calculation
            np.dot(
                np.array((
                    ( # Partial derivatives of x : dx/dr, dr/dδ and dx/dα
                        np.cos(δ)*np.cos(α),
                        r*np.sin(δ)*np.cos(α),
                        -r*np.cos(δ)*np.sin(α)
                    ),
                    ( # Partial derivatives of y : dy/dr, dy/dδ and dy/dα
                        np.cos(δ)*np.sin(α),
                        r*np.sin(δ)*np.sin(α),
                        r*np.cos(δ)*np.cos(α)
                    ),
                    ( # Partial derivatives of z : dz/dr, dz/dδ and dz/dα
                        np.sin(δ), -r*np.cos(δ), 0
                    )
                ))**2,
                np.array((Δr, Δδ, Δα))**2
            )
        )
    ))

def uvw_to_rvμδμα(x, y, z, u, v, w, Δx, Δy, Δz, Δu, Δv, Δw):
    """ Converts a UVW cartesian coordinates velocity vector (km/s) to a rvµδµα (radial velocity
        (rv; km/s), declination proper motion (μδ; mas/yr) and right ascension proper motion
        (µα; mas/yr)) spherical coordinates velocity vector (observables), along with measurement
        errors.
    """
    # Distance conversions from pc to km
    x, y, z, Δx, Δy, Δz = np.array((x, y, z, Δx, Δy, Δz)) * un.pc.to(un.km)

    return np.array((
        np.array(( # Coordinates calculation
            (u*x + v*y + z*w) / np.linalg.norm((x, y, z)),
            (w * np.linalg.norm((x, y)) - ((u*x*z + v*y*z) / np.linalg.norm((x, y)))) / (x**2 + y**2 + z**2),
            (v*x - u*y) / (x**2 + y**2) # Angular velocity conversions from rad/s to mas/yr
        )),
        np.sqrt( # Errors calculation
            np.dot(
                np.array((
                    ( # Partial derivatives of rv : d(rv)/dx, d(rv)/dy, d(rv)/dz, d(rv)/du, d(rv)/dv and d(rv)/dw)
                        (u*(y**2 + z**2) - v*x*y - w*x*z) / (x**2 + y**2 + z**2)**(3/2),
                        (v*(x**2 + z**2) - y*(u*x + w*z)) / (x**2 + y**2 + z**2)**(3/2),
                        (w*(x**2 + y**2) - z*(u*x + v*y)) / (x**2 + y**2 + z**2)**(3/2),
                        x / np.linalg.norm((x, y, z)),
                        y / np.linalg.norm((x, y, z)),
                        z / np.linalg.norm((x, y, z))
                    ),
                    ( # Partial derivatives of μδ : (d(μδ)/dx, d(μδ)/dy, d(μδ)/dz, d(μδ)/du, d(μδ)/dv and d(μδ)/dw)
                        (u*z*(2*(x**4) + (x**2)*(y**2) - (y**2)*(y**2 + z**2)) + v*x*y*z*(3*(x**2) + 3*(y**2) + z**2) - w*x*(x**2 + y**2)*(x**2 + y**2 - z**2)) / (((x**2 + y**2)**(3/2))*((x**2 + y**2 + z**2)**2)),
                        (u*x*y*z*(3*(x**2) + 3*(y**2) + z**2) - v*z*((x**4) + (x**2)*(z**2 - y**2) - 2*(y**4)) - w*y*(x**2 + y**2)*(x**2 + y**2 - z**2)) / (((x**2 + y**2)**(3/2))*((x**2 + y**2 + z**2)**2)),
                        (-u*x*(x**2 + y**2 - z**2) - v*y*(x**2 + y**2 - z**2) - 2*w*z*(x**2 + y**2)) / (np.sqrt(x**2 + y**2)*((x**2 + y**2 + z**2)**2)),
                        -(x * z) / (np.sqrt(x**2 + y**2) * (x**2 + y**2 + z**2)),
                        -(y * z) / (np.sqrt(x**2 + y**2) * (x**2 + y**2 + z**2)),
                        np.sqrt(x**2 + y**2) / (x**2 + y**2 + z**2)
                    ),
                    ( # Partial derivatives of μα : (d(μα)/dx, d(μα)/dy d(μα)/dz, d(μα)/du, d(μα)/dv and d(μα)/dw)
                        (v*(x**2 - y**2) - 2*u*x*y) / (x**2 + y**2)**2,
                        (u*(x**2 - y**2) + 2*v*x*y) / (x**2 + y**2)**2,
                        0, y/(x**2 + y**2), -z/(x**2 + y**2), 0
                    )
                ))**2,
                np.array((Δx, Δy, Δz, Δu, Δv, Δw))**2
            )
        ) # Angular velocity conversions from rad/s to mas/yr
    )) * np.array((1.0, (un.rad/un.s).to(un.mas/un.yr), (un.rad/un.s).to(un.mas/un.yr)))

def rvμδμα_to_uvw(r, δ, α, rv, μδ, μα, Δr, Δδ, Δα, Δrv, Δμδ, Δμα):
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

    return np.array((
        np.array(( # Coordinates calculation
            rv*np.cos(δ)*np.cos(α) - μα*r*np.cos(δ)*np.sin(α) - μδ*r*np.sin(δ)*np.cos(α),
            rv*np.cos(δ)*np.sin(α) + μα*r*np.cos(δ)*np.cos(α) - μδ*r*np.sin(δ)*np.sin(α),
            rv*np.sin(δ) + μδ*r*np.cos(δ)
        )),
        np.sqrt( # Errors calculation
            np.dot(
                np.array((
                    ( # Partial derivatives of u : du/dr, du/dδ, du/dα, du/d(rv), du/d(µδ) and du/d(µα)
                        -μα*np.cos(δ)*np.sin(α) - μδ*np.sin(δ)*np.cos(α),
                        -rv*np.sin(δ)*np.cos(α) + μα*r*np.sin(δ)*np.sin(α) - μδ*r*np.cos(δ)*np.cos(α),
                        -rv*np.cos(δ)*np.sin(α) - μα*r*np.cos(δ)*np.cos(α) + μδ*r*np.sin(δ)*np.sin(α),
                        np.cos(δ)*np.cos(α), -r*np.sin(δ)*np.cos(α), -r*np.cos(δ)*np.sin(α)
                    ),
                    ( # Partial derivatives of v : dv/dr, dv/dδ, dv/dα, dv/d(rv), dv/d(µδ) and dv/d(µα)
                        -μδ*np.sin(δ)*np.sin(α) + μα*np.cos(δ)*np.cos(α),
                        -rv*np.sin(δ)*np.sin(α) - μδ*r*np.cos(δ)*np.sin(α) - μα*r*np.sin(δ)*np.cos(α),
                        rv*np.cos(δ)*np.cos(α) - μδ*r*np.sin(δ)*np.cos(α) - μα*r*np.cos(δ)*np.sin(α),
                        np.cos(δ)*np.sin(α), -r*np.sin(δ)*np.sin(α), r*np.cos(δ)*np.cos(α)
                    ),
                    ( # Partial derivatives of w : (dw/dr, dw/dδ, dw/dα, dw/d(rv), dw/d(µδ) and dw/d(µα)
                        μδ*np.cos(δ), rv*np.cos(δ) - μδ*r*np.sin(δ), 0, np.sin(δ), r*np.cos(δ), 0
                    )
                ))**2,
                np.array((Δr, Δδ, Δα, Δrv, Δμδ, Δμα))**2
            )
        )
    ))
