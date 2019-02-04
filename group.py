# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Defines the Group class and the embeded Star classes. These classes contain the
    information and methods necessary to compute tracebacks of stars in a kinematic group.
"""

import numpy as np
from astropy import units as un
from time import strftime
from init import info, Series
from data import Data
from tools import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Group(list):
    """ Contains the values and related methods of a moving group and a list of Star objets that
        are part of it. Data can be obtained from the database or calculated from a raw data file.
    """
    def __init__(self, name: str, series: Series):
        """ Initializes a Group object and embedded Star objects from a simulated sample of stars
            in a moving group, raw data in the form a Python dictionary or from a database. This
            dataset is then moved backward in time from the initial to the final time. Distances
            are in pc, durations in Myr and velocities in pc/Myr.
        """
        self.series = series

        # Creation or retrieval of the group data in the database
        if series.from_database or series.to_database:
            self.model, self.created = GroupModel.get_or_create(name=name, series=series.name)

        # Initialization from database
        if series.from_database:
            GroupModel.initialize_from_database(GroupModel, self, self.model)

        # Initialization from a data or simulation
        else:
            # Group parameters
            self.name = name
            self.series_name = series.name
            self.date = strftime('%Y-%m-%d %H:%M:%S')
            self.initial_time = series.initial_time
            self.final_time = series.final_time
            self.duration = self.final_time - self.initial_time
            self.number_of_steps = series.number_of_steps + 1 # One more step to account for t = 0
            self.timestep = self.duration / series.number_of_steps
            self.time = np.linspace(self.initial_time, self.final_time, self.number_of_steps)

            # Stars parameters
            if series.from_data:
                self.stars_from_data(series)
            elif series.from_simulation:
                self.stars_from_simulation(series)
            self.number_of_stars = len(self)

            # Average velocity
            self.avg_velocity = np.sum(np.array(
                [star.velocity for star in self]), axis=0) / self.number_of_stars
            self.avg_velocity_error = np.sum(np.array(
                [star.velocity_error for star in self])**2, axis=0)**0.5 / self.number_of_stars

            # Barycenter
            self.barycenter = np.sum(np.array(
                [star.position for star in self]), axis=0) / self.number_of_stars
            self.barycenter_error = np.sum(np.array(
                [star.position_error for star in self])**2, axis=0)**0.5 / self.number_of_stars
            for star in self:
                star.get_distance()

            # Scatter
            self.get_scatter()
            self.scatter_age = self.time[np.argmin(self.scatter)]
            self.scatter_age_error = 0.0

            # Minimum spanning tree
            self.minimum_spanning_tree = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_error = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_points = np.zeros([self.number_of_steps, 2, 3])
            self.minimum_spanning_tree_age = 0.0
            self.minimum_spanning_tree_age_error = 0.0

            # Deletion of previous entries and creation of new entries in the database
            if series.to_database:
                GroupModel.save_to_database(GroupModel, self)

    def stars_from_data(self, series: Series):
        """ Creates a list of Star objects from a Python dictionary or CSV files containing the
            parameters including the name of the stars, their position (XYZ) and velocity (UVW),
            and the respective errors.
        """
        # No conversion
        # for star in series.stars:
        #     # Stars creation
        #     self.append(self.Star(name=star.name,
        #         velocity=star.velocity[0], velocity_error=star.velocity[2],
        #         position=star.position[0], position_error=star.position[2]))

        # From observables
        for star in series.stars:
            # Observables conversion into equatorial spherical coordinates
            (position_rδα, velocity_rvμδμα,
                position_rδα_error, velocity_rvμδμα_error) = observables_spherical(
                    *star.position[0], *star.velocity[0], *star.position[2], *star.velocity[2])
            # Equatorial spherical coordinates conversion into galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_uvw, velocity_uvw_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rvμδμα, *position_rδα_error, *velocity_rvμδμα_error)
            # Speed of light correction
            # position_xyz = position_xyz + velocity_uvw * (un.km/un.s).to(un.pc/un.Myr) * \
            #     position_rδα[0] / (299792458 * (un.m/un.s).to(un.pc/un.Myr))

            # Star creation
            self.append(self.Star(
                self, name=star.name,
                velocity=velocity_uvw * (un.km/un.s).to(un.pc/un.Myr),
                velocity_error=velocity_uvw_error * (un.km/un.s).to(un.pc/un.Myr),
                position=position_xyz,
                position_error=position_xyz_error))

    def stars_from_simulation(self, series: Series):
        """ Creates an artificial sample of star for a given number of stars and age based on
            the intial average position (XYZ) and velocity (UVW), and their respective error
            and scatter. The sample is then moved forward in time for the given age.
        """
        # Velocity array creation and conversions
        avg_velocity = np.array(series.avg_velocity) * (un.km/un.s).to(un.pc/un.Myr)
        avg_velocity_error = np.array(series.avg_velocity_error)
        avg_velocity_scatter = np.array(series.avg_velocity_scatter) * (un.km/un.s).to(un.pc/un.Myr)

        # Position array creation
        avg_position = np.array(series.avg_position)
        avg_position_error = np.array(series.avg_position_error)
        avg_position_scatter = np.array(series.avg_position_scatter)

        for star in range(1, series.number_of_stars + 1):
            # Velocity and a position based average values and scatters
            velocity_uvw = np.random.normal(avg_velocity, avg_velocity_scatter)
            position_xyz = velocity_uvw * series.age + np.random.normal(
                avg_position, avg_position_scatter) - (avg_velocity * series.age + avg_position)
            velocity_uvw  *= (un.pc/un.Myr).to(un.km/un.s)

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rvμδμα = galactic_uvw_equatorial_rvμδμα(*position_xyz, *velocity_uvw)[0]
            # velocity_rvμδμα = velocity_rvμδμα - np.array((5, 0.0, 0.0))
            position_rδα = galactic_xyz_equatorial_rδα(*position_xyz)[0]

            # Velocity and position conversion to observables
            position_obs, velocity_obs = spherical_observables(*position_rδα, *velocity_rvμδμα)[:2]

            # Velocity and position scrambling based on measurment errors and conversion back to
            # equatorial spherical coordinates
            (position_rδα, velocity_rvμδμα,
                position_rδα_error, velocity_rvμδμα_error) = observables_spherical(
                    *np.random.normal(position_obs, avg_position_error),
                    *np.random.normal(velocity_obs, avg_velocity_error),
                    *avg_position_error, *avg_velocity_error)

            # Velocity and position conversion back to galactic cartesian coordinates
            velocity_uvw, velocity_uvw_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rvμδμα, *position_rδα_error, *velocity_rvμδμα_error)
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)

            # Star creation
            self.append(self.Star(
                self, name='star_{}'.format(star),
                velocity=velocity_uvw * (un.km/un.s).to(un.pc/un.Myr),
                velocity_error=velocity_uvw_error * (un.km/un.s).to(un.pc/un.Myr),
                position=position_xyz, position_error=position_xyz_error))

    def get_scatter(self):
        """ Computes the xyz and total scatter of a group and their respective error for all
            timesteps, filters stars farther than 3σ from the barycenter from the calculations and
            compensates for the drift in minimal scatter age due to measurement errors.
        """
        # XYZ scatter
        self.scatter_xyz = np.std([star.position for star in self], axis=0)
        self.scatter_xyz_error = self.barycenter_error

        # 3D scatter
        self.scatter = np.sum(self.scatter_xyz**2, axis=1)**0.5 - np.sum(
            self.barycenter_error[0]**2 + self.avg_velocity_error**2 \
            * np.expand_dims(self.time, axis=0).T**2, axis=1)**0.5
        self.scatter_error = np.sum(
            (self.scatter_xyz * self.scatter_xyz_error)**2, axis=1)**0.5 / self.scatter
        # !!! Add recursive function to filters stars farther than 3σ from the barycenter here !!!

    class Star:
        """ Contains the values and related methods of a star in a kinematic group.
        """
        def __init__(self, group, **parameters):
            """ Initializes a Star objects with at least a name, velocity, velocity error, and
                intial position and position error. More parameters can be added (when initializing
                from a database for instance).
            """
            # Initialization
            self.group = group
            vars(self).update(parameters)

            # Position over time
            if not self.group.series.from_database:
                self.get_position()

        def __repr__(self):
            """ Returns a string of name of the star.
            """
            return self.name

        def get_position(self):
            """ Computes the position and position error of the star and errors for all timesteps.
                Both self.position and self.position_errro are redefined with these new values.
            """
            # Position
            self.position = self.position - self.velocity * np.expand_dims(self.group.time, axis=0).T
            self.position_error = (self.position_error**2
                + (self.velocity_error * np.expand_dims(self.group.time, axis=0).T)**2)**0.5

        def get_distance(self):
            """ Computes the relative position and distance from the barycenter and their
                respective errors for all timesteps.
            """
            # Relative position
            self.relative_position = self.position - self.group.barycenter
            self.relative_position_error = (
                self.position_error**2 + self.group.barycenter_error**2)**0.5

            # Distance
            self.distance = np.sum(self.relative_position**2, axis=1)**0.5
            self.distance_error = np.sum((self.relative_position
                * self.relative_position_error)**2, axis=1)**0.5 / self.distance
