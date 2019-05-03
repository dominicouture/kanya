# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Defines the Group class and the embeded Star classes. These classes contain the
    information and methods necessary to compute tracebacks of stars in a kinematic group.
"""

import numpy as np
from astropy import units as un
from series import info
from data import Data
from tools import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Group(list):
    """ Contains the values and related methods of a moving group and a list of Star objets that
        are part of it. Data can be obtained from the database or calculated from a raw data file.
    """

    def __init__(self, series, **values):
        """ Initializes a Group object and embedded Star objects from a simulated sample of stars
            in a moving group, raw data in the form a Python dictionary or from a database. This
            dataset is then moved backward in time from the initial to the final time. Distances
            are in pc, durations in Myr and velocities in pc/Myr.
        """

        # Initialization
        self.series = series
        vars(self).update(values)

        # From traceback
        if not series.from_database:

            # Stars from data
            if series.from_data:
                self.stars_from_data()
            # Stars from simulation
            elif series.from_simulation:
                self.stars_from_simulation()

            # Average velocity
            self.avg_velocity = np.sum(np.array(
                [star.velocity for star in self]), axis=0) / series.number_of_stars
            self.avg_velocity_error = np.sum(np.array(
                [star.velocity_error for star in self])**2, axis=0)**0.5 / series.number_of_stars
            # Average position
            self.avg_position = np.sum(np.array(
                [star.position for star in self]), axis=0) / series.number_of_stars
            self.avg_position_error = np.sum(np.array(
                [star.position_error for star in self])**2, axis=0)**0.5 / series.number_of_stars

            # Age calculation
            self.scatter()
            self.median_absolute_deviation()
            self.minimum_spanning_tree()

    def stars_from_data(self):
        """ Creates a list of Star objects from a Python dictionary or CSV files containing the
            parameters including the name of the stars, their position (XYZ) and velocity (UVW),
            and the respective errors.
        """

        # No conversion
        # for star in self.series.data:
        #     # Stars creation
        #     self.append(self.Star(name=star.name,
        #         velocity=star.velocity[0], velocity_error=star.velocity[2],
        #         position=star.position[0], position_error=star.position[2]))

        # From observables
        for star in self.series.data:

            # Observables conversion into equatorial spherical coordinates
            (position_rδα, velocity_rvμδμα,
                position_rδα_error, velocity_rvμδμα_error) = observables_spherical(
                    *star.position[0], *star.velocity[0] + np.array(
                        [self.series.rv_offset * (un.pc/un.Myr).to(un.km/un.s), 0.0, 0.0]),
                    *star.position[2], *star.velocity[2])

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

    def stars_from_simulation(self):
        """ Creates an artificial sample of star for a given number of stars and age based on
            the intial average position (XYZ) and velocity (UVW), and their respective error
            and scatter. The sample is then moved forward in time for the given age.
        """

        # Velocity array creation and conversions
        avg_velocity = self.series.avg_velocity.values * (un.km/un.s).to(un.pc/un.Myr)
        avg_velocity_error = self.series.avg_velocity_error.values
        avg_velocity_scatter = self.series.avg_velocity_scatter.values * (un.km/un.s).to(un.pc/un.Myr)

        # Position array creation
        avg_position = self.series.avg_position.values
        avg_position_error = self.series.avg_position_error.values
        avg_position_scatter = self.series.avg_position_scatter.values

        for star in range(1, self.series.number_of_stars + 1):

            # Velocity and a position based average values and scatters
            velocity_uvw = np.random.normal(avg_velocity, avg_velocity_scatter)
            position_xyz = velocity_uvw * self.series.age + (
                np.random.normal(avg_position, avg_position_scatter) -
                (avg_velocity * self.series.age + avg_position) + # Normalisation
                (15.19444, -4.93612, -1.70742223)) # beta-pictoris current average position
            velocity_uvw  *= (un.pc/un.Myr).to(un.km/un.s)

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rvμδμα = galactic_uvw_equatorial_rvμδμα(*position_xyz, *velocity_uvw)[0]
            velocity_rvμδμα = velocity_rvμδμα + np.array(
                (self.series.rv_offset * (un.pc/un.Myr).to(un.km/un.s), 0.0, 0.0))
            position_rδα = galactic_xyz_equatorial_rδα(*position_xyz)[0]

            # Velocity and position conversion to observables
            position_obs, velocity_obs = spherical_observables(*position_rδα, *velocity_rvμδμα)[:2]

            # Velocity and position scrambling based on measurment errors and conversion back to
            # equatorial spherical coordinates
            (position_rδα, velocity_rvμδμα,
                position_rδα_error, velocity_rvμδμα_error) = observables_spherical(
                    *np.random.normal(position_obs, avg_position_error),
                    *np.random.normal(velocity_obs, avg_velocity_error),
                    # *np.random.normal(position_obs, self.series.data[star].position[2]),
                    # *np.random.normal(velocity_obs, self.series.data[star].velocity[2]),
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

    def scatter(self):
        """ Computes the xyz and total scatter of a group and their respective errors for all
            timesteps, filters stars farther than 3σ from the average position and compensates
            for the drift in minimal scatter age due to measurement errors.
        """

        # Stars relative positions and distances
        for star in self:
            star.distances()

        # XYZ scatter
        self.scatter_xyz = np.std([star.position for star in self], axis=0)
        self.scatter_xyz_error = self.avg_position_error

        # 3D scatter
        # self.scatter = np.sum(self.scatter_xyz**2, axis=1)**0.5 - np.sum(
        #     self.avg_position_error[0]**2 + self.avg_velocity_error**2 \
        #     * np.expand_dims(self.series.time, axis=0).T**2, axis=1)**0.5
        self.scatter = (np.sum(self.scatter_xyz**2, axis=1) - np.sum(
            self.avg_position_error[0]**2 + self.avg_velocity_error**2 \
            * np.expand_dims(self.series.time, axis=0).T**2, axis=1))**0.5
        self.scatter_error = np.sum(
            (self.scatter_xyz * self.scatter_xyz_error)**2, axis=1)**0.5 / self.scatter
        # !!! Add recursive function to filters stars farther than 3σ from the avg_position here !!!

        # Age calculation
        self.scatter_age = self.series.time[np.argmin(self.scatter)]
        self.scatter_age_error = 0.0

    def median_absolute_deviation(self):
        """ Computes the xyz and total median absolute deviation (MAD) of a group and their
            respective errors for all timestep.
        """

        # XYZ median absolute deviation
        self.median_xyz = np.median(np.array([star.position for star in self]), axis=0)
        self.mad_xyz = np.median(
            np.abs(np.array([star.position for star in self]) - self.median_xyz), axis=0)

        # XYZ median absolute deviation
        self.mad = np.sum(self.mad_xyz**2, axis=1)**0.5

        # Age calculation
        self.mad_age = self.series.time[np.argmin(self.mad)]
        self.mad_age_error = 0.0

    def minimum_spanning_tree(self):
        """ Builds the minimal spanning tree (MST) of a group for all timseteps using a Kruskal
            algorithm and related errors and computes the total (size) of all branches, the median
            absolute deviation of the branch length and the age of the group.
        """

        # Branches creation
        self.branches = []
        for start in range(self.series.number_of_stars - 1):
            for end in range(start + 1, self.series.number_of_stars):
                self.branches.append(self.Branch(self[start], self[end]))

        # Minimum spanning tree initialization
        self.mst = np.empty(
            (self.series.number_of_steps, self.series.number_of_stars - 1), dtype=object)
        self.mst_branches = np.zeros(self.mst.shape)

        # Minimum spanning tree computation for every timestep
        for step in range(self.series.number_of_steps):

            # Sorting with regards to length
            self.branches.sort(key = lambda branch: branch.length[step])

            # Nodes, tree and branch number initialization
            for star in self:
                star.node = self.Node()
            i = 0
            j = 0

            # Branches verification and addition to tree
            while j < self.series.number_of_stars - 1:
                branch = self.branches[i]
                i += 1

                # Replaces branch start and end stars current nodes with their largest parent node
                while branch.start.node.parent != None: branch.start.node = branch.start.node.parent
                while branch.end.node.parent != None: branch.end.node = branch.end.node.parent

                # Branch confirmation if both stars have different parent nodes
                if branch.start.node != branch.end.node:
                    branch.start.node.parent = branch.end.node.parent = self.Node()
                    self.mst[step, j] = branch
                    j += 1

            # Minimum spanning tree branches length
            self.mst_branches[step] = np.vectorize(
                lambda branch: branch.length[step])(self.mst[step])

        # Minimum spanning tree size and age computation
        self.mst_mean = np.mean(self.mst_branches, axis=1)
        self.mst_mean_error = np.zeros([self.series.number_of_steps])
        self.mst_mean_age = self.series.time[np.argmin(self.mst_mean)]
        self.mst_mean_error = 0.0

        # Minimum spanning tree median absolute deviation branch length
        self.mst_median = np.median(self.mst_branches, axis=1)
        self.mst_mad = np.median(
            np.abs(self.mst_branches - np.expand_dims(self.mst_median, axis=0).T), axis=1)
        self.mst_mad_error = np.zeros([self.series.number_of_steps])
        self.mst_mad_age = self.series.time[np.argmin(self.mst_mad)]
        self.mst_mad_age_error = np.zeros([self.series.number_of_steps])

    class Branch:
        """ Connects two stars, used for the calculation of the minimum spanning tree. """

        def __init__(self, start, end):
            """ Initializes a branch and computes the distance between two Star objects,
                'start' and 'end' for all timestep.
            """

            self.start = start
            self.end = end
            self.length = np.sum((self.start.position - self.end.position)**2, axis=1)**0.5

        def __repr__(self):
            """ Returns a string of name of the branch. """

            return "'{}' to '{}' branch".format(self.start.name, self.end.name)

    class Node(object):
        """ Node of a star. """

        def __init__(self):
            """ Set the parent node of a star as None. """

            self.parent = None

    class Star:
        """ Contains the values and related methods of a star in a kinematic group. """

        def __init__(self, group, **values):
            """ Initializes a Star objects with at least a name, velocity, velocity error, intial
                position and position error. More values can be added (when initializing from a
                database for instance).
            """

            # Initialization
            self.group = group
            vars(self).update(values)

            # From traceback
            if not group.series.from_database:
                self.positions()

        def __repr__(self):
            """ Returns a string of name of the star. """

            return self.name

        def positions(self):
            """ Computes the position and position error of the star and errors for all timesteps.
                Both self.position and self.position_errro are redefined with these new values.
            """

            self.position = self.position - self.velocity * np.expand_dims(
                self.group.series.time, axis=0).T
            self.position_error = (self.position_error**2 + (self.velocity_error
                * np.expand_dims(self.group.series.time, axis=0).T)**2)**0.5

        def distances(self):
            """ Computes the relative position and distance from the average position and their
                respective errors for all timesteps.
            """

            # Relative position
            self.relative_position = self.position - self.group.avg_position
            self.relative_position_error = (
                self.position_error**2 + self.group.avg_position_error**2)**0.5

            # Distance
            self.distance = np.sum(self.relative_position**2, axis=1)**0.5
            self.distance_error = np.sum((self.relative_position
                * self.relative_position_error)**2, axis=1)**0.5 / self.distance
