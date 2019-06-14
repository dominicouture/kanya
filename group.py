# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Defines the Group class and the embeded Star classes. These classes contain the
    information and methods necessary to compute tracebacks of stars in a moving group and
    assess its age by minimizing: the 3D scatter, median absolute deviation, covariances and
    minimum spanning tree mean branch length and median absolute deviation of branch length.
"""

import numpy as np
from series import info
from data import Data
from coordinate import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Group(list):
    """ Contains the values and related methods of a moving group and a list of Star objets that
        are part of it. Data can be obtained from the database or calculated from a raw data file.
    """

    def __init__(self, series, **values):
        """ Initializes a Group object and embedded Star objects from a simulated sample of stars
            in a moving group, raw data in the form a Data object or from a database. This dataset
            is then moved backward in time from the initial to the final time, and its age is
            estimated by minimizing: the 3D scatter, median absolute deviation, covariances, and
            the minimum spanning tree branch length mean and median absolute deviation.
        """

        # Initialization
        self.series = series
        vars(self).update(values)

        # Group from traceback
        if not series.from_database:

            # Stars from data
            if series.from_data:
                self.stars_from_data()

            # Stars from simulation
            elif series.from_simulation:
                self.stars_from_simulation()

            # Average velocity and position and scatter computation
            self.get_scatter()

            # Median absolute deviation computation
            self.get_median_absolute_deviation()

            # X-U, Y-V and Z-W covariances computation
            self.get_covariances()

            # Minimum spanning tree computation
            self.get_minimum_spanning_tree()

    def stars_from_data(self):
        """ Creates a list of Star objects from a Python dictionary or CSV files containing the
            parameters including the name of the stars, their XYZ position and UVW velocity,
            and the respective errors. Radial velocity offset is also added.
        """

        # From cartesian coordinates
        # for star in self.series.data:
        #     # Stars creation
        #     self.append(self.Star(name=star.name,
        #         velocity=star.velocity.values, velocity_error=star.velocity.values,
        #         position=star.position.values, position_error=star.position.values))

        # From observables
        for star in self.series.data:

            # Observables conversion into equatorial spherical coordinates
            (position_rδα, velocity_rvμδμα,
                position_rδα_error, velocity_rvμδμα_error) = observables_spherical(
                    *star.position.values,
                    *star.velocity.values + np.array([self.series.rv_offset.value, 0.0, 0.0]),
                    *star.position.errors,
                    *star.velocity.errors)

            # Equatorial spherical coordinates conversion into galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_uvw, velocity_uvw_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rvμδμα, *position_rδα_error, *velocity_rvμδμα_error)

            # Speed of light correction
            # from astropy import units as un
            # position_xyz = position_xyz + velocity_uvw * (un.km/un.s).to(un.pc/un.Myr) * \
            #     position_rδα[0] / (299792458 * (un.m/un.s).to(un.pc/un.Myr))

            # Star creation
            self.append(self.Star(
                self, outlier=False,
                name=star.name,
                velocity=velocity_uvw,
                velocity_error=velocity_uvw_error,
                position=position_xyz,
                position_error=position_xyz_error))

    def stars_from_simulation(self):
        """ Creates an artificial sample of star for a given number of stars and age based on
            the intial average XYZ position and UVW velocity, and their respective errors
            and scatters. The sample is then moved forward in time for the given age and radial
            velocity offset is also added.
        """

        # Velocity arrays creation and conversion
        avg_velocity = self.series.avg_velocity.values
        avg_velocity_error = self.series.avg_velocity_error.values
        avg_velocity_scatter = self.series.avg_velocity_scatter.values

        # Position arrays creation
        avg_position = self.series.avg_position.values
        avg_position_error = self.series.avg_position_error.values
        avg_position_scatter = self.series.avg_position_scatter.values

        for star in range(1, self.series.number_of_stars + 1):

            # Velocity and a position based average values and scatters
            velocity_uvw = np.random.normal(avg_velocity, avg_velocity_scatter)
            position_xyz = velocity_uvw * self.series.age.values + (
                np.random.normal(avg_position, avg_position_scatter) -
                (avg_velocity * self.series.age.values + avg_position) + # Normalisation
                (15.19444, -4.93612, -1.70742223)) # β Pictoris current average position

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rvμδμα = galactic_uvw_equatorial_rvμδμα(*position_xyz, *velocity_uvw)[0]
            velocity_rvμδμα = velocity_rvμδμα + np.array([self.series.rv_offset.value, 0.0, 0.0])
            position_rδα = galactic_xyz_equatorial_rδα(*position_xyz)[0]

            # Velocity and position conversion to observables
            position_obs, velocity_obs = spherical_observables(*position_rδα, *velocity_rvμδμα)[:2]

            # Velocity and position scrambling based on measurement errors and conversion back to
            # equatorial spherical coordinates
            (position_rδα, velocity_rvμδμα,
                position_rδα_error, velocity_rvμδμα_error) = observables_spherical(
                    # *np.random.normal(position_obs, avg_position_error),
                    # *np.random.normal(velocity_obs, avg_velocity_error),
                    # *avg_position_error, *avg_velocity_error)
                    *np.random.normal(position_obs, self.series.data[star].position.errors),
                    *np.random.normal(velocity_obs, self.series.data[star].velocity.errors),
                    *self.series.data[star].position.errors,
                    *self.series.data[star].position.errors)

            # Velocity and position conversion back to galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_uvw, velocity_uvw_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rvμδμα, *position_rδα_error, *velocity_rvμδμα_error)

            # Star creation
            self.append(self.Star(
                self, outlier=False,
                name='star_{}'.format(star),
                velocity=velocity_uvw,
                velocity_error=velocity_uvw_error,
                position=position_xyz,
                position_error=position_xyz_error))

    def get_stars_coordinates(self):
        """ Filters outliers and creates self.stars list of valid stars and computes the number
            of remaining stars, the average velocity and position of stars in the group, and the
            stars' relative positions and distances from the average, regardless of whether the
            stars are outliers or not.
        """

        # Outliers filtering
        self.stars = list(filter(lambda star: not star.outlier, self))

        # Group number of stars without outliers
        self.number_of_stars = len(self.stars)

        # Average velocity
        self.avg_velocity = np.sum(np.array(
            [star.velocity for star in self.stars]), axis=0) / self.number_of_stars
        self.avg_velocity_error = np.sum(np.array(
            [star.velocity_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

        # Average position
        self.avg_position = np.sum(np.array(
            [star.position for star in self.stars]), axis=0) / self.number_of_stars
        self.avg_position_error = np.sum(np.array(
            [star.position_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

        # Stars relative position and velocity, distance and speed
        for star in self:
            star.get_relative_coordinates()

    def get_scatter(self):
        """ Computes the XYZ and total scatter of a group and their respective errors for all
            timesteps, filters stars farther than 3σ from the average position and compensates
            for the drift of the minimal scatter age due to measurement errors. The age of the
            moving group is then estimated by finding the time at which the scatter is minimal.
            If all stars are identified as outliers,
        """

        # Initial assumption that outliers are present in the group
        outliers = True
        while outliers:

            # Outliers filtering and coordinates computation
            self.get_stars_coordinates()

            # XYZ scatter
            self.scatter_xyz = np.std([star.position for star in self.stars], axis=0)
            self.scatter_xyz_error = self.avg_position_error

            # 3D scatter
            self.scatter = (np.sum(self.scatter_xyz**2, axis=1) - np.sum(
                self.avg_position_error[0]**2 + self.avg_velocity_error**2 \
                * np.expand_dims(self.series.time, axis=0).T**2, axis=1))**0.5
            self.scatter_error = np.sum(
                (self.scatter_xyz * self.scatter_xyz_error)**2, axis=1)**0.5 / self.scatter

            # Alternative 3D scatter
            # self.scatter = np.sum(self.scatter_xyz**2, axis=1)**0.5 - np.sum(
            #     self.avg_position_error[0]**2 + self.avg_velocity_error**2 \
            #     * np.expand_dims(self.series.time, axis=0).T**2, axis=1)**0.5

            # Filter stars located beyond 3σ of the average position and loop scatter computation
            for star in self.stars:
                if (star.distance > (self.scatter * 2.5)).any():
                    star.outlier = True
            outliers = np.array([star.outlier for star in self.stars])
            outliers = False if outliers.all() else outliers.any()

        # Age calculation
        self.scatter_age = self.series.time[np.argmin(self.scatter)]
        self.scatter_age_error = 0.0

    def get_median_absolute_deviation(self):
        """ Computes the XYZ and total median absolute deviation (MAD) of a group and their
            respective errors for all timesteps. The age of the moving is then estimated by
            finding the time at which the median absolute deviation is minimal.
        """

        # XYZ median absolute deviation
        self.median_xyz = np.median(np.array([star.position for star in self.stars]), axis=0)
        self.median_xyz_error = np.zeros(self.median_xyz.shape)
        self.mad_xyz = np.median(
            np.abs(np.array([star.position for star in self.stars]) - self.median_xyz), axis=0)
        self.mad_xyz_error = np.zeros(self.mad_xyz.shape)

        # XYZ median absolute deviation
        self.mad = np.sum(self.mad_xyz**2, axis=1)**0.5
        self.mad_error = np.zeros(self.mad.shape)

        # Age calculation
        self.mad_age = self.series.time[np.argmin(self.mad)]
        self.mad_age_error = 0.0

    def get_covariances(self):
        """ Computes the X-U, Y-V and Z-W absolute covariances of a group and their respective
            errors for all timesteps. The age of the moving is then estimated by finding the time
            at which the covariances is minimal.
        """

        # Covariances calculation
        self.covariance = np.absolute(np.sum(
            (np.array([star.position for star in self.stars]) - self.avg_position) *
            np.expand_dims(np.array([star.velocity for star in self.stars]) - self.avg_velocity,
                axis=1), axis=0) /self.series.number_of_stars)
        self.covariance_error = np.zeros(self.covariance.shape)

        # Age calculation
        self.covariance_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.covariance, axis=0))
        self.covariance_age_error = np.zeros(self.covariance_age.shape)

    def get_minimum_spanning_tree(self):
        """ Builds the minimal spanning tree (MST) of a group for all timseteps using a Kruskal
            algorithm, and computes errors, and the average branch length and the branch length
            median absolute absolute deviation. The age of the moving is then estimated by
            finding the time at which these measures are minimal.
        """

        # Branches creation
        self.branches = []
        self.number_of_branches = self.number_of_stars - 1
        for start in range(self.number_of_branches):
            for end in range(start + 1, self.number_of_branches + 1):
                self.branches.append(self.Branch(self.stars[start], self.stars[end]))

        # Minimum spanning tree initialization
        self.mst = np.empty(
            (self.series.number_of_steps, self.number_of_branches), dtype=object)
        self.mst_lengths = np.zeros(self.mst.shape)
        self.mst_lengths_error = np.zeros(self.mst.shape)

        # Minimum spanning tree computation for every timestep
        for step in range(self.series.number_of_steps):

            # Sort by length
            self.branches.sort(key = lambda branch: branch.length[step])

            # Nodes, tree and branch number initialization
            for star in self.stars:
                star.node = self.Node()
            i = 0
            j = 0

            # Branches verification and addition to tree
            while j < self.number_of_branches:
                branch = self.branches[i]
                i += 1

                # Replace branch start and end stars current nodes with their largest parent node
                while branch.start.node.parent != None: branch.start.node = branch.start.node.parent
                while branch.end.node.parent != None: branch.end.node = branch.end.node.parent

                # Branch confirmation if both stars have different parent nodes
                if branch.start.node != branch.end.node:
                    branch.start.node.parent = branch.end.node.parent = self.Node()
                    self.mst[step, j] = branch
                    j += 1

            # Minimum spanning tree branches length
            self.mst_lengths[step] = np.vectorize(
                lambda branch: branch.length[step])(self.mst[step])
            self.mst_lengths_error[step] = 0.0

        # Minimum spanning tree average branch length and age computation
        self.mst_mean = np.mean(self.mst_lengths, axis=1)
        self.mst_mean_error = np.zeros([self.series.number_of_steps])
        self.mst_mean_age = self.series.time[np.argmin(self.mst_mean)]
        self.mst_mean_error = 0.0

        # Minimum spanning tree branch length median absolute deviation
        self.mst_median = np.median(self.mst_lengths, axis=1)
        self.mst_median_error = np.zeros(self.mst_median.shape)
        self.mst_mad = np.median(
            np.abs(self.mst_lengths - np.expand_dims(self.mst_median, axis=0).T), axis=1)
        self.mst_mad_error = np.zeros([self.series.number_of_steps])
        self.mst_mad_age = self.series.time[np.argmin(self.mst_mad)]
        self.mst_mad_age_error = 0.0

    class Branch:
        """ Line connecting two stars used for the calculation of the minimum spanning tree. """

        def __init__(self, start, end):
            """ Initializes a Branch object and computes the distance between two Star objects,
                'start' and 'end', for all timestep.
            """

            self.start = start
            self.end = end
            self.length = np.sum((self.start.position - self.end.position)**2, axis=1)**0.5
            self.length_error = 0.0

        def __repr__(self):
            """ Returns a string of name of the branch. """

            return "'{}' to '{}' branch".format(self.start.name, self.end.name)

    class Node(object):
        """ Node of a star. """

        def __init__(self):
            """ Sets the parent node of a star as None. """

            self.parent = None

        def __repr__(self):
            """ Returns a string of name of the parent. """

            return 'None' if self.parent is None else self.parent

    class Star:
        """ Contains the values and related methods of a star in a moving group. """

        def __init__(self, group, **values):
            """ Initializes a Star object with at least a name, velocity, velocity error, initial
                position and position error. More values can be added (when initializing from a
                database for instance). If a traceback is needed, the star's position and velocity
                overtime is computed with galpy. !!! Add galactic orbits computation here !!!
            """

            # Initialization
            self.group = group
            self.outlier = False
            vars(self).update(values)

            # Positions from traceback
            if not group.series.from_database:
                self.position = self.position - self.velocity * np.expand_dims(
                    self.group.series.time, axis=0).T
                self.position_error = (self.position_error**2 + (self.velocity_error
                    * np.expand_dims(self.group.series.time, axis=0).T)**2)**0.5

        def __repr__(self):
            """ Returns a string of name of the star. """

            return self.name

        def get_relative_coordinates(self):
            """ Computes relative position and velocity, the distance and the speed from the
                average position and velocity, and their respective errors for all timesteps.
            """

            # Relative position and distance from the average position
            self.relative_position = self.position - self.group.avg_position
            self.relative_position_error = (
                self.position_error**2 + self.group.avg_position_error**2)**0.5
            self.distance = np.sum(self.relative_position**2, axis=1)**0.5
            self.distance_error = np.sum((self.relative_position
                * self.relative_position_error)**2, axis=1)**0.5 / self.distance

            # Relative velocity and speed from the average velocity
            self.relative_velocity = self.velocity - self.group.avg_velocity
            self.relative_velocity_error = (
                self.velocity_error**2 + self.group.avg_velocity_error**2)**0.5
            self.speed = np.sum(self.relative_velocity**2, axis=0)**0.5
            self.speed_error = np.sum((self.relative_velocity
                * self.relative_position_error)**2, axis=0)**0.5 / self.speed
