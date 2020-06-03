# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Defines the Group class and the embeded Star classes. These classes contain the
    information and methods necessary to compute tracebacks of stars in a moving group and
    assess its age by minimizing: the 3D scatter, median absolute deviation, position-velocity
    covariances and minimum spanning tree mean branch length and median absolute deviation of
    branch length.
"""

import numpy as np
from Traceback.data import Data
from Traceback.output import Output_Group
from Traceback.coordinate import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Group(list, Output_Group):
    """ Contains the values and related methods of a moving group and a list of Star objets that
        are part of it. Stars are can be imported from a raw data file or modeled base on
        simulation parameters
    """

    def __init__(self, series, name):
        """ Initializes a Group object and embedded Star objects from a simulated sample of stars
            in a moving group or raw data in the form a Data object. This dataset is then moved
            backwards in time from the initial time to the final time, and its age is estimated
            by minimizing: the 3D scatter, median absolute deviation, covariances, and the minimum
            spanning tree branch length mean and median absolute deviation.
        """

        # Initialization
        self.series = series
        self.name = name

        # Stars from data
        if series.from_data:
            self.stars_from_data()

        # Stars from model
        elif series.from_model:
            self.stars_from_model()

        # Average velocity and position, scatter and outliers
        self.get_scatter()

        # Median absolute deviation
        self.get_median_absolute_deviation()

        # X-U, Y-V and Z-W covariances
        self.get_covariances()

        # Minimum spanning tree
        self.get_minimum_spanning_tree()

    def stars_from_data(self):
        """ Creates a list of Star objects from a Python dictionary or CSV files containing the
            parameters including the name of the stars, their XYZ positions and UVW velocities,
            and their respective measurement errors. Radial velocity offset is also added. If
            only one group is created in the series, actual values are used. If multiple groups
            are created, positions and velocities are scrambled based on measurement errors.
        """

        # Star creation from observables
        for star in self.series.data:
            rv_offset = star.rv_offset if star.rv_offset is not None else self.series.rv_offset.value
            # rv_offset = self.series.rv_offset.value
            # print(star.name, 'rv_offset:', round(rv_offset, 3))
            # Observables conversion into equatorial spherical coordinates
            (position_rδα, velocity_rvμδμα, position_rδα_error, velocity_rvμδμα_error) = \
                observables_spherical(

                    # Position
                    *(star.position.values if self.series.number_of_groups == 1 else
                        np.random.normal(star.position.values, star.position.errors)),

                    # Velocity and radial velocity offset
                    *(star.velocity.values if self.series.number_of_groups == 1 else
                        np.random.normal(star.velocity.values, star.velocity.errors)) \
                        + np.array([rv_offset, 0.0, 0.0]),

                    # Position and velocity errors
                    *star.position.errors, *star.velocity.errors)

            # Equatorial spherical coordinates conversion into galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_uvw, velocity_uvw_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rvμδμα, *position_rδα_error, *velocity_rvμδμα_error)

            # Star creation
            self.append(self.Star(
                self, outlier=False, name=star.name,
                velocity=velocity_uvw, velocity_error=velocity_uvw_error,
                position=position_xyz, position_error=position_xyz_error))

            # Stars creation from cartesian coordinates
            # self.append(self.Star(
            #     self, outlier=False, name=star.name,
            #     velocity=star.velocity.values, velocity_error=star.velocity.values,
            #     position=star.position.values, position_error=star.position.values))

    def stars_from_model(self):
        """ Creates an artificial model of star for a given number of stars and age based on
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

        # Stars creation from a model
        for star in range(self.series.number_of_stars):

            # Velocity and a position based average values and scatters
            # Positions are projected forward in time centered around the average position
            velocity_uvw = np.random.normal(avg_velocity, avg_velocity_scatter)
            position_xyz = np.random.normal((0., 0., 0.), avg_position_scatter) + (
                velocity_uvw - avg_velocity) * self.series.age.values + avg_position

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rvμδμα = galactic_uvw_equatorial_rvμδμα(*position_xyz, *velocity_uvw)[0]
            velocity_rvμδμα = velocity_rvμδμα + np.array([self.series.rv_offset.value, 0.0, 0.0])
            position_rδα = galactic_xyz_equatorial_rδα(*position_xyz)[0]

            # Velocity and position conversion to observables
            position_obs, velocity_obs = spherical_observables(*position_rδα, *velocity_rvμδμα)[:2]

            # Observables conversion back into equatorial spherical coordinates
            # Velocity and position scrambling based on actual measurement errors
            if self.series.data_errors:
                star_errors = star - star // len(self.series.data) * len(self.series.data)
                (position_rδα, velocity_rvμδμα, position_rδα_error, velocity_rvμδμα_error) = \
                        observables_spherical(
                    *np.random.normal(position_obs, self.series.data[star_errors].position.errors),
                    *np.random.normal(velocity_obs, self.series.data[star_errors].velocity.errors),
                    *self.series.data[star_errors].position.errors,
                    *self.series.data[star_errors].velocity.errors)

            # Velocity and position scrambling based on average measurement errors
            else:
                (position_rδα, velocity_rvμδμα, position_rδα_error, velocity_rvμδμα_error) = \
                        observables_spherical(
                    *np.random.normal(position_obs, avg_position_error),
                    *np.random.normal(velocity_obs, avg_velocity_error),
                    *avg_position_error, *avg_velocity_error)

            # Velocity and position conversion back to galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_uvw, velocity_uvw_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rvμδμα, *position_rδα_error, *velocity_rvμδμα_error)

            # Star creation
            self.append(self.Star(
                self, outlier=False,
                name='star_{}'.format(star + 1),
                velocity=velocity_uvw,
                velocity_error=position_xyz_error,
                position=position_xyz,
                position_error=velocity_uvw_error))

    def get_stars_coordinates(self):
        """ Computes the number of stars that aren't ouliers, the stars average velocity and
            position, excluding outliers, and the stars relative positions and distances from
            the average, including outliers.
        """

        # Number of stars remaining, excluding outliers
        self.number_of_stars = len(self.stars)

        # Average velocity, excluding outliers
        self.avg_velocity = np.sum(np.array(
            [star.velocity for star in self.stars]), axis=0) / self.number_of_stars
        self.avg_velocity_error = np.sum(np.array(
            [star.velocity_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

        # Average position, excluding outliers
        self.avg_position = np.sum(np.array(
            [star.position for star in self.stars]), axis=0) / self.number_of_stars
        self.avg_position_error = np.sum(np.array(
            [star.position_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

        # Stars relative position and velocity, distance and speed of all stars (outliers or not)
        for star in self:
            star.get_relative_coordinates()

    def get_scatter(self):
        """ Computes the XYZ and total scatter of a group and their respective errors for all
            timesteps, filters stars farther than 3σ from the average position and compensates
            for the drift of the minimal scatter age due to measurement errors. The age of the
            moving group is then estimated by finding the time at which the scatter is minimal.
            If all stars are identified as outliers, an error is raised.
        """

        # Valid stars and outliers lists
        self.stars = list(self)
        self.number_of_stars = len(self.stars)
        self.outliers = []
        outliers = True

        # Outlier filtering
        while outliers and self.number_of_stars > int(0.8 * self.series.number_of_stars):
            # self.stars = list(filter(lambda star: not star.outlier, self)) ???

            # Coordinates computation
            self.get_stars_coordinates()

            # XYZ scatter
            self.scatter_xyz = np.std([star.position for star in self.stars], axis=0)

            # UVW scatter
            self.scatter_uvw = np.std([star.velocity for star in self.stars], axis=0)

            # 3D scatter
            self.scatter = np.sum(self.scatter_xyz**2, axis=1)**0.5

            # Alternative 3D scatters
            # self.scatter = (np.abs(np.sum(self.scatter_xyz**2, axis=1) - np.sum(
            #     self.avg_position_error[0]**2 + self.avg_velocity_error**2 \
            #     * np.expand_dims(self.series.time, axis=0).T**2, axis=1)))**0.5
            # self.scatter = (np.abs(np.sum(self.scatter_xyz**2, axis=1)**0.5 - np.sum(
            #     self.avg_position_error[0]**2 + self.avg_velocity_error**2 \
            #     * np.expand_dims(self.series.time, axis=0).T**2, axis=1))**0.5

            # Filter stars beyond the σ cutoff of the average position and loop scatter computation
            outliers = False
            for star in self.stars:
                if (star.distance > (self.scatter * self.series.cutoff)).any():
                    star.outlier = True
                    outliers = True
                    self.outliers.append(star)
                    self.stars.remove(star)
                    self.number_of_stars = len(self.stars)

        # Scatter age
        self.scatter_age = self.series.time[np.argmin(self.scatter)]
        self.scatter_min = np.min(self.scatter)

    def get_median_absolute_deviation(self):
        """ Computes the XYZ and total median absolute deviation (MAD) of a group and their
            respective errors for all timesteps. The age of the moving is then estimated by
            finding the time at which the median absolute deviation is minimal.
        """

        # XYZ median absolute deviation
        self.median_xyz = np.median(np.array([star.position for star in self.stars]), axis=0)
        self.mad_xyz = np.median(
            np.abs(np.array([star.position for star in self.stars]) - self.median_xyz), axis=0)

        # Median absolute deviation
        self.mad = np.sum(self.mad_xyz**2, axis=1)**0.5

        # Median absolute deviation age
        self.mad_age = self.series.time[np.argmin(self.mad)]
        self.mad_min = np.min(self.mad)

    # def get_median_absolute_deviation(self):
    #     """ Computes the XYZ and total median absolute deviation (MAD) of a group and their
    #         respective errors for all timesteps. The age of the moving is then estimated by
    #         finding the time at which the median absolute deviation is minimal.
    #     """
    #
    #     print(self.number_of_stars)
    #     # Jack-knife Monte Carlo
    #     total = np.zeros((100, len(self.series.time)))
    #     for i in range(100):
    #         stars = [self.stars[i] for i in np.random.choice(self.number_of_stars,
    #             int(self.number_of_stars * 0.8), replace=False)]
    #
    #         # XYZ median absolute deviation
    #         median_xyz = np.median(np.array([star.position for star in stars]), axis=0)
    #         mad_xyz = np.median(
    #             np.abs(np.array([star.position for star in stars]) - median_xyz), axis=0)
    #         total[i] = np.sum(mad_xyz**2, axis=1)**0.5
    #
    #     # Median absolute deviation
    #     self.mad = np.mean(total, axis=0)
    #
    #     # Median absolute deviation age
    #     self.mad_age = self.series.time[np.argmin(self.mad)]
    #     self.mad_min = np.min(self.mad)

    def get_covariances(self):
        """ Computes the X-U, Y-V and Z-W absolute covariances of a group and their respective
            errors for all timesteps. The age of the moving is then estimated by finding the time
            at which the covariances is minimal.
        """

        # Covariances calculation
        self.covariances = np.absolute(np.sum(
            (np.array([star.position for star in self.stars]) - self.avg_position) *
            np.expand_dims(np.array([star.velocity for star in self.stars]) - self.avg_velocity,
                axis=1), axis=0) / self.number_of_stars)

        # Covariances age
        self.covariances_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.covariances, axis=0))
        self.covariances_min = np.min(self.covariances, axis=0)

    def get_minimum_spanning_tree(self):
        """ Builds the minimum spanning tree (MST) of a group for all timesteps using a Kruskal
            algorithm, and computes the average branch length and the branch length median absolute
            absolute deviation of branch length. The age of the moving is then estimated by finding
            the time at which these value are minimal.
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

                # Replace branch start and end stars' current nodes with their largest parent node
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

        # Minimum spanning tree average branch length and age
        self.mst_mean = np.mean(self.mst_lengths, axis=1)
        self.mst_mean_age = self.series.time[np.argmin(self.mst_mean)]
        self.mst_min = np.min(self.mst_mean)

        # Minimum spanning tree branch length median absolute deviation and age
        # self.mst_median = np.median(self.mst_lengths, axis=1)
        # self.mst_mad = np.median(
        #     np.abs(self.mst_lengths - np.expand_dims(self.mst_median, axis=0).T), axis=1)
        # self.mst_mad_age = self.series.time[np.argmin(self.mst_mad)]
        # self.mst_mad_min = np.min(self.mst_mad)

        # Minimum spanning tree branch length median absolute deviation and age
        # Jack-knife Monte Carlo
        total = np.zeros((100, len(self.series.time)))
        for i in range(100):
            mst_lengths = self.mst_lengths[:, np.random.choice(self.number_of_branches,
                int(self.number_of_branches * 0.8), replace=False)]

            # Median absolute deviation
            mst_median = np.median(mst_lengths, axis=1)
            mst_mad = np.median(np.abs(mst_lengths - np.expand_dims(mst_median, axis=0).T), axis=1)
            total[i] = mst_mad

        self.mst_mad = np.mean(total, axis=0)
        self.mst_mad_age = self.series.time[np.argmin(self.mst_mad)]
        self.mst_mad_min = np.min(self.mst_mad)

    class Branch:
        """ Line connecting two stars used for the calculation of the minimum spanning tree. """

        def __init__(self, start, end):
            """ Initializes a Branch object and computes the distance between two Star objects,
                'start' and 'end', for all timestep.
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
                file for instance). If a traceback is needed, the star's position and velocity
                overtime is computed with galpy. !!! Add galactic orbits computation here !!!
            """

            # Initialization
            self.group = group
            self.outlier = False
            vars(self).update(values)

            # Positions from traceback
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

        def shift_rv(self):
            """ Shifts the radial velocity based on the gravitationnal redshift of the star and
                current value of the radial velocity.
            """

            # Gravitationnal constant (pc^3 / M_sol / Myr^2)
            G = 0.004498706056647732

            # Speed of light (pc/Myr)
            c = 306601.3937879527

            self.velocity[:,0] = a * (c + self.velocity[:,0]) - c
