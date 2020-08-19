# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Defines the Group class and the embeded Star classes. These classes contain the
    information and methods necessary to compute tracebacks of stars in a moving group and
    assess its age by minimizing: the 3D scatter, median absolute deviation, position-velocity
    covariances and minimum spanning tree mean branch length and median absolute deviation of
    branch length.
"""

import numpy as np
import galpy.util.bovy_coords as gpcooords
from galpy.orbit import Orbit
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

        # Milky Way potential
        from galpy.potential import MWPotential2014
        self.potential = MWPotential2014

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

        # Minimum spanning tree
        self.get_minimum_spanning_tree()

        # X-U, Y-V and Z-W covariances
        self.get_covariances()

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
                self, name=star.name, backward=True,
                velocity=velocity_uvw, velocity_error=velocity_uvw_error,
                position=position_xyz, position_error=position_xyz_error))

        # Stars creation from cartesian coordinates
        # for star in self.series.data:
        #     self.append(self.Star(
        #         self, name=star.name, backward=True,
        #         velocity=star.velocity.values, velocity_error=star.velocity.values,
        #         position=star.position.values, position_error=star.position.values))

    def stars_from_model(self):
        """ Creates an artificial model of star for a given number of stars and age based on
            the intial average XYZ position and UVW velocity, and their respective errors
            and scatters. The sample is then moved forward in time for the given age and radial
            velocity offset is also added.
        """

        # Model backward orbit integration
        average_model_star = self.Star(
            self, name='model_average_star', backward=True, model=True,
            velocity=self.series.avg_velocity.values, velocity_error=np.zeros(3),
            position=self.series.avg_position.values, position_error=np.zeros(3))

        # Average velocity arrays
        avg_velocity = average_model_star.velocity_new[-1]
        avg_velocity_error = self.series.avg_velocity_error.values
        avg_velocity_scatter = self.series.avg_velocity_scatter.values

        # Average position arrays
        avg_position = average_model_star.position[-1]
        avg_position_error = self.series.avg_position_error.values
        avg_position_scatter = self.series.avg_position_scatter.values

        # Stars creation from a model
        for star in range(self.series.number_of_stars):

            # Model star forward galactic orbit integration
            model_star = self.Star(
                self, name='model_star_{}'.format(star + 1), backward=False, model=True,
                velocity=np.random.normal(avg_velocity, avg_velocity_scatter), velocity_error=np.zeros(3),
                position=np.random.normal(avg_position, avg_position_scatter), position_error=np.zeros(3))

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rvμδμα = galactic_uvw_equatorial_rvμδμα(*model_star.position[-1], *model_star.velocity_new[-1])[0]
            velocity_rvμδμα += np.array([self.series.rv_offset.value, 0.0, 0.0])
            position_rδα = galactic_xyz_equatorial_rδα(*model_star.position[-1])[0]

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
                self, name='star_{}'.format(star + 1), backward=True,
                velocity=velocity_uvw, velocity_error=velocity_uvw_error,
                position=position_xyz, position_error=position_xyz_error))

    def get_stars_coordinates(self):
        """ Computes the number of stars that aren't ouliers, the stars average velocity and
            position, excluding outliers, and the stars relative positions and distances from
            the average, including outliers.
        """

        # Number of stars remaining, excluding outliers
        self.number_of_stars = len(self.stars)

        # Average velocity, excluding outliers
        self.avg_velocity_new = np.mean([star.velocity_new for star in self.stars], axis=0)
        self.avg_velocity_error = np.sum(np.array(
            [star.velocity_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

        # Average position, excluding outliers
        self.avg_position = np.mean([star.position for star in self.stars], axis=0)
        self.avg_position_error = np.sum(np.array(
            [star.position_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

        # Average rθz position, excluding outliers
        self.avg_rθz = np.mean([star.position_rθz for star in self.stars], axis=0)

        # Average μρμθw velocity, excluding outliers
        self.avg_μρμθw = np.mean([star.velocity_μρμθw for star in self.stars], axis=0)

        # Linear position and velocity, excluding outliers
        self.avg_velocity = np.mean([star.velocity for star in self.stars], axis=0)
        self.avg_position_linear = np.mean(np.array(
            [star.position_linear for star in self.stars]), axis=0)

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

        # UVW position scatter
        self.scatter_uvw = np.std([star.velocity for star in self.stars], axis=0)

        # rθz position scatter
        self.scatter_rθz = np.std([star.position_rθz for star in self.stars], axis=0)

        # μρμθw velocity scatter
        self.scatter_μρμθw = np.std([star.velocity_μρμθw for star in self.stars], axis=0)

        # Position scatter age
        self.scatter_age = self.series.time[np.argmin(self.scatter)]
        self.scatter_min = np.min(self.scatter)

        # XYZ position scatter age
        self.scatter_xyz_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.scatter_xyz, axis=0))
        self.scatter_xyz_min = np.min(self.scatter_xyz, axis=0)

        # rθz position scatter age
        self.scatter_rθz_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.scatter_rθz, axis=0))
        self.scatter_rθz_min = np.min(self.scatter_rθz, axis=0)

    def get_median_absolute_deviation(self):
        """ Computes the XYZ and total median absolute deviation (MAD) of a group and their
            respective errors for all timesteps. The age of the moving is then estimated by
            finding the time at which the median absolute deviation is minimal.
        """

        # Single median absolute deviation
        # median_xyz = np.median(np.array([star.position for star in self.stars]), axis=0)
        # self.mad_xyz = np.median(
        #     np.abs(np.array([star.position for star in self.stars]) - median_xyz), axis=0)
        # self.mad = np.sum(self.mad_xyz**2, axis=1)**0.5

        # Jack-knife Monte Carlo
        n = 500
        total_mad_xyz = np.zeros((n, len(self.series.time), 3))
        total_mad = np.zeros((n, len(self.series.time)))
        for i in range(n):
            stars = [self.stars[i] for i in np.random.choice(self.number_of_stars,
                int(self.number_of_stars * 0.8), replace=False)]

            # Median absolute deviation
            median_xyz = np.median(np.array([star.position for star in stars]), axis=0)
            total_mad_xyz[i] = np.median(
                np.abs(np.array([star.position for star in stars]) - median_xyz), axis=0)
            total_mad[i] = np.sum(total_mad_xyz[i]**2, axis=1)**0.5

        # Average median absolute deviation
        self.mad_xyz = np.mean(total_mad_xyz, axis=0)
        self.mad = np.mean(total_mad, axis=0)

        # rθz median absolute deviation
        median_rθz = np.median(np.array([star.position_rθz for star in self.stars]), axis=0)
        self.mad_rθz = np.median(
            np.abs(np.array([star.position_rθz for star in self.stars]) - median_rθz), axis=0)

        # Median absolute deviation age
        self.mad_age = self.series.time[np.argmin(self.mad)]
        self.mad_min = np.min(self.mad)

        # XYZ median absolute deviation age
        self.mad_xyz_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.mad_xyz, axis=0))
        self.mad_xyz_min = np.min(self.mad_xyz, axis=0)

        # rθz median absolute deviation age
        self.mad_rθz_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.mad_rθz, axis=0))
        self.mad_rθz_min = np.min(self.mad_rθz, axis=0)

    def get_covariances(self):
        """ Computes the X-U, Y-V and Z-W absolute covariances of a group and their respective
            errors for all timesteps. The age of the moving is then estimated by finding the time
            at which the covariances is minimal.
        """

        # XYZ positions and UVW velocities
        positions = np.array([star.position for star in self.stars]) - self.avg_position
        velocities = np.array([star.velocity_new for star in self.stars]) - self.avg_velocity_new

        # XYZ covariance matrix
        self.covariances_xyz_matrix = np.mean(np.swapaxes(
            np.array([
                [positions[:,:,0], positions[:,:,0], positions[:,:,0]],
                [positions[:,:,1], positions[:,:,1], positions[:,:,1]],
                [positions[:,:,2], positions[:,:,2], positions[:,:,2]]]) *
            np.array([
                [positions[:,:,0], positions[:,:,1], positions[:,:,2]],
                [positions[:,:,0], positions[:,:,1], positions[:,:,2]],
                [positions[:,:,0], positions[:,:,1], positions[:,:,2]]]) , 0, 1), axis=2).T

        # X-X, Y-Y, Z-Z absolute covariances ages
        self.covariances_xyz = np.absolute(self.covariances_xyz_matrix[:, (0, 1, 2), (0, 1, 2)])
        self.covariances_xyz_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.covariances_xyz, axis=0))
        self.covariances_xyz_min = np.min(self.covariances_xyz, axis=0)

        # XYZ covariance matrix determinant age
        self.covariances_xyz_matrix_det = np.linalg.det(self.covariances_xyz_matrix)
        self.covariances_xyz_matrix_det_age = self.series.time[np.argmin(self.covariances_xyz_matrix_det)]
        self.covariances_xyz_matrix_det_min = np.min(self.covariances_xyz_matrix_det)

        # XYZ cross covariance matrix
        self.cross_covariances_xyz_matrix = np.mean(np.swapaxes(
            np.array([
                [positions[:,:,0], positions[:,:,0], positions[:,:,0]],
                [positions[:,:,1], positions[:,:,1], positions[:,:,1]],
                [positions[:,:,2], positions[:,:,2], positions[:,:,2]]]) *
            np.array([
                [velocities[:,:,0], velocities[:,:,1], velocities[:,:,2]],
                [velocities[:,:,0], velocities[:,:,1], velocities[:,:,2]],
                [velocities[:,:,0], velocities[:,:,1], velocities[:,:,2]]]) , 0, 1), axis=2).T

        # Diagonal elements set to 0. and determinant
        cross_covariances_xyz_matrix_zero_diag = self.cross_covariances_xyz_matrix.copy()
        cross_covariances_xyz_matrix_zero_diag[:, (0, 1, 2), (0, 1, 2)] = 0.
        self.cross_covariances_xyz_matrix_det = np.linalg.det(cross_covariances_xyz_matrix_zero_diag)

        # Cross covariance matrix determinant age
        self.cross_covariances_xyz_matrix_det_age = self.series.time[
            np.argmin(self.cross_covariances_xyz_matrix_det)]
        self.cross_covariances_xyz_matrix_det_min = np.min(self.cross_covariances_xyz_matrix_det)

        # X-U, Y-V, Z-W absolute cross covariances
        self.cross_covariances_xyz = np.absolute(self.cross_covariances_xyz_matrix[:, (0, 1, 2), (0, 1, 2)])

        # X-U, Y-V, Z-W cross absolute covariances age
        self.cross_covariances_xyz_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.cross_covariances_xyz, axis=0))
        self.cross_covariances_xyz_min = np.min(self.cross_covariances_xyz, axis=0)

        # Cylindrical absolute cross covariances
        self.cross_covariances_rθz = np.absolute(np.mean(
            (np.array([star.position_rθz for star in self.stars]) - self.avg_rθz) *
            (np.array([star.velocity_μρμθw for star in self.stars]) - self.avg_μρμθw), axis=0))

        # Cylindrical absolute cross covariances age
        self.cross_covariances_rθz_age = np.vectorize(
            lambda step: self.series.time[step])(np.argmin(self.cross_covariances_rθz, axis=0))
        self.cross_covariances_rθz_min = np.min(self.cross_covariances_rθz, axis=0)

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

        # Minimum spanning tree average branch length age
        self.mst_mean = np.mean(self.mst_lengths, axis=1)
        self.mst_mean_age = self.series.time[np.argmin(self.mst_mean)]
        self.mst_min = np.min(self.mst_mean)

        # Single minimum spanning tree branch length median absolute deviation
        # self.mst_median = np.median(self.mst_lengths, axis=1)
        # self.mst_mad = np.median(
        #     np.abs(self.mst_lengths - np.expand_dims(self.mst_median, axis=0).T), axis=1)

        # Jack-knife Monte Carlo
        n = 500
        total = np.zeros((n, len(self.series.time)))
        for i in range(n):
            mst_lengths = self.mst_lengths[:, np.random.choice(self.number_of_branches,
                int(self.number_of_branches * 0.8), replace=False)]

            # Median absolute deviation
            mst_median = np.median(mst_lengths, axis=1)
            mst_mad = np.median(np.abs(mst_lengths - np.expand_dims(mst_median, axis=0).T), axis=1)
            total[i] = mst_mad

        # Average minimum spanning tree branch length median absolute deviation
        self.mst_mad = np.mean(total, axis=0)

        # Minimum spanning tree branch length median absolute deviation age
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
                overtime is computed with galpy.
            """

            # Initialization
            self.group = group
            self.outlier = False
            self.model = False
            vars(self).update(values)

            # Positions and errors from linear traceback
            self.position_linear = self.position - (self.velocity - Coordinate.sun_velocity) * np.expand_dims(
                self.group.series.time, axis=0).T
            self.position_error = (self.position_error**2 + (self.velocity_error
                * np.expand_dims(self.group.series.time, axis=0).T)**2)**0.5

            # Orbit integration
            self.get_orbit()

        def get_orbit(self):
            """ Computes a star's backward of forward galactic orbit using Galpy. """

            # Other position and velocity computations
            self.position_rθz_other, self.position_rθz_error_other = galactic_xyz_galactocentric_ρθz(
                *self.position)
            self.velocity_μρμθw_other, self.velocity_μρμθw_error_other = galactic_uvw_galactocentric_ρθz(
                *self.position, *self.velocity)

            # Initial position in galactocentric cylindrical coordinates (rθz)
            self.position_rθz = np.array(gpcooords.XYZ_to_galcencyl(
                *self.position,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False))

            # Initial velocity in galactocentric cylindrical coordinates (μρμθw)
            self.velocity_μρμθw = np.array(gpcooords.vxvyvz_to_galcencyl(
                *self.velocity,
                *np.array(gpcooords.XYZ_to_galcenrect(
                    *self.position,
                    Xsun=Coordinate.sun_position[0],
                    Zsun=Coordinate.sun_position[2],
                    _extra_rot=False)),
                vsun=Coordinate.sun_velocity * np.array([-1, 1, 1]),
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                galcen=False, _extra_rot=False))

            # Time for a model star or traceback
            if self.model:
                self.time = np.linspace(0., (-self.group.series.age.value if self.backward
                    else self.group.series.age.value), 1000)
            else:
                self.time = -self.group.series.time if self.backward else self.group.series.time

            # Unit conversion
            self.time *= Coordinate.sun_velocity[1] / Coordinate.sun_position[0]
            self.position_rθz /= np.array([Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])
            self.velocity_μρμθw /= Coordinate.sun_velocity[1]

            # Orbit initialization and integration
            orbit = Orbit([
                self.position_rθz[0], self.velocity_μρμθw[0], self.velocity_μρμθw[1],
                self.position_rθz[2], self.velocity_μρμθw[2], self.position_rθz[1]])
            orbit.integrate(self.time, self.group.potential, method='odeint')

            # Orbital rθz positions and unit conversion
            self.position_rθz = np.array([
                orbit.R(self.time),
                orbit.phi(self.time),
                orbit.z(self.time)]).T * np.array(
                    [Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])

            # Orbital μρμθw velocities and unit conversion
            self.velocity_μρμθw = np.array([
                orbit.vR(self.time),
                orbit.vT(self.time),
                orbit.vz(self.time)]).T * Coordinate.sun_velocity[1]

            # XYZ positions
            self.position = gpcooords.galcencyl_to_XYZ(
                *self.position_rθz.T,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)

            # UVW velocities
            self.velocity_new = gpcooords.galcencyl_to_vxvyvz(
                *self.velocity_μρμθw.T, self.position_rθz.T[1],
                vsun=Coordinate.sun_velocity * np.array([-1, 1, 1]),
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)

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

        def __repr__(self):
            """ Returns a string of name of the star. """

            return self.name

        def shift_rv(self):
            """ Shifts the radial velocity based on the gravitationnal redshift of the star and
                current value of the radial velocity.
            """

            # Gravitationnal constant (pc^3 / M_sol / Myr^2)
            G = 0.004498706056647732

            # Speed of light (pc/Myr)
            c = 306601.3937879527

            self.velocity[:,0] = a * (c + self.velocity[:,0]) - c
