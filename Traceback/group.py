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

    def __init__(self, series, number, name):
        """ Initializes a Group object and embedded Star objects from a simulated sample of stars
            in a moving group or raw data in the form a Data object. This dataset is then moved
            backwards in time from the initial time to the final time, and its age is estimated
            by minimizing: the 3D scatter, median absolute deviation, covariances, and the minimum
            spanning tree branch length mean and median absolute deviation.
        """

        # Initialization
        self.series = series
        self.number = number
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

        # Covariances
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
                    *(star.position.values if self.number == 1 else
                        np.random.normal(star.position.values, star.position.errors)),

                    # Velocity and radial velocity offset
                    *(star.velocity.values if self.number == 1 else
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
                velocity_uvw=velocity_uvw, velocity_uvw_error=velocity_uvw_error,
                position_xyz=position_xyz, position_xyz_error=position_xyz_error))

        # Stars creation from cartesian coordinates
        # for star in self.series.data:
        #     self.append(self.Star(
        #         self, name=star.name, backward=True,
        #         velocity_uvw=star.velocity.values, velocity_uvw_error=star.velocity.values,
        #         position_xyz=star.position.values, position_xyz_error=star.position.values))

    def stars_from_model(self):
        """ Creates an artificial model of star for a given number of stars and age based on
            the intial average XYZ position and UVW velocity, and their respective errors
            and scatters. The sample is then moved forward in time for the given age and radial
            velocity offset is also added.
        """

        # Model backward orbit integration
        average_model_star = self.Star(
            self, name='model_average_star', backward=True, model=True,
            velocity_uvw=self.series.velocity.values, velocity_uvw_error=np.zeros(3),
            position_xyz=self.series.position.values, position_xyz_error=np.zeros(3))

        # Average velocity arrays
        velocity = average_model_star.velocity_uvw[-1]
        velocity_error = self.series.velocity_error.values
        velocity_scatter = self.series.velocity_scatter.values

        # Average position arrays
        position = average_model_star.position_xyz[-1]
        position_error = self.series.position_error.values
        position_scatter = self.series.position_scatter.values

        # Stars creation from a model
        for star in range(self.series.number_of_stars):

            # Model star forward galactic orbit integration
            model_star = self.Star(
                self, name='model_star_{}'.format(star + 1), backward=False, model=True,
                velocity_uvw=np.random.normal(velocity, velocity_scatter), velocity_uvw_error=np.zeros(3),
                position_xyz=np.random.normal(position, position_scatter), position_xyz_error=np.zeros(3))

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rvμδμα = galactic_uvw_equatorial_rvμδμα(
                *model_star.position_xyz[-1], *model_star.velocity_uvw[-1])[0]
            velocity_rvμδμα += np.array([self.series.rv_offset.value, 0.0, 0.0])
            position_rδα = galactic_xyz_equatorial_rδα(*model_star.position_xyz[-1])[0]

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
                    *np.random.normal(position_obs, position_error),
                    *np.random.normal(velocity_obs, velocity_error),
                    *position_error, *velocity_error)

            # Velocity and position conversion back to galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_uvw, velocity_uvw_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rvμδμα, *position_rδα_error, *velocity_rvμδμα_error)

            # Star creation
            self.append(self.Star(
                self, name='star_{}'.format(star + 1), backward=True,
                velocity_uvw=velocity_uvw, velocity_uvw_error=velocity_uvw_error,
                position_xyz=position_xyz, position_xyz_error=position_xyz_error))

    class Indicator():
        """ Age indicator including its values, age at minimum and minimum. """

        def __init__(self, group, values):
            """ Initializes an age indicator. """

            self.group = group
            self.values = values
            self.age = self.group.series.time[np.argmin(self.values, axis=0)]
            self.min = np.min(self.values, axis=0)

    def get_stars_coordinates(self):
        """ Computes the number of stars that aren't ouliers, the stars average velocity and
            position, excluding outliers, and the stars relative positions and distances from
            the average, including outliers.
        """

        # Number of stars remaining, excluding outliers
        self.number_of_stars = len(self.stars)

        # Average velocity, excluding outliers
        self.velocity_uvw = np.mean([star.velocity_uvw for star in self.stars], axis=0)

        # Average position, excluding outliers
        self.position_xyz = np.mean([star.position_xyz for star in self.stars], axis=0)

        # Average rθz position, excluding outliers
        self.position_rθz = np.mean([star.position_rθz for star in self.stars], axis=0)

        # Average μρμθw velocity, excluding outliers
        self.velocity_μρμθw = np.mean([star.velocity_μρμθw for star in self.stars], axis=0)

        # Average linear position and errors, excluding outliers
        self.position_xyz_linear = np.mean(
            np.array([star.position_xyz_linear for star in self.stars]), axis=0)
        self.position_xyz_linear_error = np.sum(
            np.array([star.position_xyz_linear_error for star in self.stars])**2,
            axis=0)**0.5 / self.number_of_stars

        # Average linear velocity and errors, excluding outliers
        self.velocity_uvw_linear = np.mean([star.velocity_uvw_linear for star in self.stars], axis=0)
        self.velocity_uvw_linear_error = np.sum(np.array(
            [star.velocity_uvw_linear_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

        # Stars relative position and velocity, distance and speed, including outliers
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

            # Coordinates computation
            self.get_stars_coordinates()

            # XYZ position scatter
            self.scatter_xyz = self.Indicator(
                self, np.std([star.position_xyz for star in self.stars], axis=0))
            self.scatter_xyz_total = self.Indicator(
                self, np.sum(self.scatter_xyz.values**2, axis=1)**0.5)

            # Alternative xyz position scatter
            # self.scatter = (np.abs(np.sum(self.scatter_xyz**2, axis=1) - np.sum(
            #     self.position_xyz_linear_error[0]**2 + self.velocity_uvw_linear_error**2 \
            #     * np.expand_dims(self.series.time, axis=0).T**2, axis=1)))**0.5
            # self.sscatter_xyz_total = (np.abs(np.sum(self.scatter_xyz**2, axis=1)**0.5 - np.sum(
            #     self.position_xyz_linear_error[0]**2 + self.velocity_uvw_linear_error**2 \
            #     * np.expand_dims(self.series.time, axis=0).T**2, axis=1))**0.5

            # Filter stars beyond the σ cutoff of the average position and loop scatter computation
            outliers = False
            for star in self.stars:
                if (star.distance_xyz > (self.scatter_xyz_total.values * self.series.cutoff)).any():
                    star.outlier = True
                    outliers = True
                    self.outliers.append(star)
                    self.stars.remove(star)
                    self.number_of_stars = len(self.stars)

        # rθz position scatter
        self.scatter_rθz = self.Indicator(
            self, np.std([star.position_rθz for star in self.stars], axis=0))
        self.scatter_rθz_total = self.Indicator(
            self, np.sum(self.scatter_rθz.values**2, axis=1)**0.5)

        # UVW velocity scatter
        self.scatter_uvw = np.std([star.velocity_uvw for star in self.stars], axis=0)

        # μρμθw velocity scatter
        self.scatter_μρμθw = np.std([star.velocity_μρμθw for star in self.stars], axis=0)

    def get_median_absolute_deviation(self):
        """ Computes the XYZ and total median absolute deviation (MAD) of a group and their
            respective errors for all timesteps. The age of the moving is then estimated by
            finding the time at which the median absolute deviation is minimal.
        """

        # XYZ median absolute deviation age
        # self.mad_xyz = self.Indicator(self, np.median(
        #     np.abs(np.array([star.position_xyz for star in self.stars]) - np.median(
        #         np.array([star.position_xyz for star in self.stars]), axis=0)), axis=0))
        # self.mad_xyz_total = self.Indicator(self, np.sum(self.mad_xyz.values**2, axis=1)**0.5)

        # rθz median absolute deviation age
        # self.mad_rθz = self.Indicator(self, np.median(
        #     np.abs(np.array([star.position_rθz for star in self.stars]) - np.median(
        #         np.array([star.position_rθz for star in self.stars]), axis=0)), axis=0))
        # self.mad_rθz_total = self.Indicator(self, np.sum(self.mad_rθz.values**2, axis=1)**0.5)

        # Jack-knife Monte Carlo
        n = 500
        m = int(self.number_of_stars * 0.8)
        stars = np.array([[self.stars[i] for i in np.random.choice(
            self.number_of_stars, m, replace=False)] for j in range(n)], dtype=object).flatten()
        positions_xyz = np.array([star.position_xyz for star in stars]).reshape(
            (n, m, self.series.number_of_steps, 3)).swapaxes(0, 1)
        positions_rθz = np.array([star.position_rθz for star in stars]).reshape(
            (n, m, self.series.number_of_steps, 3)).swapaxes(0, 1)

        # XYZ median absolute deviation age
        self.mad_xyz = self.Indicator(self, np.mean(
            np.median(np.abs(positions_xyz - np.median(positions_xyz, axis=0)), axis=0), axis=0))
        self.mad_xyz_total = self.Indicator(self, np.sum(self.mad_xyz.values**2, axis=1)**0.5)

        # rθz median absolute deviation age
        self.mad_rθz = self.Indicator(self, np.mean(
            np.median(np.abs(positions_rθz - np.median(positions_rθz, axis=0)), axis=0), axis=0))
        self.mad_rθz_total = self.Indicator(self, np.sum(self.mad_rθz.values**2, axis=1)**0.5)

    def get_covariances(self):
        """ Computes the X-U, Y-V and Z-W absolute covariances of a group and their respective
            errors for all timesteps. The age of the moving is then estimated by finding the time
            at which the covariances is minimal.
        """

        # XYZ covariance matrix and determinant ages
        self.covariances_xyz_matrix = self.get_covariance_matrix('position_xyz')
        self.covariances_xyz = self.Indicator(
            self, np.abs(self.covariances_xyz_matrix[:, (0, 1, 2), (0, 1, 2)]))
        self.covariances_xyz_matrix_det = self.Indicator(
            self, np.linalg.det(self.covariances_xyz_matrix))

        # XYZ and UVW cross covariance matrix and determinant ages
        self.cross_covariances_xyz_matrix = self.get_covariance_matrix('position_xyz', 'velocity_uvw')
        self.cross_covariances_xyz = self.Indicator(
            self, np.abs(self.cross_covariances_xyz_matrix[:, (0, 1, 2), (0, 1, 2)]))
        self.cross_covariances_xyz_matrix_det = self.Indicator(
            self, np.linalg.det(self.cross_covariances_xyz_matrix))

        # rθz covariance matrix and determinant ages
        self.covariances_rθz_matrix = self.get_covariance_matrix('position_rθz')
        self.covariances_rθz = self.Indicator(
            self, np.abs(self.covariances_rθz_matrix[:, (0, 1, 2), (0, 1, 2)]))
        self.covariances_rθz_matrix_det = self.Indicator(
            self, np.linalg.det(self.covariances_rθz_matrix))

        # rθz and μρμθw cross covariance matrix and determinant ages
        self.cross_covariances_rθz_matrix = self.get_covariance_matrix('position_rθz', 'velocity_μρμθw')
        self.cross_covariances_rθz = self.Indicator(
            self, np.abs(self.cross_covariances_rθz_matrix[:, (0, 1, 2), (0, 1, 2)]))
        self.cross_covariances_rθz_matrix_det = self.Indicator(
            self, np.linalg.det(self.cross_covariances_rθz_matrix))

    def get_covariance_matrix(self, a, b=None):
        """ Computes the covariance matrix of a Star parameter, 'a', along all physical dimensions
            for all timestep. If another Star parameter, 'b', is given, the cross covariance matrix
            is given instead.
        """

        # Covariance matrix
        a = np.array([vars(star)[a] for star in self.stars]) - vars(self)[a]
        a = np.tile(a.T, (a.shape[-1], 1, 1, 1))
        if b is None:
            b = np.swapaxes(a, 0, 1)

        # Cross covariance matrix
        else:
            b = np.array([vars(star)[b] for star in self.stars]) - vars(self)[b]
            b = np.swapaxes(np.tile(b.T, (b.shape[-1], 1, 1, 1)), 0, 1)
        return np.mean(a * b, axis=3).T

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
        self.mst_mean = self.Indicator(self, np.mean(self.mst_lengths, axis=1))

        # Minimum spanning tree branch length median absolute deviation age
        # self.mst_mad = self.Indicator(self,
        #     np.median(np.abs(self.mst_lengths.T - np.median(self.mst_lengths.T, axis=0)), axis=0))

        # Jack-knife Monte Carlo
        n = 500
        m = int(self.number_of_branches * 0.8)
        mst_lengths = np.array([self.mst_lengths.T[np.random.choice(self.number_of_branches, m,
            replace=False)] for i in range(n)]).swapaxes(0, 1)

        # Minimum spanning tree branch length median absolute deviation age
        self.mst_mad = self.Indicator(self, np.mean(
            np.median(np.abs(mst_lengths - np.median(mst_lengths, axis=0)), axis=0), axis=0))

    class Branch:
        """ Line connecting two stars used for the calculation of the minimum spanning tree. """

        def __init__(self, start, end):
            """ Initializes a Branch object and computes the distance between two Star objects,
                'start' and 'end', for all timestep.
            """

            self.start = start
            self.end = end
            self.length = np.sum((self.start.position_xyz - self.end.position_xyz)**2, axis=1)**0.5

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

            # Trajectory integration
            self.get_linear()
            self.get_orbit()

        def get_linear(self):
            """ Computes a star's backward or forward linear trajectory. """

            # Linear XYZ position
            self.position_xyz_linear = self.position_xyz - (self.velocity_uvw
                + Coordinate.galactic_velocity) * np.expand_dims(self.group.series.time, axis=0).T
            self.position_xyz_linear_error = (self.position_xyz_error**2 + (self.velocity_uvw_error
                * np.expand_dims(self.group.series.time, axis=0).T)**2)**0.5

            # Linear UVW velocity
            self.velocity_uvw_linear = self.velocity_uvw
            self.velocity_uvw_linear_error = self.velocity_uvw_error

        def get_orbit(self):
            """ Computes a star's backward or forward galactic orbit using Galpy. """

            # Other position and velocity computations
            self.position_rθz_other, self.position_rθz_error_other = galactic_xyz_galactocentric_ρθz(
                *self.position_xyz)
            self.velocity_μρμθw_other, self.velocity_μρμθw_error_other = galactic_uvw_galactocentric_ρθz(
                *self.position_xyz, *self.velocity_uvw)

            # Initial position in galactocentric cylindrical coordinates (rθz)
            self.position_rθz = np.array(gpcooords.XYZ_to_galcencyl(
                *self.position_xyz,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False))

            # Initial velocity in galactocentric cylindrical cxoordinates (μρμθw)
            self.velocity_μρμθw = np.array(gpcooords.vxvyvz_to_galcencyl(
                *self.velocity_uvw,
                *np.array(gpcooords.XYZ_to_galcenrect(
                    *self.position_xyz,
                    Xsun=Coordinate.sun_position[0],
                    Zsun=Coordinate.sun_position[2],
                    _extra_rot=False)),
                vsun=Coordinate.sun_velocity,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False))

            # Somehow, adding  * np.array([-1, 1, 1]) to vsun above makes this work
            # if self.name == 'Star_30':
            #     print(self.velocity_μρμθw_other - self.velocity_μρμθw)

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
            self.position_xyz = gpcooords.galcencyl_to_XYZ(
                *self.position_rθz.T,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)

            # UVW velocities
            self.velocity_uvw = gpcooords.galcencyl_to_vxvyvz(
                *self.velocity_μρμθw.T, self.position_rθz.T[1],
                vsun=Coordinate.sun_velocity,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)

            # ξηζ positions
            self.position_ξηζ = position_rθz_ξηζ(*self.position_rθz.T,
                -self.group.series.time if self.backward else self.group.series.time)

            # ξηζ velocities
            self.velocity_ξηζ = velocity_rθz_ξηζ(*self.velocity_μρμθw.T,
                -self.group.series.time if self.backward else self.group.series.time)

            # Temporary re-assignment
            self.position_rθz = self.position_ξηζ
            self.velocity_μρμθw = self.velocity_ξηζ

        def get_relative_coordinates(self):
            """ Computes relative position and velocity, the distance and the speed from the
                average position and velocity, and their respective errors for all timesteps.
            """

            # Relative position and distance from the average group position
            self.relative_position_xyz = self.position_xyz - self.group.position_xyz
            self.distance_xyz = np.sum(self.relative_position_xyz**2, axis=1)**0.5

            # Relative velocity and speed from the average group velocity
            self.relative_velocity_uvw = self.velocity_uvw - self.group.velocity_uvw
            self.speed_uvw = np.sum(self.relative_velocity_uvw**2, axis=0)**0.5

            # Linear relative position and distance from the average group position
            self.relative_position_xyz_linear = self.position_xyz_linear - self.group.position_xyz_linear
            self.relative_position_xyz_linear_error = (
                self.position_xyz_linear_error**2 + self.group.position_xyz_linear_error**2)**0.5
            self.distance_xyz_linear = np.sum(self.relative_position_xyz_linear**2, axis=1)**0.5
            self.distance_xyz_linear_error = np.sum((self.relative_position_xyz_linear
                * self.relative_position_xyz_linear_error)**2, axis=1)**0.5 / self.distance_xyz_linear

            # Linear relative velocity and speed from the average velocity
            self.relative_velocity_uvw_linear = self.velocity_uvw_linear - self.group.velocity_uvw_linear
            self.relative_velocity_uvw_linear_error = (
                self.velocity_uvw_linear_error**2 + self.group.velocity_uvw_linear_error**2)**0.5
            self.speed_uvw_linear = np.sum(self.relative_velocity_uvw_linear_error**2, axis=0)**0.5
            self.speed_uvw_linear_error = np.sum((self.relative_velocity_uvw_linear_error
                * self.relative_position_xyz_linear_error)**2, axis=0)**0.5 / self.speed_uvw_linear

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

            self.velocity_uvw[:,0] = G * (c + self.velocity_uvw[:,0]) - c
