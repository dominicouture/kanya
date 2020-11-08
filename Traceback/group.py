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
        self.indicators = []
        self.name = name

        # Milky Way Potential
        from galpy.potential import MWPotential2014
        from galpy.potential.mwpotentials import Irrgang13I
        # self.potential = MWPotential2014
        self.potential = Irrgang13I

        # Stars from data
        if series.from_data:
            self.stars_from_data()

        # Stars from model
        elif series.from_model:
            self.stars_from_model()

        # Average velocity and position and filter outliers
        self.get_stars_coordinates()

        # Jack Knife Monte Carlo parameters
        self.set_jack_knife_Monte_Carlo()

        # Scatter
        self.get_scatter()

        # Median absolute deviation
        self.get_median_absolute_deviation()

        # Covariances
        self.get_covariances()

        # Minimum spanning tree
        self.get_minimum_spanning_tree()

    def stars_from_data(self):
        """ Creates a list of Star objects from a Python dictionary or CSV files containing the
            parameters including the name of the stars, their xyz positions and UVW velocities,
            and their respective measurement errors. Radial velocity offset is also added. If
            only one group is created in the series, actual values are used. If multiple groups
            are created, positions and velocities are scrambled based on measurement errors.
        """

        # Star creation from observables
        for star in self.series.data:
            # rv_offset = self.series.rv_offset.value
            rv_offset = star.rv_offset if star.rv_offset is not None else self.series.rv_offset.value

            # Observables conversion into equatorial spherical coordinates
            (position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error) = \
                observables_spherical(

                    # Position
                    *(star.position.values if self.number == 0 else
                        np.random.normal(star.position.values, star.position.errors)),

                    # Velocity and radial velocity offset
                    *(star.velocity.values if self.number == 0 else
                        np.random.normal(star.velocity.values, star.velocity.errors)) \
                        + np.array([rv_offset, 0.0, 0.0]),

                    # Position and velocity errors
                    *star.position.errors, *star.velocity.errors)

            # Equatorial spherical coordinates conversion into galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error)

            # Star creation
            self.append(self.Star(
                self, name=star.name, backward=True,
                velocity_xyz=velocity_xyz, velocity_xyz_error=velocity_xyz_error,
                position_xyz=position_xyz, position_xyz_error=position_xyz_error))

        # Stars creation from cartesian coordinates
        # for star in self.series.data:
        #     self.append(self.Star(
        #         self, name=star.name, backward=True,
        #         velocity_xyz=star.velocity.values, velocity_xyz_error=star.velocity.errors,
        #         position_xyz=star.position.values, position_xyz_error=star.position.errors))

    def stars_from_model(self):
        """ Creates an artificial model of star for a given number of stars and age based on
            the intial average xyz position and UVW velocity, and their respective errors
            and scatters. The sample is then moved forward in time for the given age and radial
            velocity offset is also added.
        """

        # Model backward orbit integration
        average_model_star = self.Star(
            self, name='model_average_star', backward=True, model=True,
            velocity_xyz=self.series.velocity.values, velocity_xyz_error=np.zeros(3),
            position_xyz=self.series.position.values, position_xyz_error=np.zeros(3))

        # Average velocity arrays
        velocity = average_model_star.velocity_xyz[-1]
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
                velocity_xyz=np.random.normal(velocity, velocity_scatter), velocity_xyz_error=np.zeros(3),
                position_xyz=np.random.normal(position, position_scatter), position_xyz_error=np.zeros(3))

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rδα = galactic_uvw_equatorial_rvμδμα(
                *model_star.position_xyz[-1], *model_star.velocity_xyz[-1])[0]
            velocity_rδα += np.array([self.series.rv_offset.value, 0.0, 0.0])
            position_rδα = galactic_xyz_equatorial_rδα(*model_star.position_xyz[-1])[0]

            # Velocity and position conversion to observables
            position_obs, velocity_obs = spherical_observables(*position_rδα, *velocity_rδα)[:2]

            # Observables conversion back into equatorial spherical coordinates
            # Velocity and position scrambling based on actual measurement errors
            if self.series.data_errors:
                star_errors = star - star // len(self.series.data) * len(self.series.data)
                (position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error) = \
                        observables_spherical(
                    *np.random.normal(position_obs, self.series.data[star_errors].position.errors),
                    *np.random.normal(velocity_obs, self.series.data[star_errors].velocity.errors),
                    *self.series.data[star_errors].position.errors,
                    *self.series.data[star_errors].velocity.errors)

            # Velocity and position scrambling based on average measurement errors
            else:
                (position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error) = \
                        observables_spherical(
                    *np.random.normal(position_obs, position_error),
                    *np.random.normal(velocity_obs, velocity_error),
                    *position_error, *velocity_error)

            # Velocity and position conversion back to galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error)

            # Star creation
            self.append(self.Star(
                self, name='star_{}'.format(star + 1), backward=True,
                velocity_xyz=velocity_xyz, velocity_xyz_error=velocity_xyz_error,
                position_xyz=position_xyz, position_xyz_error=position_xyz_error))

    def get_stars_coordinates(self):
        """ Computes the number of stars that aren't outliers, the stars average xyz and ξηζ
            velocity and position, excluding outliers, and the stars relative positions and
            distances from the average, including outliers. If all stars are identified as
            outliers, an error is raised.
        """

        # Valid stars and outliers lists
        self.stars = list(self)
        self.number_of_stars = len(self.stars)
        self.outliers = []
        outliers = True

        # Outlier filtering
        while outliers and self.number_of_stars > int(0.8 * self.series.number_of_stars):

            # Number of stars remaining, excluding outliers
            self.number_of_stars = len(self.stars)

            # Average position, excluding outliers
            self.position_xyz = np.mean([star.position_xyz for star in self.stars], axis=0)

            # Average linear position and errors, excluding outliers
            self.position_xyz_linear = np.mean(
                np.array([star.position_xyz_linear for star in self.stars]), axis=0)
            self.position_xyz_linear_error = np.sum(
                np.array([star.position_xyz_linear_error for star in self.stars])**2,
                axis=0)**0.5 / self.number_of_stars

            # Average ξηζ position, excluding outliers
            self.position_ξηζ = np.mean([star.position_ξηζ for star in self.stars], axis=0)

            # Average velocity, excluding outliers
            self.velocity_xyz = np.mean([star.velocity_xyz for star in self.stars], axis=0)

            # Average linear velocity and errors, excluding outliers
            self.velocity_xyz_linear = np.mean([star.velocity_xyz_linear for star in self.stars], axis=0)
            self.velocity_xyz_linear_error = np.sum(np.array(
                [star.velocity_xyz_linear_error for star in self.stars])**2, axis=0)**0.5 / self.number_of_stars

            # Average ξηζ velocity, excluding outliers
            self.velocity_ξηζ = np.mean([star.velocity_ξηζ for star in self.stars], axis=0)

            # Stars relative position and velocity, distance and speed, including outliers
            for star in self:
                star.get_relative_coordinates()

            # ξηζ position scatter
            self.scatter_ξηζ = np.std([star.position_ξηζ for star in self.stars], axis=0)

            # Filter stars beyond the σ cutoff of the average position and loop scatter computation
            outliers = False
            for star in self.stars:
                if (np.abs(star.relative_position_ξηζ) > (
                        self.scatter_ξηζ * self.series.cutoff)).any():
                    star.outlier = True
                    print(star.name)
                    outliers = True
                    self.outliers.append(star)
                    self.stars.remove(star)
                    self.number_of_stars = len(self.stars)

    def set_jack_knife_Monte_Carlo(self):
        """ Sets the parameters to compute Jack-Knife Monte-Carlo of age indicators. """

        # Select stars for Jack-Knife Monte-Carlo
        stars = np.array([np.random.choice(self.number_of_stars,
            int(self.number_of_stars * self.series.jack_knife_fraction), replace=False)
                for i in range(self.series.jack_knife_number)])

        # Create selected stars xyz and ξηζ positions and velocities arrays
        self.positions_xyz = np.array([
            star.position_xyz for star in self.stars])[stars,:,:].swapaxes(0, 1)
        self.positions_ξηζ = np.array([
            star.position_ξηζ for star in self.stars])[stars,:,:].swapaxes(0, 1)
        self.velocities_xyz = np.array([
            star.velocity_xyz for star in self.stars])[stars,:,:].swapaxes(0, 1)
        self.velocities_ξηζ = np.array([
            star.velocity_ξηζ for star in self.stars])[stars,:,:].swapaxes(0, 1)

    class Indicator():
        """ Age indicator including its values, age at minimum and minimum. Computes errors on
            values, age and minium using a Jack-Knife Monte-Carlo Method. """

        def __init__(self, group, values):
            """ Initializes an age indicator. """

            # Group and values
            self.group = group
            self.values = values
            self.group.indicators.append(self)

            # Average values
            self.value = np.atleast_1d(np.mean(self.values, axis=0))
            self.value_int_error = np.atleast_1d(np.std(self.values, axis=0))

            # Average age at the minimal value
            self.ages = np.atleast_1d(self.group.series.time[np.argmin(self.values, axis=1)])
            self.age = np.atleast_1d(self.group.series.time[np.argmin(self.value, axis=0)])
            self.age_int_error = np.atleast_1d(np.std(self.ages, axis=0))

            # Average minimal value
            self.minima = np.atleast_1d(np.min(self.values, axis=1))
            self.min = np.atleast_1d(np.min(self.value, axis=0))
            self.min_int_error = np.atleast_1d(np.std(self.minima, axis=0))

    def get_scatter(self):
        """ Computes the xyz and ξηζ scatter of a group. The age of the
            moving group is then estimated by finding the time at which the scatter is minimal.
        """

        # xyz position scatter
        self.scatter_xyz = self.Indicator(self, np.std(self.positions_xyz, axis=0))
        self.scatter_xyz_total = self.Indicator(self, np.sum(self.scatter_xyz.values**2, axis=2)**0.5)

        # ξηζ position scatter
        self.scatter_ξηζ = self.Indicator(self, np.std(self.positions_ξηζ, axis=0))
        self.scatter_ξηζ_total = self.Indicator(self, np.sum(self.scatter_ξηζ.values**2, axis=2)**0.5)

        # xyz velocity scatter
        self.scatter_velocity_xyz = np.std([star.velocity_xyz for star in self.stars], axis=0)
        self.scatter_velocity_xyz_total = np.sum(self.velocity_xyz**2, axis=0)**0.5

        # ξηζ velocity scatter
        self.scatter_velocity_ξηζ = np.std([star.velocity_ξηζ for star in self.stars], axis=0)
        self.scatter_velocity_ξηζ_total = np.sum(self.velocity_ξηζ**2, axis=0)**0.5

    def get_median_absolute_deviation(self):
        """ Computes the xyz and total median absolute deviation (MAD) of a group and their
            respective errors for all timesteps. The age of the moving is then estimated by
            finding the time at which the median absolute deviation is minimal.
        """

        # xyz median absolute deviation age
        self.mad_xyz = self.Indicator(self,
            np.median(np.abs(self.positions_xyz - np.median(self.positions_xyz, axis=0)), axis=0))
        self.mad_xyz_total = self.Indicator(self, np.sum(self.mad_xyz.values**2, axis=2)**0.5)

        # ξηζ median absolute deviation age
        self.mad_ξηζ = self.Indicator(self,
            np.median(np.abs(self.positions_ξηζ - np.median(self.positions_ξηζ, axis=0)), axis=0))
        self.mad_ξηζ_total = self.Indicator(self, np.sum(self.mad_ξηζ.values**2, axis=2)**0.5)

    def get_covariances(self):
        """ Computes the X-U, Y-V and Z-W absolute covariances of a group and their respective
            errors for all timesteps. The age of the moving is then estimated by finding the time
            at which the covariances is minimal.
        """

        # xyz position covariance matrix and determinant ages
        self.covariances_xyz_matrix = self.get_covariance_matrix('positions_xyz')
        self.covariances_xyz = self.Indicator(
            self, np.abs(self.covariances_xyz_matrix[:, :, (0, 1, 2), (0, 1, 2)])**0.5)
        self.covariances_xyz_matrix_det = self.Indicator(
            self, np.abs(np.linalg.det(self.covariances_xyz_matrix))**(1/6))
        self.covariances_xyz_matrix_trace = self.Indicator(
            self, (np.trace(self.covariances_xyz_matrix, axis1=2, axis2=3) / 3)**0.5)

        # xyz position and velocity cross covariance matrix and determinant ages
        self.cross_covariances_xyz_matrix = self.get_covariance_matrix('positions_xyz', 'velocities_xyz')
        self.cross_covariances_xyz = self.Indicator(
            self, np.abs(self.cross_covariances_xyz_matrix[:, :, (0, 1, 2), (0, 1, 2)])**0.5)
        self.cross_covariances_xyz_matrix_det = self.Indicator(
            self, np.abs(np.linalg.det(self.cross_covariances_xyz_matrix))**(1/6))
        self.cross_covariances_xyz_matrix_trace = self.Indicator(
            self, np.abs(np.trace(self.cross_covariances_xyz_matrix, axis1=2, axis2=3) / 3)**0.5)

        # ξηζ position covariance matrix and determinant ages
        self.covariances_ξηζ_matrix = self.get_covariance_matrix('positions_ξηζ')
        self.covariances_ξηζ = self.Indicator(
            self, np.abs(self.covariances_ξηζ_matrix[:, :, (0, 1, 2), (0, 1, 2)])**0.5)
        self.covariances_ξηζ_matrix_det = self.Indicator(
            self, np.abs(np.linalg.det(self.covariances_ξηζ_matrix))**(1/6))
        self.covariances_ξηζ_matrix_trace = self.Indicator(
            self, (np.trace(self.covariances_ξηζ_matrix, axis1=2, axis2=3) / 3)**0.5)

        # ξηζ position and velocity cross covariance matrix and determinant ages
        self.cross_covariances_ξηζ_matrix = self.get_covariance_matrix('positions_ξηζ', 'velocities_ξηζ')
        self.cross_covariances_ξηζ = self.Indicator(
            self, np.abs(self.cross_covariances_ξηζ_matrix[:, :, (0, 1, 2), (0, 1, 2)])**0.5)
        self.cross_covariances_ξηζ_matrix_det = self.Indicator(
            self, np.abs(np.linalg.det(self.cross_covariances_ξηζ_matrix))**(1/6))
        self.cross_covariances_ξηζ_matrix_trace = self.Indicator(
            self, np.abs(np.trace(self.cross_covariances_ξηζ_matrix, axis1=2, axis2=3) / 3)**0.5)

        # ξηζ position robust covariance matrix and determinant ages
        self.covariances_ξηζ_robust_matrix = self.get_covariance_robust_matrix('position_ξηζ')
        self.covariances_ξηζ_robust = self.Indicator(self, np.repeat(np.expand_dims(
            np.abs(self.covariances_ξηζ_robust_matrix[:, (0, 1, 2), (0, 1, 2)])**0.5, axis=0),
                self.series.jack_knife_number, axis=0))
        self.covariances_ξηζ_matrix_robust_det = self.Indicator(self, np.repeat(np.expand_dims(
            np.abs(np.linalg.det(self.covariances_ξηζ_robust_matrix))**(1/6), axis=0),
                self.series.jack_knife_number, axis=0))
        self.covariances_ξηζ_matrix_robust_trace = self.Indicator(self, np.repeat(np.expand_dims(
            (np.trace(self.covariances_ξηζ_robust_matrix, axis1=1, axis2=2) / 3)**0.5, axis=0),
                self.series.jack_knife_number, axis=0))

        # ??? xyz position robust covariance matrix and determinant ages ???
        # ??? xyz position and velocity robust cross covariance matrix and determinant ages ???
        # ??? ξηζ position and velocity robust cross covariance matrix and determinant ages ???

    def get_covariance_matrix(self, a, b=None):
        """ Computes the covariance matrix of a Star parameter, 'a', along all physical dimensions
            for all timestep. If another Star parameter, 'b', is given, the cross covariance matrix
            is given instead.
        """

        # Covariance matrix
        a = vars(self)[a] - np.mean(vars(self)[a], axis=0)
        a = np.tile(a.T, (a.shape[-1], 1, 1, 1, 1))
        if b is None:
            b = np.swapaxes(a, 0, 1)

        # Cross covariance matrix
        else:
            b = vars(self)[b] - np.mean(vars(self)[b], axis=0)
            b = np.swapaxes(np.tile(b.T, (b.shape[-1], 1, 1, 1, 1)), 0, 1)
        return np.mean(a * b, axis=4).T

    def get_covariance_robust_matrix(self, a):
        """ Computes a robust covariance determinant with SKlearn. """

        # Initialization
        from sklearn.covariance import MinCovDet
        a = np.array([vars(star)[a] for star in self.stars])
        a = np.swapaxes(a, 0, 1)

        # Covariance matrix
        robust_covariance_matrix = []
        for i in range(a.shape[0]):
            robust_covariance_matrix.append(MinCovDet(
                assume_centered=False).fit(a[i]).covariance_)
        return np.array(robust_covariance_matrix)

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

        # Jack-knife Monte Carlo branch lengths
        mst_lengths = np.array([self.mst_lengths.T[np.random.choice(self.number_of_branches,
            int(self.number_of_branches * self.series.jack_knife_fraction), replace=False)]
                for i in range(self.series.jack_knife_number)]).swapaxes(0, 1)

        # Minimum spanning tree average branch length age
        self.mst_mean = self.Indicator(self, np.mean(mst_lengths, axis=0))

        # Minimum spanning tree branch length median absolute deviation age
        self.mst_mad = self.Indicator(self,
            np.median(np.abs(mst_lengths - np.median(mst_lengths, axis=0)), axis=0))

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

            # Linear xyz position
            self.position_xyz_linear = self.position_xyz - (self.velocity_xyz
                + Coordinate.galactic_velocity) * np.expand_dims(self.group.series.time, axis=0).T
            self.position_xyz_linear_error = (self.position_xyz_error**2 + (self.velocity_xyz_error
                * np.expand_dims(self.group.series.time, axis=0).T)**2)**0.5

            # Linear UVW velocity
            self.velocity_xyz_linear = self.velocity_xyz
            self.velocity_xyz_linear_error = self.velocity_xyz_error

        def get_orbit(self):
            """ Computes a star's backward or forward galactic orbit using Galpy. """

            # New method to check previous method
            from astropy import units as u
            self.position_rδα, self.position_rδα_error = xyz_to_rδα(*self.position_xyz)
            r, δ, α = self.position_rδα[0] * u.Unit('pc'), self.position_rδα[1] * u.Unit('rad'), self.position_rδα[2] * u.Unit('rad')
            u, v, w = self.velocity_xyz * 0.9777922216731283

            # Other position and velocity computations
            self.position_rθz_other, self.position_rθz_error_other = galactic_xyz_galactocentric_ρθz(
                *self.position_xyz)
            self.velocity_rθz_other, self.velocity_rθz_error_other = galactic_uvw_galactocentric_ρθz(
                *self.position_xyz, *self.velocity_xyz)

            # Initial rθz position in galactocentric cylindrical coordinates
            self.position_rθz = np.array(gpcooords.XYZ_to_galcencyl(
                *self.position_xyz,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False))

            # Initial rθz velocity in galactocentric cylindrical coordinates
            self.velocity_rθz = np.array(gpcooords.vxvyvz_to_galcencyl(
                *self.velocity_xyz,
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
            #     print(self.velocity_rθz_other - self.velocity_rθz)

            # Time
            self.time = -self.group.series.time if self.backward else self.group.series.time

            # Unit conversion
            self.time *= Coordinate.lsr_velocity[1] / Coordinate.sun_position[0]
            self.position_rθz /= np.array([Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])
            self.velocity_rθz /= Coordinate.lsr_velocity[1]

            # Orbit initialization and integration
            orbit = Orbit([
                self.position_rθz[0], self.velocity_rθz[0], self.velocity_rθz[1],
                self.position_rθz[2], self.velocity_rθz[2], self.position_rθz[1]],
                ro=Coordinate.ro, vo=Coordinate.vo, zo=Coordinate.zo,
                solarmotion=Coordinate.sun_velocity_peculiar_kms)
            orbit.turn_physical_off()
            orbit.integrate(self.time, self.group.potential, method='odeint')

            # Orbital rθz positions and unit conversion
            self.position_rθz = np.array([
                orbit.R(self.time),
                orbit.phi(self.time),
                orbit.z(self.time)]).T * np.array(
                    [Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])

            # Orbital rθz velocities and unit conversion
            self.velocity_rθz = np.array([
                orbit.vR(self.time),
                orbit.vT(self.time),
                orbit.vz(self.time)]).T * Coordinate.lsr_velocity[1]

            # xyz positions
            self.position_xyz = gpcooords.galcencyl_to_XYZ(
                *self.position_rθz.T,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)

            # xyz velocities
            self.velocity_xyz = gpcooords.galcencyl_to_vxvyvz(
                *self.velocity_rθz.T, self.position_rθz.T[1],
                vsun=Coordinate.sun_velocity,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)

            # ξηζ positions
            self.position_ξηζ = position_rθz_ξηζ(*self.position_rθz.T,
                -self.group.series.time if self.backward else self.group.series.time)

            # ξηζ velocities
            self.velocity_ξηζ = velocity_rθz_ξηζ(*self.velocity_rθz.T,
                -self.group.series.time if self.backward else self.group.series.time)

        def get_relative_coordinates(self):
            """ Computes relative position and velocity, the distance and the speed from the
                average position and velocity, and their respective errors for all timesteps.
            """

            # Relative position and distance from the average group position
            self.relative_position_xyz = self.position_xyz - self.group.position_xyz
            self.distance_xyz = np.sum(self.relative_position_xyz**2, axis=1)**0.5

            # Relative velocity and speed from the average group velocity
            self.relative_velocity_xyz = self.velocity_xyz - self.group.velocity_xyz
            self.speed_xyz = np.sum(self.relative_velocity_xyz**2, axis=0)**0.5

            # Relative ξηζ position from the average group position
            self.relative_position_ξηζ = self.position_ξηζ - self.group.position_ξηζ
            self.distance_ξηζ = np.sum(self.relative_position_ξηζ**2, axis=1)**0.5

            # Linear relative position and distance from the average group position
            self.relative_position_xyz_linear = self.position_xyz_linear - self.group.position_xyz_linear
            self.relative_position_xyz_linear_error = (
                self.position_xyz_linear_error**2 + self.group.position_xyz_linear_error**2)**0.5
            self.distance_xyz_linear = np.sum(self.relative_position_xyz_linear**2, axis=1)**0.5
            self.distance_xyz_linear_error = np.sum((self.relative_position_xyz_linear
                * self.relative_position_xyz_linear_error)**2, axis=1)**0.5 / self.distance_xyz_linear

            # Linear relative velocity and speed from the average velocity
            self.relative_velocity_xyz_linear = self.velocity_xyz_linear - self.group.velocity_xyz_linear
            self.relative_velocity_xyz_linear_error = (
                self.velocity_xyz_linear_error**2 + self.group.velocity_xyz_linear_error**2)**0.5
            self.speed_xyz_linear = np.sum(self.relative_velocity_xyz_linear_error**2, axis=0)**0.5
            self.speed_xyz_linear_error = np.sum((self.relative_velocity_xyz_linear_error
                * self.relative_position_xyz_linear_error)**2, axis=0)**0.5 / self.speed_xyz_linear

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

            self.velocity_xyz[:,0] = G * (c + self.velocity_xyz[:,0]) - c
