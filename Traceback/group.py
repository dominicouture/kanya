# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Defines the Group class and the embeded Star classes. These classes contain the
    information and methods necessary to compute tracebacks of stars in a moving group and
    assess its age by minimizing: the 3D scatter, median absolute deviation, position-velocity
    covariances and minimum spanning tree mean branch length and median absolute deviation of
    branch length.
"""

import numpy as np
import galpy.util.coords as coords
from galpy.orbit import Orbit
from sklearn.covariance import MinCovDet
from Traceback.data import Data
from Traceback.output import Output_Group
from Traceback.coordinate import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

class Group(list, Output_Group):
    """ Contains the values and related methods of a moving group and a list of Star objets that
        are part of it. Stars are can be imported from a raw data file or modeled base on
        simulation parameters.
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

        # Stars from data
        if self.series.from_data:
            self.stars_from_data()

        # Stars from model
        elif self.series.from_model:
            self.stars_from_model()

        # Indicators
        self.set_indicators()

        # Median absolute deviation
        self.get_median_absolute_deviation()

        # Covariances
        self.get_covariances()
        self.get_covariances_robust()
        # self.get_covariances_sklearn()

        # Minimum spanning tree
        self.get_minimum_spanning_tree()

    def stars_from_data(self):
        """ Creates a list of Star objects from a Python dictionary or CSV files containing the
            parameters including the name of the stars, their xyz positions and uvw velocities,
            and their respective measurement errors. Radial velocity offset is also added. If
            only one group is created in the series, actual values are used. If multiple groups
            are created, positions and velocities are scrambled based on measurement errors.
        """

        # Radial velocity offset due to gravitational redshift
        for star in self.series.data:

            # Observables conversion into equatorial spherical coordinates
            if self.series.data.data.system.name == 'observables':

                # Compute equatorial rδα position and velocity
                position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error = position_obs_rδα(
                    *(star.position.values if self.number == 0 else
                        np.random.normal(star.position.values, star.position.errors)),
                    *(star.velocity.values if self.number == 0 else
                        np.random.normal(star.velocity.values, star.velocity.errors)),
                    *star.position.errors, *star.velocity.errors)

                # Apply radial velocity offset and error
                velocity_rδα += np.array([star.rv_offset.value, 0., 0.])
                velocity_rδα_error[0] = (velocity_rδα_error[0]**2 + star.rv_offset.error**2)**0.5

                # Compute equatorial xyz position and velocity
                position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                    *position_rδα, *position_rδα_error)
                velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                    *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error)

            # From cartesian coordinates
            if self.series.data.data.system.name == 'cartesian':

                # Compute equatorial rδα position and velocity
                position_rδα, position_rδα_error = galactic_xyz_equatorial_rδα(
                    *star.position.values, *star.position.errors)
                velocity_rδα, velocity_rδα_error = galactic_uvw_equatorial_rvμδμα(
                    *star.position.values, *star.velocity.values,
                    *star.position.errors, *star.velocity.errors)

                # Apply radial velocity offset
                velocity_rδα += np.array([rv_offset, 0., 0.])

                # Compute equatorial xyz position and velocity
                position_xyz, position_xyz_error = (star.position.values, star.position.errors)
                velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                    *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error)

            # Create star
            self.append(self.Star(
                self, name=star.name, backward=True,
                velocity_xyz=velocity_xyz, velocity_xyz_error=velocity_xyz_error,
                position_xyz=position_xyz, position_xyz_error=position_xyz_error))

        # Compute stars coordinates
        self.get_stars_coordinates()

        # Filter outliers
        self.filter_outliers()

    def stars_from_model(self):
        """ Creates an artificial model of star for a given number of stars and age based on
            the intial average xyz position and uvw velocity, and their respective errors
            and scatters. The sample is then moved forward in time for the given age and radial
            velocity offset is also added.
        """

        # Model backward orbit integration from current to star formation epoch
        self.average_model_star = self.Star(
            self, name='average_model_star',
            backward=True, model=True, age=self.series.age.value,
            velocity_xyz=self.series.velocity.values, velocity_xyz_error=np.zeros(3),
            position_xyz=self.series.position.values, position_xyz_error=np.zeros(3))

        # Average velocity at the epoch of star formation
        velocity = self.average_model_star.velocity_xyz[-1]
        velocity_error = self.series.velocity_error.values
        velocity_scatter = self.series.velocity_scatter.values

        # Average position at the epoch of star formation
        position = self.average_model_star.position_xyz[-1]
        position_error = self.series.position_error.values
        position_scatter = self.series.position_scatter.values

        # Stars creation from a model
        self.model_stars = []
        for star in range(self.series.number_of_stars):

            # Model star forward galactic orbit integration
            model_star = self.Star(
                self, name='model_star_{}'.format(star + 1),
                backward=False, model=True, age=self.series.age.value,
                velocity_xyz=np.random.normal(velocity, velocity_scatter), velocity_xyz_error=np.zeros(3),
                position_xyz=np.random.normal(position, position_scatter), position_xyz_error=np.zeros(3))
            self.model_stars.append(model_star)

            # Velocity and possition conversion to equatorial spherical coordinates
            velocity_rδα = galactic_uvw_equatorial_rvμδμα(
                *model_star.position_xyz[-1], *model_star.velocity_xyz[-1])[0]
            velocity_rδα += np.array([self.series.rv_offset.value, 0.0, 0.0])
            position_rδα = galactic_xyz_equatorial_rδα(*model_star.position_xyz[-1])[0]

            # Velocity and position conversion to observables
            position_obs, velocity_obs = position_rδα_obs(*position_rδα, *velocity_rδα)[:2]

            # Observables conversion back into equatorial spherical coordinates
            # Velocity and position scrambling based on actual measurement errors
            if self.series.data_errors:
                star_errors = star - (star // len(self.series.data)) * len(self.series.data)
                position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error = position_obs_rδα(
                    *np.random.normal(position_obs, self.series.data[star_errors].position.errors),
                    *np.random.normal(velocity_obs, self.series.data[star_errors].velocity.errors),
                    *self.series.data[star_errors].position.errors,
                    *self.series.data[star_errors].velocity.errors)

            # Velocity and position scrambling based on average measurement errors
            else:
                position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error = position_obs_rδα(
                    *np.random.normal(position_obs, position_error),
                    *np.random.normal(velocity_obs, velocity_error),
                    *position_error, *velocity_error)

            # Velocity and position conversion back to galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error)
            velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error)

            # Create star
            self.append(self.Star(
                self, name='star_{}'.format(star + 1), backward=True,
                velocity_xyz=velocity_xyz, velocity_xyz_error=velocity_xyz_error,
                position_xyz=position_xyz, position_xyz_error=position_xyz_error))

        # Compute stars coordinates
        self.get_stars_coordinates()

    def get_stars_coordinates(self):
        """ Computes the number of stars that aren't outliers, the stars average xyz and ξηζ
            velocity and position, excluding outliers, and the stars relative positions and
            distances from the average, including outliers. If all stars are identified as
            outliers, an error is raised.
        """

        # Create sample, subsample and outliers lists
        self.subsample = list(filter(lambda star: star.subsample, self))
        self.sample = list(filter(lambda star: not star.outlier, self))
        self.outliers =list(filter(lambda star: star.outlier, self))

        # Number of remaining stars
        self.subsample_size = len(self.subsample)
        self.sample_size = len(self.sample)
        self.number_of_outliers = len(self.outliers)

        # Average xyz positions and velocities, excluding outliers
        self.position_xyz = np.mean([star.position_xyz for star in self.sample], axis=0)
        self.distance_xyz = np.sum(self.position_xyz**2, axis=1)**0.5
        self.velocity_xyz = np.mean([star.velocity_xyz for star in self.sample], axis=0)
        self.speed_xyz = np.sum(self.velocity_xyz**2, axis=1)**0.5

        # Average ξηζ positions and velocities, excluding outliers
        self.position_ξηζ = np.mean([star.position_ξηζ for star in self.sample], axis=0)
        self.distance_ξηζ = np.sum(self.position_ξηζ**2, axis=1)**0.5
        self.velocity_ξηζ = np.mean([star.velocity_ξηζ for star in self.sample], axis=0)
        self.speed_ξηζ = np.sum(self.velocity_ξηζ**2, axis=1)**0.5

        # xyz position and velocity scatter, excluding outliers
        self.scatter_position_xyz = np.std([star.position_xyz for star in self.sample], axis=0)
        self.scatter_position_xyz_total = np.sum(self.scatter_position_xyz**2, axis=1)**0.5
        self.scatter_velocity_xyz = np.std([star.velocity_xyz for star in self.sample], axis=0)
        self.scatter_velocity_xyz_total = np.sum(self.scatter_velocity_xyz**2, axis=1)**0.5

        # ξηζ position and velocity scatters, excluding outliers
        self.scatter_position_ξηζ = np.std([star.position_ξηζ for star in self.sample], axis=0)
        self.scatter_position_ξηζ_total = np.sum(self.scatter_position_ξηζ**2, axis=1)**0.5
        self.scatter_velocity_ξηζ = np.std([star.velocity_ξηζ for star in self.sample], axis=0)
        self.scatter_velocity_ξηζ_total = np.sum(self.scatter_velocity_ξηζ**2, axis=1)**0.5

        # Stars relative position and velocity, distance and speed, including outliers
        for star in self:
            star.get_relative_coordinates()

    def filter_outliers(self):
        """ Filters outliers from the sample based on ξηζ position and velocity scatter over
            time. A core sample is created based on a robust covariance matrix estimator using
            the scikit-learn (sklearn) Python package, leaving other stars as part of the extended
            sample.
        """

        # Filter outliers for the first group
        if self.number == 0:

            # Iteratively validate stars coordinates
            outliers = False if self.series.cutoff is None else True
            while outliers and len(self.sample) > int(0.8 * self.series.number_of_stars):

                # Remove stars beyond the σ cutoff of the average position or velocity
                outliers = False
                for star in self.sample:
                    position_outlier = (np.abs(star.relative_position_ξηζ)
                        > (self.scatter_position_ξηζ * self.series.cutoff)).any()
                    velocity_outlier = (np.abs(star.relative_velocity_ξηζ)
                        > (self.scatter_velocity_ξηζ * self.series.cutoff)).any()
                    if position_outlier or velocity_outlier:
                        star.set_outlier(position_outlier, velocity_outlier)
                        outliers = True

                # Compute valid stars coordinates
                self.get_stars_coordinates()

            # Display a message if outliers have been found
            if self.number_of_outliers > 0:
                print(f'{self.number_of_outliers:d} outlier'
                    f"{'s' if self.number_of_outliers > 1 else ''} found in {self.series.name}:")
                for star in self.outliers:
                    print(f"{star.name} ({'P' if star.position_outlier else ''}"
                        f"{'V' if star.velocity_outlier else ''})")

            # Display message if no outliers are found
            elif self.series.cutoff is not None:
                print(f'No outliers found in {self.series.name}.')

            # Robust covariance matrix and support fraction
            if False:
                a = np.array([star.position_ξηζ for star in self.sample])
                a = np.swapaxes(a, 0, 1)
                robust_covariance_matrix = []
                support_fraction = []

                # Iteration over time
                for i in range(a.shape[0]):
                    b = MinCovDet(assume_centered=False).fit(a[i])
                    # print(i, b[i].shape, b.support_.size, b.support_.nonzero()[0].size, b.dist_.size)
                    robust_covariance_matrix.append(b.covariance_)
                    support_fraction.append(b.support_)

                # Array conversion
                robust_covariance_matrix = np.array(robust_covariance_matrix)
                support_fraction = np.array(support_fraction)
                support_fraction_all = np.sum(support_fraction, axis=0) / self.series.number_of_steps > 0.7

                # Core creation
                for i in range(self.sample_size):
                    if support_fraction_all[i]:
                        self.sample[i].subsample = True
                    else:
                        self.sample[i].subsample = False
                        self.sample[i].outlier = True

            # Temporary reasignment of subsample to stars
            self.get_stars_coordinates()

        # Use the same outliers, sample and subsample as the first group for other groups
        else:
            outliers = [star.name for star in self.series[0].outliers]
            if len(outliers) > 0:
                for star in filter(lambda star: star.name in outliers, self):
                    star.outlier == True

                # Compute stars coordinates
                self.get_stars_coordinates()

        # Covariance matrix
        # a = np.array([star.position_xyz for star in self.sample]) - self.position_xyz
        # a = np.tile(a.T, (a.shape[-1], 1, 1, 1))
        # b = np.swapaxes(a, 0, 1)
        # covariance_matrix = np.mean(a * b, axis=3).T
        # eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        # R = np.linalg.inv(eigen_vectors)

        # for star in self:
            # np.squeeze(np.matmul(np.swapaxes(R, 1, 2), np.expand_dims(star.position_xyz, axis=2)))
            # star.position_xyz = np.squeeze(np.matmul(np.expand_dims(star.position_xyz, axis=1), R))

        # self.Indicator(self, 'eigen_values', np.repeat(np.expand_dims(
        #     np.sum(eigen_values**2, axis=1)**0.5, axis=0),
        #     self.series.jackknife_number, axis=0), valid=True)

    def set_indicators(self):
        """ Sets the parameters to compute age indicators including weights based on the Mahalanobis
            distance using the empirical covariance matrix and Jackknife Monte-Carlo itrations.
        """

        # Compute Mahalanobis distance
        self.positions_xyz = np.expand_dims(np.array([
            star.position_xyz for star in self.sample]), axis=1)
        self.sample_size_jackknife = 1
        self.Mahalanobis_xyz = self.get_Mahalanobis_distance('positions_xyz')

        # Compute weights
        self.weights = np.exp(-1 * self.Mahalanobis_xyz)
        for i in range(len(self.sample)):
            self.sample[i].weight = self.weights[i]

        # Indicators list
        self.indicators = []
        for indicator in self.series.indicators:
            self.Indicator(self, indicator)

        # Select stars for Jackknife Monte-Carlo for the first group
        if self.number == 0:
            self.sample_size_jackknife = int(self.sample_size * self.series.jackknife_fraction)
            self.stars_monte_carlo = np.array([np.random.choice(self.sample_size,
                self.sample_size_jackknife, replace=False)
                    for i in range(self.series.jackknife_number)])

        # Use the same stars for Jackknife Monte-Carlo as the first group for other groups
        else:
            self.sample_size_jackknife = self.series[0].sample_size_jackknife
            self.stars_monte_carlo = self.series[0].stars_monte_carlo

        # Create selected stars xyz and ξηζ positions and velocities arrays
        self.positions_xyz = np.array([
            star.position_xyz for star in self.sample])[self.stars_monte_carlo,:,:].swapaxes(0, 1)
        self.positions_ξηζ = np.array([
            star.position_ξηζ for star in self.sample])[self.stars_monte_carlo,:,:].swapaxes(0, 1)
        self.velocities_xyz = np.array([
            star.velocity_xyz for star in self.sample])[self.stars_monte_carlo,:,:].swapaxes(0, 1)
        self.velocities_ξηζ = np.array([
            star.velocity_ξηζ for star in self.sample])[self.stars_monte_carlo,:,:].swapaxes(0, 1)

        # def cov_matrix(positions, velocities):
        #     a = positions - np.mean(positions, axis=0)
        #     a = np.tile(a.T, (a.shape[-1], 1, 1, 1, 1))
        #     b = np.swapaxes(a, 0, 1)
        #     covariance_matrix = np.mean(a * b, axis=4).T
        #     eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        #     R = np.linalg.inv(eigen_vectors)
        #
        #     return (np.squeeze(np.matmul(np.expand_dims(positions, axis=3), np.expand_dims(R, axis=0))),
        #         np.squeeze(np.matmul(np.expand_dims(velocities, axis=3), np.expand_dims(R, axis=0))))

        # self.positions_xyz, self.velocities_xyz = cov_matrix(self.positions_xyz, self.velocities_xyz)
        # self.positions_ξηζ, self.velocities_ξηζ = cov_matrix(self.positions_ξηζ, self.velocities_ξηζ)

    class Indicator():
        """ Age indicator including its values, age at minimum and minimum. Computes errors on
            values, age and minium using a Jack-knife Monte Carlo method.
        """

        def __init__(self, group, indicator):
            """ Initializes an age indicator. """

            # Initialization
            self.group = group
            self.indicator = indicator

            # Add the indicator to the group
            vars(self.group)[self.indicator.label] = self
            self.group.indicators.append(self)

        def __call__(self, values):
            """ Computes errors on values, age and minium using a Jackknife Monte Carlo method. """

            # Average values and errors
            self.values = np.atleast_3d(values)
            self.value = np.mean(self.values, axis=0)
            self.value_int_error = np.std(self.values, axis=0)

            # Average age at the minimal value and error
            self.ages = self.group.series.time[np.argmin(self.values, axis=1)]
            self.age = self.group.series.time[np.argmin(self.value, axis=0)]
            self.age_int_error = np.std(self.ages, axis=0)

            # Average minimal value and error
            self.minima = np.min(self.values, axis=1)
            self.min = np.min(self.value, axis=0)
            self.min_int_error = np.std(self.minima, axis=0)

            # Set status
            self.indicator.status = True

    def get_median_absolute_deviation(self):
        """ Computes the xyz and ξηζ, partial and total median absolute deviations (MAD) of a group
            and their respective errors for all timesteps. The age of the group is then estimated by
            finding the time at which the median absolute deviation is minimal.
        """

        # xyz median absolute deviation age
        self.mad_xyz(
            np.median(np.abs(self.positions_xyz - np.median(self.positions_xyz, axis=0)), axis=0))
        self.mad_xyz_total(np.sum(self.mad_xyz.values**2, axis=2)**0.5)

        # ξηζ median absolute deviation age
        self.mad_ξηζ(
            np.median(np.abs(self.positions_ξηζ - np.median(self.positions_ξηζ, axis=0)), axis=0))
        self.mad_ξηζ_total(np.sum(self.mad_ξηζ.values**2, axis=2)**0.5)

    def get_covariances(self):
        """ Computes the xyz and ξηζ position covariance matrices, and xyz and ξηζ position and
            velocity cross covariance matrices. The age of the group is then estimated by finding
            the time at which the variances, determinant or trace of the matrices are minimal.
        """

        # xyz position covariance matrix, determinant and trace ages
        self.covariances_xyz_matrix = self.get_covariance_matrix('positions_xyz')
        self.get_covariance_indicators(self.covariances_xyz_matrix, 'covariances_xyz')

        # xyz position and velocity cross covariance matrix, determinant and trace ages
        self.cross_covariances_xyz_matrix = self.get_covariance_matrix('positions_xyz', 'velocities_xyz')
        self.get_covariance_indicators(self.cross_covariances_xyz_matrix, 'cross_covariances_xyz')

        # ξηζ position covariance matrix, determinant and trace ages
        self.covariances_ξηζ_matrix = self.get_covariance_matrix('positions_ξηζ')
        self.get_covariance_indicators(self.covariances_ξηζ_matrix, 'covariances_ξηζ')

        # ξηζ position and velocity cross covariance matrix, determinant and trace ages
        self.cross_covariances_ξηζ_matrix = self.get_covariance_matrix('positions_ξηζ', 'velocities_ξηζ')
        self.get_covariance_indicators(self.cross_covariances_ξηζ_matrix, 'cross_covariances_ξηζ')

    def get_covariances_robust(self):
        """ Computes the xyz and ξηζ position robust covariance matrices, and xyz and ξηζ position
            and velocity robust cross covariance matrices by giving each star a different weight
            based on the Mahalanobis distance computed from the covariance and cross covariance
            matrices. The age of the group is then estimated by finding the time at which the
            variances, determinant or trace of the matrices are minimal.
        """

        # xyz position robust covariance matrix and determinant ages
        self.Mahalanobis_xyz = self.get_Mahalanobis_distance(
            'positions_xyz', covariance_matrix=self.covariances_xyz_matrix)
        self.covariances_xyz_robust_matrix = self.get_covariance_matrix(
            'positions_xyz', robust=True, Mahalanobis_distance=self.Mahalanobis_xyz)
        self.get_covariance_indicators(self.covariances_xyz_robust_matrix, 'covariances_xyz_robust')

        # xyz position and velocity robust cross covariance matrix and determinant ages
        self.Mahalanobis_cross_xyz = self.get_Mahalanobis_distance(
            'positions_xyz', 'velocities_xyz', covariance_matrix=self.cross_covariances_xyz_matrix)
        self.cross_covariances_xyz_robust_matrix = self.get_covariance_matrix(
            'positions_xyz', 'velocities_xyz', robust=True,
            Mahalanobis_distance=self.Mahalanobis_xyz)
        self.get_covariance_indicators(self.cross_covariances_xyz_robust_matrix, 'cross_covariances_xyz_robust')

        # ξηζ position robust covariance matrix, determinant and trace ages
        self.Mahalanobis_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', covariance_matrix=self.covariances_ξηζ_matrix)
        self.covariances_ξηζ_robust_matrix = self.get_covariance_matrix(
            'positions_ξηζ', robust=True, Mahalanobis_distance=self.Mahalanobis_ξηζ)
        self.get_covariance_indicators(self.covariances_ξηζ_robust_matrix, 'covariances_ξηζ_robust')

        # ξηζ position and velocity robust cross covariance matrix and determinant ages
        self.Mahalanobis_cross_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', 'velocities_ξηζ', covariance_matrix=self.cross_covariances_ξηζ_matrix)
        self.cross_covariances_ξηζ_robust_matrix = self.get_covariance_matrix(
            'positions_ξηζ', 'velocities_ξηζ', robust=True,
            Mahalanobis_distance=self.Mahalanobis_ξηζ)
        self.get_covariance_indicators(self.cross_covariances_ξηζ_robust_matrix, 'cross_covariances_ξηζ_robust')

    def get_covariances_sklearn(self):
        """ Computes the ξηζ position covariance matrix using the sklearn package. The age of the
            group is then estimated by finding the time at which the variances, determinant or
            trace of the matrix are minimal.
        """

        def get_covariance_sklearn_matrix(a):
            """ Computes a robust covariance determinant with sklearn (slow). """

            # Initialization
            from sklearn.covariance import MinCovDet
            a = np.array([vars(star)[a] for star in self.sample])
            a = np.swapaxes(a, 0, 1)

            # sklearn covariance matrix
            sklearn_covariance_matrix = []
            for i in range(a.shape[0]):
                b = MinCovDet(assume_centered=False).fit(a[i])
                sklearn_covariance_matrix.append(b.covariance_)

            return np.array(sklearn_covariance_matrix)

        # ξηζ position sklearn covariance matrix, determinant and trace ages
        self.covariances_ξηζ_sklearn_matrix = get_covariance_sklearn_matrix('position_ξηζ')
        self.covariances_ξηζ_sklearn(np.repeat(np.expand_dims(
            np.abs(self.covariances_ξηζ_sklearn_matrix[:, (0, 1, 2), (0, 1, 2)])**0.5, axis=0),
                self.series.jackknife_number, axis=0))
        self.covariances_ξηζ_matrix_det_sklearn(np.repeat(np.expand_dims(
            np.abs(np.linalg.det(self.covariances_ξηζ_sklearn_matrix))**(1/6), axis=0),
                self.series.jackknife_number, axis=0))
        self.covariances_ξηζ_matrix_trace_sklearn(np.repeat(np.expand_dims(
            np.abs(np.trace(self.covariances_ξηζ_sklearn_matrix, axis1=1, axis2=2) / 3)**0.5, axis=0),
                self.series.jackknife_number, axis=0))

    def get_covariance_matrix(self, a, b=None, robust=False, Mahalanobis_distance=None):
        """ Computes the robust covariance matrix with variable weights of a Star parameter, 'a',
            along all physical for all timestep. If another Star parameter, 'b', is given, the
            cross covariance matrix is computed instead. Weights are computed from the Mahalanobis
            distance
        """

        # Mahalanobis distance
        if robust:
            if Mahalanobis_distance is None:
                Mahalanobis_distance = self.get_Mahalanobis_distance(a, b)
            Mahalanobis_distance = np.repeat(np.expand_dims(Mahalanobis_distance, axis=3), 3, axis=3)

            # Compute weights
            weights = np.exp(-2 * Mahalanobis_distance)
            weights = np.tile(weights.T, (weights.shape[-1], 1, 1, 1, 1))

        # Covariance matrix
        a = vars(self)[a] - np.mean(vars(self)[a], axis=0)
        # a_weights = np.exp(-2. * (a / np.std(a, axis=0))**2)
        a = np.tile(a.T, (a.shape[-1], 1, 1, 1, 1))
        if b is None:
            b = np.swapaxes(a, 0, 1)

        # Cross covariance matrix
        else:
            b = vars(self)[b] - np.mean(vars(self)[b], axis=0)
            # b_weights = np.exp(-2. * (b / np.std(b, axis=0))**2)
            b = np.swapaxes(np.tile(b.T, (b.shape[-1], 1, 1, 1, 1)), 0, 1)

        return np.average(a * b, weights=weights, axis=4).T if robust else np.mean(a * b, axis=4).T

    def get_Mahalanobis_distance(self, a, b=None, covariance_matrix=None):
        """ Computes the Mahalanobis distances using the covariance matrix of a of all stars in
            the group for all jack-knife iterations.
        """

        # Compute the covariance and inverse covariance matrices
        if covariance_matrix is None:
            covariance_matrix = self.get_covariance_matrix(a, b)
        covariances_matrix_invert = np.repeat(
            np.expand_dims(np.squeeze(np.linalg.inv(covariance_matrix)), axis=0),
            self.sample_size_jackknife, axis=0)

        # Compute Mahalanobis distances
        if b is None:
            b = a
        c = np.expand_dims(vars(self)[a] - np.mean(vars(self)[a], axis=0), axis=4)
        # d = np.expand_dims(vars(self)[b] - np.mean(vars(self)[b], axis=0), axis=4)
        # d = c
        # c = (np.expand_dims(vars(self)[a] - np.mean(vars(self)[a], axis=0), axis=4) +
        #     np.expand_dims(vars(self)[b] - np.mean(vars(self)[b], axis=0), axis=4)) / 2

        return np.sqrt(np.abs(np.squeeze(np.matmul(
            np.swapaxes(c, 3, 4), np.matmul(covariances_matrix_invert, c)))))

    def get_covariance_indicators(self, covariance_matrix, indicator):
        """ Computes the covariances, and the determinant and trace of the covariance matrix for
            a given covariance matrix and indicator.
        """

        # Covariances
        vars(self)[indicator](np.abs(covariance_matrix[:, :, (0, 1, 2), (0, 1, 2)])**0.5)

        # Covariance matrix determinant
        indicator, robust = (indicator[:-7], '_robust') if indicator[-7:] == '_robust' else (indicator, '')
        vars(self)[f'{indicator}_matrix_det{robust}'](np.abs(np.linalg.det(covariance_matrix))**(1/6))

        # Covariance matrix trace
        vars(self)[f'{indicator}_matrix_trace{robust}'](np.abs(np.trace(covariance_matrix, axis1=2, axis2=3) / 3)**0.5)

    def get_minimum_spanning_tree(self):
        """ Builds the minimum spanning tree (MST) of a group for all timesteps using a Kruskal
            algorithm, and computes the average branch length and the branch length median absolute
            absolute deviation of branch length. The age of the moving is then estimated by finding
            the time at which these value are minimal.
        """

        # Branches initialization
        self.branches = []
        self.number_of_branches = self.sample_size - 1
        for start in range(self.number_of_branches):
            for end in range(start + 1, self.number_of_branches + 1):
                self.branches.append(self.Branch(self.sample[start], self.sample[end]))

        # Create minimum spanning tree
        self.mst_xyz, self.mst_xyz_lengths, self.mst_xyz_weights = self.get_branches('length_xyz')
        self.mst_ξηζ, self.mst_ξηζ_lengths, self.mst_ξηζ_weights = self.get_branches('length_ξηζ')

        # Jackknife Monte Carlo branch lengths
        sample_size_jackknife = int(self.number_of_branches * self.series.jackknife_fraction)
        branches_monte_carlo = np.array([np.random.choice(
            self.number_of_branches, sample_size_jackknife, replace=False)
                for i in range(self.series.jackknife_number)])
        mst_xyz_lengths = self.mst_xyz_lengths.T[branches_monte_carlo].swapaxes(0, 1)
        mst_xyz_weights = self.mst_xyz_weights.T[branches_monte_carlo].swapaxes(0, 1)
        mst_ξηζ_lengths = self.mst_ξηζ_lengths.T[branches_monte_carlo].swapaxes(0, 1)
        mst_ξηζ_weights = self.mst_ξηζ_weights.T[branches_monte_carlo].swapaxes(0, 1)

        # Minimum spanning tree average branch length ages
        self.mst_xyz_mean(np.mean(mst_xyz_lengths, axis=0))
        self.mst_ξηζ_mean(np.mean(mst_ξηζ_lengths, axis=0))

        # Minimum spanning tree robust average branch length ages
        self.mst_xyz_mean_robust(np.average(mst_xyz_lengths, weights=mst_xyz_weights, axis=0))
        self.mst_ξηζ_mean_robust(np.average(mst_ξηζ_lengths, weights=mst_ξηζ_weights, axis=0))

        # Minimum spanning tree branch length median absolute deviation ages
        self.mst_xyz_mad(np.median(np.abs(mst_xyz_lengths - np.median(mst_xyz_lengths, axis=0)), axis=0))
        self.mst_ξηζ_mad(np.median(np.abs(mst_ξηζ_lengths - np.median(mst_ξηζ_lengths, axis=0)), axis=0))

    def get_branches(self, length):
        """ Computes the branches, lengths and weights of a minimum spanning tree. """

        mst = np.empty((self.series.number_of_steps, self.number_of_branches), dtype=object)
        mst_lengths = np.zeros(mst.shape)
        mst_weights = np.zeros(mst.shape)

        # Minimum spanning tree computation for every timestep
        for step in range(self.series.number_of_steps):

            # Sort by length
            self.branches.sort(key=lambda branch: vars(branch)[length][step])

            # Nodes, tree and branch number initialization
            for star in self.sample:
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
                    mst[step, j] = branch
                    j += 1

            # Minimum spanning tree branches length
            mst_lengths[step] = np.vectorize(
                lambda branch: vars(branch)[length][step])(mst[step])
            mst_weights[step] = np.vectorize(
                lambda branch: branch.weight[step])(mst[step])

        return mst, mst_lengths, mst_weights

    class Branch:
        """ Line connecting two stars used for the calculation of the minimum spanning tree. """

        def __init__(self, start, end):
            """ Initializes a Branch object and computes the distance between two Star objects,
                'start' and 'end', for all timestep.
            """

            self.start = start
            self.end = end
            self.length_xyz = np.sum((self.start.position_xyz - self.end.position_xyz)**2, axis=1)**0.5
            self.length_ξηζ = np.sum((self.start.position_ξηζ - self.end.position_ξηζ)**2, axis=1)**0.5
            self.weight = np.mean(np.vstack((self.start.weight, self.end.weight)), axis=0)

        def __repr__(self):
            """ Returns a string of name of the branch. """

            return "'{}' to '{}' branch".format(self.start.name, self.end.name)

    class Node:
        """ Node of a star. """

        def __init__(self):
            """ Sets the parent node of a star as None. """

            self.parent = None

        def __repr__(self):
            """ Returns a string of name of the parent. """

            return 'None' if self.parent is None else self.parent

    class Star:
        """ Parameters and methods of a star in a moving group. """

        def __init__(self, group, **values):
            """ Initializes a Star object with at least a name, velocity, velocity error, initial
                position and position error. More values can be added (when initializing from a
                file for instance). If a traceback is needed, the star's position and velocity
                overtime is computed with galpy.
            """

            # Initialization
            self.group = group
            self.outlier = False
            self.subsample = False
            self.model = False
            self.age = None
            vars(self).update(values)

            # Forward or backward, and model or data time
            self.time = (np.copy(self.group.series.time) if not self.model
                else np.linspace(0., self.age, 100))
            self.time *= (-1 if self.backward else 1)

            # Trajectory integration
            if self.group.series.potential is None:
                self.get_linear()
            else:
                self.get_orbit()

        def set_outlier(self, position_outlier, velocity_outlier):
            """ Sets the star as an outlier. """

            self.outlier = True
            self.position_outlier = position_outlier
            self.velocity_outlier = velocity_outlier

        def get_linear(self):
            """ Computes a star's backward or forward linear trajectory. """

            # Linear xyz positions and velocities
            self.position_xyz = self.position_xyz + (self.velocity_xyz
                + Coordinate.galactic_velocity) * np.expand_dims(self.time, axis=0).T
            self.distance_xyz = np.sum(self.position_xyz**2, axis=1)**0.5
            self.velocity_xyz = np.repeat(
                np.expand_dims(self.velocity_xyz, axis=0), self.time.size, axis=0)
            self.speed_xyz = np.sum(self.velocity_xyz**2, axis=1)**0.5

            # Linear rθz positions and velocities
            position_rθz = np.array(coords.XYZ_to_galcencyl(
                *self.position_xyz.T,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False))
            velocity_rθz = np.array(coords.vxvyvz_to_galcencyl(
                *self.velocity_xyz.T,
                *np.array(coords.XYZ_to_galcenrect(
                    *self.position_xyz.T,
                    Xsun=Coordinate.sun_position[0],
                    Zsun=Coordinate.sun_position[2],
                    _extra_rot=False)).T,
                vsun=Coordinate.sun_velocity,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False))

            # Change time for forward computation
            if not self.backward and self.model:
                time = self.time - self.age
            else:
                time = self.time

            # Linear ξηζ positions and velocities
            self.position_ξηζ = position_rθz_ξηζ(*position_rθz.T, time)
            self.distance_ξηζ = np.sum(self.position_ξηζ**2, axis=1)**0.5
            self.velocity_ξηζ = velocity_rθz_ξηζ(*velocity_rθz.T, time)
            self.speed_ξηζ = np.sum(self.velocity_ξηζ**2, axis=1)**0.5

        def get_orbit(self):
            """ Computes a star's backward or forward galactic orbit using Galpy. """

            # New method to check previous method
            # from astropy import units as u
            # self.position_rδα, self.position_rδα_error = xyz_to_rδα(*self.position_xyz)
            # r, δ, α = self.position_rδα[0] * u.Unit('pc'), self.position_rδα[1] * u.Unit('rad'), self.position_rδα[2] * u.Unit('rad')
            # u, v, w = self.velocity_xyz * 0.9777922216731283

            # Other position and velocity computations
            # self.position_rθz_other, self.position_rθz_error_other = galactic_xyz_galactocentric_ρθz(
            #     *self.position_xyz)
            # self.velocity_rθz_other, self.velocity_rθz_error_other = galactic_uvw_galactocentric_ρθz(
            #     *self.position_xyz, *self.velocity_xyz)

            # Initial rθz position and velocity in galactocentric cylindrical coordinates
            position_rθz = np.array(coords.XYZ_to_galcencyl(
                *self.position_xyz,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False))
            velocity_rθz = np.array(coords.vxvyvz_to_galcencyl(
                *self.velocity_xyz,
                *np.array(coords.XYZ_to_galcenrect(
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

            # Conversion to natural units
            time = self.time * Coordinate.lsr_velocity[1] / Coordinate.sun_position[0]
            position_rθz /= np.array([Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])
            velocity_rθz /= Coordinate.lsr_velocity[1]

            # Orbit initialization and integration
            orbit = Orbit([
                position_rθz[0], velocity_rθz[0], velocity_rθz[1],
                position_rθz[2], velocity_rθz[2], position_rθz[1]],
                ro=Coordinate.ro, vo=Coordinate.vo, zo=Coordinate.zo,
                solarmotion=Coordinate.sun_velocity_peculiar_kms)
            orbit.turn_physical_off()
            orbit.integrate(time, self.group.series.potential, method='odeint')

            # rθz positions and velocities, and conversion to physical units
            position_rθz = np.array([
                orbit.R(time), orbit.phi(time), orbit.z(time)]).T * np.array(
                    [Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])
            velocity_rθz = np.array([
                orbit.vR(time), orbit.vT(time), orbit.vz(time)]).T * Coordinate.lsr_velocity[1]

            # xyz positions and velocities
            self.position_xyz = coords.galcencyl_to_XYZ(
                *position_rθz.T,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)
            self.distance_xyz = np.sum(self.position_xyz**2, axis=1)**0.5
            self.velocity_xyz = coords.galcencyl_to_vxvyvz(
                *velocity_rθz.T, position_rθz.T[1],
                vsun=Coordinate.sun_velocity,
                Xsun=Coordinate.sun_position[0],
                Zsun=Coordinate.sun_position[2],
                _extra_rot=False)
            self.speed_xyz = np.sum(self.velocity_xyz**2, axis=1)**0.5

            # Change time for forward orbital computation
            if not self.backward and self.model:
                time = self.time - self.age
            else:
                time = self.time

            # ξηζ positions and velocities
            self.position_ξηζ = position_rθz_ξηζ(*position_rθz.T, time)
            self.distance_ξηζ = np.sum(self.position_ξηζ**2, axis=1)**0.5
            self.velocity_ξηζ = velocity_rθz_ξηζ(*velocity_rθz.T, time)
            self.speed_ξηζ = np.sum(self.velocity_ξηζ**2, axis=1)**0.5

        def get_relative_coordinates(self):
            """ Computes relative position and velocity, the distance and the speed from the
                average position and velocity, and their respective errors for all timesteps.
            """

            # Relative xyz positions and velocities from the average group position
            self.relative_position_xyz = self.position_xyz - self.group.position_xyz
            self.relative_distance_xyz = np.sum(self.relative_position_xyz**2, axis=1)**0.5
            self.relative_velocity_xyz = self.velocity_xyz - self.group.velocity_xyz
            self.relative_speed_xyz = np.sum(self.relative_velocity_xyz**2, axis=0)**0.5

            # Relative ξηζ positions and velocities from the average group position
            self.relative_position_ξηζ = self.position_ξηζ - self.group.position_ξηζ
            self.relative_distance_ξηζ = np.sum(self.relative_position_ξηζ**2, axis=1)**0.5
            self.relative_velocity_ξηζ = self.velocity_ξηζ - self.group.velocity_ξηζ
            self.relative_speed_ξηζ = np.sum(self.relative_velocity_ξηζ**2, axis=1)**0.5

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
