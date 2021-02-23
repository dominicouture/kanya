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
from sklearn.covariance import MinCovDet
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

        # Stars from data
        if series.from_data:
            self.stars_from_data()

        # Stars from model
        elif series.from_model:
            self.stars_from_model()

        # Average velocity and position and filter outliers
        self.filter_outliers()

        # Indicators
        self.set_indicators()

        # Scatter
        self.get_scatter()

        # Median absolute deviation
        self.get_median_absolute_deviation()

        # Covariances
        self.get_covariances()
        self.get_covariances_robust()
        self.get_covariances_sklearn()

        # Minimum spanning tree
        self.get_minimum_spanning_tree()

    def stars_from_data(self):
        """ Creates a list of Star objects from a Python dictionary or CSV files containing the
            parameters including the name of the stars, their xyz positions and UVW velocities,
            and their respective measurement errors. Radial velocity offset is also added. If
            only one group is created in the series, actual values are used. If multiple groups
            are created, positions and velocities are scrambled based on measurement errors.
        """

        # Radial velocity offset due to gravitational redshift
        for star in self.series.data:
            print(star.name, star.rv_offset)

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

    def stars_from_model(self):
        """ Creates an artificial model of star for a given number of stars and age based on
            the intial average xyz position and UVW velocity, and their respective errors
            and scatters. The sample is then moved forward in time for the given age and radial
            velocity offset is also added.
        """

        # Model backward orbit integration from current to star formation epoch
        average_model_star = self.Star(
            self, name='average_model_star',
            backward=True, model=True, age=self.series.age.value,
            velocity_xyz=self.series.velocity.values, velocity_xyz_error=np.zeros(3),
            position_xyz=self.series.position.values, position_xyz_error=np.zeros(3))

        # Average velocity at the epoch of star formation
        velocity = average_model_star.velocity_xyz[-1]
        velocity_error = self.series.velocity_error.values
        velocity_scatter = self.series.velocity_scatter.values

        # Average position at the epoch of star formation
        position = average_model_star.position_xyz[-1]
        position_error = self.series.position_error.values
        position_scatter = self.series.position_scatter.values

        # Stars creation from a model
        for star in range(self.series.number_of_stars):

            # Model star forward galactic orbit integration
            model_star = self.Star(
                self, name='model_star_{}'.format(star + 1),
                backward=False, model=True, age=self.series.age.value,
                velocity_xyz=np.random.normal(velocity, velocity_scatter), velocity_xyz_error=np.zeros(3),
                position_xyz=np.random.normal(position, position_scatter), position_xyz_error=np.zeros(3))

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
                star_errors = star - star // len(self.series.data) * len(self.series.data)
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

        # Number of remaining stars, excluding outliers
        self.number_in_subsample = len(self.subsample)
        self.sample_size = len(self.sample)
        self.number_of_outliers = len(self.outliers)

        # Average position, excluding outliers
        self.position_xyz = np.mean([star.position_xyz for star in self.sample], axis=0)

        # Average linear position and errors, excluding outliers
        self.position_xyz_linear = np.mean(
            np.array([star.position_xyz_linear for star in self.sample]), axis=0)
        self.position_xyz_linear_error = np.sum(
            np.array([star.position_xyz_linear_error for star in self.sample])**2,
            axis=0)**0.5 / self.sample_size

        # Average ξηζ position, excluding outliers
        self.position_ξηζ = np.mean([star.position_ξηζ for star in self.sample], axis=0)

        # Average velocity, excluding outliers
        self.velocity_xyz = np.mean([star.velocity_xyz for star in self.sample], axis=0)

        # Average linear velocity and errors, excluding outliers
        self.velocity_xyz_linear = np.mean([star.velocity_xyz_linear for star in self.sample], axis=0)
        self.velocity_xyz_linear_error = np.sum(np.array(
            [star.velocity_xyz_linear_error for star in self.sample])**2, axis=0)**0.5 / self.sample_size

        # Average ξηζ velocity, excluding outliers
        self.velocity_ξηζ = np.mean([star.velocity_ξηζ for star in self.sample], axis=0)

        # Stars relative position and velocity, distance and speed, including outliers
        for star in self:
            star.get_relative_coordinates()

        # ξηζ position and velocity scatters for outlier filtering
        self.scatter_ξηζ = np.std([star.position_ξηζ for star in self.sample], axis=0)
        self.scatter_velocity_ξηζ = np.std([star.velocity_ξηζ for star in self.sample], axis=0)

    def filter_outliers(self):
        """ Filters outliers from the data based on ξηζ position and velocity scatter over time.
            Next, a core subsample is determined based on a robust covariance matrix estimator
            using the scikit-learn (Sklearn) Python package.
        """

        # Filter outliers for the first group
        if self.number == 0:

            # Compute all stars coordinates
            self.get_stars_coordinates()

            # Iteratively validate stars coordinates
            outliers = True
            while outliers and len(self.sample) > int(0.8 * self.series.number_of_stars):

                # Remove stars beyond the σ cutoff of the average position or velocity
                outliers = False
                for star in self.sample:
                    position_outlier = (np.abs(star.relative_position_ξηζ)
                        > (self.scatter_ξηζ * self.series.cutoff)).any()
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
            else:
                print(f'No outliers found in {self.series.name}.')

            # # Robust covariance matrix and support fraction
            # a = np.array([star.position_ξηζ for star in self.sample])
            # a = np.swapaxes(a, 0, 1)
            # robust_covariance_matrix = []
            # support_fraction = []
            #
            # # Iteration over time
            # for i in range(a.shape[0]):
            #     b = MinCovDet(assume_centered=False).fit(a[i])
            #     # print(i, b[i].shape, b.support_.size, b.support_.nonzero()[0].size, b.dist_.size)
            #     robust_covariance_matrix.append(b.covariance_)
            #     support_fraction.append(b.support_)
            #
            # # Array conversion
            # robust_covariance_matrix = np.array(robust_covariance_matrix)
            # support_fraction = np.array(support_fraction)
            # support_fraction_all = np.sum(support_fraction, axis=0) / self.series.number_of_steps > 0.7
            #
            # # Core creation
            # for i in range(self.sample_size):
            #     if support_fraction_all[i]:
            #         self.sample[i].subsample = True
            #     else:
            #         self.sample[i].subsample = False
            #         self.sample[i].outlier = True
            #         print(self.sample[i].name)
            #
            # # Temporary reasignment of subsample to stars
            # self.get_stars_coordinates()

        # Use the same outliers, sample and subsample as the first group for other groups
        else:
            outliers = [star.name for star in self.series[0].outliers]
            for star in filter(lambda star: star.name in outliers, self):
                star.outlier == True

            # Compute valid stars coordinates
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

        self.get_stars_coordinates()

        # self.Indicator(self, 'eigen_values', np.repeat(np.expand_dims(
        #     np.sum(eigen_values**2, axis=1)**0.5, axis=0),
        #     self.series.jack_knife_number, axis=0), valid=True)

        # Compute Mahalanobis distance
        self.positions_xyz = np.expand_dims(np.array([
            star.position_xyz for star in self.sample]), axis=1)
        self.sample_size_jack_knife = 1
        self.Mahalanobis_xyz = self.get_Mahalanobis_distance('positions_xyz')
        self.weights = np.exp(-1 * self.Mahalanobis_xyz)

        for i in range(len(self.sample)):
            self.sample[i].weight = self.weights[i]

    def set_indicators(self):
        """ Sets the parameters to compute age indicators including Jack-Knife Monte-Carlo. """

        # Indicators list
        self.indicators = []

        # Select stars for Jack-Knife Monte-Carlo for the first group
        if self.number == 0:
            self.sample_size_jack_knife = int(self.sample_size * self.series.jack_knife_fraction)
            self.stars_monte_carlo = np.array([np.random.choice(self.sample_size,
                self.sample_size_jack_knife, replace=False)
                    for i in range(self.series.jack_knife_number)])

        # Use the same stars for Jack-Knife Monte-Carlo as the first group for other groups
        else:
            self.sample_size_jack_knife = self.series[0].sample_size_jack_knife
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
            values, age and minium using a Jack-Knife Monte-Carlo method.
        """

        def __init__(self, group, name, values, verbose_name, valid=True):
            """ Initializes an age indicator. """

            # Initialization
            self.group = group
            self.name = name
            self.values = values
            self.verbose_name = np.atleast_1d(verbose_name)
            self.valid = np.atleast_1d(valid)

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

            # Add the indicator to the group
            self.group.indicators.append(self)
            vars(self.group)[self.name] = self

    def get_scatter(self):
        """ Computes the xyz and ξηζ position and velocity scatter of a group. The age of the
            moving group is then estimated by finding the time at which the position scatter is
            minimal, although the age estimate is depreciated by the use of the xyz and ξηζ
            position and velocity covariance.
        """

        # xyz position scatter
        # self.Indicator(self, 'scatter_xyz', np.std(self.positions_xyz, axis=0), valid=False)
        # self.Indicator(self, 'scatter_xyz_total', np.sum(self.scatter_xyz.values**2, axis=2)**0.5, valid=False)

        # ξηζ position scatter
        # self.Indicator(self, 'scatter_ξηζ', np.std(self.positions_ξηζ, axis=0), valid=False)
        # self.Indicator(self, 'scatter_ξηζ_total', np.sum(self.scatter_ξηζ.values**2, axis=2)**0.5, valid=False)

        # xyz velocity scatter
        self.scatter_velocity_xyz = np.std([star.velocity_xyz for star in self.sample], axis=0)
        self.scatter_velocity_xyz_total = np.sum(self.velocity_xyz**2, axis=0)**0.5

        # ξηζ velocity scatter
        self.scatter_velocity_ξηζ = np.std([star.velocity_ξηζ for star in self.sample], axis=0)
        self.scatter_velocity_ξηζ_total = np.sum(self.velocity_ξηζ**2, axis=0)**0.5

    def get_median_absolute_deviation(self):
        """ Computes the xyz and ξηζ, partial and total median absolute deviations (MAD) of a group
            and their respective errors for all timesteps. The age of the group is then estimated by
            finding the time at which the median absolute deviation is minimal.
        """

        # xyz median absolute deviation age
        self.Indicator(self, 'mad_xyz',
            np.median(np.abs(self.positions_xyz - np.median(self.positions_xyz, axis=0)), axis=0),
            np.array(['X MAD', 'Y MAD', 'Z MAD']), valid=np.full(3, False))
        self.Indicator(self, 'mad_xyz_total', np.sum(self.mad_xyz.values**2, axis=2)**0.5,
            'Total XYZ MAD', valid=False)

        # ξηζ median absolute deviation age
        self.Indicator(self, 'mad_ξηζ',
            np.median(np.abs(self.positions_ξηζ - np.median(self.positions_ξηζ, axis=0)), axis=0),
            np.array(['ξ MAD', 'η MAD', 'ζ MAD']), valid=np.full(3, False))
        self.Indicator(self, 'mad_ξηζ_total', np.sum(self.mad_ξηζ.values**2, axis=2)**0.5,
            'Total ξηζ MAD', valid=False)

    def get_covariances(self):
        """ Computes the xyz and ξηζ position covariance matrices, and xyz and ξηζ position and
            velocity cross covariance matrices. The age of the group is then estimated by finding
            the time at which the variances, determinant or trace of the matrices are minimal.
        """

        # xyz position covariance matrix, determinant and trace ages
        self.covariances_xyz_matrix = self.get_covariance_matrix('positions_xyz')
        self.get_covariance_indicators(self.covariances_xyz_matrix, 'covariances_xyz',
            [np.array(['X Variance', 'Y Variance', 'Z Variance']),
                'XYZ Covariance Matrix Determinant','XYZ Covariance Matrix Trace'],
             valid=[np.array([True, False, False]), False, False])

        # xyz position and velocity cross covariance matrix, determinant and trace ages
        self.cross_covariances_xyz_matrix = self.get_covariance_matrix('positions_xyz', 'velocities_xyz')
        self.get_covariance_indicators(self.cross_covariances_xyz_matrix, 'cross_covariances_xyz',
            [np.array(['X-U Cross Covariance', 'Y-V Cross Covariance', 'Z-W Cross Covariance']),
                'XYZ Cross Covariance Matrix Determinant','XYZ Cross Covariance Matrix Trace'],
            valid=[np.array([False, False, False]), False, False])

        # ξηζ position covariance matrix, determinant and trace ages
        self.covariances_ξηζ_matrix = self.get_covariance_matrix('positions_ξηζ')
        self.get_covariance_indicators(self.covariances_ξηζ_matrix, 'covariances_ξηζ',
            [np.array(['ξ Variance', 'η Variance', 'ζ Variance']),
                'ξηζ Covariance Matrix Determinant','ξηζ Covariance Matrix Trace'],
            valid=[np.array([True, False, False]), False, False])

        # ξηζ position and velocity cross covariance matrix, determinant and trace ages
        self.cross_covariances_ξηζ_matrix = self.get_covariance_matrix('positions_ξηζ', 'velocities_ξηζ')
        self.get_covariance_indicators(self.cross_covariances_ξηζ_matrix, 'cross_covariances_ξηζ',
            [np.array(['ξ-vξ Cross Covariance', 'η-vη Cross Covariance', 'ζ-vζ Cross Covariance']),
                'ξηζ Cross Covariance Matrix Determinant','ξηζ Cross covariance Matrix Trace'],
            valid=[np.array([False, False, False]), False, False])

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
        self.get_covariance_indicators(self.covariances_xyz_robust_matrix, 'covariances_xyz_robust',
            [np.array(['X Variance (robust)', 'Y Variance (robust)', 'Z Variance (robust)']),
                'XYZ Covariance Matrix Determinant (robust)','XYZ Covariance Matrix Trace (robust)'],
            valid=[np.array([True, False, False]), False, False])

        # xyz position and velocity robust cross covariance matrix and determinant ages
        self.Mahalanobis_cross_xyz = self.get_Mahalanobis_distance(
            'positions_xyz', 'velocities_xyz', covariance_matrix=self.cross_covariances_xyz_matrix)
        self.cross_covariances_xyz_robust_matrix = self.get_covariance_matrix(
            'positions_xyz', 'velocities_xyz', robust=True,
            Mahalanobis_distance=self.Mahalanobis_xyz)
        self.get_covariance_indicators(self.cross_covariances_xyz_robust_matrix,
            'cross_covariances_xyz_robust', [np.array(['X-U Cross Covariance (robust)',
                    'Y-V Cross Covariance (robust)', 'Z-W Cross Covariance (robust)']),
                'XYZ Cross Covariance Matrix Determinant (robust)',
                'XYZ Cross Covariance Matrix Trace (robust)'],
            valid=[np.array([False, False, False]), False, False])

        # ξηζ position robust covariance matrix, determinant and trace ages
        self.Mahalanobis_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', covariance_matrix=self.covariances_ξηζ_matrix)
        self.covariances_ξηζ_robust_matrix = self.get_covariance_matrix(
            'positions_ξηζ', robust=True, Mahalanobis_distance=self.Mahalanobis_ξηζ)
        self.get_covariance_indicators(self.covariances_ξηζ_robust_matrix, 'covariances_ξηζ_robust',
            [np.array(['ξ Variance (robust)', 'η Variance (robust)', 'ζ Variance (robust)']),
                'ξηζ Covariance Matrix Determinant (robust)','ξηζ Covariance Matrix Trace (robust)'],
            valid=[np.array([True, False, False]), False, False])

        # ξηζ position and velocity robust cross covariance matrix and determinant ages
        self.Mahalanobis_cross_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', 'velocities_ξηζ', covariance_matrix=self.cross_covariances_ξηζ_matrix)
        self.cross_covariances_ξηζ_robust_matrix = self.get_covariance_matrix(
            'positions_ξηζ', 'velocities_ξηζ', robust=True,
            Mahalanobis_distance=self.Mahalanobis_ξηζ)
        self.get_covariance_indicators(self.cross_covariances_ξηζ_robust_matrix,
            'cross_covariances_ξηζ_robust', [np.array(['ξ-vξ Cross Covariance (robust)',
                    'η-vη Cross Covariance (robust)', 'ζ-vζ Cross Covariance (robust)']),
                'ξηζ Cross Covariance Matrix Determinant (robust)',
                'ξηζ Cross Covariance Matrix Trace (robust)'],
            valid=[np.array([False, False, False]), False, False])

    def get_covariances_sklearn(self):
        """ Computes the ξηζ position covariance matrix using the sklearn package. The age of the
            group is then estimated by finding the time at which the variances, determinant or
            trace of the matrix are minimal.
        """

        def get_covariance_sklearn_matrix(a):
            """ Computes a robust covariance determinant with SKlearn (slow). """

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

        # ξηζ position covariance matrix with SKlearn, determinant and trace ages
        self.covariances_ξηζ_sklearn_matrix = get_covariance_sklearn_matrix('position_ξηζ')
        self.Indicator(self, 'covariances_ξηζ_sklearn', np.repeat(np.expand_dims(
                np.abs(self.covariances_ξηζ_sklearn_matrix[:, (0, 1, 2), (0, 1, 2)])**0.5, axis=0),
                    self.series.jack_knife_number, axis=0),
            np.array(['ξ Variance (sklearn)', 'η Variance (sklearn)', 'ζ Variance (sklearn)']),
            valid=np.array([False, False, False]))
        self.Indicator(self, 'covariances_ξηζ_sklearn_matrix_det', np.repeat(np.expand_dims(
                np.abs(np.linalg.det(self.covariances_ξηζ_sklearn_matrix))**(1/6), axis=0),
                    self.series.jack_knife_number, axis=0),
            'ξηζ Covariance Matrix Determinant (sklearn)', valid=False)
        self.Indicator(self, 'covariances_ξηζ_sklearn_matrix_trace', np.repeat(np.expand_dims(
                np.abs(np.trace(self.covariances_ξηζ_sklearn_matrix, axis1=1, axis2=2) / 3)**0.5, axis=0),
                    self.series.jack_knife_number, axis=0),
            'ξηζ Covariance Matrix Trace (sklearn)', valid=False)

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
            self.sample_size_jack_knife, axis=0)

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

    def get_covariance_indicators(self, covariance_matrix, indicator, verbose_names, valid=True):
        """ Computes the covariances, and the determinant and trace of the covariance matrix for
            a given covariance matrix and indicator.
        """

        if type(valid) == bool:
            valid = np.full(3, valid)

        # Covariances
        self.Indicator(self, indicator,
            np.abs(covariance_matrix[:, :, (0, 1, 2), (0, 1, 2)])**0.5,
            verbose_names[0], valid=valid[0])

        # Covariance matrix determinant
        self.Indicator(self, f'{indicator}_matrix_det',
            np.abs(np.linalg.det(covariance_matrix))**(1/6),
            verbose_names[1], valid=valid[1])

        # Covariance matrix trace
        self.Indicator(self, f'{indicator}_matrix_trace',
            np.abs(np.trace(covariance_matrix, axis1=2, axis2=3) / 3)**0.5,
            verbose_names[2], valid=valid[2])

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

        # Jack-knife Monte Carlo branch lengths
        sample_size_jack_knife = int(self.number_of_branches * self.series.jack_knife_fraction)
        branches_monte_carlo = np.array([np.random.choice(
            self.number_of_branches, sample_size_jack_knife, replace=False)
                for i in range(self.series.jack_knife_number)])
        mst_xyz_lengths = self.mst_xyz_lengths.T[branches_monte_carlo].swapaxes(0, 1)
        mst_xyz_weights = self.mst_xyz_weights.T[branches_monte_carlo].swapaxes(0, 1)
        mst_ξηζ_lengths = self.mst_ξηζ_lengths.T[branches_monte_carlo].swapaxes(0, 1)
        mst_ξηζ_weights = self.mst_ξηζ_weights.T[branches_monte_carlo].swapaxes(0, 1)

        # Minimum spanning tree average branch length age
        self.Indicator(self, 'mst_xyz_mean', np.mean(mst_xyz_lengths, axis=0),
            'XYZ MST Mean', valid=False)
        self.Indicator(self, 'mst_ξηζ_mean', np.mean(mst_ξηζ_lengths, axis=0),
            'ξηζ MST Mean', valid=False)

        # Robust minimum spanning tree average branch length age
        self.Indicator(self, 'mst_xyz_mean_robust',
            np.average(mst_xyz_lengths, weights=mst_xyz_weights, axis=0),
            'XYZ MST Mean (robust)', valid=False)
        self.Indicator(self, 'mst_ξηζ_mean_robust',
            np.average(mst_ξηζ_lengths, weights=mst_ξηζ_weights, axis=0),
            'ξηζ MST Mean (robust)', valid=False)

        # Minimum spanning tree branch length median absolute deviation age
        self.Indicator(self, 'mst_xyz_mad',
            np.median(np.abs(mst_xyz_lengths - np.median(mst_xyz_lengths, axis=0)), axis=0),
            'XYZ MST MAD', valid=False)
        self.Indicator(self, 'mst_ξηζ_mad',
            np.median(np.abs(mst_ξηζ_lengths - np.median(mst_ξηζ_lengths, axis=0)), axis=0),
            'ξηζ MST MAD', valid=False)

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
            self.subsample = False
            self.model = False
            self.age = None
            vars(self).update(values)

            # Trajectory integration
            self.get_linear()
            self.get_orbit()

        def set_outlier(self, position_outlier, velocity_outlier):
            """ Sets the star as an outlier. """

            self.outlier = True
            self.position_outlier = position_outlier
            self.velocity_outlier = velocity_outlier

        def get_linear(self):
            """ Computes a star's backward or forward linear trajectory. """

            # Linear xyz position
            self.position_xyz_linear = self.position_xyz - (self.velocity_xyz
                + Coordinate.galactic_velocity) * np.expand_dims(self.group.series.time, axis=0).T
            self.position_xyz_linear_error = (self.position_xyz_error**2 + (self.velocity_xyz_error
                * np.expand_dims(self.group.series.time, axis=0).T)**2)**0.5

            # Linear UVW velocity
            self.velocity_xyz_linear = np.repeat(
                np.expand_dims(self.velocity_xyz, axis=0), self.group.series.time.size, axis=0)
            self.velocity_xyz_linear_error = np.repeat(np.expand_dims(
                self.velocity_xyz_error, axis=0), self.group.series.time.size, axis=0)

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

            # Forward or backward time in Myr
            self.time_Myr = (-1 if self.backward else 1) * (self.group.series.time
                if self.age is None else np.array([0., self.group.series.age.value]))

            # Unit conversion to natural units
            self.time = self.time_Myr * Coordinate.lsr_velocity[1] / Coordinate.sun_position[0]
            self.position_rθz /= np.array([Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])
            self.velocity_rθz /= Coordinate.lsr_velocity[1]

            # Orbit initialization and integration
            orbit = Orbit([
                self.position_rθz[0], self.velocity_rθz[0], self.velocity_rθz[1],
                self.position_rθz[2], self.velocity_rθz[2], self.position_rθz[1]],
                ro=Coordinate.ro, vo=Coordinate.vo, zo=Coordinate.zo,
                solarmotion=Coordinate.sun_velocity_peculiar_kms)
            orbit.turn_physical_off()
            orbit.integrate(self.time, self.group.series.potential, method='odeint')

            # Orbital rθz positions and unit conversion to physical units
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
            self.position_ξηζ = position_rθz_ξηζ(*self.position_rθz.T, self.time_Myr)

            # ξηζ velocities
            self.velocity_ξηζ = velocity_rθz_ξηζ(*self.velocity_rθz.T, self.time_Myr)

        def get_relative_coordinates(self):
            """ Computes relative position and velocity, the distance and the speed from the
                average position and velocity, and their respective errors for all timesteps.
            """

            # Relative xyz position and distance from the average group position
            self.relative_position_xyz = self.position_xyz - self.group.position_xyz
            self.distance_xyz = np.sum(self.relative_position_xyz**2, axis=1)**0.5

            # Relative xyz velocity and speed from the average group velocity
            self.relative_velocity_xyz = self.velocity_xyz - self.group.velocity_xyz
            self.speed_xyz = np.sum(self.relative_velocity_xyz**2, axis=0)**0.5

            # Relative ξηζ position and distance from the average group position
            self.relative_position_ξηζ = self.position_ξηζ - self.group.position_ξηζ
            self.distance_ξηζ = np.sum(self.relative_position_ξηζ**2, axis=1)**0.5

            # Relative ξηζ velocity and speed from the average group velocity
            self.relative_velocity_ξηζ = self.velocity_ξηζ - self.group.velocity_ξηζ
            self.speed_ξηζ = np.sum(self.relative_velocity_ξηζ**2, axis=1)**0.5

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
