# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
group.py: Defines the Group class and the embeded Star classes. These classes contain the
information and methods necessary to compute tracebacks of stars in a moving group and assess
its age by minimizing: the 3D scatter, median absolute deviation, position-velocity covariances
and minimum spanning tree mean branch length and median absolute deviation of branch length.
"""

import galpy.util.coords as coords
from galpy.orbit import Orbit
from sklearn.covariance import MinCovDet
from time import time as get_time
from itertools import combinations
from .output import Output_Group, Output_Star
from .coordinate import *

class Group(list, Output_Group):
    """
    Contains the values and related methods of an association and a list of Star objets that are
    part of it. Stars are can be imported from data or modeled from parameters.
    """

    def __init__(self, series, number, name):
        """
        Initializes a Group in a series with a name and number, along with association size metrics.
        """

        # Initialization
        self.series = series
        self.number = number
        self.name = name

        # Initialize association size metrics
        self.metrics = []
        for metric in self.series.metrics:
            self.Metric(self, metric)

        # Add the group to the series
        self.series.append(self)

    def traceback(self):
        """
        Traces back the Galactic orbits of every star in the group, either using imported data
        or by modeling stars from parameters. Outliers are also removed from the data sample.
        """

        # Stars from data
        self.set_timer('orbits')
        if self.series.from_data:
            self.get_stars_from_data()

        # Stars from model
        elif self.series.from_model:
            self.get_stars_from_model()

        # Compute stars' coordinates
        self.get_stars_coordinates()

        # Filter outliers
        if self.series.from_data:
            self.filter_outliers()
        self.set_timer()

    def get_stars_from_data(self):
        """
        Creates a list of Star objects from a Python dictionary or CSV files containing the
        parameters including the name of the stars, their xyz positions and uvw velocities,
        and their respective measurement errors. Radial velocity shift is also added. If
        only one group is created in the series, actual values are used. If multiple groups
        are created, positions and velocities are scrambled based on measurement errors.
        """

        # Create stars from data
        index = 0
        for star in self.series.data.sample:

            # Position and velocity errors based on data
            if self.series.data_errors:
                position_errors = star.position.errors
                velocity_errors = star.velocity.errors

            # Position and velocity errors based on models
            else:
                position_errors = self.series.position_error.values
                velocity_errors = self.series.velocity_error.values

            # Radial velocity shift based on data
            if self.series.data_rv_shifts:
                rv_shift_value = star.rv_shift.value
                rv_shift_error = star.rv_shift.error

            # Radial velocity shift based on models
            else:
                rv_shift_value = self.series.rv_shift.value
                rv_shift_error = self.series.rv_shift.error

            # From an observables coordinates system
            if self.series.data.data.system.name == 'observables':

                # Convert position and velocity into equatorial spherical coordinates
                position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error = position_obs_rδα(
                    *(
                        star.position.values if self.number == 0 else
                        np.random.normal(star.position.values, position_errors)
                    ), *(
                        star.velocity.values if self.number == 0 else
                        np.random.normal(star.velocity.values, velocity_errors)
                    ), *position_errors, *velocity_errors
                )

                # Apply radial velocity shift correction and its error
                velocity_rδα -= np.array([rv_shift_value, 0.0, 0.0])
                velocity_rδα_error[0] = (velocity_rδα_error[0]**2 + rv_shift_error**2)**0.5

                # Convert position and velocity into Galactic cartesian coordinates
                position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                    *position_rδα, *position_rδα_error
                )
                velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                    *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error
                )

            # From a Galactic cartesian coordinates system
            if self.series.data.data.system.name == 'cartesian':

                # Convert position and velocity into equatorial spherical coordinates
                position_rδα, position_rδα_error = galactic_xyz_equatorial_rδα(
                    *(
                        star.position.values if self.number == 0 else
                        np.random.normal(star.position.values, position_errors)
                    ), *star.position.errors
                )
                velocity_rδα, velocity_rδα_error = galactic_uvw_equatorial_rvμδμα(
                    *(
                        star.position.values if self.number == 0 else
                        np.random.normal(star.position.values, position_errors)
                    ), *(
                        star.velocity.values if self.number == 0 else
                        np.random.normal(star.velocity.values, velocity_errors)
                    ), *position_errors, *velocity_errors)

                # Apply radial velocity shift correction and its error
                velocity_rδα -= np.array([rv_shift_value, 0.0, 0.0])
                velocity_rδα_error[0] = (velocity_rδα_error[0]**2 + rv_shift_error**2)**0.5

                # Convert position and velocity back into Galactic cartesian coordinates
                position_xyz, position_xyz_error = (star.position.values, position_errors)
                velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                    *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error
                )

            # Create star
            index += 1
            self.append(
                self.Star(
                    self, index=index, name=star.name, time=self.series.time,
                    velocity_xyz=velocity_xyz, velocity_xyz_error=velocity_xyz_error,
                    position_xyz=position_xyz, position_xyz_error=position_xyz_error
                )
            )

            # Update progress bar
            self.series.progress_bar.update(1)

    def get_stars_from_model(self):
        """
        Creates an artificial model of star for a given number of stars and age based on the
        the initial average xyz position and uvw velocity, and their respective errors and
        scatters. The sample is then moved forward in time for the given age and radial velocity
        shift is also added.
        """

        # Model average star's backward orbit integration from the current-day epoch
        # to the epoch of star formation
        self.average_model_star = self.Star(
            self, name='average_model_star', time=self.series.model_time,
            velocity_xyz=self.series.velocity.values, velocity_xyz_error=np.zeros(3),
            position_xyz=self.series.position.values, position_xyz_error=np.zeros(3)
        )

        # Average velocity at the epoch of star formation
        velocity = self.average_model_star.velocity_xyz[-1]
        velocity_error = self.series.velocity_error.values
        velocity_scatter = self.series.velocity_scatter.values

        # Average position at the epoch of star formation
        position = self.average_model_star.position_xyz[-1]
        position_error = self.series.position_error.values
        position_scatter = self.series.position_scatter.values

        # Create stars from a model
        self.model_stars = []
        for star in range(self.series.number_of_stars):

            # Model star's forward orbit integration from the epoch of star formation
            # to the current-day epoch
            model_star = self.Star(
                self, name='model_star_{}'.format(star + 1), time=self.series.model_time[::-1],
                velocity_xyz=np.random.normal(velocity, velocity_scatter),
                velocity_xyz_error=np.zeros(3),
                position_xyz=np.random.normal(position, position_scatter),
                position_xyz_error=np.zeros(3)
            )
            self.model_stars.append(model_star)

            # Convert position and velocity into equatorial spherical coordinates
            position_rδα = galactic_xyz_equatorial_rδα(*model_star.position_xyz[-1])[0]
            velocity_rδα = galactic_uvw_equatorial_rvμδμα(
                *model_star.position_xyz[-1], *model_star.velocity_xyz[-1]
            )[0]

            # Convert position and velocity into observables
            position_obs, velocity_obs = position_rδα_obs(*position_rδα, *velocity_rδα)[:2]

            # Scramble position and velocity based on data
            if self.series.data_errors:
                errors = star - (
                    star // len(self.series.data.sample)
                ) * len(self.series.data.sample)
                position_obs_error = self.series.data.sample[errors].position.errors
                velocity_obs_error = self.series.data.sample[errors].velocity.errors

            # Scramble position and velocity based on models
            else:
                position_obs_error = position_error
                velocity_obs_error = velocity_error

            # Convert position and velocity back into equatorial spherical coordinates
            position_rδα, velocity_rδα, position_rδα_error, velocity_rδα_error = position_obs_rδα(
                *np.random.normal(position_obs, position_error),
                *np.random.normal(velocity_obs, velocity_error),
                *position_obs_error, *velocity_obs_error
            )

            # Radial velocity shift based on data
            if self.series.data_rv_shifts:
                rv_shift = star - (
                    star // len(self.series.data.sample)
                ) * len(self.series.data.sample)
                rv_shift_value = self.series.data.sample[rv_shift].rv_shift.value
                rv_shift_error = self.series.data.sample[rv_shift].rv_shift.error

            # Radial velocity shift based on models
            else:
                rv_shift_value = self.series.rv_shift.value
                rv_shift_error = self.series.rv_shift.error

            # Apply radial velocity shift bias and its error
            velocity_rδα += np.array([rv_shift_value, 0.0, 0.0])
            velocity_rδα_error[0] = (velocity_rδα_error[0]**2 + rv_shift_error**2)**0.5

            # Convert position and velocity back into Galactic cartesian coordinates
            position_xyz, position_xyz_error = equatorial_rδα_galactic_xyz(
                *position_rδα, *position_rδα_error
            )
            velocity_xyz, velocity_xyz_error = equatorial_rvμδμα_galactic_uvw(
                *position_rδα, *velocity_rδα, *position_rδα_error, *velocity_rδα_error
            )

            # Create star
            self.append(
                self.Star(
                    self, index=star + 1, name='star_{}'.format(star + 1), time=self.series.time,
                    velocity_xyz=velocity_xyz, velocity_xyz_error=velocity_xyz_error,
                    position_xyz=position_xyz, position_xyz_error=position_xyz_error
                )
            )

            # Update progress bar
            self.series.progress_bar.update(1)

    def get_stars_coordinates(self):
        """
        Computes the number of stars that aren't outliers, the stars average xyz and ξηζ
        velocity and position, excluding outliers, and the stars relative positions and
        distances from the average, including outliers. If all stars are identified as
        outliers, an error is raised.
        """

        # Create sample, subsample and outliers lists
        self.sample = list(filter(lambda star: not star.outlier, self))
        self.subsample = list(filter(lambda star: star.subsample, self))
        self.outliers =list(filter(lambda star: star.outlier, self))

        # Number of remaining stars
        self.sample_size = len(self.sample)
        self.subsample_size = len(self.subsample)
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
        """
        Filters outliers from the sample based on ξηζ position and velocity scatter over time.
        A subsample is created based on a robust covariances matrix estimator using the
        scikit-learn (sklearn) Python package, leaving other stars as part of the extended
        sample.
        """

        # Iteratively validate stars coordinates in the first group
        if self.number == 0:
            if self.series.cutoff is not None:
                outliers = True
                while outliers and len(self.sample) > int(0.8 * len(self)):

                    # Remove stars beyond the σ cutoff of the average position or velocity
                    outliers = False
                    for star in self.sample:
                        position_outlier = (
                            np.abs(star.relative_position_ξηζ)
                            > (self.scatter_position_ξηζ * self.series.cutoff)
                        ).any()
                        velocity_outlier = (
                            np.abs(star.relative_velocity_ξηζ)
                            > (self.scatter_velocity_ξηζ * self.series.cutoff)
                        ).any()
                        if position_outlier or velocity_outlier:
                            star.set_outlier(position_outlier, velocity_outlier)
                            outliers = True

                    # Compute valid stars coordinates
                    self.get_stars_coordinates()

                # Create a message if a least one outlier have been found
                if self.number_of_outliers > 0:
                    self.outliers_messages = [
                        f'{self.number_of_outliers:d} outlier'
                        f"{'s' if self.number_of_outliers > 1 else ''} "
                        f"found in '{self.series.name}' series during traceback:"
                    ]

                    # Create a message for every outlier found
                    for star in self.outliers:
                        outlier_type = (
                            'Position' if star.position_outlier else
                            'Velocity' if star.velocity_outlier else ''
                        )
                        if outlier_type == 'Position':
                            outlier_sigma = np.max(
                                np.abs(star.relative_position_ξηζ) / self.scatter_position_ξηζ
                            )
                        if outlier_type == 'Velocity':
                            outlier_sigma = np.max(
                                np.abs(star.relative_velocity_ξηζ) / self.scatter_velocity_ξηζ
                            )
                        self.outliers_messages.append(
                            f'{star.name}: {outlier_type} > {outlier_sigma:.1f}σ'
                        )

                # Create message if no outliers have been found and cutoff is not None
                else:
                    self.outliers_messages = [f"No outliers found in '{self.series.name}' series."]

            # Leave no message if cutoff is None
            else:
                self.outliers_messages = []

            # Robust covariances matrix and support fraction
            if False:
                a = np.swapaxes(np.array([star.position_ξηζ for star in self.sample]), 0, 1)
                robust_covariances_matrix = []
                support_fraction = []

                # Iteration over time
                for step in range(self.series.number_of_steps):
                    MCD = MinCovDet(assume_centered=False).fit(a[step])
                    robust_covariances_matrix.append(MCD.covariance_)
                    support_fraction.append(MCD.support_)
                    # print(step, MCD.support_.size, MCD.support_.nonzero()[0].size, MCD.dist_.size)

                # Array conversion
                robust_covariances_matrix = np.array(robust_covariances_matrix)
                support_fraction = np.array(support_fraction)
                support_fraction_all = (
                    np.sum(support_fraction, axis=0) / self.series.number_of_steps > 0.7
                )

                # Subsample creation
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

    def get_covariance_errors(self):
        """
        Computes the covariance matrices on the positions and velocities of every star, in
        every group, as a function of time, and add this value every star in the group.
        """

        # xyz position covariance matrices
        self.positions_xyz = np.array(
            [[star.position_xyz for star in group.sample] for group in self.series[1:]]
        )
        position_xyz_cov_matrices = self.get_covariances_matrix('positions_xyz')
        for star in range(len(self.sample)):
            self.sample[star].position_xyz_error = position_xyz_cov_matrices[star]

        # xyz velocity covariance matrices
        self.velocities_xyz = np.array(
            [[star.velocity_xyz for star in group.sample] for group in self.series[1:]]
        )
        velocity_xyz_cov_matrices = self.get_covariances_matrix('velocities_xyz')
        for star in range(len(self.sample)):
            self.sample[star].velocity_xyz_error = velocity_xyz_cov_matrices[star]

        # ξηζ position covariance matrices
        self.positions_ξηζ = np.array(
            [[star.position_ξηζ for star in group.sample] for group in self.series[1:]]
        )
        position_ξηζ_cov_matrices = self.get_covariances_matrix('positions_ξηζ')
        for star in range(len(self.sample)):
            self.sample[star].position_ξηζ_error = position_ξηζ_cov_matrices[star]

        # ξηζ velocity covariance matrices
        self.velocities_ξηζ = np.array(
            [[star.velocity_ξηζ for star in group.sample] for group in self.series[1:]]
        )
        velocity_ξηζ_cov_matrices = self.get_covariances_matrix('velocities_ξηζ')
        for star in range(len(self.sample)):
            self.sample[star].velocity_ξηζ_error = velocity_ξηζ_cov_matrices[star]

    class Metric():
        """
        Association size metric including its values, age at minimum and minimum. Computes
        errors on values, age and minium using a jackknife Monte Carlo method.
        """

        def __init__(self, group, metric):
            """Initializes the association size metric."""

            # Initialization
            self.group = group
            self.metric = metric

            # Add the metric to the group
            vars(self.group)[self.metric.label] = self
            self.group.metrics.append(self)

        def __call__(self, values):
            """Computes errors on values, age and minium using a jackknife Monte Carlo method."""

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
            self.metric.status = True

        def __repr__(self):
            """"""
            return '{}: {} ± {} Myr'.format(
                self.metric.name,
                self.age,
                self.age_int_error
            )

    def chronologize(self):
        """
        Computes the kinematic age of the group by finding the epoch of minimal association size
        using several size metrics. Parameters used to compute association size metrics, including
        weights based on Mahalanobis distances computed with empirical covariances matrices, and
        jackknife Monte Carlo itrations.
        """

        # Compute Mahalanobis distance
        if len(self) > 0:
            self.positions_xyz = np.array([star.position_xyz for star in self.sample])[:,None]
            self.number_of_stars_iteration = 1
            self.Mahalanobis_xyz = self.get_Mahalanobis_distance('positions_xyz')

            # Compute weights
            self.weights = np.exp(-1 * self.Mahalanobis_xyz)
            for i in range(len(self.sample)):
                self.sample[i].weight = self.weights[i]

            # !!! For the first group, randomly select stars for the jackknife Monte Carlo !!!
            if True:
                self.number_of_stars_iteration = int(
                    self.sample_size * self.series.iteration_fraction
                )
                self.stars_monte_carlo = np.array(
                    [
                        np.random.choice(
                            self.sample_size, self.number_of_stars_iteration, replace=False
                        ) for i in range(self.series.number_of_iterations)
                    ]
                ).T

            # For other groups, use the same stars for the jackknife Monte Carlo as the first group
            else:
                self.number_of_stars_iteration = self.series[0].number_of_stars_iteration
                self.stars_monte_carlo = self.series[0].stars_monte_carlo

            # Create selected stars xyz and ξηζ positions and velocities arrays
            self.positions_xyz = np.array(
                [star.position_xyz for star in self.sample]
            )[self.stars_monte_carlo]
            self.positions_ξηζ = np.array(
                [star.position_ξηζ for star in self.sample]
            )[self.stars_monte_carlo]
            self.velocities_xyz = np.array(
                [star.velocity_xyz for star in self.sample]
            )[self.stars_monte_carlo]
            self.velocities_ξηζ = np.array(
                [star.velocity_ξηζ for star in self.sample]
            )[self.stars_monte_carlo]

            # Set principal component analysis
            if self.series.pca:
                self.set_principal_component_analysis(version=1)

            # Compute empirical covariances metrics
            if self.series.cov_metrics:
                self.set_timer('cov_metrics')
                self.get_covariances()

            # Compute robust covariances metrics
            if self.series.cov_robust_metrics:
                self.set_timer('cov_robust_metrics')
                self.get_covariances_robust()

            # Compute sklearn covariances metrics
            if self.series.cov_sklearn_metrics:
                self.set_timer('cov_sklearn_metrics')
                self.get_covariances_sklearn()

            # Compute median absolute deviation metrics
            if self.series.mad_metrics:
                self.set_timer('mad_metrics')
                self.get_median_absolute_deviation()

            # Compute minimum spanning tree metrics
            if self.series.mst_metrics:
                self.set_timer('mst_metrics')
                self.get_minimum_spanning_tree()
            self.set_timer()

    def set_principal_component_analysis(self, version):
        """
        Redefines the components of XYZ and ξηζ positions and velocities as the first, second and
        third principal components by rotating the positions and velocities axes.
        """

        # Principal component analysis (First version)
        if version==1:
            def rotate(positions, velocities):
                """
                Rotates positions and velocities axes along the main components of the positions.
                """

                # Rotation matrix computation
                a = positions - np.mean(positions, axis=0)
                a = np.tile(a.T, (a.shape[-1], 1, 1, 1, 1))
                b = np.swapaxes(a, 0, 1)
                covariances_matrix = np.mean(a * b, axis=4).T
                eigen_values, eigen_vectors = np.linalg.eig(covariances_matrix)
                rotation_matrix = np.linalg.inv(eigen_vectors)[None]

                # Use the same rotation at all times
                for i in range(1, rotation_matrix.shape[2]):
                    rotation_matrix[:,:,i] = rotation_matrix[:,:,0]

                # Rotation positions and velocities
                return (
                    np.squeeze(np.matmul(positions[:,:,:,None], rotation_matrix)),
                    np.squeeze(np.matmul(velocities[:,:,:,None], rotation_matrix))
                )

        # Rotate along principal components
        self.positions_xyz, self.velocities_xyz = rotate(
            self.positions_xyz, self.velocities_xyz
        )
        self.positions_ξηζ, self.velocities_ξηζ = rotate(
            self.positions_ξηζ, self.velocities_ξηζ
        )

        # Principal component analysis (Second version)
        if version==2:
            a = np.array([star.position_xyz for star in self.sample]) - self.position_xyz
            a = np.tile(a.T, (a.shape[-1], 1, 1, 1))
            b = np.swapaxes(a, 0, 1)
            covariances_matrix = np.mean(a * b, axis=3).T
            eigen_values, eigen_vectors = np.linalg.eig(covariances_matrix)
            rotation_matrix = np.linalg.inv(eigen_vectors)

            # Rotate star positions
            for star in self:
                star.position_xyz = np.squeeze(
                    np.matmul(
                        star.position_xyz[:,None], rotation_matrix
                    )
                )
                star.position_xyz_other = np.squeeze(
                    np.matmul(
                        np.swapaxes(rotation_matrix, 1, 2), star.position_xyz[:,:,None]
                        )
                    )

        # Compute eigen values association size metric
        self.Metric(
            self, 'eigen_values', np.repeat(
                (np.sum(eigen_values**2, axis=1)**0.5)[None],
                self.series.number_of_iterations, axis=0
            )
        )

    def get_covariances(self):
        """
        Computes the empirical xyz and ξηζ position covariances matrices, and xyz and ξηζ
        position and velocity cross covariances matrices. The age of the group is then
        estimated by finding the epoch when the covariances, determinant or trace of the
        covariances matrices are minimal.
        """

        # xyz position covariances matrix, determinant and trace ages
        self.covariances_xyz_matrix = self.get_covariances_matrix('positions_xyz')
        self.get_covariances_metrics(
            self.covariances_xyz_matrix, self.covariances_xyz,
            self.covariances_xyz_matrix_det, self.covariances_xyz_matrix_trace
        )

        # xyz position and velocity cross covariances matrix, determinant and trace ages
        self.cross_covariances_xyz_matrix = self.get_covariances_matrix(
            'positions_xyz', 'velocities_xyz'
        )
        self.get_covariances_metrics(
            self.cross_covariances_xyz_matrix, self.cross_covariances_xyz,
            self.cross_covariances_xyz_matrix_det, self.cross_covariances_xyz_matrix_trace
        )

        # Update progress bar
        self.series.progress_bar.update(1)

        # ξηζ position covariances matrix, determinant and trace ages
        self.covariances_ξηζ_matrix = self.get_covariances_matrix('positions_ξηζ')
        self.get_covariances_metrics(
            self.covariances_ξηζ_matrix, self.covariances_ξηζ,
            self.covariances_ξηζ_matrix_det, self.covariances_ξηζ_matrix_trace
        )

        # ξηζ position and velocity cross covariances matrix, determinant and trace ages
        self.cross_covariances_ξηζ_matrix = self.get_covariances_matrix(
            'positions_ξηζ', 'velocities_ξηζ'
        )
        self.get_covariances_metrics(
            self.cross_covariances_ξηζ_matrix, self.cross_covariances_ξηζ,
            self.cross_covariances_ξηζ_matrix_det, self.cross_covariances_ξηζ_matrix_trace
        )

        # Update progress bar
        self.series.progress_bar.update(1)

    def get_covariances_robust(self):
        """
        Computes the robust xyz and ξηζ position covariances matrices, and xyz and ξηζ position
        and velocity cross covariances matrices by giving each star a different weight based
        on the Mahalanobis distance computed from the covariance and cross covariances matrices.
        The age of the group is then estimated by finding the epoch when the variances,
        determinant or trace of the covariances matrices are minimal.
        """

        # xyz position robust covariances matrix and determinant ages
        self.Mahalanobis_xyz = self.get_Mahalanobis_distance(
            'positions_xyz', covariances_matrix=self.covariances_xyz_matrix
        )
        self.covariances_xyz_matrix_robust = self.get_covariances_matrix(
            'positions_xyz', robust=True, Mahalanobis_distance=self.Mahalanobis_xyz
        )
        self.get_covariances_metrics(
            self.covariances_xyz_matrix_robust, self.covariances_xyz_robust,
            self.covariances_xyz_matrix_det_robust, self.covariances_xyz_matrix_trace_robust
        )

        # Update progress bar
        self.series.progress_bar.update(2)

        # xyz position and velocity robust cross covariances matrix and determinant ages
        self.Mahalanobis_cross_xyz = self.get_Mahalanobis_distance(
            'positions_xyz', 'velocities_xyz',
            covariances_matrix=self.cross_covariances_xyz_matrix
        )
        self.cross_covariances_xyz_matrix_robust = self.get_covariances_matrix(
            'positions_xyz', 'velocities_xyz',
            robust=True, Mahalanobis_distance=self.Mahalanobis_xyz
        )
        self.get_covariances_metrics(
            self.cross_covariances_xyz_matrix_robust,
            self.cross_covariances_xyz_robust,
            self.cross_covariances_xyz_matrix_det_robust,
            self.cross_covariances_xyz_matrix_trace_robust
        )

        # Update progress bar
        self.series.progress_bar.update(2)

        # ξηζ position robust covariances matrix, determinant and trace ages
        self.Mahalanobis_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', covariances_matrix=self.covariances_ξηζ_matrix
        )
        self.covariances_ξηζ_matrix_robust = self.get_covariances_matrix(
            'positions_ξηζ', robust=True, Mahalanobis_distance=self.Mahalanobis_ξηζ
        )
        self.get_covariances_metrics(
            self.covariances_ξηζ_matrix_robust, self.covariances_ξηζ_robust,
            self.covariances_ξηζ_matrix_det_robust, self.covariances_ξηζ_matrix_trace_robust
        )

        # Update progress bar
        self.series.progress_bar.update(2)

        # ξηζ position and velocity robust cross covariances matrix and determinant ages
        self.Mahalanobis_cross_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', 'velocities_ξηζ',
            covariances_matrix=self.cross_covariances_ξηζ_matrix
        )
        self.cross_covariances_ξηζ_matrix_robust = self.get_covariances_matrix(
            'positions_ξηζ', 'velocities_ξηζ',
            robust=True, Mahalanobis_distance=self.Mahalanobis_ξηζ
        )
        self.get_covariances_metrics(
            self.cross_covariances_ξηζ_matrix_robust,
            self.cross_covariances_ξηζ_robust,
            self.cross_covariances_ξηζ_matrix_det_robust,
            self.cross_covariances_ξηζ_matrix_trace_robust
        )

        # Update progress bar
        self.series.progress_bar.update(2)

        # xyz Mahalanobis distance mean and median
        self.mahalanobis_xyz_mean(np.mean(self.Mahalanobis_xyz, axis=0))
        self.mahalanobis_xyz_median(np.median(self.Mahalanobis_xyz, axis=0))

        # ξηζ Mahalanobis distance mean and median
        self.mahalanobis_ξηζ_mean(np.mean(self.Mahalanobis_ξηζ, axis=0))
        self.mahalanobis_ξηζ_median(np.median(self.Mahalanobis_ξηζ, axis=0))

        # Update progress bar
        self.series.progress_bar.update(1)

    def get_covariances_sklearn(self):
        """
        Computes the xyz and ξηζ position covariances matrix using the sklearn package. The
        age of the group is then estimated by finding the epoch when the variances, determinant
        or trace of the matrix are minimal.
        """

        # xyz position sklearn covariances matrix, determinant and trace ages
        self.covariances_xyz_matrix_sklearn = self.get_covariances_matrix(
            'position_xyz', sklearn=True
        )
        self.get_covariances_metrics(
            self.covariances_xyz_matrix_sklearn, self.covariances_xyz_sklearn,
            self.covariances_xyz_matrix_det_sklearn, self.covariances_xyz_matrix_trace_sklearn
        )

        # ξηζ position sklearn covariances matrix, determinant and trace ages
        self.covariances_ξηζ_matrix_sklearn = self.get_covariances_matrix(
            'position_ξηζ', sklearn=True
        )
        self.get_covariances_metrics(
            self.covariances_ξηζ_matrix_sklearn, self.covariances_ξηζ_sklearn,
            self.covariances_ξηζ_matrix_det_sklearn, self.covariances_ξηζ_matrix_trace_sklearn
        )

    def get_covariances_matrix(
        self, a, b=None, robust=False, sklearn=False, Mahalanobis_distance=None
    ):
        """
        Computes the covariances matrix with variable weights of a Star parameter, 'a', along
        all physical dimensions for all timestep. If another Star parameter, 'b', is given, the
        cross covariances matrix is computed instead. If 'robust' is True, a robust estimator
        is computed using weights from the Mahalanobis distance. If 'sklearn' is True, a robust
        covariance estimator with sklearn's minimum covariance determinant (MCD).
        """

        # Sklearn covariance estimator
        if sklearn:
            a = np.swapaxes(np.array([vars(star)[a] for star in self.sample]), 0, 1)
            covariances_matrix = []
            support_fraction = []
            for step in range(self.series.number_of_steps):
                MCD = MinCovDet(assume_centered=False).fit(a[step])
                covariances_matrix.append(MCD.covariance_)
                support_fraction.append(MCD.support_)

                # Update progress bar
                if step % 5 == 0:
                    self.series.progress_bar.update(1)

            return np.repeat(
                np.array(covariances_matrix)[None],
                self.series.number_of_iterations, axis=0
            )

        # Weights from the Mahalanobis distance
        else:
            if robust:
                if Mahalanobis_distance is None:
                    Mahalanobis_distance = self.get_Mahalanobis_distance(a, b)
                Mahalanobis_distance = np.repeat(Mahalanobis_distance[:,:,:,None], 3, axis=3)

                # Weights based on the Mahalanobis distance
                weights = np.exp(-2 * Mahalanobis_distance)
                weights = np.tile(weights.T, (weights.shape[-1], 1, 1, 1, 1))

            # Covariances matrix
            a = vars(self)[a] - np.mean(vars(self)[a], axis=0)
            # a_weights = np.exp(-2. * (a / np.std(a, axis=0))**2)
            a = np.tile(a.T, (a.shape[-1], 1, 1, 1, 1))
            if b is None:
                b = np.swapaxes(a, 0, 1)

            # Cross covariances matrix
            else:
                b = vars(self)[b] - np.mean(vars(self)[b], axis=0)
                # b_weights = np.exp(-2. * (b / np.std(b, axis=0))**2)
                b = np.swapaxes(np.tile(b.T, (b.shape[-1], 1, 1, 1, 1)), 0, 1)

            return (
                np.average(a * b, weights=weights, axis=4).T if robust
                else np.mean(a * b, axis=4).T
            )

    def get_Mahalanobis_distance(self, a, b=None, covariances_matrix=None):
        """
        Computes the Mahalanobis distances using the covariances matrix of a of all stars in
        the group for all jackknife iterations.
        """

        # Compute the covariances matrix and inverse covariances matrix
        if covariances_matrix is None:
            covariances_matrix = self.get_covariances_matrix(a, b)
        covariances_matrix_invert = np.repeat(
            np.linalg.inv(covariances_matrix)[None],
            self.number_of_stars_iteration, axis=0
        )

        # Compute Mahalanobis distances
        if b is None:
            b = a
        c = (vars(self)[a] - np.mean(vars(self)[a], axis=0))[:,:,:,:,None]
        # d = (vars(self)[b] - np.mean(vars(self)[b], axis=0))[:,:,:,:,None]
        # d = c
        # c = ((
        #     (vars(self)[a] - np.mean(vars(self)[a], axis=0)) +
        #     (vars(self)[b] - np.mean(vars(self)[b], axis=0))) / 2)[:,:,:,:,None]

        return np.sqrt(
            np.abs(
                np.squeeze(
                    np.matmul(np.swapaxes(c, 3, 4), np.matmul(covariances_matrix_invert, c)),
                    axis=(3, 4)
                )
            )
        )

    def get_covariances_metrics(
        self, covariances_matrix, covariances,
        covariances_matrix_det, covariances_matrix_trace
    ):
        """Computes the covariances, and the determinant and trace of the covariances matrix."""

        # Covariances
        covariances(np.abs(covariances_matrix[:, :, (0, 1, 2), (0, 1, 2)])**0.5)

        # Covariances matrix determinant
        covariances_matrix_det(np.abs(np.linalg.det(covariances_matrix))**(1 / 6))
        # covariances_matrix_det(np.abs(np.linalg.det(covariances_matrix))**(1 / 2))

        # Covariances matrix trace
        covariances_matrix_trace(np.abs(np.trace(covariances_matrix, axis1=2, axis2=3) / 3)**0.5)

    def get_median_absolute_deviation(self):
        """
        Computes the xyz and ξηζ, partial and total median absolute deviations (MAD) of a group
        and their respective errors for all timesteps. The age of the group is then estimated
        by finding the epoch when the median absolute deviation is minimal.
        """

        # xyz median absolute deviation ages
        self.mad_xyz(
            np.median(
                np.abs(self.positions_xyz - np.median(self.positions_xyz, axis=0)), axis=0
            )
        )
        self.mad_xyz_total(np.sum(self.mad_xyz.values**2, axis=2)**0.5)

        # ξηζ median absolute deviation ages
        self.mad_ξηζ(
            np.median(
                np.abs(self.positions_ξηζ - np.median(self.positions_ξηζ, axis=0)), axis=0
            )
        )
        self.mad_ξηζ_total(np.sum(self.mad_ξηζ.values**2, axis=2)**0.5)

        # Update progress bar
        self.series.progress_bar.update(1)

    def get_minimum_spanning_tree(self):
        """
        Builds the minimum spanning trees (MST) of a group for all timesteps using a Kruskal
        algorithm, and computes the average branch length and the branch length median absolute
        absolute deviation of branch length. The age of the association is then estimated by
        finding the epoch when these values are minimal.
        """

        # Set the number of minimum spanning tree branches
        self.number_of_branches = self.sample_size - 1
        self.number_of_branches_monte_carlo = (
            int(self.sample_size * self.series.iteration_fraction) - 1
        )

        # Create branches for all possible pairs of stars
        self.branches = []
        branch_indices = []
        for start, end in combinations(range(self.sample_size), 2):
            self.branches.append(self.Branch(self.sample[start], self.sample[end]))
            branch_indices.append((start, end))
        self.branches = np.array(self.branches, dtype=object)
        branch_indices = np.array(branch_indices, dtype=int)

        # Find the corresponding branches for every jackknife Monte Carlo iteration
        self.branches_monte_carlo = (
            np.all(
                np.any(
                    branch_indices[:,None,:,None] == self.stars_monte_carlo.T[None,:,None,:],
                    axis=3
                ), axis=2
            )
        )

        # Find xyz position and velocity minimum spanning tree branches
        self.get_branches('position', 'xyz')
        self.get_branches('velocity', 'xyz')

        # xyz minimum spanning tree average branch length and median absolute deviation ages
        if self.series.mst_metrics:
            self.mst_xyz_mean(np.mean(self.mst_position_xyz_values, axis=0))
            self.mst_xyz_mean_robust(
                np.average(
                    self.mst_position_xyz_values, weights=self.mst_position_xyz_weights, axis=0
                )
            )
            self.mst_xyz_mad(
                np.median(
                    np.abs(
                        self.mst_position_xyz_values -
                        np.median(self.mst_position_xyz_values, axis=0)
                    ), axis=0
                )
            )

        # Find ξηζ position and velocity minimum spanning tree branches
        self.get_branches('position', 'ξηζ')
        self.get_branches('velocity', 'ξηζ')

        # ξηζ minimum spanning tree average branch length and median absolute deviation ages
        if self.series.mst_metrics:
            self.mst_ξηζ_mean(np.mean(self.mst_position_ξηζ_values, axis=0))
            self.mst_ξηζ_mean_robust(
                np.average(
                    self.mst_position_ξηζ_values, weights=self.mst_position_ξηζ_weights, axis=0
                )
            )
            self.mst_ξηζ_mad(
                np.median(
                    np.abs(
                        self.mst_position_ξηζ_values -
                        np.median(self.mst_position_ξηζ_values, axis=0)
                    ), axis=0
                )
            )

            # xyz minimum spanning tree 90 percentile
            mst_position_xyz_values_sorted = np.sort(self.mst_position_xyz_values, axis=0)
            mst_position_xyz_values_sorted90 = mst_position_xyz_values_sorted[
                int(self.number_of_branches * 0.2):int(self.number_of_branches * 0.80) - 1,:
            ]
            self.mst_xyz_90(np.mean(mst_position_xyz_values_sorted90, axis=0))

    def get_branches(self, coord, system):
        """
        Computes the branches, values and weights of a minimum spanning tree for every timestep
        and jackknife Monte Carlo iteration. The full minimum spanning tree is also computed for
        every timestep.
        """

        # Sort branches by coordinate for every timestep
        branch_coord = {'position': 'length', 'velocity': 'speed'}
        value = f'{branch_coord[coord]}_{system}'
        branches_order = np.argsort(
            np.array([vars(branch)[value] for branch in self.branches], dtype=float), axis=0
        ).T
        branches_sorted = self.branches[branches_order]
        branches_monte_carlo_sorted = np.moveaxis(
            self.branches_monte_carlo[branches_order], -1, 0
        )

        # Initialize minimum spanning trees, values and weights
        mst = np.empty((self.series.number_of_steps, self.number_of_branches), dtype=object)
        if self.series.mst_metrics and coord == 'position':
            mst_monte_carlo = np.empty(
                (
                    self.series.number_of_iterations,
                    self.series.number_of_steps,
                    self.number_of_branches_monte_carlo
                ), dtype=object
            )
            mst_values = np.zeros(mst_monte_carlo.shape)
            mst_weights = np.zeros(mst_monte_carlo.shape)

        # Validate branches for every timestep
        for step in range(self.series.number_of_steps):
            mst[step] = self.validate_branches(branches_sorted[step], self.number_of_branches)

            # Validate branches for every jackknife Monte Carlo iteration
            if self.series.mst_metrics and coord == 'position':
                for iteration in range(self.series.number_of_iterations):
                    mst_monte_carlo[iteration, step] = self.validate_branches(
                        branches_sorted[step, branches_monte_carlo_sorted[iteration, step]],
                        self.number_of_branches_monte_carlo
                    )
                    mst_values[iteration, step] = np.array(
                        [vars(branch)[value][step] for branch in mst_monte_carlo[iteration, step]]
                    )
                    mst_weights[iteration, step] = np.array(
                        [branch.weight[step] for branch in mst_monte_carlo[iteration, step]]
                    )

                # Update progress bar
                if step % 6 == 0:
                    self.series.progress_bar.update(1)

        # Set minimum spanning tree, values and weights
        vars(self)[f'mst_{coord}_{system}'] = mst
        if self.series.mst_metrics and coord == 'position':
            vars(self)[f'mst_monte_carlo_{coord}_{system}'] = mst_monte_carlo
            vars(self)[f'mst_{coord}_{system}_values'] = mst_values.T.swapaxes(1, 2)
            vars(self)[f'mst_{coord}_{system}_weights'] = mst_weights.T.swapaxes(1, 2)

    def validate_branches(self, branches, number_of_branches):
        """
        Validate a list sorted branches using a Kruskal algorithm up to a given number of branches.
        Returns a list of validated branches.
        """

        # Set stars nodes
        for star in self.sample:
            star.node = self.Node()

        # Initialize minimum spanning tree, and test and valid indices
        mst = np.empty(number_of_branches, dtype=object)
        test_index = 0
        valid_index = 0

        # Branches verification and addition to tree
        while valid_index < number_of_branches:
            branch = branches[test_index]
            test_index += 1

            # Find branch start and end stars largest parent node
            while branch.start.node.parent != None:
                branch.start.node = branch.start.node.parent
            while branch.end.node.parent != None:
                branch.end.node = branch.end.node.parent

            # Validate branch if both stars have different parent nodes
            if branch.start.node != branch.end.node:
                branch.start.node.parent = branch.end.node.parent = self.Node()
                mst[valid_index] = branch
                valid_index += 1

        return mst

    class Branch:
        """Line connecting two stars used for the calculation of the minimum spanning tree."""

        def __init__(self, start, end):
            """
            Initializes a Branch object and computes the distance between two Star objects,
            'start' and 'end', for all timestep.
            """

            # Set start and end branches
            self.start = start
            self.end = end

            # Compute branch lengths, speeds and weigth
            self.length_xyz = np.sum(
                (self.start.position_xyz - self.end.position_xyz)**2, axis=1
            )**0.5
            self.length_ξηζ = np.sum(
                (self.start.position_ξηζ - self.end.position_ξηζ)**2, axis=1
            )**0.5
            self.speed_xyz = np.sum(
                (self.start.velocity_xyz - self.end.velocity_xyz)**2, axis=1
            )**0.5
            self.speed_ξηζ = np.sum(
                (self.start.velocity_ξηζ - self.end.velocity_ξηζ)**2, axis=1
            )**0.5
            self.weight = np.mean(np.vstack((self.start.weight, self.end.weight)), axis=0)

        def __repr__(self):
            """Returns a string of name of the branch."""

            return "'{}' to '{}' branch".format(self.start.name, self.end.name)

    class Node:
        """Node of a star."""

        def __init__(self):
            """Sets the parent node of a star as None."""

            self.parent = None

        def __repr__(self):
            """Returns a string of name of the parent."""

            return 'None' if self.parent is None else self.parent

    class Star(Output_Star):
        """Parameters and methods of a star in a moving group."""

        def __init__(self, group, **values):
            """
            Initializes a Star object with at least a name, integration times, initial veloctiy,
            velocity error, initial position and position error. More values can be added (when
            initializing from a file for instance). If a traceback is needed, the star's positions
            and velocities over time is computed with a linear approximation or galpy, in both xyz
            and ξηζ coordinate systems. The initial position and velocity corresponds to the first
            timestep in the integration time array, even if it is nonzero.
            """

            # Initialization
            self.group = group
            self.outlier = False
            self.subsample = False
            vars(self).update(values)

            # Trajectory integration
            if self.group.series.potential is None:
                self.get_linear()
            else:
                self.get_orbit()

            # !!! Temporary xyz position and velocity errors setting !!!
            # self.position_xyz_error = np.repeat(
            #     self.position_xyz_error[None], self.group.series.number_of_steps, axis=0
            # )
            # self.velocity_xyz_error = np.repeat(
            #     self.velocity_xyz_error[None], self.group.series.number_of_steps, axis=0
            # )

        def set_outlier(self, position_outlier, velocity_outlier):
            """Sets the star as an outlier."""

            self.outlier = True
            self.position_outlier = position_outlier
            self.velocity_outlier = velocity_outlier

        def get_linear(self):
            """Computes a star's backward or forward linear trajectory in space."""

            # xyz positions and velocities
            self.position_xyz = (
                self.position_xyz + (self.time - self.time[0])[:,None] * (
                    self.velocity_xyz + Coordinate.sun_velocity
                )
            )
            self.velocity_xyz = np.repeat(self.velocity_xyz[None], self.time.size, axis=0)
            self.distance_xyz = np.sum(self.position_xyz**2, axis=1)**0.5
            self.speed_xyz = np.sum(self.velocity_xyz**2, axis=1)**0.5

            # rθz positions and velocities
            position_rθz = np.array(
                coords.XYZ_to_galcencyl(
                    *self.position_xyz.T,
                    Xsun=Coordinate.sun_position[0],
                    Zsun=Coordinate.sun_position[2],
                    _extra_rot=False
                )
            )
            velocity_rθz = np.array(
                coords.vxvyvz_to_galcencyl(
                    *self.velocity_xyz.T,
                    *np.array(
                        coords.XYZ_to_galcenrect(
                            *self.position_xyz.T,
                            Xsun=Coordinate.sun_position[0],
                            Zsun=Coordinate.sun_position[2],
                            _extra_rot=False
                        )
                    ).T,
                    vsun=Coordinate.sun_velocity,
                    Xsun=Coordinate.sun_position[0],
                    Zsun=Coordinate.sun_position[2],
                    _extra_rot=False
                )
            )

            # rθz positions and velocities
            # position_rθz = galactic_xyz_galactocentric_ρθz(*self.position_xyz)
            # velocity_rθz = galactic_uvw_galactocentric_ρθz(*self.position_xyz, *self.velocity_xyz)

            # ξηζ positions and velocities
            self.position_ξηζ = position_ρθz_ξηζ(*position_rθz.T, self.time)
            self.velocity_ξηζ = velocity_ρθz_ξηζ(*velocity_rθz.T)
            self.distance_ξηζ = np.sum(self.position_ξηζ**2, axis=1)**0.5
            self.speed_ξηζ = np.sum(self.velocity_ξηζ**2, axis=1)**0.5

        def get_orbit(self):
            """Computes the star's galactic orbit using Galpy over time."""

            # Initial rθz position and velocity
            position_rθz = galactic_xyz_galactocentric_ρθz(*self.position_xyz)
            velocity_rθz = galactic_uvw_galactocentric_ρθz(*self.position_xyz, *self.velocity_xyz)

            # Conversion to Galpy's natural units
            time = self.time * Coordinate.lsr_velocity[1] / Coordinate.sun_position[0]
            position_rθz /= np.array([Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])
            velocity_rθz /= Coordinate.lsr_velocity[1]

            # Orbit integration
            orbit = Orbit(
                [
                    position_rθz[0], velocity_rθz[0], velocity_rθz[1],
                    position_rθz[2], velocity_rθz[2], position_rθz[1]
                ]
            )
            orbit.turn_physical_off()
            orbit.integrate(time, self.group.series.potential, method='odeint')

            # rθz positions and velocities, and conversion to physical units
            position_rθz = np.array(
                [orbit.R(time), orbit.phi(time), orbit.z(time)]
            ).T * np.array([Coordinate.sun_position[0], 1., Coordinate.sun_position[0]])
            velocity_rθz = np.array(
                [orbit.vR(time), orbit.vT(time), orbit.vz(time)]
            ).T * Coordinate.lsr_velocity[1]

            # xyz positions and velocities
            self.position_xyz = galactocentric_ρθz_galactic_xyz(*position_rθz.T)
            self.velocity_xyz = galactocentric_ρθz_galactic_uvw(*position_rθz.T, *velocity_rθz.T)
            self.distance_xyz = np.sum(self.position_xyz**2, axis=1)**0.5
            self.speed_xyz = np.sum(self.velocity_xyz**2, axis=1)**0.5

            # ξηζ positions and velocities
            self.position_ξηζ = position_ρθz_ξηζ(*position_rθz.T, self.time)
            self.velocity_ξηζ = velocity_ρθz_ξηζ(*velocity_rθz.T)
            self.distance_ξηζ = np.sum(self.position_ξηζ**2, axis=1)**0.5
            self.speed_ξηζ = np.sum(self.velocity_ξηζ**2, axis=1)**0.5

        def get_relative_coordinates(self):
            """
            Computes relative position and velocity, the distance and the speed from the
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
            """Returns a string of name of the star."""

            return self.name

    def set_timer(self, operation=None):
        """Records the operation time."""

        # Add the timer of the previous operation
        if self.series.timer:
            if 'previous_operation' in vars(self):
                if self.previous_operation is not None:

                    # Compute operation time
                    operation_time = get_time() - self.previous_time
                    if self.previous_operation in self.series.timers.keys():
                        self.series.timers[self.previous_operation] += operation_time
                    else:
                        self.series.timers[self.previous_operation] = operation_time

            # Save preivous operation and time
            self.previous_operation = operation
            self.previous_time = get_time()
