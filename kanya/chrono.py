# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""chrono.py: Defines the methods used to compute the age of the group."""

import numpy as np
from tqdm import tqdm
from sklearn.covariance import MinCovDet
from itertools import combinations

class Chrono():
    """Contains the methods used to compute the age of the group."""

    class Metric():
        """
        Association size metric including its average values, age error, age at minimum and
        minimum.
        """

        def __init__(self, group, metric):
            """Initializes an average association size metric."""

            # Initialization
            self.group = group
            self.label = eval(metric['label'])
            self.name = np.array(eval(metric['name']))
            self.latex_short = np.array(eval(metric['latex_short']))
            self.latex_long = np.array(eval(metric['latex_long']))
            self.units = np.array(eval(metric['units']))
            self.order = int(metric['order'])
            self.valid = np.array(eval(metric['valid']))
            self.age_shift = np.array(eval(metric['age_shift']))
            self.status = False
            self.ndim = self.valid.size

            # Add the association size metric to the gorup
            vars(self.group)[self.label] = self
            self.group.metrics.append(self)

        def __call__(self):
            """
            Computes the values and errors of an association size metric. If the group has
            been initialized from data, the first group (index 0) is used for values, age and
            mininum. Other groups are used to compute uncertainty due to measurement errors.
            If only one group is present, the internal error is used as the total error.
            """

            # Average association size metric for stars from data
            if self.status and self.group.from_data:

                # Value and errors
                self.value = vars(self.group[0])[self.label].value
                self.value_int_error = vars(self.group[0])[self.label].value_int_error
                self.value_ext_error = (
                    np.std([vars(group)[self.label].value for group in self.group[1:]], axis=0)
                    if self.group.number_of_groups > 1 else np.zeros(self.value.shape)
                )
                self.values = (
                    np.array([vars(group)[self.label].values for group in self.group[1:]])
                    if self.group.number_of_groups > 1
                    else vars(self.group[0])[self.label].values[None]
                )
                self.value_error = np.std(self.values, axis=(0, 1))
                self.value_error_quad = (self.value_int_error**2 + self.value_ext_error**2)**0.5

                # Age and errors
                self.age = vars(self.group[0])[self.label].age
                self.age_int_error = vars(self.group[0])[self.label].age_int_error
                self.age_ext_error = (
                    np.std([vars(group)[self.label].age for group in self.group[1:]], axis=0)
                    if self.group.number_of_groups > 1 else np.zeros(self.age.shape)
                )
                self.ages = (
                    np.array([vars(group)[self.label].ages for group in self.group[1:]])
                    if self.group.number_of_groups > 1
                    else vars(self.group[0])[self.label].ages[None]
                )
                self.ages += self.age - np.mean(self.ages, axis=(0, 1))[None,None,:]
                self.age_error = np.std(self.ages, axis=(0, 1))
                self.age_error_quad = (self.age_int_error**2 + self.age_ext_error**2)**0.5
                # self.ages = self.group.time[np.argmin(self.values, axis=2)]
                # self.ages += self.age - np.mean(self.ages, axis=(0, 1))[None,None,:]
                # self.age_error = np.std(self.ages, axis=(0, 1))

                # Minimum and errors
                self.min = vars(self.group[0])[self.label].min
                self.min_int_error = vars(self.group[0])[self.label].min_int_error
                self.min_ext_error = (
                    np.std([vars(group)[self.label].min for group in self.group[1:]], axis=0)
                    if self.group.number_of_groups > 1 else np.zeros(self.min.shape)
                )
                self.minima = (
                    np.array([vars(group)[self.label].minima for group in self.group[1:]])
                    if self.group.number_of_groups > 1
                    else vars(self.group[0])[self.label].minima[None]
                )
                self.min_error = np.std(self.minima, axis=(0, 1))
                self.min_error_quad = (self.min_int_error**2 + self.min_ext_error**2)**0.5

                # Age shift based on models
                self.age_adjusted = self.age - self.age_shift

                # Minimum change
                self.min_change = (self.min / self.value[0] - 1.) * 100.

            # Average association size metric for stars from a model
            elif self.status and self.group.from_model:
                self.values = np.mean(
                    [vars(group)[self.label].values for group in self.group], axis=0
                )

                # Value and errors
                self.value = np.mean(
                    [vars(group)[self.label].value for group in self.group], axis=0
                )
                self.value_int_error = np.mean(
                    [vars(group)[self.label].value_int_error for group in self.group], axis=0
                )
                self.value_ext_error = np.std(
                    [vars(group)[self.label].value for group in self.group], axis=0
                )
                self.values = np.array([vars(group)[self.label].values for group in self.group])
                self.value_error = np.std(self.values, axis=(0, 1))
                self.value_error_quad = (self.value_int_error**2 + self.value_ext_error**2)**0.5

                # Age and errors
                self.age = np.mean(
                    [vars(group)[self.label].age for group in self.group], axis=0
                )
                self.age_int_error = np.mean(
                    [vars(group)[self.label].age_int_error for group in self.group], axis=0
                )
                self.age_ext_error = np.std(
                    [vars(group)[self.label].age for group in self.group], axis=0
                )
                self.ages = np.array([vars(group)[self.label].ages for group in self.group])
                self.age_error = np.std(self.ages, axis=(0, 1))
                self.age_error_quad = (self.age_int_error**2 + self.age_ext_error**2)**0.5

                # Minimum and errors
                self.min = np.mean(
                    [vars(group)[self.label].min for group in self.group], axis=0
                )
                self.min_int_error = np.mean(
                    [vars(group)[self.label].min_int_error for group in self.group], axis=0
                )
                self.min_ext_error = np.std(
                    [vars(group)[self.label].min for group in self.group], axis=0
                )
                self.minima = np.array([vars(group)[self.label].minima for group in self.group])
                self.min_error = np.std(self.minima, axis=(0, 1))
                self.min_error_quad = (self.min_int_error**2 + self.min_ext_error**2)**0.5

                # Age shift based on models
                self.age_shift = self.group.age.value - self.age
                self.age_adjusted = self.group.age.value

                # Minimum change
                self.min_change = (self.min / self.value[0] - 1.) * 100.

            # Null average association size metric
            else:
                null_1d = np.full((self.ndim,), np.nan)
                null_2d = np.full((self.group.number_of_steps, self.ndim), np.nan)
                null_3d = np.full(
                    (
                        self.group.number_of_groups - (1 if self.group.from_data else 0),
                        self.group.number_of_iterations,
                        self.ndim
                    ), np.nan
                )
                null_4d = np.full(
                    (
                        self.group.number_of_groups - (1 if self.group.from_data else 0),
                        self.group.number_of_iterations,
                        self.group.number_of_steps,
                        self.ndim
                    ), np.nan
                )

                # Value and errors
                self.value = null_2d
                self.value_int_error = null_2d
                self.value_ext_error = null_2d
                self.values = null_4d
                self.value_error = null_2d
                self.value_error_quad = null_2d

                # Age and errors
                self.age = null_1d
                self.age_int_error = null_1d
                self.age_ext_error = null_1d
                self.ages = null_3d
                self.age_error = null_1d
                self.age_error_quad = null_1d

                # Minimum and errors
                self.min = null_1d
                self.min_int_error = null_1d
                self.min_ext_error = null_1d
                self.minima = null_3d
                self.min_error = null_1d
                self.min_error_quad = null_1d

                # Age shift based on models
                self.age_shift = null_1d
                self.age_adjusted = null_1d

                # Minimum change
                self.min_change = null_1d

            # Box size (Myr) converted to the corresponding number of steps
            # box_size = 1. # Transform in to parameter
            # box_size = 0.01
            # box_size = int(
            #     box_size * self.group.number_of_steps /
            #     (self.group.final_time.value - self.group.initial_time.value)
            # )
            # if box_size > 1:
            #     box = np.squeeze(np.tile(
            #         np.ones(box_size) / box_size, (1 if self.values.ndim == 1 else 3, 1)
            #     )).T
            #     box = np.ones(box_size) / box_size

                # Smoothing with moving average
                # if self.values.ndim == 1:
                #     self.values = np.convolve(self.values, box, mode='same')
                # else:
                #     self.values = np.apply_along_axis(
                #         lambda x: np.convolve(x, box, mode='same'), axis=0, arr=self.values
                #     )

    class Metric_Group():
        """Association size metric including its values, ages, errors, and minima."""

        def __init__(self, group, metric):
            """Initializes the association size metric."""

            # Initialization
            self.group = group
            self.metric = metric

            # Add the metric to the group
            vars(self.group)[self.metric.label] = self
            self.group.metrics.append(self)

        def __call__(self, values, value=None):
            """
            Computes values, errors using a jackknife Monte Carlo method, ages by finding the
            epoch of minimal association size, and minima. The value is computed by averaging
            jackknife Monte Carlo iterations or by using the given value.
            """

            # Average values and errors
            self.values = np.atleast_3d(values)
            self.value = np.mean(self.values, axis=0) if value is None else np.atleast_2d(value).T
            self.value_int_error = np.std(self.values, axis=0)

            # Average age at the minimal value and error
            self.ages = self.group.time[np.argmin(self.values, axis=1)]
            self.age = self.group.time[np.argmin(self.value, axis=0)]
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

    def chronologize(
        self, size_metrics=None, cov_metrics=None, cov_robust_metrics=None,
        cov_sklearn_metrics=None, mad_metrics=None, tree_metrics=None, logging=True
    ):
        """
        Computes the kinematic age of the group by finding the epoch of minimal association
        size using several association size metrics, saved in 5D arrays (number_of_groups,
        number_in_core, number_of_iterations, number_of_steps + 1, 1 or 3).
        """

        # Set association size metric parameters
        for parameter in (
            'size_metrics', 'cov_metrics', 'cov_robust_metrics',
            'cov_sklearn_metrics', 'mad_metrics', 'tree_metrics'
        ):
            if locals()[parameter] is None:
                vars(self)[parameter] = self.set_boolean(vars(self.config)[parameter])
            else:
                vars(self)[parameter] = parameter
                self.check_type(vars(self)[parameter], parameter, 'boolean')

        # Set number of iterations parameter
        self.number_of_iterations = self.set_integer(self.config.number_of_iterations)

        # Check if number of iterations is greater than 0
        self.stop(
            self.number_of_iterations <= 0, 'ValueError',
            "'number_of_iterations' must be greater to 0 ({} given).", self.number_of_iterations
        )

        # Set iteration fraction parameter
        self.iteration_fraction = self.set_quantity(self.config.iteration_fraction).value

        # Check if iteration fraction is between 0 and 1
        self.stop(
            self.iteration_fraction <= 0 or self.iteration_fraction > 1, 'ValueError',
            "'iteration_fraction' must be between 0 and 1 ({} given).", self.iteration_fraction
        )

        # Set principal component analysis parameter
        self.pca = self.set_boolean(self.config.pca)

        # Association size metrics logging and set progress bar
        if self.size_metrics:
            self.log(
                "Computing size metrics of '{}' group.",
                self.name, display=True, logging=logging
            )
            self.progress_bar = tqdm(
                total=self.number_of_groups * sum(
                    [
                        2 * self.cov_metrics,
                        9 if self.cov_robust_metrics else 0,
                        (self.number_of_steps // 5 + 1) * 2 if self.cov_sklearn_metrics else 0,
                        self.mad_metrics,
                        (self.number_of_steps // 6 + 1) * 2 if self.tree_metrics else 0
                    ]
                ), unit=' size metric', bar_format=(
                    '{desc}{percentage:3.0f}% |{bar}| '
                    ' {elapsed} {remaining} '
                )
            )

            # Set logging message
            for group in self:
                message = f"Computing size metrics of '{group.name}' group"

                # Association size metrics computation logging and progress bar
                self.log(f'{message}.', display=False, logging=logging)
                self.progress_bar.set_description(desc=message, refresh=True)

                # Compute association size metrics
                group.chronologize()

            # Association size metrics logging and set progress bar
            message = f"Size metrics of '{self.name}' group succesfully computed"
            self.log(f'{message}.', display=False, logging=logging)
            self.progress_bar.set_description(message, refresh=True)
            self.progress_bar.close()
            del self.progress_bar

            # Compute average association size metrics
            for metric in self.metrics:
                metric()

        # Logging Should add a flag to see if the age has been computed. The traceback
        # is a requirement for the chronoligization, not a reason to stop.
        # else:
        #     self.log(
        #         "Kinematic age could not be computed because '{}' group has not been traced back.",
        #         self.name, level='info', display=True, logging=logging
        #     )

    def chronologize_Group(self):
        """
        Computes the kinematic age of the group by finding the epoch of minimal association size
        using several size metrics. Parameters used to compute association size metrics, including
        weights based on Mahalanobis distances computed with empirical covariances matrices.
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
                    self.sample_size * self.iteration_fraction
                )
                self.stars_monte_carlo = np.array(
                    [
                        np.random.choice(
                            self.sample_size, self.number_of_stars_iteration, replace=False
                        ) for i in range(self.number_of_iterations)
                    ]
                ).T

            # For other groups, use the same stars for the jackknife Monte Carlo as the first group
            else:
                self.number_of_stars_iteration = self.number_of_stars_iteration
                self.stars_monte_carlo = self.stars_monte_carlo

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
            if self.pca:
                self.set_principal_component_analysis(version=1)

            # Compute empirical covariances metrics
            if self.cov_metrics:
                self.set_timer('cov_metrics')
                self.get_covariances()

            # Compute robust covariances metrics
            if self.cov_robust_metrics:
                self.set_timer('cov_robust_metrics')
                self.get_covariances_robust()

            # Compute sklearn covariances metrics
            if self.cov_sklearn_metrics:
                self.set_timer('cov_sklearn_metrics')
                self.get_covariances_sklearn()

            # Compute median absolute deviation metrics
            if self.mad_metrics:
                self.set_timer('mad_metrics')
                self.get_median_absolute_deviation()

            # Compute tree branches metrics
            if self.tree_metrics:
                self.set_timer('tree_metrics')
                self.get_tree_branches()
            self.set_timer()

    def set_principal_component_analysis(self, version):
        """
        Redefines the components of XYZ and ξηζ positions and velocities as the first, second
        and third principal components by rotating the positions and velocities axes.
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
                self.number_of_iterations, axis=0
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
        self.progress_bar.update(1)

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
        self.progress_bar.update(1)

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
        self.progress_bar.update(2)

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
        self.progress_bar.update(2)

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
        self.progress_bar.update(2)

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
        self.progress_bar.update(2)

        # xyz Mahalanobis distance mean and median
        self.mahalanobis_xyz_mean(np.mean(self.Mahalanobis_xyz, axis=0))
        self.mahalanobis_xyz_median(np.median(self.Mahalanobis_xyz, axis=0))

        # ξηζ Mahalanobis distance mean and median
        self.mahalanobis_ξηζ_mean(np.mean(self.Mahalanobis_ξηζ, axis=0))
        self.mahalanobis_ξηζ_median(np.median(self.Mahalanobis_ξηζ, axis=0))

        # Update progress bar
        self.progress_bar.update(1)

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

    def get_covariances_matrix_old(
        self, a, b=None, robust=False, sklearn=False, Mahalanobis_distance=None
    ):
        """
        Computes the covariances matrix with variable weights of a Star parameter, 'a', along all
        physical dimensions. If another Star parameter, 'b', is given, the cross covariances matrix
        is computed instead. If 'robust' is True, a robust estimator is computed using weights from
        the Mahalanobis distance. If 'sklearn' is True, a robust covariance estimator with sklearn's
        minimum covariance determinant (MCD).
        """

        # Sklearn covariance estimator
        if sklearn:
            a = np.swapaxes(np.array([vars(star)[a] for star in self.sample]), 0, 1)
            covariances_matrix = []
            support_fraction = []
            for step in range(self.number_of_steps):
                MCD = MinCovDet(assume_centered=False).fit(a[step])
                covariances_matrix.append(MCD.covariance_)
                support_fraction.append(MCD.support_)

                # Update progress bar
                if step % 5 == 0:
                    self.progress_bar.update(1)

            return np.repeat(
                np.array(covariances_matrix)[None],
                self.number_of_iterations, axis=0
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
        Computes the Mahalanobis distances using the covariances matrix of every stars in the group.
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
        and their respective errors. The age of the group is then estimated by finding the epoch
        when the median absolute deviation is minimal.
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
        self.progress_bar.update(1)

    def get_tree_branches(self):
        """
        Finds every tree branches connecting all stars in the group, and computes the position
        and velocity, xyz and ξηζ, full and minimum spanning trees (MSTs). The age of the group
        is then estimated by finding the epoch when the mean, robust mean, and median absolute
        deviation tree branch lengths are minimal.
        """

        # Set the number of tree branches
        self.number_of_branches = self.sample_size - 1
        self.number_of_branches_monte_carlo = (
            int(self.sample_size * self.iteration_fraction) - 1
        )

        # Create branches and find branch indices for every possible pairs of stars
        self.branches = []
        branch_indices = []
        for start, end in combinations(range(self.sample_size), 2):
            self.branches.append(self.Branch(self.sample[start], self.sample[end]))
            branch_indices.append((start, end))
        self.branches = np.array(self.branches, dtype=object)
        branch_indices = np.array(branch_indices, dtype=int)

        # Find corresponding branches for jackknife Monte Carlo
        self.branches_monte_carlo = (
            np.all(
                np.any(
                    branch_indices[:,None,:,None] == self.stars_monte_carlo.T[None,:,None,:],
                    axis=3
                ), axis=2
            )
        )

        # Find xyz position and velocity full trees
        self.get_full_tree('position', 'xyz')
        self.get_full_tree('velocity', 'xyz')

        # Find xyz position and velocity full trees
        self.get_full_tree('position', 'ξηζ')
        self.get_full_tree('velocity', 'ξηζ')

        # Find xyz position and velocity minimum spanning trees
        self.get_minimum_spanning_tree('position', 'xyz')
        self.get_minimum_spanning_tree('velocity', 'xyz')

        # Find ξηζ position and velocity minimum spanning trees
        self.get_minimum_spanning_tree('position', 'ξηζ')
        self.get_minimum_spanning_tree('velocity', 'ξηζ')

    def get_full_tree(self, coord, system):
        """
        Finds the values and weights of the full tree, and computes mean, robust mean, and median
        absolute absolute deviation branch lengths ages.
        """

        # Select coordinates
        branch_coord = {'position': 'length', 'velocity': 'speed'}
        value = f'{branch_coord[coord]}_{system}'

        # Initialize full tree branch values and weights, and remove branches over 1σ in length
        branches_value = np.array([vars(branch)[value] for branch in self.branches]).T
        branches_weight = (
            (branches_value - np.mean(branches_value, axis=1)[:,None]) /
            np.std(branches_value, axis=1)[:,None]
        ) < 1.0

        # Initialize full tree branch values and weights for jackknife Monte Carlo, and remove
        # branches over 1σ in length
        branches_values = np.array(
            [
                [
                    vars(branch)[value]
                    for branch in self.branches[self.branches_monte_carlo[:,iteration]]
                ] for iteration in range(self.number_of_iterations)
            ]
        ).swapaxes(1, 2)
        branches_weights = (
            (branches_values - np.mean(branches_values, axis=2)[:,:,None]) /
            np.std(branches_values, axis=2)[:,:,None]
        ) < 1.0

        # Mean, robust mean, and median absolute deviation full tree branch lengths ages
        if self.tree_metrics and coord == 'position':
            vars(self)[f'branches_{system}_mean'](
                np.mean(branches_values, axis=2), np.mean(branches_value, axis=1)
            )
            vars(self)[f'branches_{system}_mean_robust'](
                np.average(branches_values, weights=branches_weights, axis=2),
                np.average(branches_value, weights=branches_weight, axis=1)
            )
            vars(self)[f'branches_{system}_mad'](
                np.median(
                    np.abs(branches_values - np.median(branches_values, axis=2)[:,:,None]), axis=2
                ), np.median(
                    np.abs(branches_value - np.median(branches_value, axis=1)[:,None]), axis=1
                )
            )

    def get_minimum_spanning_tree(self, coord, system):
        """
        Finds branches, values, and weights of the minimum spanning tree (MST), and computes mean,
        robust mean, and median absolute absolute deviation branch lengths ages.
        """

        # Select coordinates
        branch_coord = {'position': 'length', 'velocity': 'speed'}
        value = f'{branch_coord[coord]}_{system}'

        # Sort branches by coordinate
        branches_order = np.argsort(
            np.array([vars(branch)[value] for branch in self.branches], dtype=float), axis=0
        ).T
        branches_sorted = self.branches[branches_order]
        branches_monte_carlo_sorted = np.moveaxis(self.branches_monte_carlo[branches_order], -1, 0)

        # Initialize minimum spanning tree branches, values and weights
        mst = vars(self)[f'mst_{coord}_{system}'] = np.empty(
            (self.number_of_steps, self.number_of_branches), dtype=object
        )
        mst_value = np.zeros(mst.shape)
        mst_weight = np.zeros(mst.shape)

        # Validate minimum spanning tree branches
        for step in range(self.number_of_steps):
            mst[step] = self.validate_branches(branches_sorted[step], self.number_of_branches)
            mst_value[step] = np.array([vars(branch)[value][step] for branch in mst[step]])
            mst_weight[step] = np.array([branch.weight[step] for branch in mst[step]])

        # Initialize minimum spanning tree branches, values and weights for jackknife Monte Carlo
        if self.tree_metrics and coord == 'position':
            mst_monte_carlo = vars(self)[f'mst_monte_carlo_{coord}_{system}'] = np.empty(
                (
                    self.number_of_iterations,
                    self.number_of_steps,
                    self.number_of_branches_monte_carlo
                ), dtype=object
            )
            mst_values = np.zeros(mst_monte_carlo.shape)
            mst_weights = np.zeros(mst_monte_carlo.shape)

            # Validate minimum spanning tree branches for jackknife Monte Carlo
            for step in range(self.number_of_steps):
                for iteration in range(self.number_of_iterations):
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
                    self.progress_bar.update(1)

            # Mean, robust mean, and median absolute deviation minimum spanning tree
            # branch lengths ages
            vars(self)[f'mst_{system}_mean'](
                np.mean(mst_values, axis=2), np.mean(mst_value, axis=1)
            )
            vars(self)[f'mst_{system}_mean_robust'](
                np.average(mst_values, weights=mst_weights, axis=2),
                np.average(mst_value, weights=mst_weight, axis=1)
            )
            vars(self)[f'mst_{system}_mad'](
                np.median(np.abs(mst_values - np.median(mst_values, axis=2)[:,:,None]), axis=2),
                np.median(np.abs(mst_value - np.median(mst_value, axis=1)[:,None]), axis=1)
            )

    def validate_branches(self, branches, number_of_branches):
        """
        Validate a list sorted branches using a Kruskal algorithm up to a given number of branches
        for minimum spanning tree (MST) computation. Returns a list of validated branches.
        """

        # Set stars nodes
        for star in self.sample:
            star.node = self.Node()

        # Initialize the minimum spanning tree and test and validate branches
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
        """Line connecting two stars used for the calculation of the tree."""

        def __init__(self, start, end):
            """
            Initializes a Branch object and computes the distance between two Star objects,
            'start' and 'end'.
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
