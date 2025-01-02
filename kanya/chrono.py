# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chrono.py: Defines the methods used to compute the age of the group by finding the epoch when
several association size metrics are minimal.
"""

import numpy as np
from tqdm import tqdm
from itertools import combinations
import multiprocessing as mp
from .tools import Iterate

class Chrono():
    """
    Contains the methods used to compute the age of the group. This chronoligization is done
    by finding the epoch when several association size metrics is minimal. This process, and the
    computation of the errors the traceback ages, is done within the Metric subclass. Only members
    flagged as part of the core of the group are used in the computation of the association size
    metrics, including:

      - Empirical, robust, and sklearn's spatial covariance matrix diagonal, determinant and trace
      - Empirical, robust, and sklearn's spatial-kinematic cross-covariance matrix diagonal,
        determinant and trace
      - Median absolute deviation
      - Minimum spanning tree and full tree mean, robust mean, and median absolute deviation of
        branch lengths
    """

    class Metric():
        """
        Association size metric, including its associated values, ages, errors, and minima.
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

            # Add the association size metric to the group
            vars(self.group)[self.label] = self
            self.group.metrics.append(self)

        def __call__(self, value, values):
            """
            Computes the values, errors, ages and minima of an association size metric. If the
            group has been initialized from data, the value is used to compute the age and mininum,
            and Monte Carlo values are used only to compute uncertainties. For models, the value
            is None, and it is instead computed by averaging the Monte Carlo values. Ages are
            computed by finding the epochs that minimizes the values (i.e. the association sizes).
            """

            # Find values and compute errors
            self.value = value if value is not None else np.mean(values, axis=(0, 1))
            self.value = self.value[..., None] if self.value.ndim == 1 else self.value
            self.values = values[..., None] if values.ndim == 3 else values
            self.value_int_error = np.mean(np.std(self.values, axis=1), axis=0)
            self.value_ext_error = np.mean(np.std(self.values, axis=0), axis=0)
            self.value_error = np.std(self.values, axis=(0, 1))
            self.value_error_quad = (self.value_int_error**2 + self.value_ext_error**2)**0.5

            # Find ages and compute errors
            self.age = self.group.time[np.argmin(self.value, axis=-2)]
            self.ages = self.group.time[np.argmin(self.values, axis=-2)]
            self.age_int_error = np.mean(np.std(self.ages, axis=1), axis=0)
            self.age_ext_error = np.mean(np.std(self.ages, axis=0), axis=0)
            self.age_error = np.std(self.ages, axis=(0, 1))
            self.age_error_quad = (self.age_int_error**2 + self.age_ext_error**2)**0.5

            # Find minima and compute errors
            self.min = np.min(self.value, axis=-2)
            self.mins = np.min(self.values, axis=-2)
            self.min_int_error = np.mean(np.std(self.mins, axis=1), axis=0)
            self.min_ext_error = np.mean(np.std(self.mins, axis=0), axis=0)
            self.min_error = np.std(self.mins, axis=(0, 1))
            self.min_error_quad = (self.min_int_error**2 + self.min_ext_error**2)**0.5

            # Compute adjusted ages
            self.age_adjusted = self.age - self.age_shift
            self.ages_adjusted = self.ages - self.age_shift

            # Compute changes at minima
            self.min_change = (self.min / self.value[0] - 1.0) * 100.0
            self.mins_change = (self.mins / self.values[..., 0, :] - 1.0) * 100.0

            # Set status
            self.status = True

        def __repr__(self):
            """Creates a string with the name of the Metric, its age and error."""

            return '\n'.join(
                [
                    f'{self.name[i]}: {self.age[i]:.1f} ± {self.age_error[i]:.1f} Myr'
                    for i in range(self.name.size)
                ]
            )

        def set_null(self):
            """Set the parameters of the association size metric as null."""

            # Null association size metric
            if not self.status:
                null_1d = np.full((self.ndim,), np.nan)
                null_2d = np.full((self.group.number_of_steps, self.ndim), np.nan)
                null_3d = np.full(
                    (self.group.number_monte_carlo, self.group.number_jackknife, self.ndim), np.nan
                )
                null_4d = np.full(
                    (
                        self.group.number_monte_carlo, self.group.number_jackknife,
                        self.group.number_of_steps, self.ndim
                    ), np.nan
                )

                # Set null values and errors
                self.value = null_2d
                self.values = null_4d
                self.value_int_error = null_2d
                self.value_ext_error = null_2d
                self.value_error = null_2d
                self.value_error_quad = null_2d

                # Set null ages and errors
                self.age = null_1d
                self.ages = null_3d
                self.age_int_error = null_1d
                self.age_ext_error = null_1d
                self.age_error = null_1d
                self.age_error_quad = null_1d

                # Set null minima and errors
                self.min = null_1d
                self.mins = null_3d
                self.min_int_error = null_1d
                self.min_ext_error = null_1d
                self.min_error = null_1d
                self.min_error_quad = null_1d

                # Set null adjusted age
                self.age_adjusted = null_1d
                self.ages_adjusted = null_3d

                # Set changes at minima
                self.min_change = null_1d
                self.mins_change = null_3d

        def smooth(self):
            """Smooths the values with a convolution."""

            # Box size (Myr) converted to the corresponding number of steps
            box_size = 0.1
            box_size = int(
                box_size * self.group.number_of_steps /
                (self.group.final_time.value - self.group.initial_time.value)
            )
            if box_size > 1:
                box = np.squeeze(np.tile(np.ones(box_size) / box_size, (self.ndim, 1))).T
                box = np.ones(box_size) / box_size

                # Smoothing with moving average
                if self.ndim == 1:
                    self.values = np.convolve(self.values, box, mode='same')
                else:
                    self.values = np.apply_along_axis(
                        lambda x: np.convolve(x, box, mode='same'), axis=0, arr=self.values
                    )

    def chronologize(
        self, size_metrics=None, cov_metrics=None, cov_robust_metrics=None,
        cov_sklearn_metrics=None, mad_metrics=None, tree_metrics=None, logging=True
    ):
        """
        Computes the kinematic age of the group (i.e. chronologize) by finding the epoch of
        minimal association size using several association size metrics, saved in 4D arrays
        (number_monte_carlo, number_jackknife, number_of_steps + 1, 1 or 3). Parameters used
        to compute association size metrics, including weights based on Mahalanobis distances,
        are computed with the empirical covariance matrices.
        """

        # Start process pool
        mp.set_start_method('fork', force=True)
        self.pool = mp.Pool()

        # Choose behaviour, if groups have already been chronologized
        if self.traceback_present:
            if self.chrono_present:
                forced = self.choose(
                    f"'{self.name}' group has already been chronologized.", 1, forced
                )

                # Delete groups
                if forced:
                    self.chrono_present = False
                    self.log(
                        "Existing '{}' group will be overwritten.",
                        self.name, logging=logging
                    )
                    self.chronologize(logging=logging)

                # Cancel chrono
                else:
                    self.log(
                        "'{}' group was not chronologized because it has "
                        "already been chronologized.", self.name, logging=logging
                    )

            # Configure chrono
            else:
                self.configure_chrono(
                    size_metrics=size_metrics, cov_metrics=cov_metrics,
                    cov_robust_metrics=cov_robust_metrics, cov_sklearn_metrics=cov_sklearn_metrics,
                    mad_metrics=mad_metrics, tree_metrics=tree_metrics, logging=logging
                )

            # Logging
            if self.size_metrics:
                self.log(
                    "Computing size metrics of '{}' group.",
                    self.name, display=True, logging=logging
                )

                # Set progress bar
                self.progress_bar = tqdm(
                    total=sum(
                        [
                            4 * self.cov_metrics, 9 if self.cov_robust_metrics else 0,
                            2 * self.number_of_iterations * (self.number_of_steps // 5 + 1)
                            if self.cov_sklearn_metrics else 0, 2 * self.mad_metrics,
                            self.number_monte_carlo * self.number_jackknife * 2 + 18
                            if self.tree_metrics else 0
                        ]
                    ), leave=False
                    # unit=' size metric',
                    # bar_format=(
                    #     '{desc}{percentage:3.0f}% |{bar}| '
                    #     ' {elapsed} {remaining} '
                    # ),
                )

                # Logging and progress bar
                # message = f"Computing size metrics of '{self.name}' group"
                # self.log(f'{message}.', display=False, logging=logging)
                # self.progress_bar.set_description(desc=message, refresh=True)

                # Set principal component analysis
                if self.pca:
                    self.set_principal_component_analysis(version=1)

                # Compute empirical covariance metrics
                if self.cov_metrics:
                    self.set_timer('cov_metrics')
                    self.get_covariance()

                # Compute robust covariance metrics
                if self.cov_robust_metrics:
                    self.set_timer('cov_robust_metrics')
                    self.get_covariance_robust()

                # Compute sklearn covariance metrics
                if self.cov_sklearn_metrics:
                    self.set_timer('cov_sklearn_metrics')
                    self.get_covariance_sklearn()

                # Compute median absolute deviation metrics
                if self.mad_metrics:
                    self.set_timer('mad_metrics')
                    self.get_median_absolute_deviation()

                # Compute tree branches metrics
                if self.tree_metrics:
                    self.set_timer('tree_metrics')
                    self.get_tree_branches()
                self.set_timer()

                # Set unused metrics as null
                for metric in self.metrics:
                    metric.set_null()

                # Set chrono as present
                self.chrono_present = True

                # Logging and progress bar
                # message = f"Size metrics of '{self.name}' group succesfully computed"
                # self.log(f'{message}.', display=False, logging=logging)
                # self.progress_bar.set_description(message, refresh=True)

                # Delete progress bar
                self.progress_bar.close()
                del self.progress_bar

                # Logging
                self.log(
                    "Size metrics of '{}' group succesfully computed.",
                    self.name, display=True, logging=logging
                )

            # Logging
            else:
                self.log(
                    "'{}' group was not chronologized because no size metrics were selected.",
                    self.name, level='info', display=True, logging=logging
                )

        # Logging
        else:
            self.log(
                "'{}' group was not chronologized because it has not been traced back.",
                self.name, level='info', display=True, logging=logging
            )

        # Close process pool
        self.pool.close()

    def configure_chrono(
        self, size_metrics=None, cov_metrics=None, cov_robust_metrics=None,
        cov_sklearn_metrics=None, mad_metrics=None, tree_metrics=None, logging=True
    ):
        """Checks chrono configuration parameters."""

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

        # Set number of jackknife parameter
        self.number_jackknife = self.set_integer(self.config.number_jackknife, 0)

        # Set the jackknife fraction parameter
        self.fraction_jackknife = self.set_quantity(self.config.fraction_jackknife).value

        # Check if the jackknife fraction is between 0 and 1
        self.stop(
            self.fraction_jackknife <= 0 or self.fraction_jackknife > 1, 'ValueError',
            "'fraction_jackknife' must be between 0 and 1 ({} given).", self.fraction_jackknife
        )

        # Set the number of jackknife iterations and the jackknife fraction
        # if the number of jackknife iterations is null
        if self.number_jackknife == 0:
            self.number_jackknife = 1
            self.fraction_jackknife = 1

        # Set principal component analysis parameter
        self.pca = self.set_boolean(self.config.pca)

        # Compute the xyz Mahalanobis distances
        # !!! New distances for every Monte Carlo and jackknife iteration, and xyz and ξηζ !!!
        # !!! positions and velocity !!!
        self.number_of_stars_iteration = 1
        self.Mahalanobis_xyz = self.get_Mahalanobis_distance(self.position_xyz[0, self.core, None])

        # Compute weights
        self.weights = np.exp(-self.Mahalanobis_xyz)

        # Compute the number of stars per jackknife Monte Carlo iteration
        self.number_of_stars_iteration = int(
            self.number_in_core * self.fraction_jackknife
        )

        # Randomly select stars for the jackknife Monte Carlo
        self.stars_monte_carlo = np.array(
            [
                [
                    np.random.choice(
                        self.number_in_core, self.number_of_stars_iteration, replace=False
                    ) for i in range(self.number_jackknife)
                ] for j in range(self.number_monte_carlo)
            ]
        )

        # Select coordinate and system
        for coord in ('position', 'velocity'):
            for system in ('xyz', 'ξηζ'):
                label = f'{coord}_{system}'

                # Select stars from the core and jackknife iterations
                if self.from_data:
                    vars(self)[f'{label}_0'] = vars(self)[label][0, self.core]
                    if self.number_monte_carlo > 0:
                        vars(self)[f'{label}_monte_carlo'] = np.take_along_axis(
                            vars(self)[label][1:, None, self.core],
                            self.stars_monte_carlo[..., None, None], axis=2
                        )
                    else:
                        vars(self)[f'{label}_monte_carlo'] = np.take_along_axis(
                            vars(self)[f'{label}_0'][None],
                            self.stars_monte_carlo[..., None, None], axis=2
                        )

                # Select stars from jackknife iterations
                elif self.from_model:
                    vars(self)[f'{label}_monte_carlo'] = (
                        vars(self)[label][:, self.stars_monte_carlo]
                    )

        # Logging
        self.log("'{}' group ready to be chronologized.", self.name, logging=logging)

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
                covariance_matrix = np.mean(a * b, axis=4).T
                eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
                rotation_matrix = np.linalg.inv(eigen_vectors)[None]

                # Use the same rotation at all times
                for i in range(1, rotation_matrix.shape[2]):
                    rotation_matrix[:,:,i] = rotation_matrix[:,:,0]

                # Rotation positions and velocities
                return (
                    np.squeeze(np.matmul(positions[:,:,:,None], rotation_matrix)),
                    np.squeeze(np.matmul(velocities[:,:,:,None], rotation_matrix)),
                    eigen_values
                )

        # Rotate along principal components
        self.position_xyz, self.velocity_xyz, eigen_values_xyz = rotate(
            self.position_xyz, self.velocity_xyz
        )
        self.position_ξηζ, self.velocity_ξηζ, eigen_values_ξηζ = rotate(
            self.position_ξηζ, self.velocity_ξηζ
        )

        # Principal component analysis (Second version)
        if version==2:
            a = np.array([star.position_xyz for star in self.sample]) - self.position_xyz
            a = np.tile(a.T, (a.shape[-1], 1, 1, 1))
            b = np.swapaxes(a, 0, 1)
            covariance_matrix = np.mean(a * b, axis=3).T
            eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
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
                self.number_jackknife, axis=0
            )
        )

    def get_covariance(self):
        """
        Computes the empirical xyz and ξηζ position covariance matrices, and xyz and ξηζ
        position and velocity cross covariance matrices. The age of the group is then
        estimated by finding the epoch when the covariance, determinant or trace of the
        covariance matrices are minimal.
        """

        # Select system
        for system in ('xyz', 'ξηζ'):

            # Position covariance matrix, determinant and trace ages
            vars(self)[f'covariance_{system}_matrix'] = self.get_covariance_matrix(
                vars(self)[f'position_{system}_0']
            ) if self.from_data else None
            vars(self)[f'covariance_{system}_matrix_monte_carlo'] = self.get_covariance_matrix(
                np.moveaxis(vars(self)[f'position_{system}_monte_carlo'], 2, 0)
            )
            self.get_covariance_metrics('covariance', system)

            # Update progress bar
            self.progress_bar.update(1)

            # Position and velocity cross covariance matrix, determinant and trace ages
            vars(self)[f'cross_covariance_{system}_matrix'] = self.get_covariance_matrix(
                vars(self)[f'position_{system}_0'], vars(self)[f'velocity_{system}_0']
            ) if self.from_data else None
            vars(self)[f'cross_covariance_{system}_matrix_monte_carlo'] = self.get_covariance_matrix(
                np.moveaxis(vars(self)[f'position_{system}_monte_carlo'], 2, 0),
                np.moveaxis(vars(self)[f'velocity_{system}_monte_carlo'], 2, 0)
            )
            self.get_covariance_metrics('cross_covariance', system)

            # Update progress bar
            self.progress_bar.update(1)

    def get_covariance_robust(self):
        """
        Computes the robust xyz and ξηζ position covariance matrices, and xyz and ξηζ position
        and velocity cross covariance matrices by giving each star a different weight based
        on the Mahalanobis distance computed from the covariance and cross covariance matrices.
        The age of the group is then estimated by finding the epoch when the variances,
        determinant or trace of the covariance matrices are minimal.
        """

        # xyz position robust covariance matrix and determinant ages
        self.Mahalanobis_xyz = self.get_Mahalanobis_distance(
            'positions_xyz', covariance_matrix=self.covariance_xyz_matrix
        )
        self.covariance_xyz_matrix_robust = self.get_covariance_matrix(
            'positions_xyz', robust=True, Mahalanobis_distance=self.Mahalanobis_xyz
        )
        self.get_covariance_metrics(
            self.covariance_xyz_matrix_robust, self.covariance_xyz_robust,
            self.covariance_xyz_matrix_det_robust, self.covariance_xyz_matrix_trace_robust
        )

        # Update progress bar
        self.progress_bar.update(2)

        # xyz position and velocity robust cross covariance matrix and determinant ages
        self.Mahalanobis_cross_xyz = self.get_Mahalanobis_distance(
            'positions_xyz', 'velocities_xyz',
            covariance_matrix=self.cross_covariance_xyz_matrix
        )
        self.cross_covariance_xyz_matrix_robust = self.get_covariance_matrix(
            'positions_xyz', 'velocities_xyz',
            robust=True, Mahalanobis_distance=self.Mahalanobis_xyz
        )
        self.get_covariance_metrics(
            self.cross_covariance_xyz_matrix_robust,
            self.cross_covariance_xyz_robust,
            self.cross_covariance_xyz_matrix_det_robust,
            self.cross_covariance_xyz_matrix_trace_robust
        )

        # Update progress bar
        self.progress_bar.update(2)

        # ξηζ position robust covariance matrix, determinant and trace ages
        self.Mahalanobis_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', covariance_matrix=self.covariance_ξηζ_matrix
        )
        self.covariance_ξηζ_matrix_robust = self.get_covariance_matrix(
            'positions_ξηζ', robust=True, Mahalanobis_distance=self.Mahalanobis_ξηζ
        )
        self.get_covariance_metrics(
            self.covariance_ξηζ_matrix_robust, self.covariance_ξηζ_robust,
            self.covariance_ξηζ_matrix_det_robust, self.covariance_ξηζ_matrix_trace_robust
        )

        # Update progress bar
        self.progress_bar.update(2)

        # ξηζ position and velocity robust cross covariance matrix and determinant ages
        self.Mahalanobis_cross_ξηζ = self.get_Mahalanobis_distance(
            'positions_ξηζ', 'velocities_ξηζ',
            covariance_matrix=self.cross_covariance_ξηζ_matrix
        )
        self.cross_covariance_ξηζ_matrix_robust = self.get_covariance_matrix(
            'positions_ξηζ', 'velocities_ξηζ',
            robust=True, Mahalanobis_distance=self.Mahalanobis_ξηζ
        )
        self.get_covariance_metrics(
            self.cross_covariance_ξηζ_matrix_robust,
            self.cross_covariance_ξηζ_robust,
            self.cross_covariance_ξηζ_matrix_det_robust,
            self.cross_covariance_ξηζ_matrix_trace_robust
        )

        # Update progress bar
        self.progress_bar.update(2)

        # ??? Should these even exist ???
        # xyz Mahalanobis distance mean and median
        self.mahalanobis_xyz_mean(np.mean(self.Mahalanobis_xyz, axis=0))
        self.mahalanobis_xyz_median(np.median(self.Mahalanobis_xyz, axis=0))

        # ξηζ Mahalanobis distance mean and median
        self.mahalanobis_ξηζ_mean(np.mean(self.Mahalanobis_ξηζ, axis=0))
        self.mahalanobis_ξηζ_median(np.median(self.Mahalanobis_ξηζ, axis=0))

        # Update progress bar
        self.progress_bar.update(1)

    def get_covariance_sklearn(self):
        """
        Computes the xyz and ξηζ position covariance matrices using the sklearn package. The
        age of the group is then estimated by finding the epoch when the variances, determinant
        or trace of the matrix are minimal.
        """

        # xyz position sklearn covariance matrix, determinant and trace ages
        self.covariance_xyz_matrix_sklearn = self.get_covariance_matrix(
            'position_xyz', sklearn=True
        )
        self.get_covariance_metrics(
            self.covariance_xyz_matrix_sklearn, self.covariance_xyz_sklearn,
            self.covariance_xyz_matrix_det_sklearn, self.covariance_xyz_matrix_trace_sklearn
        )

        # ξηζ position sklearn covariance matrix, determinant and trace ages
        self.covariance_ξηζ_matrix_sklearn = self.get_covariance_matrix(
            'position_ξηζ', sklearn=True
        )
        self.get_covariance_metrics(
            self.covariance_ξηζ_matrix_sklearn, self.covariance_ξηζ_sklearn,
            self.covariance_ξηζ_matrix_det_sklearn, self.covariance_ξηζ_matrix_trace_sklearn
        )

    def get_covariance_metrics(self, metric, system, robust=False, sklearn=False):
        """
        Computes the diagonal terms, the determinant, and trace of the covariance matrix, and the
        association size metrics associated to these values.
        """

        # Find labels and covariance matrices
        label = f'{metric}_{system}'
        method = '_robust' if robust else '_sklearn' if sklearn else ''
        covariance_matrix = vars(self)[f'{label}_matrix{method}']
        covariance_matrix_monte_carlo = vars(self)[f'{label}_matrix_monte_carlo{method}']

        # Covariance diagonal ages
        vars(self)[f'{label}{method}'](
            np.abs(np.diagonal(covariance_matrix, axis1=-2, axis2=-1))**0.5
            if self.from_data else None,
            np.abs(np.diagonal(covariance_matrix_monte_carlo, axis1=-2, axis2=-1))**0.5
        )

        # Covariance matrix determinant age
        vars(self)[f'{label}_matrix_det{method}'](
            np.abs(np.linalg.det(covariance_matrix))**(1 / 6)
            if self.from_data else None,
            np.abs(np.linalg.det(covariance_matrix_monte_carlo))**(1 / 6)
        )

        # Covariance matrix trace age
        vars(self)[f'{label}_matrix_trace{method}'](
            np.abs(np.trace(covariance_matrix, axis1=-2, axis2=-1) / 3)**0.5
            if self.from_data else None,
            np.abs(np.trace(covariance_matrix_monte_carlo, axis1=-2, axis2=-1) / 3)**0.5
        )

    def get_median_absolute_deviation(self):
        """
        Computes the xyz and ξηζ, partial and total median absolute deviations (MAD) of a group
        and their respective errors. The age of the group is then estimated by finding the epoch
        when the median absolute deviation is minimal.
        """

        # Median absolute deviation ages
        for system in ('xyz', 'ξηζ'):
            vars(self)[f'mad_{system}'](
                np.median(
                    np.abs(
                        vars(self)[f'position_{system}_0'] -
                        np.median(vars(self)[f'position_{system}_0'], axis=0)[None]
                    ), axis=0
                ),
                np.median(
                    np.abs(
                        vars(self)[f'position_{system}_monte_carlo'] -
                        np.median(vars(self)[f'position_{system}_monte_carlo'], axis=2)[:, :, None]
                    ), axis=2
                )
            )

            # Total median absolute deviation
            vars(self)[f'mad_{system}_total'](
                np.sum(vars(self)[f'mad_{system}'].value**2, axis=-1)**0.5,
                np.sum(vars(self)[f'mad_{system}'].values**2, axis=-1)**0.5
            )

            # Update progress bar
            self.progress_bar.update(1)

    def get_tree_branches(self):
        """
        Finds every tree branches connecting all stars in the group, computes their xyz and ξηζ
        lengths and speeds, and finds the full and minimum spanning trees (MSTs). The age of the
        group is estimated by finding the epoch when the mean, robust mean, and median absolute
        deviation tree branch lengths are minimal.
        """

        # Find the branch indices for every possible pair of stars in the core
        self.branches = np.array(
            [(start, end) for start, end in combinations(range(self.number_in_core), 2)]
        )

        # Find the corresponding Monte Carlo branch indices
        self.branches_monte_carlo = self.branches[
            np.apply_along_axis(
                np.nonzero, 2, np.all(
                    np.any(
                        self.branches[None, None, ..., None] ==
                        self.stars_monte_carlo[..., None, None, :], axis=-1
                    ), axis=-1
                )
            )[..., 0, :]
        ]

        # Find the xyz and ξηζ position and velocity full and minimum spanning trees
        for coord in ('position', 'velocity'):
            for system in ('xyz', 'ξηζ'):
                self.get_full_tree(coord, system)
                self.get_minimum_spanning_tree(coord, system)

    def get_full_tree(self, coord, system):
        """
        Computes the values and weights of the branches of full tree, and computes the mean,
        robust mean, and median absolute absolute deviation full tree branch lengths ages.
        """

        # Select coordinates
        branch_coord = {'position': 'length', 'velocity': 'speed'}

        # Compute branches values
        values = vars(self)[f'branches_{branch_coord[coord]}_{system}_values'] = np.sum(
            (
                vars(self)[f'{coord}_{system}_0'][self.branches[:, 0]] -
                vars(self)[f'{coord}_{system}_0'][self.branches[:, 1]]
            )**2, axis=-1
        )**0.5

        # Compute branches weights
        weights = vars(self)[
            f'branches_{branch_coord[coord]}_{system}_weights'
        ] = np.ones(values.shape)

        # Update progress bar
        if self.size_metrics and self.tree_metrics:
            self.progress_bar.update(1)

        # Compute Monte Carlo branches values
        if self.size_metrics and self.tree_metrics and coord == 'position':
            values_monte_carlo = vars(self)[
                f'branches_{branch_coord[coord]}_{system}_values_monte_carlo'
            ] = np.sum(
                (
                    np.take_along_axis(
                        vars(self)[f'{coord}_{system}'][1:, None, self.core],
                        self.branches_monte_carlo[..., 0, None, None], axis=2
                    ) - np.take_along_axis(
                        vars(self)[f'{coord}_{system}'][1:, None, self.core],
                        self.branches_monte_carlo[..., 1, None, None], axis=2
                    )
                )**2, axis=-1
            )**0.5

            # Compute Monte Carlo branches weights
            weights_monte_carlo = vars(self)[
                f'branches_{branch_coord[coord]}_{system}_weights_monte_carlo'
            ] = np.ones(values_monte_carlo.shape)

            # Mean, robust mean, and median absolute deviation full tree branch lengths ages
            vars(self)[f'branches_{system}_mean'](
                np.mean(values, axis=0),
                np.mean(values_monte_carlo, axis=2)
            )
            vars(self)[f'branches_{system}_mean_robust'](
                np.average(values, weights=weights, axis=0),
                np.average(values_monte_carlo, weights=weights_monte_carlo, axis=2)
            )
            vars(self)[f'branches_{system}_mad'](
                np.median(
                    np.abs(values - np.median(values, axis=0)[None]), axis=0
                ),
                np.median(
                    np.abs(
                        values_monte_carlo -
                        np.median(values_monte_carlo, axis=2)[..., None, :]
                    ), axis=2
                )
            )

            # Update progress bar
            self.progress_bar.update(1)

    def get_minimum_spanning_tree(self, coord, system):
        """
        Finds the branches, values, and weights of the minimum spanning tree (MST), and computes
        the mean, robust mean, and median absolute absolute deviation minimum spanning tree branch
        lengths ages.
        """

        # Select coordinates
        branch_coord = {'position': 'length', 'velocity': 'speed'}
        values = vars(self)[f'branches_{branch_coord[coord]}_{system}_values']
        weights = vars(self)[f'branches_{branch_coord[coord]}_{system}_weights']

        # Set the number of branches in the minimum spanning tree
        self.number_of_branches = self.number_in_core - 1
        self.number_of_branches_monte_carlo = self.number_of_stars_iteration - 1

        # Sort branches by values
        branches_order = np.argsort(values, axis=0)
        branches_sorted = self.branches[branches_order]

        # Update progress bar
        if self.size_metrics and self.tree_metrics:
            self.progress_bar.update(1)

        # Validate minimum spanning tree's branches
        indices = np.array(
            self.pool.starmap(
                validate_branches, zip(
                    np.moveaxis(branches_sorted, 1, 0),
                    Iterate(self.number_of_branches),
                    Iterate(self.number_in_core)
                )
            )
        ).T

        # Update progress bar
        if self.size_metrics and self.tree_metrics:
            self.progress_bar.update(1)

        # Find minimum spanning tree branches, values and weights
        vars(self)[f'mst_branches_{coord}_{system}'] = np.take_along_axis(
            branches_sorted, indices[..., None], axis=0
        )
        indices = np.take_along_axis(branches_order, indices, axis=0)
        mst_values = np.take_along_axis(values, indices, axis=0)
        mst_weights = np.take_along_axis(weights, indices, axis=0)

        # Select Monte Carlo coordinates
        if self.size_metrics and self.tree_metrics and coord == 'position':
            values_monte_carlo = vars(self)[
                f'branches_{branch_coord[coord]}_{system}_values_monte_carlo'
            ]
            weights_monte_carlo = vars(self)[
                f'branches_{branch_coord[coord]}_{system}_weights_monte_carlo'
            ]

            # Sort branches by Monte Carlo values
            branches_monte_carlo_order = np.argsort(values_monte_carlo, axis=2)
            branches_monte_carlo_sorted = np.take_along_axis(
                self.branches_monte_carlo[..., None, :],
                branches_monte_carlo_order[..., None], axis=2
            )

            # Update progress bar
            self.progress_bar.update(1)

            # Validate Monte Carlo minimum spanning tree's branches
            indices = np.zeros(
                (
                    self.number_monte_carlo, self.number_jackknife,
                    self.number_of_branches_monte_carlo, self.number_of_steps
                ), dtype=int
            )
            for group in range(self.number_monte_carlo):
                for iteration in range(self.number_jackknife):
                    indices[group, iteration] = np.array(
                        self.pool.starmap(
                            func=validate_branches,
                            iterable=zip(
                                np.moveaxis(branches_monte_carlo_sorted[group, iteration], 1, 0),
                                Iterate(self.number_of_branches_monte_carlo),
                                Iterate(self.number_in_core)
                            )
                        )
                    ).T

                    # Update progress bar
                    self.progress_bar.update(1)

            # Find Monte_carlo minimum spanning tree branches, values and weights
            vars(self)[f'mst_branches_{coord}_{system}_monte_carlo'] = np.take_along_axis(
                branches_monte_carlo_sorted, indices[..., None], axis=2
            )
            indices = np.take_along_axis(branches_monte_carlo_order, indices, axis=2)
            mst_values_monte_carlo = np.take_along_axis(values_monte_carlo, indices, axis=2)
            mst_weights_monte_carlo = np.take_along_axis(weights_monte_carlo, indices, axis=2)

            # Mean, robust mean, and median absolute deviation minimum spanning tree
            # branch lengths ages
            vars(self)[f'mst_{system}_mean'](
                np.mean(mst_values, axis=0), np.mean(mst_values_monte_carlo, axis=2)
            )
            vars(self)[f'mst_{system}_mean_robust'](
                np.average(mst_values, weights=mst_weights, axis=0),
                np.average(mst_values_monte_carlo, weights=mst_weights_monte_carlo, axis=2)
            )
            vars(self)[f'mst_{system}_mad'](
                np.median(np.abs(mst_values - np.median(mst_values, axis=0)[None]), axis=0),
                np.median(
                    np.abs(
                        mst_values_monte_carlo -
                        np.median(mst_values_monte_carlo, axis=2)[..., None, :]
                    ), axis=2
                )
            )

            # Update progress bar
            self.progress_bar.update(1)

def validate_branches(branches, number_of_branches, number_in_core):
    """
    Validate a list sorted branches using a Kruskal algorithm up to a given number of branches
    for minimum spanning tree (MST) computation. Returns a list of validated branch indices.
    """

    class Node:
        """Node of a minimum spanning tree."""

        def __init__(self):
            """Sets the parent node as None."""

            self.parent = None

    # Set nodes
    nodes = [Node() for i in range(number_in_core)]

    # Initialize the minimum spanning tree and test and validate branches
    mst = np.zeros(number_of_branches, dtype=int)
    test_index = 0
    valid_index = 0

    # Branches verification and addition to tree
    while valid_index < number_of_branches:
        branch = branches[test_index]

        # Find branch start (0) and end (1) stars largest parent node
        while nodes[branch[0]].parent != None:
            nodes[branch[0]] = nodes[branch[0]].parent
        while nodes[branch[1]].parent != None:
            nodes[branch[1]] = nodes[branch[1]].parent

        # Validate branch if both stars have different parent nodes
        if nodes[branch[0]] != nodes[branch[1]]:
            nodes[branch[0]].parent = nodes[branch[1]].parent = Node()
            mst[valid_index] = test_index
            valid_index += 1
        test_index += 1

    return mst
