# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""chrono.py: Defines the methods used to compute the age of the group."""

import numpy as np
from tqdm import tqdm
from sklearn.covariance import MinCovDet
from itertools import combinations

class Chrono():
    """
    Contains the methods used to compute the age of the group. This chronoligization is done
    by minimizing by finding the epoch when several association size metrics is minimal. This
    process, and the computation of the errors the traceback ages, is done within the Metric
    subclass. Only members flagged as part of the core of the group are used in the computation
    of the association size metrics, including:

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
                            4 * self.cov_metrics,
                            9 if self.cov_robust_metrics else 0,
                            (self.number_of_steps // 5 + 1) * 2 if self.cov_sklearn_metrics else 0,
                            2 * self.mad_metrics,
                            (self.number_of_steps // 6 + 1) * 2 if self.tree_metrics else 0
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

        # ??? These steps could be done in the Traceback class, only if robust is selected ???
        # Compute the xyz Mahalanobis distances
        self.number_of_stars_iteration = 1
        self.Mahalanobis_xyz = self.get_Mahalanobis_distance(self.position_xyz[0, self.core, None])

        # Compute weights
        self.weights = np.exp(-self.Mahalanobis_xyz)

        # Randomly select stars for the jackknife Monte Carlo
        self.number_of_stars_iteration = int(
            self.number_in_core * self.fraction_jackknife
        )
        self.stars_jackknife = np.array(
            [
                np.random.choice(
                    self.number_in_core, self.number_of_stars_iteration, replace=False
                ) for i in range(self.number_jackknife)
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
                        vars(self)[f'{label}_monte_carlo'] = (
                            vars(self)[label][1:, self.core][:, self.stars_jackknife]
                        )
                    else:
                        vars(self)[f'{label}_monte_carlo'] = (
                            vars(self)[f'{label}_0'][None][:, self.stars_jackknife]
                        )

                # Select stars from jackknife iterations
                elif self.from_model:
                    vars(self)[f'{label}_monte_carlo'] = (
                        vars(self)[label][:, self.stars_jackknife]
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
        Finds every tree branches connecting all stars in the group, and computes the position
        and velocity, xyz and ξηζ, full and minimum spanning trees (MSTs). The age of the group
        is then estimated by finding the epoch when the mean, robust mean, and median absolute
        deviation tree branch lengths are minimal.
        """

        # Set the number of tree branches
        self.number_of_branches = self.sample_size - 1
        self.number_of_branches_monte_carlo = (
            int(self.sample_size * self.fraction_jackknife) - 1
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
                    branch_indices[:,None,:,None] == self.stars_jackknife.T[None,:,None,:],
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
                    for branch in self.branches[self.branches_monte_carlo[:, iteration]]
                ] for iteration in range(self.number_jackknife)
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
                    self.number_jackknife,
                    self.number_of_steps,
                    self.number_of_branches_monte_carlo
                ), dtype=object
            )
            mst_values = np.zeros(mst_monte_carlo.shape)
            mst_weights = np.zeros(mst_monte_carlo.shape)

            # Validate minimum spanning tree branches for jackknife Monte Carlo
            for step in range(self.number_of_steps):
                for iteration in range(self.number_jackknife):
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
        """Line connecting two stars."""

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
