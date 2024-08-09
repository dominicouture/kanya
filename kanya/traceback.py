# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""traceback.py: Defines the methods used for the traceback analysis of the group."""

from galpy.orbit import Orbit
from .coordinate import *
from sklearn.covariance import MinCovDet

class Traceback():
    """
    Contains the methods used for the traceback analysis of a group. When tracing back stars from
    data, the first group (index 0) has no Monte Carlo measurement errors applied on positions and
    velocities, and the only difference between the remaining groups is the measurements errors.
    When tracing back stars from models, the initial positions, initial velocities (based on the
    initial position and velocity scatters, the current-day positions and velocities, the age, and
    the number of stars), and the mesurement errors are different all between groups.
    """

    def traceback(self, mode=None, forced=None, logging=True):
        """
        Traces back the Galactic orbit of every star in the group, either using imported data
        or by modeling groups from parameters. Outliers are also removed from the data sample.
        Positions and velocities are saved in 4D arrays (number_monte_carlo, number_of_stars,
        number_of_steps + 1, 3).
        """

        # Choose behaviour, if groups have already been traced back
        if self.traceback_present:
            forced = self.choose(f"'{self.name}' group has already been traced back.", 1, forced)

            # Delete groups
            if forced:
                self.traceback_present = False
                self.log("Existing '{}' group will be overwritten.", self.name, logging=logging)
                self.traceback(logging=logging)

            # Cancel traceback
            else:
                self.configure_mode()
                self.log(
                    "'{}' group was not traced back from {} because it has already been traced "
                    "back.", self.name, 'data' if self.from_data else 'a model', logging=logging
                )

        # Configure mode
        else:
            self.configure_mode(mode=mode)

            # Configure traceback
            self.configure_traceback(logging=logging)

            # Logging, if no groups are to be traced back from a model
            if self.from_model and self.number_of_groups == 0:
                self.log(
                    "No '{}' groups to be traced back from a model.",
                    self.name, display=True, logging=logging
                )

            # Logging, if at least one group is to be traced back
            else:
                self.log(
                    "Tracing back '{}' group from {}.",
                    self.name, 'data' if self.from_data else 'a model',
                    display=True, logging=logging
                )

                # Set progress bar
                # self.progress_bar = tqdm(
                #     total=self.number_monte_carlo * self.number_of_stars, unit=' orbit',
                #     leave=False, bar_format=(
                #         '{desc}{percentage:3.0f}% |{bar}| '
                #         '{n_fmt}/{total_fmt} {elapsed} {remaining} '
                #     )
                # )

                # Logging and progress bar
                # for number in range(self.number_monte_carlo):
                #     message = f"Tracing back '{self.name}-{number}' group"
                #     self.log(f'{message}.', display=False, logging=logging)
                #     self.progress_bar.set_description(desc=message, refresh=True)

                # Stars from data
                self.set_timer('orbits')
                if self.from_data:
                    self.get_stars_from_data()

                    # Compute stars' coordinates
                    self.get_stars_coordinates()

                    # Filter outliers
                    self.filter_outliers(logging=logging)

                    # Compute covariance error matrices
                    self.get_stars_errors()

                # Stars from model
                elif self.from_model:
                    self.get_stars_from_model()

                    # Compute stars' coordinates
                    self.get_stars_coordinates()

                    # Compute covariance error matrices
                    self.get_stars_errors()

                # Logging message and progress bar
                self.set_timer()
                message = f"'{self.name}' group succesfully traced back"
                self.log(f'{message}.', display=True, logging=logging)
                # self.progress_bar.set_description(message, refresh=True)
                # self.progress_bar.close()
                # del self.progress_bar

                # Set traceback as present
                self.traceback_present = True

    def configure_mode(self, mode=None):
        """
        Checks if selected mode, either 'from_data' or 'from_model' is valid, and if one and no
        more than one mode has been selected. A 'mode' argument can be provided to override the
        'from_data' and 'from_model' parameters.
        """

        # Set mode from data or model
        if mode is None:
            self.from_data = self.set_boolean(self.config.from_data)
            self.from_model = self.set_boolean(self.config.from_model)

        # Check the type of mode
        else:
            self.check_type(mode, 'mode', ('string', 'None'))

            # Set mode from data or model
            if mode.lower() == 'from_data':
                self.from_data = True
                self.from_model = False
            elif mode.lower() == 'from_model':
                self.from_data = False
                self.from_model = True
            else:
                self.stop(True, 'ValueError', "Could not understand mode '{}'", mode)

        # Check if at least one mode has been selected
        self.stop(
            self.from_data == False and self.from_model == False, 'ValueError',
            "Either traceback '{}' from data or a model (None selected).", self.name
        )

        # Check if no more than one mode has been selected
        self.stop(
            self.from_data == True and self.from_model == True,
            'ValueError', "No more than one traceback mode ('from_data' or 'from_model') "
            "can be selected for '{}'", self.name
        )

    def configure_traceback(self, logging=True):
        """Checks traceback configuration parameters for 'From data' and 'From model' modes."""

        # Set number of groups parameter
        self.number_monte_carlo = self.set_integer(
            self.config.number_monte_carlo, 0 if self.from_data else 1
        )

        # Set number of steps parameter, including the initial step at t = 0
        self.number_of_steps = self.set_integer(self.config.number_of_steps, 1)

        # Set initial time parameter
        self.initial_time = self.set_quantity(self.config.initial_time)

        # Set final time parameter
        self.final_time = self.set_quantity(self.config.final_time)

        # Check if initial and final times are equal
        if self.final_time.value == self.initial_time.value:
            self.log(
                "The initial and final times are equal ({} given).",
                str(self.initial_time), level='warning', display=True, logging=logging
            )

        # Set duration, timesteps and integration times parameters
        self.duration = abs(self.final_time - self.initial_time)
        self.timestep = self.duration / (self.number_of_steps - 1)
        self.time = np.linspace(
            self.initial_time.value, self.final_time.value, self.number_of_steps
        )

        # Set direction of the orbit integration parameters
        self.forward = self.final_time.value >= self.initial_time.value
        self.backward = not self.forward

        # Set data errors parameter
        self.data_errors = self.set_boolean(self.config.data_errors)

        # Set data radial velocity shifts parameter
        self.data_rv_shifts = self.set_boolean(self.config.data_rv_shifts)

        # Set radial velocity shift parameter
        self.rv_shift = self.set_quantity(self.config.rv_shift)

        # Set cutoff parameter
        self.cutoff = self.set_quantity(self.config.cutoff, none=True).value

        # Check if cutoff is greater than 0
        if self.cutoff is not None:
            self.stop(
                self.cutoff <= 0.0, 'ValueError',
                "'cutoff' must be greater than 0 ({} given).", self.cutoff
            )

        # Set sample parameter
        self.sample = self.set_string(self.config.sample, none=True)

        # Check if the sample is valid
        if self.sample is not None:
            self.sample = self.sample.lower()
            valid_samples = ('full', 'rejected', 'input', 'extended', 'core')
            self.stop(
                self.sample not in valid_samples, 'ValueError',
                "'sample' must be {} ({} given).", enumerate_strings(*valid_samples, 'None'),
                self.sample
            )

        # Set potential parameter
        self.potential = self.set_string(self.config.potential, none=True)

        # Check if the potential is convertible to a galpy.potential object
        if self.potential is not None:
            try:
                exec(
                    f'from galpy.potential.mwpotentials import {self.potential} as potential',
                    globals()
                )
            except:
                self.stop(True, 'ValueError', "'potenial' is invalid ({} given).", self.potential)
            self.potential = potential

        # Set timer parameter
        self.timer = self.set_boolean(self.config.timer)
        self.timers = {} if self.timer else None

        # Data configuration
        if self.from_data:
            self.configure_data(logging=logging)

        # Model configuration
        if self.from_model:
            self.configure_model(logging=logging)

        # Set default covariance errors
        self.default_cov_error = np.tile(
            np.diag(np.full(3, 1e-15)),
            (self.number_of_stars, self.number_of_steps, 1, 1)
        )

        # Logging
        self.log("'{}' group ready for traceback.", self.name, logging=logging)

    def configure_data(self, logging=True):
        """
        Checks if traceback and output from data is possible and creates a Data object from a
        CSV file, or a Python dictionary, list, tuple or np.ndarray. Model parameters are also
        set to None or False if the traceback is from data. This allows to use data errors
        without resetting model parameters.
        """

        # Logging
        if self.from_data:
            self.log("Initializing '{}' group from data.", self.name, logging=logging)

        # Import data
        from .data import Data
        self.data = Data(self)

        # Set number_of_stars parameter
        if self.from_data:
            self.number_of_stars = len(self.data.sample)

            # Check if the number of stars is greater than or equal to 1
            self.stop(
                self.number_of_stars < 1, 'ValueError',
                "The sample size must be greater than or equal to 1 ({} star in the {} sample).",
                self.number_of_stars, self.sample
            )

            # Set age parameter
            self.age = None

            # Set model position and velocity parameters
            for parameter in (*self.config.position_parameters, *self.config.velocity_parameters):
                vars(self)[parameter] = None

            # Set model position and velocity errors
            if not self.data_errors:
                self.position_error = self.set_coordinate(self.config.position_error)
                self.velocity_error = self.set_coordinate(self.config.velocity_error)

            # Set model stars integration times
            self.model_time = None

    def configure_model(self, logging=True):
        """Checks if traceback and output from a model is possible."""

        # Logging
        self.log("Initializing '{}' group from a model.", self.name, logging=logging)

        # Set number of stars parameter
        self.number_of_stars = self.set_integer(self.config.number_of_stars, 1, mode=True)

        # Set age parameter
        self.age = self.set_quantity(self.config.age, mode=True)

        # Check if age is equal to the initial time
        self.stop(
            self.age.value == self.initial_time.value, 'ValueError',
            "'age' cannot be equal to 'initial_time' ({} and {} given).",
            self.age, self.initial_time
        )

        # Set model position parameters
        self.position = self.set_coordinate(self.config.position, mode=True)
        self.position_error = self.set_coordinate(self.config.position_error, mode=True)
        self.position_scatter = self.set_coordinate(self.config.position_scatter, mode=True)

        # Set model velocity parameters
        self.velocity = self.set_coordinate(self.config.velocity, mode=True)
        self.velocity_error = self.set_coordinate(self.config.velocity_error, mode=True)
        self.velocity_scatter = self.set_coordinate(self.config.velocity_scatter, mode=True)

        # Configure data to use actual error measurements or radial velocity shift
        if self.data_errors or self.data_rv_shifts:
            self.configure_data()

            # Check if the sample size is greater than or equal to 1
            self.stop(
                len(self.data.sample) < 1, 'ValueError', "If 'data_errors' or 'data_rv_shift' "
                "are True, the sample size must be greater than 0 ({} star in the {} sample).",
                len(self.data.sample), self.sample
            )

        # Set data to None since measurement errors and radial velocity shifts are modeled
        else:
            self.data = None

        # Set model stars integration times
        self.model_time = np.linspace(
            self.initial_time.value, self.age.value,
            int(abs(self.age.value - self.initial_time.value) / self.timestep.value) + 1
        )

    def get_stars_from_data(self):
        """
        Computes the initial XYZ Galactic positions and UVW space velocities of stars in the group
        from data, either in spherical of cartesian coordinates. Radial velocity shifts are applied,
        and Monte Carlo iterations are also computed to measure the error on position and velocity
        over the orbital integration. Orbits are then integrated.
        """

        # Star names from data
        self.star_designations = np.array([star.name for star in self.data.sample])

        # Position and velocity from data
        position = np.array([star.position.values for star in self.data.sample])
        velocity = np.array([star.velocity.values for star in self.data.sample])

        # Position and velocity errors from data or model
        if self.data_errors:
            position_errors = np.array([star.position.errors for star in self.data.sample])
            velocity_errors = np.array([star.velocity.errors for star in self.data.sample])
        else:
            position_errors = np.repeat(
                self.position_error.values[None],
                self.number_of_stars, axis=0
            )
            velocity_errors = np.repeat(
                self.velocity_error.values[None],
                self.number_of_stars, axis=0
            )

        # Monte Carlo position and velocity
        position_monte_carlo = np.random.normal(
            position, position_errors,
            (self.number_monte_carlo,) + position.shape
        )
        velocity_monte_carlo = np.random.normal(
            velocity, velocity_errors,
            (self.number_monte_carlo,) + velocity.shape
        )

        # Concatenate position and velocity
        position = np.concatenate((position[None], position_monte_carlo))
        velocity = np.concatenate((velocity[None], velocity_monte_carlo))

        # Convert xyz cartesian galactic coordinates to πδα spherical equatorial coordinates
        if self.data.data.system.name == 'cartesian':
            position = position_xyz_to_πδα(*position.T).T
            velocity = velocity_xyz_to_πδα(*position.T, *velocity.T).T

        # Radial velocity shift from data or model
        if self.data_rv_shifts:
            rv_shift_values = np.array([star.rv_shift.value for star in self.data.sample])
            rv_shift_errors = np.array([star.rv_shift.error for star in self.data.sample])
        else:
            rv_shift_values = np.repeat(self.rv_shift.value, self.number_of_stars)
            rv_shift_errors = np.repeat(self.rv_shift.error, self.number_of_stars)

        # Apply radial velocity shift corrections
        velocity[..., 0] -= rv_shift_values

        # Convert πδα spherical equatorial coordinates to xyz cartesian galactic coordinates
        self.position_xyz = position_πδα_to_xyz(*position.T).T
        self.velocity_xyz = velocity_πδα_to_xyz(*position.T, *velocity.T).T

        # Compute xyz and ξηζ positions and velocities (group, star, time, axis)
        self.position_xyz, self.velocity_xyz, \
        self.position_ξηζ, self.velocity_ξηζ = self.get_orbits(
            self.position_xyz, self.velocity_xyz, self.time, self.potential
        )

        # Set the number of Monte Carlo iterations, if needed
        if self.number_monte_carlo == 0:
            self.number_monte_carlo = 1
            self.position_xyz = np.repeat(self.position_xyz, 2, axis=0)
            self.velocity_xyz = np.repeat(self.velocity_xyz, 2, axis=0)
            self.position_ξηζ = np.repeat(self.position_ξηζ, 2, axis=0)
            self.velocity_ξηζ = np.repeat(self.velocity_ξηζ, 2, axis=0)

        # Set the core and outliers
        self.core = np.ones(self.number_of_stars, dtype=bool)
        self.outliers = np.zeros(self.number_of_stars, dtype=bool)

        # Set the number in the core, and the number of outliers
        self.number_in_core = self.number_of_stars
        self.number_of_outliers = 0

    def get_stars_from_model(self):
        """
        Creates an artificial model of stars for a given number of stars and age based on the
        the initial average xyz position and uvw velocity, and their respective errors and
        scatters. The sample is then moved forward in time for the given age, and the radial
        velocity shift is applied.
        """

        # Integrate average model star's backward orbit from the current-day epoch
        self.average_model_star = self.Star(
            self, name='average_model_star', time=self.model_time,
            position_xyz=self.position.values, velocity_xyz=self.velocity.values
        )

        # Create stars from a model
        self.model_stars = []
        for star in range(self.number_of_stars):
            index = star - (star // len(self.data.sample)) * len(self.data.sample)

            # Integrate model star's forward orbit from the epoch of star formation
            self.model_stars.append(
                self.Star(
                    self, name='model_star_{}'.format(star + 1), time=self.model_time[::-1],
                    position_xyz=np.random.normal(
                        self.average_model_star.position_xyz[-1],
                        self.position_scatter.values
                    ), velocity_xyz=np.random.normal(
                        self.average_model_star.velocity_xyz[-1],
                        self.velocity_scatter.values
                    )
                )
            )

            # Convert xyz cartesian galactic coordinates to πδα spherical equatorial coordinates
            position_πδα = position_xyz_to_πδα(*self.model_stars[star].position_xyz[-1])
            velocity_πδα = velocity_xyz_to_πδα(
                *self.model_stars[star].position_xyz[-1], *self.model_stars[star].velocity_xyz[-1]
            )

            # Select position and velocity errors from data or model
            if self.data_errors:
                position_πδα_error = self.data.sample[index].position.errors
                velocity_πδα_error = self.data.sample[index].velocity.errors
            else:
                position_πδα_error = self.position_error.values
                velocity_πδα_error = self.velocity_error.values

            # Scramble position and velocity
            position_πδα = np.random.normal(position_πδα, position_πδα_error)
            velocity_πδα = np.random.normal(velocity_πδα, velocity_πδα_error)

            # Select radial velocity shift from data or model
            if self.data_rv_shifts:
                rv_shift_value = self.data.sample[index].rv_shift.value
                rv_shift_error = self.data.sample[index].rv_shift.error
            else:
                rv_shift_value = self.rv_shift.value
                rv_shift_error = self.rv_shift.error

            # Apply radial velocity shift bias
            velocity_πδα += np.array((rv_shift_value, 0.0, 0.0))

            # Convert πδα spherical equatorial coordinates to xyz cartesian galactic coordinates
            position_xyz = position_πδα_to_xyz(*position_πδα)
            velocity_xyz = velocity_πδα_to_xyz(*position_πδα, *velocity_πδα)

            # Create star
            self.append(
                self.Star(
                    self, index=star + 1, name=f'star_{star + 1}', time=self.time,
                    position_xyz=position_xyz, velocity_xyz=velocity_xyz
                )
            )

            # Update progress bar
            # self.progress_bar.update(1)

    def get_stars_coordinates(self):
        """
        Computes the stars' and group's average xyz and ξηζ average position, velocity, distance,
        and speed, and xyz and ξηζ position, velocity, distance, and speed scatters, excluding
        outliers. The stars' relative xyz and ξηζ position, velocity, distance, and speed,
        including outliers, along with their respective averages, excluding outliers, are also
        computed.
        """

        # Norm definition
        norm = {'position': 'distance', 'velocity': 'speed'}

        def compute_coordinates(operation, coord, system):
            """
            Computes the norm, average, average norm, and group average norm, and, if needed,
            the scatter, scatter norm, and group scatter norm. The norm corresponds to the
            distance for position coordinates and to the speed for velocity coordinates.
            """

            # Select function
            func = np.mean if operation in ('average', 'relative_average') else np.std

            # Average or scatter : (group, time, axis)
            vars(self)[f'{operation}_{coord}_{system}'] = func(
                vars(self)[f'{coord}_{system}'][:, self.core], axis=-3
            )

            # Norm verage or scatter : (group, time)
            vars(self)[f'{operation}_{norm[coord]}_{system}'] = func(
                vars(self)[f'{norm[coord]}_{system}'][:, self.core], axis=-2
            )

            # Group norm average or scatter : (group, time)
            vars(self)[f'{operation}_group_{norm[coord]}_{system}'] = np.sum(
                vars(self)[f'{operation}_{coord}_{system}']**2, axis=-1
            )**0.5

        # Select coordinate and system
        for coord in ('position', 'velocity'):
            for system in ('xyz', 'ξηζ'):

                # Norm : (group, star, time)
                vars(self)[f'{norm[coord]}_{system}'] = np.sum(
                    vars(self)[f'{coord}_{system}']**2, axis=-1
                )**0.5

                # Average : (group, time)
                compute_coordinates('average', coord, system)

                # Scatter : (group, time)
                compute_coordinates('scatter', coord, system)

                # Relative value : (group, star, time, axis)
                vars(self)[f'relative_{coord}_{system}'] = (
                    vars(self)[f'{coord}_{system}'] -
                    vars(self)[f'average_{coord}_{system}'][:, None]
                )

                # Relative norm : (group, star, time)
                vars(self)[f'relative_{norm[coord]}_{system}'] = np.sum(
                    vars(self)[f'relative_{coord}_{system}']**2, axis=-1
                )**0.5

                # Relative average : (gruop, time)
                compute_coordinates('relative_average', coord, system)

                # Relative scatter : (group, time), same as the absolute scatter
                vars(self)[f'relative_scatter_{coord}_{system}'] = vars(self)[
                    f'scatter_{coord}_{system}'
                ]
                vars(self)[f'relative_scatter_{norm[coord]}_{system}'] = vars(self)[
                    f'scatter_{norm[coord]}_{system}'
                ]
                vars(self)[f'relative_scatter_group_{norm[coord]}_{system}'] = vars(self)[
                    f'scatter_group_{norm[coord]}_{system}'
                ]

    def filter_outliers(self, logging=True):
        """
        Filters outliers from the sample based on ξηζ position and velocity scatter over time.
        A subcore is created based on a robust covariance matrix estimator using the scikit-
        learn (sklearn) Python package, leaving other stars as part of the core sample.
        """

        # Set position and velocity outliers
        self.position_outliers = np.zeros(self.number_of_stars, dtype=bool)
        self.velocity_outliers = np.zeros(self.number_of_stars, dtype=bool)

        # Set the number of position and velocity outliers
        self.number_of_position_outliers = np.count_nonzero(self.position_outliers)
        self.number_of_velocity_outliers = np.count_nonzero(self.position_outliers)

        # Iteratively find outliers
        if self.cutoff is not None:
            while self.number_in_core > int(0.8 * self.number_of_stars):

                # Find new position or velocity outliers beyond the cutoff
                position_outliers = ~self.position_outliers & np.any(
                    np.abs(self.relative_position_ξηζ[0]) >
                    (self.scatter_position_ξηζ[0] * self.cutoff), axis=(1, 2)
                )
                velocity_outliers = ~self.velocity_outliers & np.any(
                    np.abs(self.relative_velocity_ξηζ[0]) >
                    (self.scatter_velocity_ξηζ[0] * self.cutoff), axis=(1, 2)
                )
                outliers = position_outliers | velocity_outliers

                # Continue the search, if new outliers were found
                if (~self.outliers & outliers).any():
                    self.core[outliers] = False
                    self.outliers[outliers] = True
                    self.position_outliers[position_outliers] = True
                    self.velocity_outliers[velocity_outliers] = True

                    # Update the number in the core, and the number of outliers
                    self.number_in_core = np.count_nonzero(self.core)
                    self.number_of_outliers = np.count_nonzero(self.outliers)
                    self.number_of_position_outliers = np.count_nonzero(self.position_outliers)
                    self.number_of_velocity_outliers = np.count_nonzero(self.position_outliers)

                    # Re-compute stars' coordinates
                    self.get_stars_coordinates()

                # Stop the search, if no outliers were found
                else:
                    break

            # Create a message if a least one outlier have been found
            if self.number_of_outliers > 0:
                self.outliers_messages = [
                    f'{self.number_of_outliers:d} outlier'
                    f"{'s' if self.number_of_outliers > 1 else ''} "
                    f"found in '{self.name}' group during traceback:"
                ]

                # Create a message for every outlier found
                for star in self.outliers.nonzero()[0]:
                    outlier_type = (
                        'Position' if self.position_outliers[star] else
                        'Velocity' if self.velocity_outliers[star] else ''
                    )
                    if outlier_type == 'Position':
                        outlier_sigma = np.max(
                            np.abs(self.relative_position_ξηζ[0, star]) /
                            self.scatter_position_ξηζ[0]
                        )
                    elif outlier_type == 'Velocity':
                        outlier_sigma = np.max(
                            np.abs(self.relative_velocity_ξηζ[0, star]) /
                            self.scatter_velocity_ξηζ[0]
                        )
                    self.outliers_messages.append(
                        f'{self.star_designations[star]}: {outlier_type} > {outlier_sigma:.1f}σ'
                    )

            # Create message if no outliers have been found and cutoff is not None
            else:
                self.outliers_messages = [f"No outliers found in '{self.name}' group."]

            # Outliers logging
            for message in self.outliers_messages:
                self.log(message, display=True, logging=logging)

        # Robust covariance matrix and support fraction
        if False:
            a = np.swapaxes(np.array([star.position_ξηζ for star in self.sample]), 0, 1)
            robust_covariance_matrix = []
            support_fraction = []

            # Iteration over time
            for step in range(self.number_of_steps):
                MCD = MinCovDet(assume_centered=False).fit(a[step])
                robust_covariance_matrix.append(MCD.covariance_)
                support_fraction.append(MCD.support_)
                # print(step, MCD.support_.size, MCD.support_.nonzero()[0].size, MCD.dist_.size)

            # Array conversion
            robust_covariance_matrix = np.array(robust_covariance_matrix)
            support_fraction = np.array(support_fraction)
            support_fraction_all = (
                np.sum(support_fraction, axis=0) / self.number_of_steps > 0.7
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

    def get_stars_errors(self):
        """
        Computes the covariance matrices of the xyz and ξηζ position and velocity of every star,
        in every group, as a function of time, and their respective errors.
        """

        # Norm definition
        norm = {'position': 'distance', 'velocity': 'speed'}

        def compute_errors(coord, system, operation=None, group=False):
            """
            Computes the covariance matrix, error, norm error, and, if needed, the group
            error for an operation.
            """

            # Set index
            i = 1 if self.from_data else 0

            # Set label
            label = '_'.join((operation, coord, system) if operation != None else (coord, system))

            # Covariance matrix: (star, time, axix, axis) or (time, axix, axis)
            if self.number_monte_carlo > 0:
                vars(self)[f'{label}_cov_error'] = self.get_covariance_matrix(vars(self)[label][i:])
            elif vars(self)[label].ndim == 4:
                vars(self)[f'{label}_cov_error'] = self.default_cov_error
            else:
                vars(self)[f'{label}_cov_error'] = self.default_cov_error[0]

            # Error: (star, time, axix) or (time, axix)
            vars(self)[f'{label}_error'] = np.diagonal(
                vars(self)[f'{label}_cov_error'], axis1=-2, axis2=-1
            )**0.5

            # Norm error: (star, time) or (time)
            label = label.replace(coord, norm[coord])
            vars(self)[f'{label}_error'] = np.std(vars(self)[label][i:], axis=0)

            # Group norm error: (time)
            if group:
                label = label.replace(operation, f'{operation}_group')
                vars(self)[f'{label}_error'] = np.std(vars(self)[label][i:], axis=0)

        # Select coordinate and system
        for coord in ('position', 'velocity'):
            for system in ('xyz', 'ξηζ'):

                # Covariance matrix, error, and norm error
                compute_errors(coord, system)

                # Average covariance matrix, error, norm error, and group norm error
                compute_errors(coord, system, operation='average', group=True)

                # Scatter covariance matrix, error, norm error, and group norm error
                compute_errors(coord, system, operation='scatter', group=True)

                # Relative covariance matrix, error, and norm error
                compute_errors(coord, system, operation='relative')

                # Relative average covariance matrix, error, norm error, and group norm error
                compute_errors(coord, system, operation='relative_average', group=True)

                # Relative scatter covariance matrix, error, norm error, and group norm error
                vars(self)[f'relative_scatter_{coord}_{system}_cov_error'] = vars(self)[
                    f'scatter_{coord}_{system}_cov_error'
                ]
                vars(self)[f'relative_scatter_{coord}_{system}_error'] = vars(self)[
                    f'scatter_{coord}_{system}_error'
                ]
                vars(self)[f'relative_scatter_{norm[coord]}_{system}_error'] = vars(self)[
                    f'scatter_{norm[coord]}_{system}_error'
                ]
                vars(self)[f'relative_scatter_group_{norm[coord]}_{system}_error'] = vars(self)[
                    f'scatter_group_{norm[coord]}_{system}_error'
                ]

    def get_orbits(self, position_xyz, velocity_xyz, time, potential):
        """
        Computes the stars' Galactic orbit using Galpy with the given potential at the given time.
        The XYZ and ξηζ positions and velocities are computed.
        """

        # Linear trajectories
        if potential is None:

            # Compute xyz positions and velocities
            position_xyz = (
                position_xyz + (velocity_xyz + Coordinate.sun_velocity) *
                (time - time[0])[:, None]
            )
            velocity_xyz = np.repeat(velocity_xyz[None], time.size, axis=0)

            # Compute rθh positions and velocities
            position_rθh = position_xyz_to_rθh(*position_xyz.T).T
            velocity_rθh = velocity_xyz_to_rθh(*position_xyz.T, *velocity_xyz.T).T

        # Orbital trajectories
        else:

            # Compute initial rθh position and velocity
            position_rθh = position_xyz_to_rθh(*position_xyz.T).T
            velocity_rθh = velocity_xyz_to_rθh(*position_xyz.T, *velocity_xyz.T).T

            # Convert time, and initial rθh position and velocity to Galpy's natural units
            time = time * Coordinate.lsr_velocity[1] / Coordinate.sun_position[0]
            position_rθh[..., [0, 2]] = position_rθh[..., [0, 2]] / Coordinate.sun_position[0]
            velocity_rθh /= Coordinate.lsr_velocity[1]

            # Integrate orbit
            orbit = Orbit(
                np.array(
                    [
                        position_rθh.T[0], velocity_rθh.T[0], velocity_rθh.T[1],
                        position_rθh.T[2], velocity_rθh.T[2], position_rθh.T[1]
                    ]
                ).T
            )
            orbit.turn_physical_off()
            orbit.integrate(time, potential, method='odeint', progressbar=True)

            # Compute rθh positions and velocities
            position_rθh = np.array((orbit.R(time).T, orbit.phi(time).T, orbit.z(time).T)).T
            velocity_rθh = np.array((orbit.vR(time).T, orbit.vT(time).T, orbit.vz(time).T)).T

            # Convert time, and rθh positions and velocities back to physical units
            time = time * Coordinate.sun_position[0] / Coordinate.lsr_velocity[1]
            position_rθh[..., [0, 2]] = position_rθh[..., [0, 2]] * Coordinate.sun_position[0]
            velocity_rθh *= Coordinate.lsr_velocity[1]

            # Compute xyz positions and velocities (group, star, time, axis)
            position_xyz = position_rθh_to_xyz(*position_rθh.T).T
            velocity_xyz = velocity_rθh_to_xyz(*position_rθh.T, *velocity_rθh.T).T

        # Compute ξηζ positions and velocities (group, star, time, axis)
        position_ξηζ = position_rθh_to_ξηζ(*position_rθh.T, time).T
        velocity_ξηζ = velocity_rθh_to_ξηζ(*velocity_rθh.T).T

        return position_xyz, velocity_xyz, position_ξηζ, velocity_ξηζ

    def get_covariance_matrix(
        self, a, b=None, robust=False, sklearn=False, Mahalanobis_distance=None
    ):
        """
        Computes covariance matrices for the values and samples in 'a', an array where the first
        dimension represents the samples and the last dimension represents the values. In between,
        there's an arbitrary number of extra dimensions (samples, ..., values). The output is an
        array with the same number of dimensions. However, the dimension representing samples is
        removed and an extra dimension representing values is added at the end, so that the last
        two dimensions represent the covariance matrices (..., values, values). If 'b' is not None,
        the cross covariance matrices are computed instead. If 'robust' is True, a robust estimator
        is computed using weights from the Mahalanobis distance. If 'sklearn' is True, a robust
        covariance estimator with sklearn's minimum covariance determinant (MCD). If 'a' or 'b' are
        strings instead of arrays, the function assumes the strings represent the names of the
        arrays in the group. 'a' and 'b' must have at least 2 dimensions.

          - Pour robuste et sklearn, ça peut attendre.
          - On pourrait créer un fonction self.get_weights
        """

        # Find a and b arrays, if needed
        if type(a) == str:
            a = vars(self)[a]
        if type(b) == str:
            b = vars(self)[b]

        # Sklearn covariance estimator
        if sklearn and False:
            a = np.swapaxes(np.array([vars(star)[a] for star in self.sample]), 0, 1)
            covariance_matrix = []
            support_fraction = []
            for step in range(self.number_of_steps):
                MCD = MinCovDet(assume_centered=False).fit(a[step])
                covariance_matrix.append(MCD.covariance_)
                support_fraction.append(MCD.support_)

                # Update progress bar
                if step % 5 == 0:
                    self.progress_bar.update(1)

            return np.repeat(
                np.array(covariance_matrix)[None],
                self.number_jackknife, axis=0
            )

        # Weights from the Mahalanobis distance
        else:
            if robust:
                if Mahalanobis_distance is None:
                    Mahalanobis_distance = self.get_Mahalanobis_distance(a, b)
                Mahalanobis_distance = np.repeat(Mahalanobis_distance[..., None], 3, axis=-1)

                # Weights based on the Mahalanobis distance
                weights = np.exp(-2 * Mahalanobis_distance)
                weights = np.repeat(weights[..., None], weights.shape[-1], axis=-1)

            # Covariance matrices
            a = a - np.mean(a, axis=0) # a_weights = np.exp(-2. * (a / np.std(a, axis=0))**2)
            a = np.repeat(a[..., None], a.shape[-1], axis=-1)
            # d = np.tile(a[..., None], (1,) * a.ndim + (a.shape[-1],))
            if b is None:
                b = np.swapaxes(a, -1, -2)

            # Cross covariance matrices
            else:
                b = b - np.mean(b, axis=0) # b_weights = np.exp(-2. * (b / np.std(b, axis=0))**2)
                b = np.swapaxes(np.repeat(b[..., None], b.shape[-1], axis=-1), -1, -2)

            return (
                np.average(a * b, weights=weights, axis=0) if robust
                else np.mean(a * b, axis=0)
            )

    def get_Mahalanobis_distance(self, a, b=None, covariance_matrix=None):
        """
        Computes the Mahalanobis distances using the covariance matrix of every stars in the group.
        """

        # Find a and b arrays, if needed
        if type(a) == str:
            a = vars(self)[a]
        if type(b) == str:
            b = vars(self)[b]

        # Compute the covariance matrix
        if covariance_matrix is None:
            covariance_matrix = self.get_covariance_matrix(a, b)

        # Compute the inverse covariance matrix
        covariance_matrix_invert = np.repeat(
            np.linalg.inv(covariance_matrix)[None],
            self.number_of_stars_iteration, axis=0
        )

        # Compute the Mahalanobis distances !!! Needs to be checked !!!
        if b is None:
            b = a
        c = (a - np.mean(a, axis=0))[..., None]
        # d = (b - np.mean(b, axis=0))[..., None]
        # d = c
        # c = (((a - np.mean(a, axis=0)) + (b - np.mean(b, axis=0))) / 2)[..., None]

        return np.abs(
            np.squeeze(
                np.matmul(np.swapaxes(c, -2, -1), np.matmul(covariance_matrix_invert, c)),
                axis=(-2, -1)
            )
        )**0.5
