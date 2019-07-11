# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" output.py: Defines functions to create data output such as plots of size indicators over time,
    2D and 3D scatters at a given time, histograms, color mesh, etc.
"""

import numpy as np
from os import path
from matplotlib import rcParams, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from Traceback.collection import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

def save_figure(plt, name, file_path=None, forced=False, default=False, cancel=False):
    """ Checks whether a path already exists and asks for user input if it does. The base path
        is assumed to be the output directory. Also, if the path does not have an extension, a
        '.pdf' extension is added.
    """

    # file_path parameter
    file_path = file_path if file_path is not None else output(create=True) + '/'

    # Check if file_path parameter is a string, which must be done before the directory call
    stop(type(file_path) != str, 'TypeError', "'file_path' must be a string ({} given).",
        type(file_path))

    # file_path redefined as the absolute path, default name and directory creation
    file_path = path.join(directory(output(), path.dirname(file_path), 'file_path', create=True),
        path.basename(file_path)) if path.basename(file_path) != '' else '{}.pdf'.format(name)

    # Check if there's an extension and add a '.pdf' extension, if needed
    if path.splitext(file_path)[1] == '':
        file_path += '.pdf'

    # Check if a file already exists
    if path.exists(file_path):
        choice = None
        stop(type(forced) != bool, 'TypeError',
            "'forced' must be a boolean ({} given).", type(forced))
        if not forced:
            stop(type(default) != bool, 'TypeError',
                "'default' must be a default ({} given).", type(default))
            if not default:
                stop(type(cancel) != bool, 'TypeError',
                    "'cancel' must be a boolean ({} given).", type(cancel))
                if not cancel:

                    # User input
                    while choice == None:
                        choice = input(
                            "A file already exists at '{}'. Do you wish to overwrite (Y), "
                            "keep both (K) or cancel (N)? ".format(file_path)).lower()

                        # Loop over question if input could not be interpreted
                        if choice not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                            print("Could not understand '{}'.".format(choice))
                            choice = None

                # Saving cancellation
                if cancel or choice in ('n', 'no'):

                    # Logging
                    log("'{}': figure not saved because a file already exists at '{}'.",
                        name, file_path)

            # Default name and saving
            if default or choice in ('k', 'keep'):
                from Traceback.tools import default_name
                file_path = default_name(file_path)
                plt.savefig(file_path)

                # Logging
                log("'{}': file name changed and figure saved at '{}'.", name, file_path)

        # Existing file deletion and saving
        if forced or choice in ('y', 'yes'):
            from os import remove
            remove(file_path)
            plt.savefig(file_path)

            # Logging
            log("'{}': existing file located at '{}' deleted and replaced.", name, file_path)

    # Saving
    else:
        plt.savefig(file_path)

        # Logging
        log("'{}': figure saved at '{}'.", name, file_path)

class Output_Series():
    """ Defines output methods from a Series object. """

    def create_size_indicators_plot(self, secondary=False,
            forced=False, default=False, cancel=False):
        """ Creates a plot of scatter, median absolute deviation, and minimum spanning tree
            branches length mean and median absolute deviation over the entire duration of the
            data.
        """

        # Figure initialization
        rcParams.update({'font.family': 'serif', 'font.size': '14'})
        plt.figure(figsize=(12, 8), facecolor='w')

        # Scatter
        plt.plot(self.time, self.scatter / self.scatter[0], 'k-', linewidth=2.0,
            label='Scatter age: ({} ± {}) Myr'.format(self.scatter_age, self.scatter_age_error))

        # Median absolute deviation
        plt.plot(self.time, self.mad / self.mad[0], '-.', color='0.2', linewidth=2.0,
            label='MAD age: ({} ± {}) Myr'.format(self.mad_age, self.mad_age_error))

        # Minimum spanning tree mean branch length
        plt.plot(self.time, self.mst_mean / self.mst_mean[0], '--', color='0.4', linewidth=2.0,
            label='MST mean age: ({} ± {}) Myr'.format(self.mst_mean_age, self.mst_mean_age_error))

        # Minimum spanning tree median absolute deviation branch length
        plt.plot(self.time, self.mst_mad / self.mst_mad[0], 'g:', color='0.6', linewidth=2.0,
            label='MST MAD age: ({} ± {}) Myr'.format(self.mst_mad_age, self.mst_mad_age_error))

        # Secondary lines
        self.stop(type(secondary) != bool, 'TypeError',
            "'secondary' must be a boolean ({} given).", type(secondary))
        if secondary:
            i = 0
            plot_i = np.arange(0, len(self), 20)
            for group in self:
                if i in plot_i:
                    plt.plot(self.time, group.scatter, '-', color='0.7', linewidth=0.5)
                i += 1

        # Title, legend and axis formatting
        if self.from_data:
            plt.title("Size indicators of β-Pictoris (without outliners) over {} Myr\n"
                "with {} km/s redshift correction and actual measurement errors\n".format(
                    self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
        elif self.from_model:
            plt.title("Average size indicators of {} moving group simulations with kinematics "
                "similar to β Pictoris\n over {} Myr with {} km/s redshift correction and "
                "actual measurement errors of Gaia DR2\n".format(self.number_of_groups,
                    self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
        plt.legend()
        plt.xlabel('Time (Myr)')
        plt.ylabel('Relative Scatter, MAD, MST mean and MST MAD')
        plt.xlim(0, self.final_time.value)
        plt.ylim(0, 1.5)
        # plt.xticks([14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0])
        # plt.yticks([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

        # Save and show figure
        save_figure(plt, self.name, 'Size_indicator_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_plot(self, forced=False, default=False, cancel=False):
        """ Creates a plot of X-U, Y-V and Z-W covariances. """

        # Figure initialization
        rcParams.update({'font.family': 'serif', 'font.size': '14'})
        plt.figure(figsize=(12, 8), facecolor='w')

        # Covariance normalization
        covariances = self.covariances / self.covariances[0]

        # X-U covariance
        plt.plot(self.time, covariances[:,0], '-', color='0.0', linewidth=2.0,
            label='X-U covariance age: ({} ± {}) Myr'.format(self.xu_age, self.xu_age_error))

        # Y-V covariance
        plt.plot(self.time, covariances[:,1], '-.', color='0.2', linewidth=2.0,
            label='Y-V covariance age: ({} ± {}) Myr'.format(self.yv_age, self.yv_age_error))

        # Z-W covariance
        plt.plot(self.time, covariances[:,2], '--', color='0.4', linewidth=2.0,
            label='Z-W covariance age: ({} ± {}) Myr'.format(self.zw_age, self.zw_age_error))

        # Title, legend and axis formatting
        if self.from_data:
            plt.title("X-U, Y-V and Z-W covariances of β Pictoris (without outliners) over {} Myr\n"
                "with {} km/s redshift correction and actual measurement errors\n".format(
                    self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
        elif self.from_model:
            plt.title("X-U, Y-V and Z-W covariances of {} moving group simulations with kinematics "
                "similar to β Pictoris \n over {} Myr with {} km/s redshift correction and actual "
                "measurement errors of Gaia DR2\n".format(self.number_of_groups,
                    self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
        plt.legend()
        plt.xlabel('Time (Myr)')
        plt.ylabel('Covariance')
        plt.xlim(0, self.final_time.value)
        plt.ylim(0, 1.5)

        # Show figure
        save_figure(plt, self.name, 'Covariances_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_histogram_ages(self, forced=False, default=False, cancel=False):
        """ Creates an histogram of ages computed in a series. """

        # Figure initialization
        rcParams.update({'font.family': 'serif', 'font.size': '14'})
        plt.figure(figsize=(12, 8), facecolor='w')

        # Histogram plotting
        ages = [group.scatter_age for group in self]
        plt.hist(ages, bins='auto') # bins=np.arange(21.975, 26.025, 0.05)

        # Title and axis formatting
        plt.title(
            'Distribution of {} moving groups age,\n'
            'without measurement errors. Average age: ({} ± {}) Myr\n'.format(
                len(self), np.round(np.mean(ages), 3), np.round(np.std(ages), 3)))
        plt.xlabel('Age (Myr)')
        plt.ylabel('Number of groups')

        # Show figure
        save_figure(plt, 'Age Histogram', 'Age Historgram.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

class Output_Group():
    """ Defines output methods from a Group object. """

    def create_2D_scatter(self, i, j, step=None, age=None, errors=False, labels=False, mst=False,
            forced=False, default=False, cancel=False):
        """ Creates a 2D scatter plot of star positions in i and j at a given 'step' or 'age' in
            Myr. If 'age' doesn't match a step, the closest step is used instead. 'age' overrules
            'steps' if both are given. 'labels' adds the stars' name and 'mst' adds the minimun
            spanning tree branches.
        """

        # Axis initialization
        axis = {'x': 0, 'y': 1, 'z': 2}
        keys = tuple(axis.keys())

        # X axis
        self.series.stop(type(i) != str, "X axis 'i' must be a string ({} given).", type(i))
        self.series.stop(i.lower() not in keys, 'ValueError',
            "X axis 'i' must be an axis key ('x', 'y' or 'z', {} given).", i)
        i = axis[i.lower()]

        # Y axis
        self.series.stop(type(j) != str, "Y axis 'j' must be a string ({} given).", type(j))
        self.series.stop(j.lower() not in keys, 'ValueError',
            "Y axis 'j' must be an axis key ('x', 'y' or 'z', {} given).", j)
        j = axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Figure initialization
        rcParams.update({'font.family': 'serif', 'font.size': 14, 'lines.markersize': 4})
        plt.figure(figsize=(12, 8), facecolor='w')

        # Scatter
        plt.scatter(
            [star.position[step, i] for star in self.stars],
            [star.position[step, j] for star in self.stars], marker='o', color='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.stars:
                position = star.position[step]
                error = star.position_error[step]

                # Horizontal error bars
                plt.plot(
                    (position[i] - error[i], position[i] + error[i]),
                    (position[j], position[j]), c='0.1', linewidth=0.7)

                # Vertical error bars
                plt.plot(
                    (position[i], position[i]),
                    (position[j] - error[j], position[j] + error[j]), c='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.stars:
                plt.text(star.position[step, i] + 1, star.position[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=7)

        # Branches creation
        self.series.stop(type(mst) != bool, 'TypeError', "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                plt.plot(
                    (branch.start.position[step, i], branch.end.position[step, i]),
                    (branch.start.position[step, j], branch.end.position[step, j]), c='b')

        # Title and axis formatting
        plt.title("{} and {} positions of stars in β Pictoris at {} Myr "
            "wihtout outliers.\n".format(keys[i].upper(), keys[j].upper(), age))
        plt.xlabel('{} (pc)'.format(keys[i].upper()))
        plt.ylabel('{} (pc)'.format(keys[j].upper()))

        # Show figure
        save_figure(plt, self.name, '2D_Scatter_{}_{}{}.pdf'.format(
                self.name, keys[i].upper(), keys[j].upper()),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_3D_scatter(self, step=None, age=None, errors=False, labels=False, mst=False,
            forced=False, default=False, cancel=False):
        """ Creates a 3D scatter plot of star positions in x, y and z at a given 'step' or 'age'
            in Myr. If 'age' doesn't match a step, the closest step is used instead. 'age'
            overrules 'step' if both are given. 'labels' adds the stars' name and 'mst' adds the
            minimum spanning tree branches.
        """

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(step * self.series.timestep.value, 2)

        # Figure initialization
        rcParams.update({'font.family': 'serif', 'font.size': '14', 'lines.markersize': 4})
        fig = plt.figure(figsize=(12, 8), facecolor='w')
        ax = fig.add_subplot(111, projection='3d')

        # Scatter
        ax.scatter(
            [star.relative_position[step, 0] for star in self.stars],
            [star.relative_position[step, 1] for star in self.stars],
            [star.relative_position[step, 2] for star in self.stars], marker='o', c='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.stars:
                position = star.relative_position[step]
                error = star.relative_position_error[step]

                # X axis error bars
                ax.plot(
                    (position[0] - error[0], position[0] + error[0]),
                    (position[1], position[1]), (position[2], position[2]), c='0.1', linewidth=0.7)

                # Y axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1] - error[1], position[1] + error[1]),
                    (position[2], position[2]), c='0.1', linewidth=0.7)

                # Z axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1], position[1]),
                    (position[2] - error[2], position[2] + error[2]), c='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.stars:
                ax.text(star.position[step, 0] + 2, star.position[step, 1] + 2,
                    star.position[step, 2] + 2, star.name, horizontalalignment='left', fontsize=7)

        # Branches creation
        self.series.stop(type(mst) != bool, 'TypeError', "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (branch.start.relative_position[step, 0],
                        branch.end.relative_position[step, 0]),
                    (branch.start.relative_position[step, 1],
                        branch.end.relative_position[step, 1]),
                    (branch.start.relative_position[step, 2],
                        branch.end.relative_position[step, 2]), c='b')

        # Title and axis formatting
        plt.title("Minimum spanning tree of stars in β Pictoris at {} Myr.\n".format(age))
        ax.set_xlabel('\n X (pc)')
        ax.set_ylabel('\n Y (pc)')
        ax.set_zlabel('\n Z (pc)')
        # ax.set_xlim3d(-70, 70)
        # ax.set_ylim3d(-30, 30)
        # ax.set_zlim3d(-30, 30)

        # Show figure
        save_figure(plt, self.name, '3D_Scatter_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_scatter(self, i, j, step=None, age=None, errors=False, labels=False,
            forced=False, default=False, cancel=False):
        """ Creates a covariance scatter of star positions and velocities in i and j at a given
            'step' or 'age' in Myr. If 'age' doesn't match a step, the closest step is used
            instead. 'age' overrules 'steps' if both are given. 'labels' adds the stars' name
            and 'mst' adds the minimum spanning tree branches.
        """

        # Axex initialization
        position_axis = {'x': 0, 'y': 1, 'z': 2}
        velocity_axis = {'u': 0, 'v': 1, 'w': 2}
        position_keys = tuple(position_axis.keys())
        velocity_keys = tuple(velocity_axis.keys())

        # Position axis
        self.series.stop(type(i) != str, "Position axis 'i' must be a string ({} given).", type(i))
        self.series.stop(i.lower() not in position_keys, 'ValueError',
            "Position axis 'i' must be an postion axis key ('x', 'y' or 'z', {} given).", i)
        i = position_axis[i.lower()]

        # Velocity axis
        self.series.stop(type(j) != str, "Velocity axis 'j' must be a string ({} given).", type(j))
        self.series.stop(j.lower() not in velocity_keys, 'ValueError',
            "Velocity axis 'j' must be an postion axis key ('u', 'v' or 'w', {} given).", j)
        j = velocity_axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Figure initialization
        rcParams.update({'font.family': 'serif', 'font.size': 14, 'lines.markersize': 4})
        plt.figure(figsize=(12, 8), facecolor='w')

        # Scatter
        plt.scatter(
            [star.position[step, i] for star in self.stars],
            [star.velocity[j] for star in self.stars], marker='o', color='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.stars:
                position = star.position[step]
                position_error = star.position_error[step]
                velocity = star.velocity
                velocity_error = star.velocity_error

                # Position (horizontal) error bars
                plt.plot(
                    (position[i] - position_error[i], position[i] + position_error[i]),
                    (velocity[j], velocity[j]),
                    c='0.1', linewidth=0.7)

                # Velocity (vertical) error bars
                plt.plot(
                    (position[i], position[i]),
                    (velocity[j] - velocity_error[j], velocity[j] + velocity_error[j]),
                    c='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.stars:
                plt.text(star.position[step, i] + 1, star.velocity[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=7)

        # Title and axis formatting
        plt.title("{} and {} covariance of stars in β Pictoris at {} Myr "
            "wihtout outliers.\n".format(position_keys[i].upper(), velocity_keys[j].upper(), age))
        plt.xlabel('{} (pc)'.format(position_keys[i].upper()))
        plt.ylabel('{} (pc/Myr)'.format(velocity_keys[j].upper()))

        # Show figure
        save_figure(plt, self.name, 'Covariances_Scatter_{}_{}{}.pdf'.format(
                self.name, position_keys[i].upper(), velocity_keys[j].upper()),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

def plot_age_error(forced=False, default=False, cancel=False):
    """ Creates a plot of ages obtained for diffrent measurement errors on radial velocity
        and offset due to gravitationnal redshift.
    """

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # + 0.0 km/s points
    plt.errorbar(
        [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        [23.789, 23.763, 23.700, 23.174, 21.390, 15.400, 8.761],
        yerr=[0.268, 0.286, 0.356, 0.626, 0.960, 1.138, 1.290],
        fmt='o', color='0.0', label='+ 0.0 km/s')

    # + 0.5 km/s points
    plt.errorbar(
        [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        [20.658, 20.601, 20.571, 20.196, 19.076, 15.423, 8.804],
        yerr=[0.283, 0.288, 0.316, 0.487, 0.819, 1.143, 1.319],
        fmt='D', color='0.2', label='+ 0.5 km/s')

    # + 1.0 km/s points
    plt.errorbar(
        [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        [18.101, 18.087, 18.055, 17.827, 16.988, 14.376, 8.930],
        yerr=[0.331, 0.332, 0.342, 0.448, 0.636, 0.973, 1.066],
        fmt='^', color='0.4', label='+ 1.0 km/s')

    # β Pictoris typical error line
    plt.axvline(x=1.0112, ymin=0.0, ymax = 25, linewidth=1, color='k', ls='dashed')
    plt.text(1.05, 24, 'β Pictoris typical RV error.', horizontalalignment='left', fontsize=8)

    # Title, legend and axis formatting
    plt.title(
        "Measured age of a simulation of 1000 24 Myr-old groups \n"
        "over the measurement error on RV (other errors typical of Gaia DR2)\n")
    plt.legend(loc='upper right')
    plt.xlabel('Error on RV (km/s)')
    plt.ylabel('Age (Myr)')
    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    plt.yticks([6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
    plt.xlim(-0.1, 4.1)
    plt.ylim(6, 24.5)

    # Show figure
    save_figure(plt, 'Error-RV Offset Plot', 'Error-RV Offset Plot.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_histogram(ages, initial_scatter, number_of_stars, number_of_groups, age,
        forced=False, default=False, cancel=False):
    """ Creates an histogram of ages computed by multiple tracebacks. """

    # Check if ages are valid
    stop(type(ages) not in (tuple, list),
        "'ages' must either must be a tuple or list ({} given)", type(ages))
    for age in ages:
        stop(type(age) not in (int, float), 'TypeError',
            "All 'ages' must be an integer or float ({} given).", type(age))
        stop(age < 0, 'ValueError',
            "All 'ages' must be greater than or equal to 0.0 ({} given).", type(age))

    # Check if age is valid
    stop(type(age) not in (int, float), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Check if initial scatter is valid
    stop(type(initial_scatter) not in (int, float), 'TypeError',
        "'initial_scatter' must be an integer or float ({} given).", type(initial_scatter))
    stop(initial_scatter < 0, 'ValueError', "'initial_scatter' must be greater than "
        "or equal to 0.0 ({} given).", type(initial_scatter))

    # Check if number_of_stars is valid
    stop(type(number_of_stars) not in (int, float), 'TypeError',
        "'number_of_stars' must an integer or float ({} given).", type(number_of_stars))
    stop(number_of_stars % 1 != 0, 'ValueError',
        "'number_of_stars' must be convertible to an integer ({} given).", number_of_stars)
    number_of_stars = int(number_of_stars)

    # Check if number_of_groups is valid
    stop(type(number_of_groups) not in (int, float), 'TypeError',
        "'number_of_groups' must an integer or float ({} given).", type(number_of_groups))
    stop(number_of_groups % 1 != 0, 'ValueError',
        "'number_of_groups' must be convertible to an integer ({} given).", number_of_groups)
    number_of_groups = int(number_of_groups)

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '15'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # Histogram plotting
    hist, bin_edges = np.histogram(ages, density=True)
    plt.hist(ages, bins='auto', density=True) # bins=np.arange(21.975, 26.025, 0.05)

    # Title  and axis formatting
    plt.title(
        'Distribution of ages ({} groups, {} Myr, {} stars,\ninitial scatter = {} pc, {})'.format(
            number_of_groups, age, number_of_stars, initial_scatter,
            'calculated age = ({} ± {}) Myr'.format(
                np.round(np.average(ages), 3), np.round(np.std(ages), 3))))
    plt.xlabel('Age (Myr)')
    plt.ylabel('Number of groups')

    # Save figure
    save_figure(plt, 'Histogram', 'Distribution of ages ({} groups, {} Myr, {} stars, initial '
            'scatter = {} pc).pdf'.format(number_of_groups, age, number_of_stars, initial_scatter),
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_color_mesh(ages, age, initial_scatter, number_of_stars, number_of_groups,
        forced=False, default=False, cancel=False):
    """ Creates a color mesh of ages over the initial scatter and number_of_stars.
        !!! Créer un pour passer d'un array numpy de shape (n, 3) à un color mesh + smoothing !!!
        !!! Genre create_color_mesh(x, y, z, smoothing). !!!
    """

    # Check if ages are valid
    stop(type(ages) not in (tuple, list),
        "'ages' must either must be a tuple or list ({} given)", type(ages))
    for age in ages:
        stop(type(age) not in (int, float), 'TypeError',
            "All 'ages' must be an integer or float ({} given).", type(age))
        stop(age < 0, 'ValueError',
            "All 'ages' must be greater than or equal to 0.0 ({} given).", type(age))

    # Check if age is valid
    stop(type(age) not in (int, float), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Check if initial scatter is valid
    stop(type(initial_scatter) not in (int, float), 'TypeError',
        "'initial_scatter' must be an integer or float ({} given).", type(initial_scatter))
    stop(initial_scatter < 0, 'ValueError', "'initial_scatter' must be greater than "
        "or equal to 0.0 ({} given).", type(initial_scatter))

    # Check if number_of_stars is valid
    stop(type(number_of_stars) not in (int, float), 'TypeError',
        "'number_of_stars' must an integer or float ({} given).", type(number_of_stars))
    stop(number_of_stars % 1 != 0, 'ValueError',
        "'number_of_stars' must be convertible to an integer ({} given).", number_of_stars)
    number_of_stars = int(number_of_stars)

    # Check if number_of_groups is valid
    stop(type(number_of_groups) not in (int, float), 'TypeError',
        "'number_of_groups' must an integer or float ({} given).", type(number_of_groups))
    stop(number_of_groups % 1 != 0, 'ValueError',
        "'number_of_groups' must be convertible to an integer ({} given).", number_of_groups)
    number_of_groups = int(number_of_groups)

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # Mesh plotting
    x, y = np.meshgrid(initial_scatter, number_of_stars)
    grid_x, grid_y = np.mgrid[5:25:300j, 30:200:300j]
    grid_z = griddata(np.array([(i, j) for i in initial_scatter for j in number_of_stars]),
        ages.T.flatten(), (grid_x, grid_y), method='linear')
    plt.pcolormesh(grid_x, grid_y, grid_z, cmap=plt.cm.Greens_r, vmin=0, vmax=2)
    plt.colorbar()

    # Title and axis formatting
    plt.title('Scatter on age (Myr) over the initial scatter (pc)\n'
        'and the number of stars ({} groups, {} Myr)'.format(number_of_groups, age))
    plt.xlabel('Initial scatter (pc)')
    plt.ylabel('Number of stars')
    plt.xticks([5.0, 10.0, 15.0, 20.0, 25.0])

    # Save figure
    save_figure(plt, 'Scatter on age ({} Myr)'.format(age),
        'Scatter on age ({} Myr).png'.format(age), forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_minimum_error_plots(forced=False, default=False, cancel=False):
    """ Creates a plot of the minimal possible errors with 0.0 measurement errors. """

    # Data
    a5 = np.array([
        [0.387, 0.787, 1.152, 1.572, 1.918],
        [0.330, 0.683, 0.994, 1.321, 1.677],
        [0.287, 0.590, 0.905, 1.227, 1.517],
        [0.267, 0.538, 0.783, 1.091, 1.350],
        [0.207, 0.409, 0.632, 0.811, 1.024],
        [0.148, 0.310, 0.426, 0.603, 0.737]])

    a10 = np.array([
        [0.388, 0.770, 1.167, 1.570, 1.978],
        [0.341, 0.653, 1.059, 1.382, 1.608],
        [0.290, 0.583, 0.882, 1.203, 1.509],
        [0.280, 0.550, 0.804, 1.085, 1.342],
        [0.200, 0.412, 0.625, 0.811, 1.009],
        [0.148, 0.298, 0.439, 0.616, 0.714]])

    a24 = np.array([
        [0.401, 0.780, 1.188, 1.586, 1.943],
        [0.331, 0.689, 0.983, 1.348, 1.667],
        [0.292, 0.578, 0.894, 1.175, 1.495],
        [0.270, 0.560, 0.820, 1.090, 1.351],
        [0.215, 0.410, 0.604, 0.829, 1.046],
        [0.148, 0.279, 0.436, 0.597, 0.698]])

    a50 = np.array([
        [0.380, 0.774, 1.195, 1.514, 1.975],
        [0.337, 0.683, 0.986, 1.369, 1.667],
        [0.294, 0.617, 0.859, 1.194, 1.474],
        [0.272, 0.544, 0.806, 1.098, 1.379],
        [0.213, 0.421, 0.633, 0.815, 1.034],
        [0.144, 0.293, 0.446, 0.573, 0.698]])

    a100 = np.array([
        [0.385, 0.786, 1.162, 1.586, 1.992],
        [0.335, 0.685, 0.972, 1.287, 1.673],
        [0.304, 0.596, 0.888, 1.197, 1.518],
        [0.262, 0.547, 0.836, 1.104, 1.370],
        [0.217, 0.421, 0.621, 0.855, 1.042],
        [0.147, 0.297, 0.432, 0.604, 0.729]])

    initial_scatter = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    numbers_of_star = np.array([30, 40, 50, 60, 100, 200])
    number_of_groups = 1000

    # create_color_mesh(initial_scatter, numbers_of_star, a24, 5.0, number_of_groups)
    # create_color_mesh(initial_scatter, numbers_of_star, a24, 10.0, number_of_groups)
    # create_color_mesh(initial_scatter, numbers_of_star, a24, 24.0, number_of_groups)
    # create_color_mesh(initial_scatter, numbers_of_star, a24, 50.0, number_of_groups)
    # create_color_mesh(initial_scatter, numbers_of_star, a24, 100.0, number_of_groups)
    # create_color_mesh(
        # initial_scatter, numbers_of_star, (a5+a10+a24+a50+a100)/5, 'x', number_of_groups)

    # Plotting
    errors = (0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
        4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    ages = (24.001, 23.966, 23.901, 23.74, 23.525, 22.224, 20.301, 18.113, 15.977, 11.293, 7.995,
        5.803, 4.358, 3.364, 2.665, 2.204, 1.756, 1.257, 0.933, 0.735, 0.580, 0.488, 0.346, 0.262,
        0.192, 0.160, 0.134)
    plt.plot(errors, ages, '.-', color='0.0', linewidth=1.0)

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # Title and axis
    plt.xlabel('Error on UVW velcotiy (km/s)')
    plt.ylabel('Age of minimal scatter (Myr)')

    # Show figure
    save_figure(plt, 'Minium Error Plot', 'minimum_error_plot.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()
