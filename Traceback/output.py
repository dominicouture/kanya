# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" output.py: Defines functions to create data output such as plots of size indicators over time,
    2D and 3D scatters at a given time, histograms, color mesh, etc.
"""

import numpy as np
from os import path
from matplotlib import pyplot as plt, lines, ticker as tkr
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import linregress
from Traceback.collection import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Format ticks label with commas instead of dots for French language publications
format_ticks = tkr.FuncFormatter(lambda x, pos: str(round(float(x), 1)))

# Set pyplot rc parameters
plt.rc('font', serif='Latin Modern Math', family='serif', size='12')
plt.rc('mathtext', fontset='custom', it='Latin Modern Roman:italic', rm='Latin Modern Math:roman')
plt.rc('lines', markersize=4)
plt.rc('pdf', fonttype=42)

# Set colors
greys = ('#000000', '#333333', '#666666', '#999999', '#cccccc')
greens = ('#000000', '#005600', '#00a000', '#00df00', '#56f056')
blues = ('#000000', '#002b56', '#0050a0', '#0080ff', '#40a0ff')
oranges = ('#000000', '#562b00', '#a05000', '#ff8000', 'ffa040')
magentas = ('#000000', '#560056', '#a000a0', '#df00df', '#f056f0')
colors = (greens[1], greens[2], blues[2], blues[3], blues[4])

def save_figure(name, file_path=None, forced=False, default=False, cancel=False):
    """ Checks whether a path already exists and asks for user input if it does. The base path
        is assumed to be the output directory. Also, if the path does not have an extension, a
        '.pdf' extension is added.
    """

    # Padding
    padding = 0.01

    # file_path parameter
    file_path = file_path if file_path is not None else output(create=True) + '/'

    # Check if file_path parameter is a string, which must be done before the directory call
    stop(type(file_path) != str, 'TypeError', "'file_path' must be a string ({} given).",
        type(file_path))

    # file_path redefined as the absolute path, default name and directory creation
    file_path = path.join(directory(output(), path.dirname(file_path), 'file_path', create=True),
        path.basename(file_path) if path.basename(file_path) != '' else f'{name}.pdf')

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
                    log("'{}': file not saved because a file already exists at '{}'.",
                        name, file_path)

            # Default name and saving
            if default or choice in ('k', 'keep'):
                from Traceback.tools import default_name
                file_path = default_name(file_path)
                plt.savefig(file_path, bbox_inches='tight', pad_inches=padding)

                # Logging
                log("'{}': file name changed and file saved at '{}'.", name, file_path)

        # Existing file deletion and saving
        if forced or choice in ('y', 'yes'):
            from os import remove
            remove(file_path)
            plt.savefig(file_path, bbox_inches='tight', pad_inches=padding)

            # Logging
            log("'{}': existing file located at '{}' deleted and replaced.", name, file_path)

    # Saving
    else:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=padding)
        # plt.savefig(file_path)

        # Logging
        log("'{}': file saved at '{}'.", name, file_path)

class Output_Series():
    """ Defines output methods from a Series object. """

    def plot_indicator(self, ax, time, indicator, index, label, linestyle, color):
        """ Draws the indicator's value over time on a given axis along with an enveloppe to
            display the uncertainty. The display is further customized with the 'label',
            'linestyle' and 'color' parameters.
        """

        ax.plot(time, indicator.value.T[index],
            linestyle=linestyle, color=color, linewidth=1.5, label=(
                f'{label} : ({indicator.age[index]:.2f}'
                f' ± {indicator.age_error[index]:.2f}) Myr'))
        ax.fill_between(time,
            indicator.value.T[index] - indicator.value_error.T[index],
            indicator.value.T[index] + indicator.value_error.T[index],
            color=color, alpha=0.3, linewidth=0.)

    def create_mad_xyz_plot(self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of the xyz median absolute deviation over the entire duration of
            the data.
        """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Plot total xyz MAD, x MAD, y and z MAD
        self.plot_indicator(ax, -self.time, self.mad_xyz_total, 0, 'MAD${{}}_{{XYZ}}$', '-', colors[0])
        self.plot_indicator(ax, -self.time, self.mad_xyz, 0, 'MAD${{}}_{{x}}$', '-', colors[2])
        self.plot_indicator(ax, -self.time, self.mad_xyz, 1, 'MAD${{}}_{{y}}$', '--', colors[3])
        self.plot_indicator(ax, -self.time, self.mad_xyz, 2, 'MAD${{}}_{{z}}$', ':', colors[4])

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("XYZ MAD of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("Average XYZ MAD of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=2, fontsize=7, fancybox=False)
        ax.set_xlabel('Time (Myr)', fontsize=8)
        ax.set_ylabel('Size (pc)', fontsize=8)
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 40.)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f'MAD_xyz_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_mad_ξηζ_plot(self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of the ξηζ median absolute deviation over the entire duration of
            the data.
        """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Plot total ξηζ MAD, ξ MAD, η MAD and ζ MAD
        self.plot_indicator(ax, -self.time, self.mad_ξηζ_total, 0, 'MAD${{}}_{{ξ^\prime η^\prime ζ^\prime}}$', '-', colors[0])
        self.plot_indicator(ax, -self.time, self.mad_ξηζ, 0, 'MAD${{}}_{{ξ^\prime}}$', '-', colors[2])
        self.plot_indicator(ax, -self.time, self.mad_ξηζ, 1, 'MAD${{}}_{{η^\prime}}$', '--', colors[3])
        self.plot_indicator(ax, -self.time, self.mad_ξηζ, 2, 'MAD${{}}_{{ζ^\prime}}$', ':', colors[4])

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("ξηζ MAD of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("Average ξηζ MAD of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=2, fontsize=7, fancybox=False)
        ax.set_xlabel('Time (Myr)', fontsize=8)
        ax.set_ylabel('Size (pc)', fontsize=8)
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 40.)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f'MAD_ξηζ_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_xyz_plot(self, robust=False, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of X-X, Y-Y and Z-Z covariances, and determinant and trace. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Select empirical or robust indicators
        if robust:
            covariances_xyz_matrix_det = self.covariances_xyz_matrix_det_robust
            covariances_xyz_matrix_trace = self.covariances_xyz_matrix_trace_robust
            covariances_xyz = self.covariances_xyz_robust
        else:
            covariances_xyz_matrix_det = self.covariances_xyz_matrix_det
            covariances_xyz_matrix_trace = self.covariances_xyz_matrix_trace
            covariances_xyz = self.covariances_xyz

        # Plot xyz covariance matrix determinant and trace, x-x covariance, y-y covariance and z-z covariance
        self.plot_indicator(
            ax, -self.time, covariances_xyz_matrix_det, 0, 'Det${{}}_{{XYZ}}$', '-', colors[0])
        self.plot_indicator(
            ax, -self.time, covariances_xyz_matrix_trace, 0, 'Trace${{}}_{{XYZ}}$', '--', colors[1])
        self.plot_indicator(ax, -self.time, covariances_xyz, 0, '$X$-$X$', '-', colors[2])
        self.plot_indicator(ax, -self.time, covariances_xyz, 1, '$Y$-$Y$', '--', colors[3])
        self.plot_indicator(ax, -self.time, covariances_xyz, 2, '$Z$-$Z$', ':', colors[4])

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if robust:
                if self.from_data:
                    ax.set_title(" X-X, Y-Y and Z-Z robust covariances of β Pictoris (without outliners) "
                        "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                        "errors\n".format(
                            self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
                elif self.from_model:
                    ax.set_title(" X-X, Y-Y and Z-Z robust covariances of {} moving group simulations "
                        "with kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                        "correction and actual measurement errors of Gaia DR2\n".format(
                            self.number_of_groups, self.duration.value,
                            round(self.rv_offset.to('km/s').value, 2)))
            else:
                if self.from_data:
                    ax.set_title("X-X, Y-Y and Z-Z covariances of β Pictoris (without outliners) "
                        "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                        "errors\n".format(
                            self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
                elif self.from_model:
                    ax.set_title("X-X, Y-Y and Z-Z covariances of {} moving group simulations with "
                        "kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                        "correction and actual measurement errors of Gaia DR2\n".format(
                            self.number_of_groups, self.duration.value,
                            round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=2, fontsize=7, fancybox=False)
        ax.set_xlabel('Time (Myr)', fontsize=8)
        ax.set_ylabel('Size (pc)', fontsize=8)
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 40.)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f"Covariances_xyz_{self.name}{'_robust' if robust else ''}.pdf",
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_ξηζ_plot(self, robust=False, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of ξ-ξ, η-η and ζ-ζ covariances, and determinant and trace. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Select empirical or robust indicators
        if robust:
            covariances_ξηζ_matrix_det = self.covariances_ξηζ_matrix_det_robust
            covariances_ξηζ_matrix_trace = self.covariances_ξηζ_matrix_trace_robust
            covariances_ξηζ = self.covariances_ξηζ_robust
        else:
            covariances_ξηζ_matrix_det = self.covariances_ξηζ_matrix_det
            covariances_ξηζ_matrix_trace = self.covariances_ξηζ_matrix_trace
            covariances_ξηζ = self.covariances_ξηζ

        # Plot total ξηζ covariance matrix determinant and trace, ξ-ξ covariance, η-η covariance and ζ-ζ covariance
        self.plot_indicator(
            ax, -self.time, covariances_ξηζ_matrix_det, 0, 'Det${{}}_{{ξ^\prime η^\prime ζ^\prime}}$', '-', colors[0])
        self.plot_indicator(
            ax, -self.time, covariances_ξηζ_matrix_trace, 0, 'Trace${{}}_{{ξ^\prime η^\prime ζ^\prime}}$', '--', colors[1])
        self.plot_indicator(ax, -self.time, covariances_ξηζ, 0, '$ξ^\prime$-$ξ^\prime$', '-', colors[2])
        self.plot_indicator(ax, -self.time, covariances_ξηζ, 1, '$η^\prime$-$η^\prime$', '--', colors[3])
        self.plot_indicator(ax, -self.time, covariances_ξηζ, 2, '$ζ^\prime$-$ζ^\prime$', ':', colors[4])

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if robust:
                if self.from_data:
                    ax.set_title(" ξ-ξ, η-η and ζ-ζ robust covariances of β Pictoris (without outliners) "
                        "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                        "errors\n".format(
                            self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
                elif self.from_model:
                    ax.set_title(" ξ-ξ, η-η and ζ-ζ robust covariances of {} moving group simulations "
                        "with kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                        "correction and actual measurement errors of Gaia DR2\n".format(
                            self.number_of_groups, self.duration.value,
                            round(self.rv_offset.to('km/s').value, 2)))
            else:
                if self.from_data:
                    ax.set_title(" ξ-ξ, η-η and ζ-ζ covariances of β Pictoris (without outliners) "
                        "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                        "errors\n".format(
                            self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
                elif self.from_model:
                    ax.set_title(" ξ-ξ, η-η and ζ-ζ covariances of {} moving group simulations with "
                        "kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                        "correction and actual measurement errors of Gaia DR2\n".format(
                            self.number_of_groups, self.duration.value,
                            round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=2, fontsize=7, fancybox=False)
        ax.set_xlabel('Time (Myr)', fontsize=8)
        ax.set_ylabel('Size (pc)', fontsize=8)
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 40.)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f"Covariances_ξηζ_{self.name}{'_robust' if robust else ''}.pdf",
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_cross_covariances_xyz_plot(self, robust=False, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of x-u, y-v and z-w cross covariances, and determinant and trace. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Select empirical or robust indicators
        if robust:
            cross_covariances_xyz_matrix_det = self.cross_covariances_xyz_matrix_det_robust
            cross_covariances_xyz_matrix_trace = self.cross_covariances_xyz_matrix_trace_robust
            cross_covariances_xyz = self.cross_covariances_xyz_robust
        else:
            cross_covariances_xyz_matrix_det = self.cross_covariances_xyz_matrix_det
            cross_covariances_xyz_matrix_trace = self.cross_covariances_xyz_matrix_trace
            cross_covariances_xyz = self.cross_covariances_xyz

        # Plot xyz cross covariance matrix determinant and trace, x-u cross covariance, y-v cross covariance and z-w cross covariance
        self.plot_indicator(
            ax, -self.time, cross_covariances_xyz_matrix_det, 0, 'Det${{}}_{{XYZ}}$', '-', colors[0])
        self.plot_indicator(
            ax, -self.time, cross_covariances_xyz_matrix_trace, 0, 'Trace${{}}_{{XYZ}}$', '--', colors[1])
        self.plot_indicator(ax, -self.time, cross_covariances_xyz, 0, '$X$-$U$', '-', colors[2])
        self.plot_indicator(ax, -self.time, cross_covariances_xyz, 1, '$Y$-$V$', '--', colors[3])
        self.plot_indicator(ax, -self.time, cross_covariances_xyz, 2, '$Z$-$W$', ':', colors[4])

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("X-U, Y-V and Z-W cross covariances of β Pictoris (without outliners) "
                    "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                    "errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("X-U, Y-V and Z-W cross covariances of {} moving group simulations "
                    "with kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=2, fontsize=7, fancybox=False)
        ax.set_xlabel('Time (Myr)', fontsize=8)
        ax.set_ylabel('Size (pc)', fontsize=8)
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 10.)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f"Cross_covariances_xyz_{self.name}{'_robust' if robust else ''}.pdf",
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_cross_covariances_ξηζ_plot(self, robust=False, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of ξ-vξ, η-vη and ζ-vζ cross covariances, and determinant and trace. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Select empirical or robust indicators
        if robust:
            cross_covariances_ξηζ_matrix_det = self.cross_covariances_ξηζ_matrix_det_robust
            cross_covariances_ξηζ_matrix_trace = self.cross_covariances_ξηζ_matrix_trace_robust
            cross_covariances_ξηζ = self.cross_covariances_ξηζ_robust
        else:
            cross_covariances_ξηζ_matrix_det = self.cross_covariances_ξηζ_matrix_det
            cross_covariances_ξηζ_matrix_trace = self.cross_covariances_ξηζ_matrix_trace
            cross_covariances_ξηζ = self.cross_covariances_ξηζ

        # Plot xyz cross covariance matrix determinant and trace, x-u cross covariance, y-v cross covariance and z-w cross covariance
        self.plot_indicator(
            ax, -self.time, cross_covariances_ξηζ_matrix_det, 0, 'Det${{}}_{{ξ^\prime η^\prime ζ^\prime}}$', '-', colors[0])
        self.plot_indicator(
            ax, -self.time, cross_covariances_ξηζ_matrix_trace, 0, 'Trace${{}}_{{ξ^\prime η^\prime ζ^\prime}}$', '--', colors[1])
        self.plot_indicator(ax, -self.time, cross_covariances_ξηζ, 0, '$ξ^\prime$-$\dot{ξ}^\prime$', '-', colors[2])
        self.plot_indicator(ax, -self.time, cross_covariances_ξηζ, 1, '$η^\prime$-$\dot{η}^\prime$', '--', colors[3])
        self.plot_indicator(ax, -self.time, cross_covariances_ξηζ, 2, '$ζFla\prime$-$\dot{ζ}^\prime$', ':', colors[4])

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("ξ-vξ, η-vη and ζ-vζ cross covariances of β Pictoris (without "
                    "outliners) over {} Myr\nwith {} km/s redshift correction and actual "
                    "measurement errors\n".format(
                        self.duration.value,  round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("ξ-vξ, η-vη and ζ-vζ cross covariances of {} moving group simu"
                    "lations with kinematics similar to β Pictoris \n over {} Myr with {} km/s "
                    "redshift correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=2, fontsize=7, fancybox=False)
        ax.set_xlabel('Time (Myr)', fontsize=8)
        ax.set_ylabel('Size (pc)', fontsize=8)
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 10.)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f"Cross_covariances_ξηζ_{self.name}{'_robust' if robust else ''}.pdf",
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_ξηζ_sklearn_plot(self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of ξ-ξ, η-η and ζ-ζ sklearn covariances, and determinant and trace. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # sklearn ξηζ covariance matrix determinant
        ax.plot(-self.time, self.covariances_ξηζ_matrix_det_sklearn.value,
            linestyle='-', color=colors[0], linewidth=1.5, label=(
                f'Det${{}}_{{ξ\prime η\prime ζ\prime}}$ (sklearn) : ({self.covariances_ξηζ_matrix_det_sklearn.age[0]:.2f} '
                f'± {self.covariances_ξηζ_matrix_det_sklearn.age_error[0]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.covariances_ξηζ_matrix_det_sklearn.value - self.covariances_ξηζ_matrix_det_sklearn.value_error,
            self.covariances_ξηζ_matrix_det_sklearn.value + self.covariances_ξηζ_matrix_det_sklearn.value_error,
            color=colors[0], alpha=0.3, linewidth=0.)

        # sklearn ξηζ covariance matrix trace
        ax.plot(-self.time, self.covariances_ξηζ_matrix_trace_sklearn.value,
            linestyle='--', color=colors[1], linewidth=1.5, label=(
                f'Trace${{}}_{{ξ\prime η\prime ζ\prime}}$ (sklearn) : ({self.covariances_ξηζ_matrix_trace_sklearn.age[0]:.2f} '
                f'± {self.covariances_ξηζ_matrix_trace_sklearn.age_error[0]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.covariances_ξηζ_matrix_trace_sklearn.value - self.covariances_ξηζ_matrix_trace_sklearn.value_error,
            self.covariances_ξηζ_matrix_trace_sklearn.value + self.covariances_ξηζ_matrix_trace_sklearn.value_error,
            color=colors[1], alpha=0.3, linewidth=0.)

        # sklearn ξ-ξ covariance
        ax.plot(-self.time, self.covariances_ξηζ_sklearn.value[:,0],
            linestyle='-', color=colors[2], linewidth=1.5, label=(
                f'$ξ\prime$-$ξ\prime$ (sklearn) : ({self.covariances_ξηζ_sklearn.age[0]:.2f}'
                f' ± {self.covariances_ξηζ_sklearn.age_error[0]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.covariances_ξηζ_sklearn.value[:,0].T - self.covariances_ξηζ_sklearn.value_error[:,0].T,
            self.covariances_ξηζ_sklearn.value[:,0].T + self.covariances_ξηζ_sklearn.value_error[:,0].T,
            color=colors[2], alpha=0.3, linewidth=0.)

        # sklearn η-η covariance
        ax.plot(-self.time, self.covariances_ξηζ_sklearn.value[:,1],
            linestyle='--', color=colors[3], linewidth=1.5, label=(
                f'$η\prime$-$η\prime$ (sklearn) : ({self.covariances_ξηζ_sklearn.age[1]:.2f} '
                f'± {self.covariances_ξηζ_sklearn.age_error[1]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.covariances_ξηζ_sklearn.value[:,1].T - self.covariances_ξηζ_sklearn.value_error[:,1].T,
            self.covariances_ξηζ_sklearn.value[:,1].T + self.covariances_ξηζ_sklearn.value_error[:,1].T,
            color=colors[3], alpha=0.3, linewidth=0.)

        # sklearn ζ-ζ covariance
        ax.plot(-self.time, self.covariances_ξηζ_sklearn.value[:,2],
            linestyle=':', color=colors[4], linewidth=1.5, label=(
                f'$ζ\prime$-$ζ\prime$ (sklearn) : ({self.covariances_ξηζ_sklearn.age[2]:.2f} '
                f'± {self.covariances_ξηζ_sklearn.age_error[2]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.covariances_ξηζ_sklearn.value[:,2].T - self.covariances_ξηζ_sklearn.value_error[:,2].T,
            self.covariances_ξηζ_sklearn.value[:,2].T + self.covariances_ξηζ_sklearn.value_error[:,2].T,
            color=colors[4], alpha=0.3, linewidth=0.)

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title(" ξ-ξ, η-η and ζ-ζ sklearn covariances of β Pictoris (without outliners) "
                    "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                    "errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title(" ξ-ξ, η-η and z-z sklearn covariances of {} moving group simulations with "
                    "kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=4, fontsize=6)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Size (pc)')
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 30.)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f'Covariances_ξηζ_sklearn_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_det_mad_mst_plot(self, secondary=False, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of xyz covariance matrix determinant, xyz total median absolute
            deviation, and minimum spanning tree branches length mean and median absolute
            deviation over the entire duration of the data.
        """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # xyz covariance matrix determinant
        ax.plot(-self.time, self.covariances_xyz_matrix_det.value,
            linestyle='-', color=colors[0], linewidth=1.5, label=(
                f'Det${{}}_{{xyz}}$ : ({self.covariances_xyz_matrix_det.age[0]:.2f}'
                f' ± {self.covariances_xyz_matrix_det.age_error[0]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.covariances_xyz_matrix_det.value - self.covariances_xyz_matrix_det.value_error,
            self.covariances_xyz_matrix_det.value + self.covariances_xyz_matrix_det.value_error,
            color=colors[0], alpha=0.3, linewidth=0.)

        # xyz median absolute deviation
        ax.plot(-self.time, self.mad_xyz_total.value,
            linestyle='--', color=colors[1], linewidth=1.5, label=(
                f'MAD${{}}_{{xyz}}$ : ({self.mad_xyz_total.age[0]:.2f}'
                f' ± {self.mad_xyz_total.age_error[0]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.mad_xyz_total.value - self.mad_xyz_total.value_error,
            self.mad_xyz_total.value + self.mad_xyz_total.value_error,
            color=colors[1], alpha=0.3, linewidth=0.)

        # xyz minimum spanning tree mean branch length
        ax.plot(-self.time, self.mst_xyz_mean.value,
            linestyle='-', color=colors[2], linewidth=1.5, label=(
                f'Mean MST : ({self.mst_xyz_mean.age[0]:.2f}'
                f' ± {self.mst_xyz_mean.age_error[0]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.mst_xyz_mean.value - self.mst_xyz_mean.value_error,
            self.mst_xyz_mean.value + self.mst_xyz_mean.value_error,
            color=colors[2], alpha=0.3, linewidth=0.)

        # xyz minimum spanning tree median absolute deviation branch length
        ax.plot(-self.time, self.mst_xyz_mad.value,
            linestyle='--', color=colors[3], linewidth=1.5, label=(
                f'MAD MST : ({self.mst_xyz_mad.age[0]:.2f}'
                f' ± {self.mst_xyz_mad.age_error[0]:.2f}) Myr'))
        ax.fill_between(-self.time,
            self.mst_xyz_mad.value - self.mst_xyz_mad.value_error,
            self.mst_xyz_mad.value + self.mst_xyz_mad.value_error,
            color=colors[3], alpha=0.3, linewidth=0.)

        # Secondary lines
        self.stop(type(secondary) != bool, 'TypeError',
            "'secondary' must be a boolean ({} given).", type(secondary))
        if secondary:
            i = 0
            plot_i = np.arange(0, len(self), 20)
            for group in self:
                if i in plot_i:
                    ax.plot(-self.time, group.covariances_xyz_matrix_det.value,
                        linestyle='-', color='k', alpha=0.5, linewidth=0.5)
                i += 1

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("Size of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("Average size of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=1, fontsize=6)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Size (pc)')
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 40.)
        # ax.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f'Size_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_det_mad_mst_cross_covariances_plots(self, other, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of xyz covariance matrix determinant, xyz total median absolute
            deviation, and minimum spanning tree branches length mean and median absolute
            deviation over the entire duration of the data.
        """

        # Check if 'other' is valid
        self.stop(type(other) != type(self), 'TypeError',
            "'other' must be a Series object ({} given).", type(other))

        # Figure initialization
        self.check_traceback()
        fig, (ax0, ax1) = plt.subplots(
            ncols=2, constrained_layout=True, figsize=(10, 4.2), facecolor='w')

        # xyz covariance matrix determinant
        ax0.plot(-self.time, self.covariances_xyz_matrix_det.value,
            linestyle='-', color=colors[1], linewidth=1.5, label=(
                f'Det${{}}_{{xyz}}$ : ({self.covariances_xyz_matrix_det.age[0]:.2f} '
                f'± {self.covariances_xyz_matrix_det.age_error[0]:.2f}) Myr'))
        ax0.fill_between(-self.time,
            self.covariances_xyz_matrix_det.value - self.covariances_xyz_matrix_det.value_error,
            self.covariances_xyz_matrix_det.value + self.covariances_xyz_matrix_det.value_error,
            color=colors[1], alpha=0.3, linewidth=0.)

        # xyz median absolute deviation
        ax0.plot(-self.time, self.mad_xyz_total.value,
            linestyle='-.', color=colors[2], linewidth=1.5, label=(
                f'MAD${{}}_{{xyz}}$ : ({self.mad_xyz_total.age[0]:.2f} '
                f'± {self.mad_xyz_total.age_error[0]:.2f}) Myr'))
        ax0.fill_between(-self.time,
            self.mad_xyz_total.value - self.mad_xyz_total.value_error,
            self.mad_xyz_total.value + self.mad_xyz_total.value_error,
            color=colors[2], alpha=0.3, linewidth=0.)

        # xyz minimum spanning tree mean branch length
        ax0.plot(-self.time, self.mst_xyz_mean.value,
            linestyle='--', color=colors[3], linewidth=1.5, label=(
                f'Mean MST : ({self.mst_xyz_mean.age[0]:.2f} '
                f'± {self.mst_xyz_mean.age_error[0]:.2f}) Myr'))
        ax0.fill_between(-self.time,
            self.mst_xyz_mean.value - self.mst_xyz_mean.value_error,
            self.mst_xyz_mean.value + self.mst_xyz_mean.value_error,
            color=colors[3], alpha=0.3, linewidth=0.)

        # xyz minimum spanning tree median absolute deviation branch length
        ax0.plot(-self.time, self.mst_xyz_mad.value,
            linestyle=':', color=colors[4], linewidth=1.5, label=(
                f'MAD MST : ({self.mst_xyz_mad.age[0]:.2f} '
                f'± {self.mst_xyz_mad.age_error[0]:.2f}) Myr'))
        ax0.fill_between(-self.time,
            self.mst_xyz_mad.value - self.mst_xyz_mad.value_error,
            self.mst_xyz_mad.value + self.mst_xyz_mad.value_error,
            color=colors[4], alpha=0.3, linewidth=0.)

        # xyz covariance matrix determinant (other)
        ax0.plot(other.time, other.covariances_xyz_matrix_det.value,
            linestyle='-', color=greens[1], linewidth=1.5, label=(
                f'Det${{}}_{{xyz}}$ : ({other.covariances_xyz_matrix_det.age[0]:.2f} '
                f'± {other.covariances_xyz_matrix_det.age_error[0]:.2f}) Myr'))
        ax0.fill_between(other.time,
            other.covariances_xyz_matrix_det.value - other.covariances_xyz_matrix_det.value_error,
            other.covariances_xyz_matrix_det.value + other.covariances_xyz_matrix_det.value_error,
            color=greens[1], alpha=0.3, linewidth=0.)

        # xyz median absolute deviation (other)
        ax0.plot(other.time, other.mad_xyz_total.value,
            linestyle='-.', color=greens[2], linewidth=1.5, label=(
                f'MAD${{}}_{{xyz}}$ : ({other.mad_xyz_total.age[0]:.2f} '
                f'± {other.mad_xyz_total.age_error[0]:.2f}) Myr'))
        ax0.fill_between(other.time,
            other.mad_xyz_total.value - other.mad_xyz_total.value_error,
            other.mad_xyz_total.value + other.mad_xyz_total.value_error,
            color=greens[2], alpha=0.3, linewidth=0.)

        # xyz minimum spanning tree mean branch length (other)
        ax0.plot(other.time, other.mst_xyz_mean.value,
            linestyle='--', color=greens[3], linewidth=1.5, label=(
                f'Mean MST : ({other.mst_xyz_mean.age[0]:.2f} '
                f'± {other.mst_xyz_mean.age_error[0]:.2f}) Myr'))
        ax0.fill_between(other.time,
            other.mst_xyz_mean.value - other.mst_xyz_mean.value_error,
            other.mst_xyz_mean.value + other.mst_xyz_mean.value_error,
            color=greens[3], alpha=0.3, linewidth=0.)

        # xyz minimum spanning tree median absolute deviation branch length (other)
        ax0.plot(other.time, other.mst_xyz_mad.value,
            linestyle=':', color=greens[4], linewidth=1.5, label=(
                f'MAD MST : ({other.mst_xyz_mad.age[0]:.2f} '
                f'± {other.mst_xyz_mad.age_error[0]:.2f}) Myr'))
        ax0.fill_between(other.time,
            other.mst_xyz_mad.value - other.mst_xyz_mad.value_error,
            other.mst_xyz_mad.value + other.mst_xyz_mad.value_error,
            color=greens[4], alpha=0.3, linewidth=0.)

        # Legend and axes formatting
        ax0.legend(loc=4, fontsize=9.5)
        ax0.set_xlabel('Time (Myr)')
        ax0.set_ylabel("Size (pc)")
        ax0.set_xlim(-self.final_time.value + 20., 0.)
        ax0.set_ylim(0., 40.)
        # ax0.yaxis.set_major_formatter(format_ticks)

        # x-u cross covariance
        ax1.plot(-self.time, self.cross_covariances_xyz.value[:,0],
            linestyle='-', color=colors[1], linewidth=1.5, label=(
                f'$X$-$U$ : ({self.cross_covariances_xyz.age[0]:.2f} '
                f'± {self.cross_covariances_xyz.age_error[0]:.2f}) Myr'))
        ax1.fill_between(-self.time,
            self.cross_covariances_xyz.value[:,0] - self.cross_covariances_xyz.value_error[:,0],
            self.cross_covariances_xyz.value[:,0] + self.cross_covariances_xyz.value_error[:,0],
            color=colors[1], alpha=0.3, linewidth=0.)

        # y-v cross covariance
        ax1.plot(-self.time, self.cross_covariances_xyz.value[:,1],
            linestyle='-.', color=colors[2], linewidth=1.5, label=(
                f'$Y$-$V$ : ({self.cross_covariances_xyz.age[1]:.2f} '
                f'± {self.cross_covariances_xyz.age_error[1]:.2f}) Myr'))
        ax1.fill_between(-self.time,
            self.cross_covariances_xyz.value[:,1] - self.cross_covariances_xyz.value_error[:,1],
            self.cross_covariances_xyz.value[:,1] + self.cross_covariances_xyz.value_error[:,1],
            color=colors[2], alpha=0.3, linewidth=0.)

        # z-w cross covariance
        ax1.plot(-self.time, self.cross_covariances_xyz.value[:,2],
            linestyle='--', color=colors[3], linewidth=1.5, label=(
                f'$Z$-$W$ : ({self.cross_covariances_xyz.age[2]:.2f} '
                f'± {self.cross_covariances_xyz.age_error[2]:.2f}) Myr'))
        ax1.fill_between(-self.time,
            self.cross_covariances_xyz.value[:,2] - self.cross_covariances_xyz.value_error[:,2],
            self.cross_covariances_xyz.value[:,2] + self.cross_covariances_xyz.value_error[:,2],
            color=colors[3], alpha=0.3, linewidth=0.)

        # x-u cross covariance (other)
        ax1.plot(other.time, other.cross_covariances_xyz.value[:,0],
            linestyle='-', color=greens[1], linewidth=1.5, label=(
                f'$X-U$ : ({other.cross_covariances_xyz.age[0]:.2f} '
                f'± {other.cross_covariances_xyz.age_error[0]:.2f}) Myr'))
        ax1.fill_between(other.time,
            other.cross_covariances_xyz.value[:,0] - other.cross_covariances_xyz.value_error[:,0],
            other.cross_covariances_xyz.value[:,0] + other.cross_covariances_xyz.value_error[:,0],
            color=greens[1], alpha=0.3, linewidth=0.)

        # y-v cross covariance (other)
        ax1.plot(other.time, other.cross_covariances_xyz.value[:,1],
            linestyle='-.', color=greens[2], linewidth=1.5, label=(
                f'$Y-V$ : ({other.cross_covariances_xyz.age[1]:.2f} '
                f'± {other.cross_covariances_xyz.age_error[1]:.2f}) Myr'))
        ax1.fill_between(other.time,
            other.cross_covariances_xyz.value[:,1] - other.cross_covariances_xyz.value_error[:,1],
            other.cross_covariances_xyz.value[:,1] + other.cross_covariances_xyz.value_error[:,1],
            color=greens[2], alpha=0.3, linewidth=0.)

        # z-w cross covariance (other)
        ax1.plot(other.time, other.cross_covariances_xyz.value[:,2],
            linestyle='--', color=greens[3], linewidth=1.5, label=(
                f'$Z-W$ : ({other.cross_covariances_xyz.age[2]:.2f} '
                f'± {other.cross_covariances_xyz.age_error[2]:.2f}) Myr'))
        ax1.fill_between(other.time,
            other.cross_covariances_xyz.value[:,2] - other.cross_covariances_xyz.value_error[:,2],
            other.cross_covariances_xyz.value[:,2] + other.cross_covariances_xyz.value_error[:,2],
            color=greens[3], alpha=0.3, linewidth=0.)

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                fig.suptitle("XYZ cross covariances of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                fig.suptitle("Average XYZ cross covariances of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax1.legend(loc=4, fontsize=9.5)
        ax1.set_xlabel('Time (Myr)')
        ax1.set_ylabel('Size (pc)')
        ax1.set_xlim(-self.final_time.value + 20., 0.)
        ax1.set_ylim(0., 10.)
        # ax1.yaxis.set_major_formatter(format_ticks)

        # Save figure
        save_figure(self.name, f'Size_covariances_xyz_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_age_distribution(self, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of the distribution of ages computed in a series, including the
            effects of measurement errors and the jack-knife Monte-Carlo.
        """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Plot histogram
        ages = [group.covariances_xyz_matrix_det.age[0] for group in self]
        ax.hist(ages, bins=np.linspace(16, 24, 33), density=True, color=blues[3], alpha=0.66, label='Indicator')
        # bins=np.arange(21.975, 26.025, 0.05)

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                f'Distribution of {self.number_of_groups} moving groups age,\n'
                f'Average age: ({self.covariances_xyz_matrix_det.age[0]:.2f} '
                f'± {self.covariances_xyz_matrix_det.age_error[0]:.2f}) Myr\n')

        # Axes formatting
        ax.set_xlabel('Age (Myr)')
        ax.set_ylabel('Density')
        ax.tick_params(direction='in', top=True, right=True)

        # Save figure
        save_figure(self.name, f'Age_distribution_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def show_metrics(self):
        """ Displays all metrics of a series. """

        def convert(value):
            """ Returns a converted, rounded string for display. """

            return str(np.round(value, 2)).replace('[', '').replace(']', '')

        def create_line(indicator, valid=True):
            """ Creates a single metric (i) line. """

            valid = np.copy(indicator.valid) if valid else np.invert(indicator.valid)

            for i in filter(lambda i: valid[i], np.arange(valid.size)):
                print(f"{indicator.name[i]:<50}"
                    f"{convert(indicator.age_ajusted[i]):>15}"
                    f"{convert(indicator.age_int_error[i]):>20}"
                    f"{convert(indicator.age_ext_error[i]):>20}"
                    f"{convert(indicator.age_error[i]):>15}"
                    f"{convert(indicator.min_change[i]):>20}"
                    f"{convert(indicator.age_offset[i]):>15}")

        # Set precision and order
        np.set_printoptions(precision=2)
        order = np.argsort([i.order for i in self.indicators])

        # Line list
        lines = []

        # Display header
        print(f"{'':-<155}")
        print(f"{'Metric':<50}{'Age':>15}{'Jackknife Error':>20}{'Measurement Error':>20}"
            f"{'Total Error':>15}{'Minimum Change':>20}{'Offset':>15}")
        print(f"{'[Myr]':>65}{'[Myr]':>20}{'[Myr]':>20}{'[Myr]':>15}{'[%]':>20}{'[Myr]':>15}")
        print(f"{'':-<155}")

        # Display errors on age of valid indicators
        print('Valid\n' f"{'':-<155}")
        for i in order:
            create_line(self.indicators[i], valid=True)
        print(f"{'':-<155}")

        # Display errors on age of invalid indicators
        print('Rejected\n' f"{'':-<155}")
        for i in order:
            create_line(self.indicators[i], valid=False)
        print(f"{'':-<155}")

    def convert_to_LaTeX_table(self, table):
        """ Converts a table or array to a LaTeX compatible format. """

        for row in table:
            print(' & '.join([str(i) for i in row], end='\n') + '\\\\')

    def create_table_observations(self):
        """ Creates a table of observations: ra, dec, pmra, pmdec, parallax and radial velocity. """

        # Name and spectral type
        for row in self.data:
            line = [row.name, row.type]

            # Radius and mass
            # line += [row.radius.get_LaTeX(), row.mass.get_LaTeX()]

            # Position and velocity
            position = row.position.to(['mas', 'deg', 'deg'])
            velocity = row.velocity.to(['km/s', 'mas/yr', 'mas/yr'])
            line += [
                position[2].convert_deg(True), position[1].convert_deg(False),
                velocity[2].get_LaTeX(), velocity[1].get_LaTeX(),
                position[0].get_LaTeX(), velocity[0].get_LaTeX()]

            # line += [i.get_LaTeX() for i in (position[0], position[1], position[2])]

            # Velocity

            # line += [i.get_LaTeX() for i in (velocity[0], velocity[1], velocity[2])]

            print(' & '.join([str(i) for i in line]) + '\\\\')


class Output_Group():
    """ Defines output methods from a Group object. """

    def get_indicator(self, indicator, index=None):
        """ Retrieves the proprer Series.indicator instance from a string and index. """

        # Indicator instance
        self.series.stop(type(indicator) != str, 'TypeError',
            "'indicator' must be a string or None ({} given).", type(indicator))
        self.series.stop(indicator not in [indicator.label for indicator in self.series.indicators],
            'ValueError', "'indicator' must be a valid indicator key ({} given).", indicator)
        indicator = vars(self.series)[indicator]

        # indicator index
        if index is not None:
            self.series.stop(type(index) != int, 'TypeError',
                "'index' must be an integer or None ({} given).", type(index))
            self.series.stop(index > indicator.value.size - 1, 'ValueError',
                "'index' is too large for this indicator ({} given, {} in size).", index, indicator.value.size)
        else:
            self.series.stop(indicator.value.size > 1, 'ValueError',
                "No 'index' is provided (indicator is {} in size).", indicator.value.size)

        return (indicator, index if indicator.value.size > 1 else 0)

    def get_epoch(self, age=None, indicator=None, index=None):
        """ Computes the time index of the epoch for a given age or, indicator and dimensional index. """

        # Index from age
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer, float or None ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Index from the epoch of minimal of an indicator
        elif indicator is not None:
            indicator, index = self.get_indicator(indicator, index)
            age = indicator.age_ajusted[index]
        else:
            self.series.stop(True, 'ValueError',
                "Either 'age' or 'indicator' must not be None ({} given).", age, index)

        return int(age / self.series.final_time.value * self.series.number_of_steps)

    def trajectory_xyz(
            self, title=True, labels=False, indicator=None, index=None,
            forced=False, default=False, cancel=False):
        """ Draw the xyz trajectories of stars in the group. """

        # Figure initialization
        fig = plt.figure(figsize=(5, 4.75), facecolor='w', dpi=300)
        ax1 = fig.add_axes([0.10, 0.34, 0.61, 0.61])
        ax2 = fig.add_axes([0.72, 0.34, 0.23, 0.61])
        ax3 = fig.add_axes([0.10, 0.10, 0.61, 0.23])

        # Check labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))

        # Birth epoch
        birth = self.get_epoch(indicator=indicator, index=index)

        # Plot stars trajectories
        for ax, x, y in ((ax1, 1, 0), (ax2, 2, 0), (ax3, 1, 2)):
            for star in self:
                ax.plot(
                    star.position_xyz.T[x] / 1000,
                    star.position_xyz.T[y] / 1000,
                    color='r' if star.outlier else 'k', linewidth=0.4, zorder=1)

                # Plot stars current positions
                ax.scatter(
                    np.array([star.position_xyz[0,x]]) / 1000,
                    np.array([star.position_xyz[0,y]]) / 1000,
                    s=6, marker='s', color='k', zorder=2)

                # Plot stars birth positions
                if birth is not None:
                    ax.scatter(
                        np.array([star.position_xyz[birth,x]]) / 1000,
                        np.array([star.position_xyz[birth,y]]) / 1000,
                        s=6, marker='o', color='r' if star.outlier else 'b', zorder=2)

                # Add stars name
                if self.series.from_data and labels:
                    ax.text(
                        np.array(star.position_xyz.T[0,x]) / 1000,
                        np.array(star.position_xyz.T[0,y]) / 1000,
                        star.name, horizontalalignment='left', fontsize=7)

            # Plot average model star trajectory
            if self.series.from_model:
                ax.plot(
                    self.average_model_star.position_xyz.T[x] / 1000,
                    self.average_model_star.position_xyz.T[y] / 1000,
                    color='g', linewidth=1.0, zorder=3)

                # Plot average model star birth position
                ax.scatter(
                    np.array([self.average_model_star.position_xyz[-1,x]]) / 1000,
                    np.array([self.average_model_star.position_xyz[-1,y]]) / 1000,
                    s=6, marker='*', color='g', zorder=3)

                # Plot average model star current position
                ax.scatter(
                    np.array([self.average_model_star.position_xyz[0,x]]) / 1000,
                    np.array([self.average_model_star.position_xyz[0,y]]) / 1000,
                    s=6, marker='s', color='g', zorder=3)

                # Plot model stars trajectories
                for star in self.model_stars:
                    ax.plot(
                        star.position_xyz.T[x] / 1000,
                        star.position_xyz.T[y] / 1000,
                        color='b', linewidth=0.4, zorder=2)

                    # Plot model stars birth positions
                    ax.scatter(
                        np.array([star.position_xyz[0,x]]) / 1000,
                        np.array([star.position_xyz[0,y]]) / 1000,
                        s=6, marker='*', color='b', zorder=2)

                    # Plot model stars current positions
                    ax.scatter(
                        np.array([star.position_xyz[-1,x]]) / 1000,
                        np.array([star.position_xyz[-1,y]]) / 1000,
                        s=6, marker='s', color='b', zorder=2)

            # Draw vertical and horizontal lines through the Sun's position at the current epoch
            ax.axhline(0., color='0.4', linestyle=':', zorder=0)
            ax.axvline(0., color='0.4', linestyle=':', zorder=0)

        # Draw circles around the galactic center
        for r in range(1, 16):
            ax1.add_artist(plt.Circle((0, 8.122), r, color='0.8', fill=False, linestyle=':', zorder=0))

        # Axes formatting
        ax1.set_xticklabels([])
        ax1.set_ylabel('$X$ (kpc)')
        ax2.set_xlabel('$Z$ (kpc)')
        ax2.set_yticklabels([])
        ax3.set_xlabel('$Y$ (kpc)')
        ax3.set_ylabel('$Z$ (kpc)')

        # Invert y axis
        ax1.invert_xaxis()
        ax3.invert_xaxis()

        # Axes limits
        ax1.set_xlim(1, -13)
        ax1.set_ylim(-1, 13)
        ax2.set_xlim(-0.3, 0.3)
        ax2.set_ylim(-1, 13)
        ax3.set_xlim(1, -13)
        ax3.set_ylim(-0.3, 0.3)

        # Axes ticks
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        ax3.tick_params(direction='in')

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            fig.suptitle("Trajectories xyz of stars in βPMG")

        # Save figure
        save_figure(self.name, f'Trajectory_xyz_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)

    def trajectory_ξηζ(
            self, title=True, labels=False, indicator=None, index=None,
            forced=False, default=False, cancel=False):
        """ Draws the ξηζ trajectories of stars in the group. """

        # Figure initialization
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax1 = fig.add_axes([0.10, 0.34, 0.61, 0.61])
        ax2 = fig.add_axes([0.72, 0.34, 0.23, 0.61])
        ax3 = fig.add_axes([0.10, 0.10, 0.61, 0.23])
        # ax1 = fig.add_axes([0.05, 0.25, 0.61, 0.61])
        # ax2 = fig.add_axes([0.72, 0.25, 0.23, 0.61])
        # ax3 = fig.add_axes([0.05, 0.05, 0.61, 0.23])

        # Check labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))

        # Birth epoch
        birth = self.get_epoch(indicator=indicator, index=index)

        # Plot stars trajectories
        for ax, x, y in ((ax1, 0, 1), (ax2, 2, 1), (ax3, 0, 2)):
            for star in self:
                ax.plot(
                    star.position_ξηζ.T[x],
                    star.position_ξηζ.T[y],
                    color='r' if star.outlier else 'k', linewidth=0.4, zorder=1)

                # Plot stars current positions
                ax.scatter(
                    np.array([star.position_ξηζ[0,x]]),
                    np.array([star.position_ξηζ[0,y]]),
                    s=6, marker='s', color='k', zorder=2)

                # Plot stars birth positions
                if birth is not None:
                    ax.scatter(
                        np.array([star.position_ξηζ[birth,x]]),
                        np.array([star.position_ξηζ[birth,y]]),
                        s=6, marker='o', color='r' if star.outlier else 'b', zorder=2)

                # Add stars name
                if self.series.from_data and labels:
                    ax.text(
                        np.array(star.position_ξηζ.T[0,x]),
                        np.array(star.position_ξηζ.T[0,y]),
                        star.name, horizontalalignment='left', fontsize=7)

            # Plot average model star trajectory
            if self.series.from_model:
                ax.plot(
                    self.average_model_star.position_ξηζ.T[x],
                    self.average_model_star.position_ξηζ.T[y],
                    color='g', linewidth=1.0, zorder=3)

                # Plot average model star birth position
                ax.scatter(
                    np.array([self.average_model_star.position_ξηζ[-1,x]]),
                    np.array([self.average_model_star.position_ξηζ[-1,y]]),
                    s=6, marker='*', color='g', zorder=3)

                # Plot average model star current position
                ax.scatter(
                    np.array([self.average_model_star.position_ξηζ[0,x]]),
                    np.array([self.average_model_star.position_ξηζ[0,y]]),
                    s=6, marker='s', color='g', zorder=3)

                # Model stars
                for star in self.model_stars:
                    ax.plot(
                        star.position_ξηζ.T[x],
                        star.position_ξηζ.T[y],
                        color='b', linewidth=0.4, zorder=2)

                    # Plot model stars birth positions
                    ax.scatter(
                        np.array([star.position_ξηζ[0,x]]),
                        np.array([star.position_ξηζ[0,y]]),
                        s=6, marker='*', color='b', zorder=2)

                    # Plot model stars current positions
                    ax.scatter(
                        np.array([star.position_ξηζ[-1,x]]),
                        np.array([star.position_ξηζ[-1,y]]),
                        s=6, marker='s', color='b', zorder=2)

        # Axes formatting
        ax1.set_xticklabels([])
        ax1.set_ylabel('$η\prime$ (pc)', fontsize=8)
        ax2.set_xlabel('$ζ\prime$ (pc)', fontsize=8)
        ax2.set_yticklabels([])
        ax3.set_xlabel('$ξ\prime$ (pc)', fontsize=8)
        ax3.set_ylabel('$ζ\prime$ (pc)', fontsize=8)

        # Axes limits
        ax1.set_xlim(-225, 60)
        ax1.set_ylim(-45, 110)
        ax2.set_xlim(-40, 49)
        ax2.set_ylim(-45, 110)
        ax3.set_xlim(-225, 60)
        ax3.set_ylim(-40, 49)

        # Axes ticks
        ax1.tick_params(direction='in', top=True, right=True, labelsize=8)
        ax2.tick_params(direction='in', top=True, right=True, labelsize=8)
        ax3.tick_params(direction='in', top=True, right=True, labelsize=8)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            fig.suptitle("Trajectories ξηζ of stars in βPMG")

        # Save figure
        save_figure(self.name, f'Trajectory_ξηζ_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def trajectory_txyz(
            self, title=True, indicator=None, index=None,
            forced=False, default=False, cancel=False):
        """ Draws the xyz trajectories as a function of time of stars. """

        # Figure initialization
        fig = plt.figure(figsize=(8, 7.5), facecolor='w')
        ax1 = fig.add_subplot(224)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(222)
        ax4 = fig.add_subplot(221)

        # Birth epoch
        birth = [self.get_epoch(indicator='covariances_xyz', index=index) for index in range(3)]

        # Plot stars xyz trajectories
        for ax, y in ((ax1, 2), (ax2, 1), (ax3, 0)):
            for star in self:
                color = '#ff0000' if star.outlier else greys[0]
                ax.plot(
                    self.series.time, (star.position_xyz - self.position_xyz)[:,y],
                    color=color, linewidth=0.4, zorder=1)

            # Show vectical dashed line
            ax.axvline(x=self.series.time[birth[y]], color='0.5', linewidth=1.0, linestyle='--')

            # Average model star
            if self.series.from_model:
                position_xyz = np.mean([star.position_xyz[:,y] for star in self.model_stars], axis=0)
                average_model_star_position_xyz = (
                    self.average_model_star.position_xyz[:,y] - position_xyz[::-1])

                # Plot average model star trajectory
                ax.plot(
                    -self.average_model_star.time,
                    average_model_star_position_xyz,
                    color='g', linewidth=1.0, zorder=3)

                # Plot average model star birth and current positions
                for t, x, m in ((-1, -1, '*'), (0, 0, 's')):
                    ax.scatter(
                        -np.array([self.average_model_star.time[t]]),
                        np.array([average_model_star_position_xyz[x]]),
                        s=6, marker=m, color='g', zorder=4)

                # Model stars
                for star in self.model_stars:
                    model_star_position_xyz = star.position_xyz[:,y] - position_xyz

                    # Plot model stars trajectories
                    ax.plot(
                        star.time[::-1], model_star_position_xyz,
                        color='b', linewidth=0.4, zorder=2)

                    # Plot model stars birth and current positions
                    for t, x, m in ((-1, 0, '*'), (0, -1, 's')):
                        ax.scatter(
                            np.array([star.time[t]]),
                            np.array([model_star_position_xyz[x]]),
                            s=6, marker=m, color='b', zorder=2)

        # Plot average xyz trajectories
        for y, ls, label in ((0, '-', '$<X>$'), (1, '--', '$<Y>$'), (2, '-.', '$<Z>$')):
            ax4.plot(
                self.series.time, self.position_xyz[:,y] / 1000, color=greys[y * 2],
                linewidth=1.5, linestyle=ls, label=label, zorder=1)

            # Plot average model star trajectory
            if self.series.from_model:
                average_model_star_position_xyz = self.average_model_star.position_xyz[:,y] / 1000
                ax4.plot(
                    -self.average_model_star.time,
                    average_model_star_position_xyz,
                    color=greens[y + 2], linewidth=1.5, linestyle=ls, zorder=3)

                # Plot average model star birth and current positions
                for t, x, m in ((-1, -1, '*'), (0, 0, 's')):
                    ax4.scatter(
                        -np.array([self.average_model_star.time[t]]),
                        np.array([average_model_star_position_xyz[x]]),
                        s=6, marker=m, color=greens[y + 2], zorder=4)

                # Plot model stars trajectories
                model_star_position_xyz = np.mean(
                    [star.position_xyz[:,y] for star in self.model_stars], axis=0) / 1000
                ax4.plot(
                    self.model_stars[0].time[::-1], model_star_position_xyz,
                    color=blues[y + 2], linewidth=1.5, linestyle=ls, zorder=2)

                # Plot model stars birth and current positions
                for t, x, m in ((-1, 0, '*'), (0, -1, 's')):
                    ax4.scatter(
                        np.array([self.model_stars[0].time[t]]),
                        np.array([model_star_position_xyz[x]]),
                        s=6, marker=m, color=blues[y + 2], zorder=2)

        # Legend and axes formatting
        ax1.set_xlabel('Time (Myr)')
        ax2.set_xlabel('Time (Myr)')
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax1.set_ylabel('$Z - <Z>$ (pc)')
        ax2.set_ylabel('$Y - <Y>$ (pc)')
        ax3.set_ylabel('$X - <X>$ (pc)')
        ax4.set_ylabel('$<XYZ>$ (kpc)')
        ax4.legend(loc=2, fontsize=9)

        # Axes limits
        ax1.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))
        ax2.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))
        ax3.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))
        ax4.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))

        # Axes ticks
        ax1.tick_params(direction='in', top=True, right=True)
        ax2.tick_params(direction='in', top=True, right=True)
        ax3.tick_params(direction='in', top=True, right=True)
        ax4.tick_params(direction='in', top=True, right=True)

        # Set axes location
        ax1.yaxis.set_label_position('right')
        ax3.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax3.yaxis.tick_right()

        # Tight layout
        fig.tight_layout(h_pad=0.5, w_pad=1.0)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(f'XYZ trajectories of stars in {self.name}')

        # Save figure
        save_figure(self.name, f"Trajectory_txyz_{self.name}.pdf",
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def trajectory_tξηζ(self, title=True, forced=False, default=False, cancel=False):
        """ Draws the ξηζ trajectories as a function of time of stars. """

        # Figure initialization
        fig = plt.figure(figsize=(8, 7.5), facecolor='w')
        ax1 = fig.add_subplot(224)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(222)
        ax4 = fig.add_subplot(221)

        # Birth epoch
        birth = [self.get_epoch(indicator='covariances_ξηζ', index=index) for index in range(3)]

        # Plot stars ξηζ trajectories
        for ax, y in ((ax1, 2), (ax2, 1), (ax3, 0)):
            for star in self:
                color = '#ff0000' if star.outlier else greys[0]
                ax.plot(
                    self.series.time, (star.position_ξηζ - self.position_ξηζ)[:,y],
                    color=color, linewidth=0.4, zorder=1)

            # Show vectical dashed line
            ax.axvline(x=self.series.time[birth[y]], color='0.5', linewidth=1.0, linestyle='--')

            # Plot average model star trajectory
            if self.series.from_model:
                position_ξηζ = np.mean([star.position_ξηζ[:,y] for star in self.model_stars], axis=0)
                average_model_star_position_ξηζ = (
                    self.average_model_star.position_ξηζ[:,y] - position_ξηζ[::-1])
                ax.plot(
                    -self.average_model_star.time,
                    average_model_star_position_ξηζ,
                    color='g', linewidth=1.0, zorder=3)

                # Plot average model star birth and current positions
                for t, x, m in ((-1, -1, '*'), (0, 0, 's')):
                    ax.scatter(
                        -np.array([self.average_model_star.time[t]]),
                        np.array([average_model_star_position_ξηζ[x]]),
                        s=6, marker=m, color='g', zorder=4)

                # Plot model stars trajectories
                for star in self.model_stars:
                    model_star_position_ξηζ = star.position_ξηζ[:,y] - position_ξηζ
                    ax.plot(
                        star.time[::-1], model_star_position_ξηζ,
                        color='b', linewidth=0.4, zorder=2)

                    # Plot model stars birth and current positions
                    for t, x, m in ((-1, 0, '*'), (0, -1, 's')):
                        ax.scatter(
                            np.array([star.time[t]]),
                            np.array([model_star_position_ξηζ[x]]),
                            s=6, marker=m, color='b', zorder=2)

        # Plot average ξηζ trajectories
        for y, ls, label in ((0, '-', '$<ξ\prime>$'), (1, '--', '$<η\prime>$'), (2, '-.', '$<ζ\prime>$')):
            ax4.plot(
                self.series.time, self.position_ξηζ[:,y], color=greys[y * 2],
                linewidth=1.5, linestyle=ls, label=label, zorder=1)

            # Plot average model star trajectory
            if self.series.from_model:
                average_model_star_position_ξηζ = self.average_model_star.position_ξηζ[:,y]
                ax4.plot(
                    -self.average_model_star.time,
                    average_model_star_position_ξηζ,
                    color=greens[y + 2], linewidth=1.5, linestyle=ls, zorder=3)

                # Plot average model star birth and current positions
                for t, x, m in ((-1, -1, '*'), (0, 0, 's')):
                    ax4.scatter(
                        -np.array([self.average_model_star.time[t]]),
                        np.array([average_model_star_position_ξηζ[x]]),
                        s=6, marker=m, color=greens[y + 2], zorder=4)

                # Plot model stars trajectories
                model_star_position_ξηζ = np.mean(
                    [star.position_ξηζ[:,y] for star in self.model_stars], axis=0)
                ax4.plot(
                    self.model_stars[0].time[::-1], model_star_position_ξηζ,
                    color=blues[y + 2], linewidth=1.5, linestyle=ls, zorder=2)

                # Plot model stars birth and current positions
                for t, x, m in ((-1, 0, '*'), (0, -1, 's')):
                    ax4.scatter(
                        np.array([self.model_stars[0].time[t]]),
                        np.array([model_star_position_ξηζ[x]]),
                        s=6, marker=m, color=blues[y + 2], zorder=2)

        # Legend and axes formatting
        ax1.set_xlabel('Time (Myr)')
        ax2.set_xlabel('Time (Myr)')
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax1.set_ylabel('$ζ\prime - <ζ\prime>$ (pc)')
        ax2.set_ylabel('$η\prime - <η\prime>$ (pc)')
        ax3.set_ylabel('$ξ\prime - <ξ\prime>$ (pc)')
        ax4.set_ylabel('$<ξ\prime η\prime ζ\prime>$ (pc)')
        ax4.legend(loc=2, fontsize=9, fancybox=False)

        # Axes limits
        ax1.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))
        ax2.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))
        ax3.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))
        ax4.set_xlim(np.min(self.series.time) - 1, np.max(self.series.time))

        # Axes ticks
        ax1.tick_params(direction='in', top=True, right=True)
        ax2.tick_params(direction='in', top=True, right=True)
        ax3.tick_params(direction='in', top=True, right=True)
        ax4.tick_params(direction='in', top=True, right=True)

        # Set axes location
        ax1.yaxis.set_label_position('right')
        ax3.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax3.yaxis.tick_right()

        # Tight layout
        fig.tight_layout(h_pad=0.5, w_pad=1.0)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(f'ξηζ trajectories of stars in {self.name}')

        # Save figure
        save_figure(self.name, f'Trajectory_tξηζ_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_map(self, labels=False, title=True, forced=False, default=False, cancel=False):
        """ Creates a Mollweide projection of a traceback. For this function to work, uvw
            velocities must not compensated for the sun velocity and computing xyz positions.
        """

        # Figure initialization
        fig = plt.figure(figsize=(8, 4), facecolor='w')
        ax = fig.add_subplot(111, projection="mollweide")

        # Coordinates computation
        from Traceback.coordinate import galactic_xyz_equatorial_rδα
        positions = np.array([[galactic_xyz_equatorial_rδα(*star.position_xyz[step])[0] \
            for step in range(self.series.number_of_steps)] for star in self])
        alphas = np.vectorize(lambda α: α - (2 * np.pi if α > np.pi else 0.0))(positions[:,:,2])
        deltas = positions[:,:,1]

        # Trajectories
        for star in range(len(self)):
            color = '#2635ff' if not self[star].outlier else '#ff2634'

            # Identify discontinuties
            discontinuties = (np.abs(alphas[star, 1:] - alphas[star, :-1]) > 3 * np.pi / 2).nonzero()[0] + 1

            # Create individual segments
            segments = []
            lower_limit = 0
            for upper_limit in discontinuties:
                segments.append(range(lower_limit, upper_limit))
                lower_limit = upper_limit
            segments.append(np.arange(lower_limit, alphas.shape[1]))

            # Plot individual segments
            for i in segments:
                ax.plot(alphas[star, i], deltas[star, i], color=color, lw=1, zorder=2)

            # Labels
            if labels:
                ax.text(alphas[star, 0] + 0.1, deltas[star, 0] + 0.1, star + 1,
                    horizontalalignment='left', fontsize=7, zorder=2)

        # Scatter
        ax.scatter(alphas[:,0], deltas[:,0], marker='.', color='k', zorder=3)

        # Proper motion arrows
        for star in self.series.data:
            ax.arrow(
                star.position.values[2] - (2 * np.pi if star.position.values[2] > np.pi else 0.0),
                star.position.values[1], -star.velocity.values[2]/4, -star.velocity.values[1]/4,
                head_width=0.03, head_length=0.03, color='k', zorder=4)

        # Axes formatting
        ax.grid(zorder=1)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title('Mollweide projection of tracebacks.')

        # Save figure
        save_figure(self.name, f'Mollweide_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_scatter(
            self, i, j, step=None, age=None, errors=False, labels=False, mst=False,
            title=True, forced=False, default=False, cancel=False):
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
        fig = plt.figure(figsize=(6, 5.5), facecolor='w')
        ax = fig.add_subplot(111)

        # Scatter
        ax.scatter(
            [star.position_xyz[step, i] for star in self.sample],
            [star.position_xyz[step, j] for star in self.sample], marker='o', color='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.sample:
                position = star.position_xyz[step]
                error = star.position_xyz_error[step]

                # Horizontal error bars
                ax.plot(
                    (position[i] - error[i], position[i] + error[i]),
                    (position[j], position[j]), color='0.1', linewidth=0.7)

                # Vertical error bars
                ax.plot(
                    (position[i], position[i]),
                    (position[j] - error[j], position[j] + error[j]), color='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.sample:
                ax.text(star.position_xyz[step, i] + 1, star.position_xyz[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=7)

        # Branches creation
        self.series.stop(type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (branch.start.position[step, i], branch.end.position[step, i]),
                    (branch.start.position[step, j], branch.end.position[step, j]), color='b')

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("{} and {} positions of stars in β Pictoris at {} Myr "
                "wihtout outliers.\n".format(keys[i].upper(), keys[j].upper(), age))

        # Axes formatting
        ax.set_xlabel(f'${keys[i].lower()}$ (pc)')
        ax.set_ylabel(f'${keys[j].lower()}$ (pc)')

        # Save figure
        save_figure(self.name,
            f'2D_Scatter_{self.name}_{keys[i].upper()}{keys[j].upper()}_at_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_3D_scatter(
            self, step=None, age=None, errors=False, labels=False, mst=False,
            title=True, forced=False, default=False, cancel=False):
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
        fig = plt.figure(figsize=(6, 5.5), facecolor='w')
        ax = fig.add_subplot(111, projection='3d')

        # Scatter
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.sample],
            [star.relative_position_xyz[step, 1] for star in self.sample],
            [star.relative_position_xyz[step, 2] for star in self.sample], marker='o', color='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.sample:
                position = star.relative_position_xyz[step]
                error = star.relative_position_xyz_error[step]

                # X axis error bars
                ax.plot(
                    (position[0] - error[0], position[0] + error[0]),
                    (position[1], position[1]), (position[2], position[2]), color='0.1', linewidth=0.7)

                # Y axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1] - error[1], position[1] + error[1]),
                    (position[2], position[2]), color='0.1', linewidth=0.7)

                # Z axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1], position[1]),
                    (position[2] - error[2], position[2] + error[2]), color='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.sample:
                ax.text(star.position_xyz[step, 0] + 2, star.position_xyz[step, 1] + 2,
                    star.position_xyz[step, 2] + 2, star.name, horizontalalignment='left', fontsize=7)

        # Branches creation
        self.series.stop(type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (branch.start.relative_position_xyz[step, 0],
                        branch.end.relative_position_xyz[step, 0]),
                    (branch.start.relative_position_xyz[step, 1],
                        branch.end.relative_position_xyz[step, 1]),
                    (branch.start.relative_position_xyz[step, 2],
                        branch.end.relative_position_xyz[step, 2]), color='b')

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(f'Minimum spanning tree of stars in β Pictoris at {age:.1f} Myr\n')

        # Axes formatting
        ax.set_xlabel('\n $x$ (pc)')
        ax.set_ylabel('\n $y$ (pc)')
        ax.set_zlabel('\n $z$ (pc)')

        # Save figure
        save_figure(self.name, f'3D_Scatter_{self.name}_at_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_and_3D_scatter(self, ages, title=True, forced=False, default=False, cancel=False):
        """ Creates a three 4-axis columns of xy, xz and yz 2D scatters, and a 3D scatter at three
            ages definied by a list or tuple.
        """

        # Check if ages is valid
        self.series.stop(type(ages) not in (tuple, list), 'TypeError',
            "'ages' must a tuple or a list ({} given).", type(ages))
        self.series.stop(len(ages) != 3, 'ValueError',
            "'ages' must be have a length of 3 ({} given).", ages)

        # Figure initialization
        self.fig = plt.figure(figsize=(9, 11.15), facecolor='w')

        # Axes creation
        row1 = 0.795
        row2 = 0.545
        row3 = 0.295
        row4 = 0.035
        col1 = 0.070
        col2 = 0.398
        col3 = 0.730
        self.create_2D_axis('x', 'y', age=ages[0], index=1, left=col1, bottom=row1)
        self.create_2D_axis('x', 'y', age=ages[1], index=2, left=col2, bottom=row1)
        self.create_2D_axis('x', 'y', age=ages[2], index=3, left=col3, bottom=row1)
        self.create_2D_axis('x', 'z', age=ages[0], index=4, left=col1, bottom=row2)
        self.create_2D_axis('x', 'z', age=ages[1], index=5, left=col2, bottom=row2)
        self.create_2D_axis('x', 'z', age=ages[2], index=6, left=col3, bottom=row2)
        self.create_2D_axis('y', 'z', age=ages[0], index=7, left=col1, bottom=row3)
        self.create_2D_axis('y', 'z', age=ages[1], index=8, left=col2, bottom=row3)
        self.create_2D_axis('y', 'z', age=ages[2], index=9, left=col3, bottom=row3)
        self.create_3D_axis(age=ages[0], index=10, left=0.0535, bottom=row4)
        self.create_3D_axis(age=ages[1], index=11, left=0.381, bottom=row4)
        self.create_3D_axis(age=ages[2], index=12, left=0.712, bottom=row4)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title('XY, XZ, YZ and 3D scatters at '
                f'{ages[0]:.1f}, {ages[1]:.1f} and {ages[2]:.1f}Myr.\n')

        # Save figure
        save_figure(self.name, '2D_Scatter_{}_{}_{}_{}_Myr.pdf'.format(self.name, *ages),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_axis(self, i, j, step=None, age=None, index=1, left=0, bottom=0):
        """ Creates a 2D axis. """

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

        # Axis initialization
        ax = self.fig.add_subplot(4, 3, index, position=[left, bottom, 0.255, 0.20])

        # Star scatter
        ax.scatter(
            [star.relative_position_xyz[step, i] for star in self.sample],
            [star.relative_position_xyz[step, j] for star in self.sample],
            marker='o', s=8, color='k')

        # Outliers scatter
        ax.scatter(
            [star.relative_position_xyz[step, i] for star in self.outliers],
            [star.relative_position_xyz[step, j] for star in self.outliers],
            marker='o', s=8, color='#ff2634')

        # Error bars
        for star in self:
            position = star.relative_position_xyz[step]
            error = star.position_xyz_error[step]
            color = 'k' if not star.outlier else '#ff2634'

            # Horizontal error bars
            ax.plot(
                (position[i] - error[i], position[i] + error[i]),
                (position[j], position[j]), color=color, linewidth=0.7)

            # Vertical error bars
            ax.plot(
                (position[i], position[i]),
                (position[j] - error[j], position[j] + error[j]), color=color, linewidth=0.7)

        # Axes formatting
        ax.set_xlabel(f'${keys[i].upper()}$ (pc)')
        ax.set_ylabel(f'${keys[j].upper()}$ (pc)', labelpad=-12.)

        # Limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

    def create_3D_axis(self, step=None, age=None, index=1, left=0, bottom=0):
        """ Creates a 3D axis. """

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
        ax = self.fig.add_subplot(
            4, 3, index, projection='3d', position=[left, bottom, 0.29, 0.215])

        # Star scatter
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.sample],
            [star.relative_position_xyz[step, 1] for star in self.sample],
            [star.relative_position_xyz[step, 2] for star in self.sample], marker='o', color='0.0')

        # Outlier scatter
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.outliers],
            [star.relative_position_xyz[step, 1] for star in self.outliers],
            [star.relative_position_xyz[step, 2] for star in self.outliers], marker='o', color='#ff2634')

        # Branches creation
        for branch in self.mst[step]:
            ax.plot(
                (branch.start.relative_position_xyz[step, 0],
                    branch.end.relative_position_xyz[step, 0]),
                (branch.start.relative_position_xyz[step, 1],
                    branch.end.relative_position_xyz[step, 1]),
                (branch.start.relative_position_xyz[step, 2],
                    branch.end.relative_position_xyz[step, 2]), color='#2635ff')

        # Axes formatting
        ax.view_init(azim=45)
        ax.set_xlabel('$x$ (pc)')
        ax.set_ylabel('$y$ (pc)')
        ax.set_zlabel('$z$ (pc)')

        # Axes limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)

    def create_cross_covariances_scatter(
            self, i, j, step=None, age=None, errors=False, labels=False,
            title=True, forced=False, default=False, cancel=False):
        """ Creates a cross covariance scatter of star positions and velocities in i and j at a
            given 'step' or 'age' in Myr. If 'age' doesn't match a step, the closest step is used
            instead. 'age' overrules 'steps' if both are given. 'labels' adds the stars' name
            and 'mst' adds the minimum spanning tree branches.
        """

        # Axes initialization
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
        fig = plt.figure(figsize=(6, 5.5), facecolor='w')
        ax = fig.add_subplot(111)

        # Scatter
        ax.scatter(
            [star.position_xyz[step, i] for star in self.sample],
            [star.velocity_xyz[step, j] for star in self.sample], marker='o', color='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.sample:
                position = star.position_xyz[step]
                position_error = star.position_xyz_error[step]
                velocity = star.velocity_xyz[step]
                velocity_error = star.velocity_xyz_error[step]

                # Position (horizontal) error bars
                ax.plot(
                    (position[i] - position_error[i], position[i] + position_error[i]),
                    (velocity[j], velocity[j]),
                    color='0.1', linewidth=0.7)

                # Velocity (vertical) error bars
                ax.plot(
                    (position[i], position[i]),
                    (velocity[j] - velocity_error[j], velocity[j] + velocity_error[j]),
                    color='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.sample:
                ax.text(star.position_xyz[step, i] + 1, star.velocity_xyz[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=7)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("{} and {} covariance of stars in β Pictoris at {} Myr wihtout "
                "outliers.\n".format(position_keys[i].upper(), velocity_keys[j].upper(), age))

        # Axes formatting
        ax.set_xlabel(f'{position_keys[i].upper()} (pc)')
        ax.set_ylabel(f'{velocity_keys[j].upper()} (pc/Myr)')

        # Save figure
        save_figure(self.name, f'Covariances_Scatter_{self.name}_'
                f'{position_keys[i].upper()}-{velocity_keys[j].upper()}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_age_distribution(
            self, title=True, indicator=None, index=None,
            forced=False, default=False, cancel=False):
        """ Creates a plot of the distribution of jack-knife Monte-Carlo ages computed in a group. """

        # Figure initialization
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w')
        ax = fig.add_subplot(111)

        # Retrieve ages
        indicator, index = self.get_indicator(indicator, index)
        indicator_name = indicator.label
        ages = indicator.ages
        if ages.ndim == 2:
            ages = ages[self.number]
        elif ages.ndim == 3:
            ages = ages[self.number,:,index]

        # Plot uncorrected histogram and gaussian curve
        x = np.linspace(8, 36, 1000)
        μ = indicator.age[index]
        σ = indicator.age_int_error[index]
        print(μ, σ)
        gauss = np.exp(-0.5 * ((x - μ) / σ)**2) / np.sqrt(2 * np.pi) / σ
        i, = (gauss > 0.001).nonzero()
        ax.plot(x[i], gauss[i], color=blues[3], zorder=0.8, label='$\\xi^\\prime$ variance')
        ax.hist(
            ages, bins=np.linspace(12, 32, 81), density=True,
            color=blues[3], alpha=0.3, zorder=0.8)
        ax.vlines(μ, ymin=0.0, ymax=np.max(gauss), color=blues[3], linestyle='--', zorder=0.8)

        # Plot corrected histogram and gaussian curve
        x = np.linspace(8, 36, 1000)
        μ = indicator.age_ajusted[index]
        σ = (indicator.age_int_error[index]**2 + 1.56**2)**0.5
        print(μ, σ)
        gauss = np.exp(-0.5 * ((x - μ) / σ)**2) / np.sqrt(2 * np.pi) / σ
        i, = (gauss > 0.001).nonzero()
        ax.plot(x[i], gauss[i], color=blues[2], zorder=0.9, label='Corrected $\\xi^\\prime$ variance')
        ages = (ages - indicator.age[index]) * (σ / indicator.age_int_error[index]) + μ
        ax.hist(
            ages, bins=np.linspace(12, 32, 81), density=True,
            color=blues[2], alpha=0.3, zorder=0.9)
        # ax.fill_between(x[i], np.zeros_like(x[i]), gauss[i], color=blues[2], alpha=0.3, linewidth=0., zorder=0.3)
        ax.vlines(μ, ymin=0.0, ymax=np.max(gauss), color=blues[2], linestyle='--', zorder=0.9)

        # Plot gaussian curve from Miret-Roig et al. (2020)
        μ = 18.5
        σ1, σ2 = 2.4, 2.0
        x1, x2 = np.arange(μ - 10, μ, 0.005), np.arange(μ, μ + 10, 0.001)
        gauss1 = np.exp(-0.5 * ((x1 - μ) / σ1)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        gauss2 = np.exp(-0.5 * ((x2 - μ) / σ2)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        x = np.concatenate((x1, x2))
        gauss = np.concatenate((gauss1, gauss2))
        i, = (gauss > 0.001).nonzero()
        ax.plot(x[i], gauss[i], color=greens[3], zorder=0.7, label='Miret-Roig et al. (2020)')
        ax.fill_between(x[i], np.zeros_like(x[i]), gauss[i], color=greens[1], alpha=0.3, linewidth=0., zorder=0.7)
        ax.vlines(μ, ymin=0.0, ymax=np.max(gauss), color=greens[1], linestyle='--', zorder=0.7)

        # Plot gaussian curve from Crundall et al. (2019)
        μ = 18.3
        σ1, σ2 = 1.2, 1.3
        x1, x2 = np.arange(μ - 10, μ, 0.005), np.arange(μ, μ + 10, 0.001)
        gauss1 = np.exp(-0.5 * ((x1 - μ) / σ1)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        gauss2 = np.exp(-0.5 * ((x2 - μ) / σ2)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        x = np.concatenate((x1, x2))
        gauss = np.concatenate((gauss1, gauss2))
        i, = (gauss > 0.001).nonzero()
        ax.plot(x[i], gauss[i], color=greens[2], zorder=0.6, label='Crundall et al. (2019)')
        ax.fill_between(x[i], np.zeros_like(x[i]), gauss[i], color=greens[2], alpha=0.3, linewidth=0., zorder=0.6)
        ax.vlines(μ, ymin=0.0, ymax=np.max(gauss), color=greens[2], linestyle='--', zorder=0.6)

        # Show a shaded area for LDB and isochrone ages
        LDB_range = np.array([20, 26])
        ax.fill_between(LDB_range, 0, 1, color='0.5', alpha=0.2,  linewidth=0., transform=ax.get_xaxis_transform(), zorder=0.5)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                f'Distribution of {self.series.jackknife_number} jack knife Monte-Carlo,\n'
                f'Average age: ({indicator.age[0]:.2f} '
                f'± {indicator.age_int_error[0]:.2f}) Myr\n')

        # Axes formatting
        ax.set_xlabel('Age (Myr)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.set_xlim(13, 29)
        ax.set_xticks([14., 16., 18., 20., 22., 24., 26., 28.])
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        ax.legend(loc=1, fontsize=8, fancybox=False)

        # Save figure
        save_figure(self.name, f'Age_distribution_Jack_knife_{self.name}_{indicator_name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

def create_histogram(ages, initial_scatter, number_of_stars, number_of_groups, age,
        title=True, forced=False, default=False, cancel=False):
    """ Creates an histogram of ages computed by multiple tracebacks. """

    # Check if ages are valid
    stop(type(ages) not in (tuple, list),
        "'ages' must either must be a tuple or list ({} given)", type(ages))
    for age in ages:
        stop(type(age) not in (int, float), 'TypeError',
            "All 'ages' must be an integer or float ({} given).", type(age))
        stop(age < 0, 'ValueError',
            "All 'ages' must be greater than or equal to 0.0 ({} given).", type(age))

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

    # Check if age is valid
    stop(type(age) not in (int, float), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Figure initialization
    fig = plt.figure(figsize=(6, 5.5), facecolor='w')
    ax = fig.add_subplot(111)

    # Histogram plotting
    hist, bin_edges = np.histogram(ages, density=True)
    ax.hist(ages, bins='auto', density=True) # bins=np.arange(21.975, 26.025, 0.05)

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title(
            "Distribution of ages ({} groups, {} Myr, {} stars,\ninitial scatter = "
                "{} pc, {})".format(number_of_groups, age, number_of_stars, initial_scatter,
                    'calculated age = ({} ± {}) Myr'.format(
                        np.round(np.average(ages), 3), np.round(np.std(ages), 3))))

    # Axes formatting
    ax.set_xlabel('Age (Myr)')
    ax.set_ylabel('Number of groups')

    # Save figure
    save_figure(self.name, f'Distribution of ages for {number_of_groups} groups, {age:.1f}Myr, '
            f'{number_of_stars} stars, initial scatter = {initial_scatter}pc.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_color_mesh(initial_scatter, number_of_stars, errors, age, number_of_groups, method,
        title=True, forced=False, default=False, cancel=False):
    """ Creates a color mesh of errors over the initial scatter and number_of_stars.
        !!! Créer une fonction pour passer d'un array Numpy de shape (n, 3) à un !!!
        !!! color mesh + smoothing, genre create_color_mesh(x, y, z, smoothing). !!!
    """

    # Check if initial scatter is valid
    stop(type(initial_scatter) not in (tuple, list, np.ndarray),
        "'initial_scatter' must either must be a tuple or list ({} given)", type(initial_scatter))
    for scatter in np.array(initial_scatter).flatten():
        stop(type(scatter) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'initial_scatter' must be an integer or float ({} given).", type(scatter))
        stop(age < 0, 'ValueError',
            "All 'initial_scatter' must be greater than or equal to 0.0 ({} given).", type(scatter))

    # Check if number_of_stars is valid
    stop(type(number_of_stars) not in (tuple, list, np.ndarray),
        "'number_of_stars' must either must be a tuple or list ({} given)", type(errors))
    for star in np.array(number_of_stars).flatten():
        stop(type(star) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'initial_scatter' must be an integer or float ({} given).", type(star))
        stop(star < 0, 'ValueError',
            "All 'initial_scatter' must be greater than or equal to 0.0 ({} given).", type(star))
        stop(star % 1 != 0, 'ValueError',
            "All 'number_of_stars' must be convertible to an integer ({} given).", star)

    # Check if errors are valid
    stop(type(errors) not in (tuple, list, np.ndarray),
        "'errors' must either must be a tuple or list ({} given)", type(errors))
    for error in np.array(errors).flatten():
        stop(type(error) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'errors' must be an integer or float ({} given).", type(error))
        stop(error < 0, 'ValueError',
            "All 'errors' must be greater than or equal to 0.0 ({} given).", type(error))

    # Check if age is valid
    stop(type(age) not in (int, float, np.int64, np.float64), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Check if number_of_groups is valid
    stop(type(number_of_groups) not in (int, float, np.int64, np.float64), 'TypeError',
        "'number_of_groups' must an integer or float ({} given).", type(number_of_groups))
    stop(number_of_groups % 1 != 0, 'ValueError',
        "'number_of_groups' must be convertible to an integer ({} given).", number_of_groups)
    number_of_groups = int(number_of_groups)

    # Check if method is valid
    stop(type(method) != str, 'TypeError', "'method' must a string ({} given).", type(method))

    # Figure initialization
    fig = plt.figure(figsize=(5, 4), facecolor='w')
    ax = fig.add_subplot(111)

    # Mesh plotting
    x, y = np.meshgrid(initial_scatter, number_of_stars)
    grid_x, grid_y = np.mgrid[0:20:81j, 20:100:81j]
    grid_z = griddata(np.array([(i, j) for i in initial_scatter for j in number_of_stars]),
        errors.T.flatten(), (grid_x, grid_y), method='linear')
    ax.pcolormesh(grid_x, grid_y, grid_z, cmap=plt.cm.PuBu_r, vmin=0, vmax=6)
    fig.colorbar(mappable=plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 6.0), cmap=plt.cm.PuBu_r),
        ax=ax, ticks=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], format='%0.1f')

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title('Scatter on age (Myr) over the initial scatter (pc)\n'
            f'and the number of stars ({number_of_groups} groups, {age:.1f}Myr)')

    # Axes formatting
    ax.set_xlabel('Initial scatter (pc)')
    ax.set_ylabel('Number of stars')
    ax.set_xticks([0., 5., 10., 15., 20.])
    ax.set_yticks([20., 40., 60., 80., 100.])

    # Save figure
    save_figure(self.name, f'Scatter on age ({age:.1f}Myr, {method}).pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def plot_age_error(title=True, forced=False, default=False, cancel=False):
    """ Creates a plot of ages obtained for diffrent measurement errors on radial velocity
        and offset due to gravitationnal redshift.
    """

    # Figure initialization
    fig = plt.figure(figsize=(7, 5.8), facecolor='w')
    ax = fig.add_subplot(111)

    # + 0.0 km/s points
    ax.errorbar(
        np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        np.array([
            23.824, 23.506, 22.548, 21.238, 19.454, 17.639,
            16.008, 14.202, 12.67, 11.266, 9.732, 8.874, 8.044]),
        yerr=np.array([
            0.376, 0.517, 0.850, 1.062, 1.204, 1.383,
            1.534, 1.612, 1.544, 1.579, 1.576, 1.538, 1.504]),
        fmt='o', color='0.0', ms=6.0, elinewidth=1.0, label='$\\Delta v_{r,grav}$ = 0,0 km/s')

    # + 0.5 km/s points
    ax.errorbar(
        np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        [19.858, 19.655, 19.116, 19.292, 17.246, 15.988,
            14.749, 13.577, 12.379, 11.222, 10.229, 9.210, 8.446],
        yerr=[0.376, 0.425, 0.641, 0.773, 0.992, 1.136,
            1.129, 1.251, 1.338, 1.331, 1.272, 1.345, 1.323],
        fmt='D', color='0.33', ms=6.0, elinewidth=1.0, label='$\\Delta v_{r,grav}$ = 0,5 km/s')

    # + 1.0 km/s points
    ax.errorbar(
        np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        [16.87, 16.743, 16.404, 15.884, 15.26, 14.522,
            13.529, 12.619, 11.751, 10.847, 9.982, 9.353, 8.461],
        yerr=[0.379, 0.453, 0.583, 0.685, 0.864, 0.93,
            0.951, 1.032, 1.147, 1.035, 1.142, 1.187, 1.149],
        fmt='^', color='0.67', ms=6.0, elinewidth=1.0, label='$\\Delta v_{r,grav}$ = 1,0 km/s')

    # β Pictoris typical error line
    ax.axvline(x=1.0105, ymin=0.0, ymax = 25, linewidth=1, color='k', ls='dashed')
    ax.text(1.15, 6.95, 'Erreur de mesure\nsur $v_r$ typique des\nmembres de $\\beta\\,$PMG',
        horizontalalignment='left', fontsize=14)

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title(
            "Measured age of a simulation of 1000 24 Myr-old groups\n"
            "over the measurement error on RV (other errors typical of Gaia DR2)\n")

    # Legend and axes formatting
    ax.legend(loc=1, fontsize=14)
    ax.xaxis.set_major_formatter(format_ticks)
    ax.yaxis.set_major_formatter(format_ticks)
    ax.set_xlabel('Error on RV (km/s)', fontsize=14)
    ax.set_ylabel('Age (Myr)', fontsize=14)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(6, 24.5)

    # Save figure
    save_figure(self.name, f'Errors_RV_offset_plot_{self.name}.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_minimum_error_plots(title=True, forced=False, default=False, cancel=False):
    """ Creates a plot of the error on the age of minimal scatter as a function of the error
        on the uvw velocity.
    """

    # Figure initialization
    fig = plt.figure(figsize=(6, 5.5), facecolor='w')
    ax = fig.add_subplot(111)

    # Plotting
    errors = (0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1., 1.5, 2., 2.5, 3., 3.5,
        4., 4.5, 5., 6., 7., 8., 9., 10., 12., 14., 16., 18., 20.)
    ages = (24.001, 23.966, 23.901, 23.74, 23.525, 22.224, 20.301, 18.113, 15.977, 11.293, 7.995,
        5.803, 4.358, 3.364, 2.665, 2.204, 1.756, 1.257, 0.933, 0.735, 0.580, 0.488, 0.346, 0.262,
        0.192, 0.160, 0.134)
    ax.plot(errors, ages, '.-', color='0.0', linewidth=1.0)

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title('Impact of UVW velocity on the age of minimal scatter.')

    # Axes formatting
    ax.set_xlabel('Error on UVW velocity (km/s)')
    ax.set_ylabel('Age at minimal XYZ scatter (Myr)')

    # Save figure
    save_figure(self.name, f'Minimum_error_plot_{self.name}.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()
