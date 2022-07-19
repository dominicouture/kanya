# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" output.py: Defines functions to create data output such as plots of association size metrics
    over time, 2D and 3D scatters at a given time, histograms, color mesh, etc.
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
# ax.xaxis.set_major_formatter(format_ticks)
# ax.yaxis.set_major_formatter(format_ticks)
format_ticks = tkr.FuncFormatter(lambda x, pos: str(round(float(x), 1)))

# Set pyplot rc parameters
plt.rc('font', serif='Latin Modern Math', family='serif', size='8')
plt.rc('mathtext', fontset='custom', it='Latin Modern Roman:italic', rm='Latin Modern Math:roman')
plt.rc('lines', markersize=4)
plt.rc('pdf', fonttype=42)

# Set colors
greys = ('#000000', '#333333', '#666666', '#999999', '#cccccc')
greens = ('#000000', '#005600', '#00a000', '#00df00', '#56f056')
blues = ('#000000', '#002b56', '#0050a0', '#0080ff', '#40a0ff')
oranges = ('#000000', '#562b00', '#a05000', '#ff8000', '#ffa040')
magentas = ('#000000', '#560056', '#a000a0', '#df00df', '#f056f0')
colors = (greens[1], greens[2], blues[2], blues[3], blues[4])

def save_figure(name, file_path=None, tight=True, forced=False, default=False, cancel=False):
    """ Checks whether a path already exists and asks for user input if it does. The base path
        is assumed to be the output directory. Also, if the path does not have an extension, a
        '.pdf' extension is added.
    """

    def save(file_path, tight):
        """ Saves figure with or without tight layout and some padding. """

        # Check 'tight' argument
        stop(
            type(forced) != bool, 'TypeError',
            "'forced' must be a boolean ({} given).", type(forced))

        # Save figure
        if tight:
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0.01)
        else:
            plt.savefig(file_path)

    # file_path parameter
    file_path = file_path if file_path is not None else output(create=True) + '/'

    # Check if file_path parameter is a string, which must be done before the directory call
    stop(
        type(file_path) != str, 'TypeError',
        "'file_path' must be a string ({} given).", type(file_path))

    # file_path redefined as the absolute path, default name and directory creation
    file_path = path.join(
        directory(output(), path.dirname(file_path), 'file_path', create=True),
        path.basename(file_path) if path.basename(file_path) != '' else f'{name}.pdf')

    # Check if there's an extension and add a '.pdf' extension, if needed
    if path.splitext(file_path)[1] == '':
        file_path += '.pdf'

    # Check if a file already exists
    if path.exists(file_path):
        choice = None
        stop(
            type(forced) != bool, 'TypeError',
            "'forced' must be a boolean ({} given).", type(forced))
        if not forced:
            stop(
                type(default) != bool, 'TypeError',
                "'default' must be a default ({} given).", type(default))
            if not default:
                stop(
                    type(cancel) != bool, 'TypeError',
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

                # Cancel save
                if cancel or choice in ('n', 'no'):

                    # Logging
                    log(
                        "'{}': file not saved because a file already exists at '{}'.",
                        name, file_path)

            # Set default name and save figure
            if default or choice in ('k', 'keep'):
                from Traceback.tools import default_name
                file_path = default_name(file_path)
                save(file_path, tight)

                # Logging
                log("'{}': file name changed and file saved at '{}'.", name, file_path)

        # Delete existing file and save figure
        if forced or choice in ('y', 'yes'):
            from os import remove
            remove(file_path)
            save(file_path, tight)

            # Logging
            log("'{}': existing file located at '{}' deleted and replaced.", name, file_path)

    # Save figure
    else:
        save(file_path, tight)

        # Logging
        log("'{}': file saved at '{}'.", name, file_path)

class Output_Series():
    """ Output methods for a series of groups. """

    def initialize_figure_metric(self):
        """ Initializes a figure and an axis to plot association size metrics. """

        # Initialize figure
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
        ax = fig.add_axes([0.103, 0.103, 0.895, 0.895])

        return fig, ax

    def check_robust_sklearn_metric(self, robust, sklearn):
        """ Checks if 'robust' and 'sklearn' arguments are valid to select to proper association
            size metrics.
        """

        # Check 'robust' and 'sklearn' arguments
        self.stop(
            type(robust) != bool, 'TypeError',
            "'robust' must be a boolean ({} given).", type(robust))
        self.stop(
            type(sklearn) != bool, 'TypeError',
            "'sklearn' must be a boolean ({} given).", type(sklearn))
        self.stop(
            robust and sklearn, 'ValueError',
            "'robust' and 'sklearn' cannot both be True.")

    def plot_metric(self, ax, metric, index, linestyle, color, zorder=0.5, secondary=False):
        """ Plots the association size metric's value over time on a given axis along with an
            enveloppe to display the uncertainty. The display is further customized with the
            'linestyle' and 'color' parameters. If 'secondary' is True, secondary lines are
            displayed as well.
        """

        # Plot the value of the metric over time
        ax.plot(
            -self.time, metric.value.T[index],
            linewidth=1.0, linestyle=linestyle, color=color, zorder=zorder, label=(
                f'{metric.latex_name[index]} : ({metric.age[index]:.1f}'
                f' ± {metric.age_error[index]:.1f}) Myr'))

        # Plot an enveloppe to display the uncertainty
        ax.fill_between(
            -self.time,
            metric.value.T[index] - metric.value_error.T[index],
            metric.value.T[index] + metric.value_error.T[index],
            color=color, alpha=0.3, linewidth=0., zorder=zorder - 0.5)

        # Plot secondary lines
        self.stop(
            type(secondary) != bool, 'TypeError',
            "'secondary' must be a boolean ({} given).", type(secondary))
        if secondary:
            values = metric.values.reshape((
                metric.values.shape[0] * metric.values.shape[1],
                metric.values.shape[2], metric.values.shape[3]))
            for i in np.unique(np.round(
                    np.linspace(0, self.number_of_groups * self.jackknife_number - 1, 20))):
                ax.plot(
                    -self.time, values[int(i),:,index],
                    linestyle=linestyle, color=color, alpha=0.4,
                    linewidth=0.5, zorder=zorder - 0.25)

    def set_title_metric(self, ax, title, metric):
        """ Sets a title for association size metrics plots if 'title' is True. """

        self.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title(
                    "{} of {}\n over {:.1f} Myr with a {:.1f} km/s radial "
                    "velocity correction\n".format(
                        metric, self.name, self.duration.value,
                        self.rv_shift.to('km/s').value),
                    fontsize=8)
            elif self.from_model:
                ax.set_title(
                    "Average {} of {} simulated associations over {:.1f} Myr\n"
                    "with kinematics similar to {} and a {:.1f} km/s radial velocity "
                    "bias\n".format(
                        metric, self.number_of_groups, self.duration.value,
                        self.name, self.rv_shift.to('km/s').value),
                    fontsize=8)

    def set_axis_metric(self, ax, hide_x=False, hide_y=False):
        """ Sets the parameters of an axis and its figure to plot association size metrics. """

        # Set legend
        legend = ax.legend(loc=2, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.8))
        legend.get_frame().set_edgecolor((0.0, 0.0, 0.0, 1.0))
        legend.get_frame().set_linewidth(0.5)

        # Set labels
        ax.set_xlabel('Epoch (Myr)', fontsize=8)
        ax.set_ylabel('Association size (pc)', fontsize=8)

        # Set limits
        ax.set_xlim(-self.final_time.value + 14, -self.initial_time.value)
        ax.set_ylim(-1., 39.)

        # Set ticks
        ax.set_xticks([0., -5., -10., -15., -20., -25., -30., -35.])
        ax.set_yticks([0.,  5.,  10.,  15.,  20.,  25.,  30.,  35.])
        ax.tick_params(top=True, right=True, which='both', direction='in', width=0.5, labelsize=8)

        # Set spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Hide labels and tick labels, if needed
        if hide_x:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        if hide_y:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    def create_covariances_xyz_plot(
            self, robust=False, sklearn=False, title=False,
            forced=False, default=False, cancel=False):
        """ Creates a plot of x-x, y-y and z-z covariances, and the determinant and the trace of
            the covariances matrix. If either 'robust' or 'sklearn' is True, the robust or sklearn
            covariances matrix is used. Otherwise, the empirical covariances matrix is used.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Check 'robust' and 'sklearn' arguments
        self.check_robust_sklearn_metric(robust, sklearn)

        # Select empirical, robust or sklearn association size metrics
        if sklearn:
            covariances_xyz_matrix_det = self.covariances_xyz_matrix_det_sklearn
            covariances_xyz_matrix_trace = self.covariances_xyz_matrix_trace_sklearn
            covariances_xyz = self.covariances_xyz_sklearn
        elif robust:
            covariances_xyz_matrix_det = self.covariances_xyz_matrix_det_robust
            covariances_xyz_matrix_trace = self.covariances_xyz_matrix_trace_robust
            covariances_xyz = self.covariances_xyz_robust
        else:
            covariances_xyz_matrix_det = self.covariances_xyz_matrix_det
            covariances_xyz_matrix_trace = self.covariances_xyz_matrix_trace
            covariances_xyz = self.covariances_xyz

        # Plot xyz covariances matrix determinant and trace, and x-x, y-y and z-z covariances
        self.plot_metric(ax, covariances_xyz_matrix_det, 0, '-', colors[0], 0.8)
        self.plot_metric(ax, covariances_xyz_matrix_trace, 0, '--', colors[1], 0.6)
        self.plot_metric(ax, covariances_xyz, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, covariances_xyz, 1, '--', colors[3], 0.7)
        self.plot_metric(ax, covariances_xyz, 2, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$XYZ$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} covariances")

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'covariances_xyz_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_ξηζ_plot(
            self, robust=False, sklearn=False, title=False,
            forced=False, default=False, cancel=False):
        """ Creates a plot of ξ'-ξ', η'-η' and ζ'-ζ' covariances, and the determinant and the trace
            of the covariances matrix. If either 'robust' or 'sklearn' is True, the robust or
            sklearn covariances matrix is used. Otherwise, the empirical covariances matrix is used.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Check 'robust' and 'sklearn' arguments
        self.check_robust_sklearn_metric(robust, sklearn)

        # Select empirical, robust or sklearn association size metrics
        if sklearn:
            covariances_ξηζ_matrix_det = self.covariances_ξηζ_matrix_det_sklearn
            covariances_ξηζ_matrix_trace = self.covariances_ξηζ_matrix_trace_sklearn
            covariances_ξηζ = self.covariances_ξηζ_sklearn
        elif robust:
            covariances_ξηζ_matrix_det = self.covariances_ξηζ_matrix_det_robust
            covariances_ξηζ_matrix_trace = self.covariances_ξηζ_matrix_trace_robust
            covariances_ξηζ = self.covariances_ξηζ_robust
        else:
            covariances_ξηζ_matrix_det = self.covariances_ξηζ_matrix_det
            covariances_ξηζ_matrix_trace = self.covariances_ξηζ_matrix_trace
            covariances_ξηζ = self.covariances_ξηζ

        # Plot total ξ'η'ζ' covariances matrix determinant and trace,
        # and ξ'-ξ', η'-η' and ζ'-ζ' covariances
        self.plot_metric(ax, covariances_ξηζ_matrix_det, 0, '-', colors[0], 0.8)
        self.plot_metric(ax, covariances_ξηζ_matrix_trace, 0, '--', colors[1], 0.6)
        self.plot_metric(ax, covariances_ξηζ, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, covariances_ξηζ, 1, '--', colors[3], 0.7)
        self.plot_metric(ax, covariances_ξηζ, 2, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$ξ^\prime η^\prime ζ^\prime$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} covariances")

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'covariances_ξηζ_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_cross_covariances_xyz_plot(
            self, robust=False, sklearn=False, title=False,
            forced=False, default=False, cancel=False):
        """ Creates a plot of x-u, y-v and z-w cross covariances, and the determinant and the trace
            of the cross covariances matrix between positions and velocities.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Check 'robust' and 'sklearn' arguments
        self.check_robust_sklearn_metric(robust, sklearn)

        # Select empirical, robust or sklearn association size metrics
        if sklearn:
            cross_covariances_xyz_matrix_det = self.cross_covariances_xyz_matrix_det
            cross_covariances_xyz_matrix_trace = self.cross_covariances_xyz_matrix_trace
            cross_covariances_xyz = self.cross_covariances_xyz
        if robust:
            cross_covariances_xyz_matrix_det = self.cross_covariances_xyz_matrix_det_robust
            cross_covariances_xyz_matrix_trace = self.cross_covariances_xyz_matrix_trace_robust
            cross_covariances_xyz = self.cross_covariances_xyz_robust
        else:
            cross_covariances_xyz_matrix_det = self.cross_covariances_xyz_matrix_det
            cross_covariances_xyz_matrix_trace = self.cross_covariances_xyz_matrix_trace
            cross_covariances_xyz = self.cross_covariances_xyz

        # Plot total xyz cross covariances matrix determinant and trace,
        # and x-u, y-v and z-w cross covariances
        self.plot_metric(ax, cross_covariances_xyz_matrix_det, 0, '-', colors[0], 0.8)
        self.plot_metric(ax, cross_covariances_xyz_matrix_trace, 0, '--', colors[1], 0.6)
        self.plot_metric(ax, cross_covariances_xyz, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, cross_covariances_xyz, 1, '--', colors[3], 0.7)
        self.plot_metric(ax, cross_covariances_xyz, 2, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$X-U$, $Y-V$ and $Z-W$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} cross covariances")

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'Cross_covariances_xyz_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_cross_covariances_ξηζ_plot(
            self, robust=False, sklearn=False, title=False,
            forced=False, default=False, cancel=False):
        """ Creates a plot of ξ'-vξ', η'-vη' and ζ'-vζ' cross covariances, and the determinant and
            the trace of the cross covariances matrix between position and velocities.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Check 'robust' and 'sklearn' arguments
        self.check_robust_sklearn_metric(robust, sklearn)

        # Select empirical, robust or sklearn association size metrics
        if sklearn:
            cross_covariances_xyz_matrix_det = self.cross_covariances_xyz_matrix_det
            cross_covariances_xyz_matrix_trace = self.cross_covariances_xyz_matrix_trace
            cross_covariances_xyz = self.cross_covariances_xyz
        elif robust:
            cross_covariances_ξηζ_matrix_det = self.cross_covariances_ξηζ_matrix_det_robust
            cross_covariances_ξηζ_matrix_trace = self.cross_covariances_ξηζ_matrix_trace_robust
            cross_covariances_ξηζ = self.cross_covariances_ξηζ_robust
        else:
            cross_covariances_ξηζ_matrix_det = self.cross_covariances_ξηζ_matrix_det
            cross_covariances_ξηζ_matrix_trace = self.cross_covariances_ξηζ_matrix_trace
            cross_covariances_ξηζ = self.cross_covariances_ξηζ

        # Plot total ξηζ cross covariances matrix determinant and trace,
        # and ξ'-vξ', η'-vη' and ζ'-vζ' cross covariances
        self.plot_metric(ax, cross_covariances_ξηζ_matrix_det, 0, '-', colors[0], 0.8)
        self.plot_metric(ax, cross_covariances_ξηζ_matrix_trace, 0, '--', colors[1], 0.6)
        self.plot_metric(ax, cross_covariances_ξηζ, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, cross_covariances_ξηζ, 1, '--', colors[3], 0.7)
        self.plot_metric(ax, cross_covariances_ξηζ, 2, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$ξ^\prime-\dot{ξ}^\prime$, '
            '$η^\prime-\dot{η}^\prime$ and $ζ^\prime-\dot{ζ}^\prime$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} cross covariances")

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'Cross_covariances_ξηζ_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_mad_xyz_plot(self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of the total xyz median absolute deviation (MAD), and x, y and z
            components of the MAD.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot total xyz MAD, x MAD, y and z MAD
        self.plot_metric(ax, self.mad_xyz_total, 0, '-', colors[0], 0.8)
        self.plot_metric(ax, self.mad_xyz, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, self.mad_xyz, 1, '--', colors[3], 0.7)
        self.plot_metric(ax, self.mad_xyz, 2, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(ax, title, '$XYZ$ MAD')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'MAD_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_mad_ξηζ_plot(self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of the total ξηζ median absolute deviation (MAD), and ξ', η' and ζ'
            components of the MAD.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot total ξ'η'ζ' MAD, ξ' MAD, η' MAD and ζ' MAD
        self.plot_metric(ax, self.mad_ξηζ_total, 0, '-', colors[0], 0.8)
        self.plot_metric(ax, self.mad_ξηζ, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, self.mad_ξηζ, 1, '--', colors[3], 0.7)
        self.plot_metric(ax, self.mad_ξηζ, 2, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(ax, title, '$ξ^\prime η^\prime ζ^\prime$ MAD')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'MAD_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_xyz_mst_plot(self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of the mean branch length (both empirical and robust) and median
            absolute deviation of the xyz minimum spanning tree (MST).
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot association size metrics
        self.plot_metric(ax, self.mst_xyz_mean, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, self.mst_xyz_mean_robust, 0, '--', colors[3], 0.7)
        self.plot_metric(ax, self.mst_xyz_mad, 0, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(ax, title, '$XYZ$ MST')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'MST_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_ξηζ_mst_plot(self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of the mean branch length (both empirical and robust) and median
            absolute deviation of the ξηζ minimum spanning tree (MST).
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot association size metrics
        self.plot_metric(ax, self.mst_ξηζ_mean, 0, '-', colors[2], 0.9)
        self.plot_metric(ax, self.mst_ξηζ_mean_robust, 0, '--', colors[3], 0.7)
        self.plot_metric(ax, self.mst_ξηζ_mad, 0, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(ax, title, '$ξ^\prime η^\prime ζ^\prime$ MST')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        save_figure(
            self.name, f'MST_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_mad_ξηζ_plot(
            self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of ξ'-ξ', η'-η' and ζ'-ζ' covariances, and the determinant and the trace
            of the covariances matrix, and  the total ξηζ median absolute deviation (MAD), and ξ',
            η' and ζ' components of the MAD.
        """

        # Initialize figure
        self.check_traceback()
        fig = plt.figure(figsize=(3.345, 6.520), facecolor='w', dpi=300)
        ax1 = fig.add_axes([0.100, 0.526, 0.899, 0.461])
        ax2 = fig.add_axes([0.100, 0.050, 0.899, 0.461])

        # Plot total ξ'η'ζ' covariances matrix determinant and trace,
        # and ξ'-ξ', η'-η' and ζ'-ζ' covariances
        self.plot_metric(ax1, self.covariances_ξηζ_matrix_det, 0, '-', colors[0], 0.8)
        self.plot_metric(ax1, self.covariances_ξηζ_matrix_trace, 0, '--', colors[1], 0.6)
        self.plot_metric(ax1, self.covariances_ξηζ, 0, '-', colors[2], 0.9)
        self.plot_metric(ax1, self.covariances_ξηζ, 1, '--', colors[3], 0.7)
        self.plot_metric(ax1, self.covariances_ξηζ, 2, ':', colors[4], 0.5)

        # Plot total ξ'η'ζ' MAD, ξ' MAD, η' MAD and ζ' MAD
        self.plot_metric(ax2, self.mad_ξηζ_total, 0, '-', colors[0], 0.8)
        self.plot_metric(ax2, self.mad_ξηζ, 0, '-', colors[2], 0.9)
        self.plot_metric(ax2, self.mad_ξηζ, 1, '--', colors[3], 0.7)
        self.plot_metric(ax2, self.mad_ξηζ, 2, ':', colors[4], 0.5)

        # Set title
        self.set_title_metric(ax1, title, '$ξ^\prime η^\prime ζ^\prime$ covariances and MAD')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax1, hide_x=True)
        self.set_axis_metric(ax2)

        # Save figure
        save_figure(
            self.name, f'covariances_MAD_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_det_mad_mst_cross_covariances_xyz_plots(
            self, other, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of xyz determinant of the covariances matrix, xyz total median absolute
            deviation, mean branch length of the xyz minimum spanning tree, and x-u, y-v and z-w
            cross covariances.
        """

        # Check if 'other' is valid
        self.stop(
            type(other) != type(self), 'TypeError',
            "'other' must be a Series object ({} given).", type(other))

        # Initialize figure
        self.check_traceback()
        other.check_traceback()
        fig, (ax0, ax1) = plt.subplots(
            ncols=2, constrained_layout=True, figsize=(6.66, 3.33), facecolor='w', dpi=300)

        # Plot association size metrics (self)
        self.plot_metric(ax0, self.covariances_xyz_matrix_det, 0, '-', blues[1])
        self.plot_metric(ax0, self.mad_xyz_total, 0, '--', blues[2])
        self.plot_metric(ax0, self.mst_xyz_mean, 0, '-.', blues[3])
        self.plot_metric(ax0, self.mst_xyz_mad, 0, ':', blues[4])
        self.plot_metric(ax1, self.cross_covariances_xyz, 0, '-', blues[1])
        self.plot_metric(ax1, self.cross_covariances_xyz, 1, '--', blues[2])
        self.plot_metric(ax1, self.cross_covariances_xyz, 2, ':', blues[3])

        # Plot association size metrics (other)
        other.plot_metric(ax0, other.covariances_xyz_matrix_det, 0, '-', greens[1])
        other.plot_metric(ax0, other.mad_xyz_total, 0, '--', greens[2])
        other.plot_metric(ax0, other.mst_xyz_mean, 0, '-.', greens[3])
        other.plot_metric(ax0, other.mst_xyz_mad, 0, ':', greens[4])
        other.plot_metric(ax1, other.cross_covariances_xyz, 0, '-', greens[1])
        other.plot_metric(ax1, other.cross_covariances_xyz, 1, '--', greens[2])
        other.plot_metric(ax1, other.cross_covariances_xyz, 2, ':', greens[3])

        # Set title
        self.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                fig.suptitle(
                    '$XYZ$ covariances, MAD, MST and cross covariances of {}\n and {} over '
                    '{:.1f} Myr with a {:.1f} km/s radial velocity correction\n'.format(
                        self.name, other.name, self.duration.value,
                        self.rv_shift.to('km/s').value),
                    fontsize=8)
            elif self.from_model:
                fig.suptitle(
                    'Average $XYZ$ covariances, MAD, MST and cross covariances of {}'
                    'simulated associations over {:.1f} Myr\n with kinematics similar to'
                    "{} and {}, and a {:.1f} km/s radial velocity bias\n".format(
                        self.number_of_groups, self.duration.value,
                        self.name, other.name, self.rv_shift.to('km/s').value),
                    fontsize=8)

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax0)
        self.set_axis_metric(ax1)

        # Save figure
        save_figure(
            self.name, f'covariances_MAD_MST_Cross_covariannces_xyz_{self.name}_{other.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_age_distribution(
            self, title=False, forced=False, default=False, cancel=False):
        """ Creates a plot of the distribution of ages computed in a series, including the
            effects of measurement errors and the jack-knife Monte Carlo.
        """

        # Initialize figure
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
        ax = fig.add_subplot(111, frame_on=True)

        # Plot histogram
        ages = [group.covariances_xyz_matrix_det.age[0] for group in self]
        ax.hist(
            ages, bins=np.linspace(16, 24, 33), density=True,
            color=blues[3], alpha=0.7, label='metric')
        # bins=np.arange(21.975, 26.025, 0.05)

        # Set title
        self.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                f'Distribution of {self.number_of_groups} moving groups age,\n'
                f'Average age: ({self.covariances_xyz_matrix_det.age[0]:.2f} '
                f'± {self.covariances_xyz_matrix_det.age_error[0]:.2f}) Myr\n', fontsize=8)

        # Set labels
        ax.set_xlabel('Age (Myr)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

        # Set ticks
        ax.tick_params(top=True, right=True, which='both', direction='in', width=0.5, labelsize=8)

        # Set spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Save figure
        save_figure(
            self.name, f'age_distribution_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def show_metrics(self):
        """ Creates a table of all association size metrics of a series. """

        def convert(value):
            """ Returns a converted, rounded string for display. """

            return str(np.round(value, 2)).replace('[', '').replace(']', '')

        def create_line(metric, valid=True):
            """ Creates a single metric (i) line. """

            valid = np.copy(metric.valid) if valid else np.invert(metric.valid)

            for i in filter(lambda i: valid[i], np.arange(valid.size)):
                print(
                    f'{metric.name[i]:<50}'
                    f'{convert(metric.age[i]):>15}'
                    f'{convert(metric.age_int_error[i]):>20}'
                    f'{convert(metric.age_ext_error[i]):>20}'
                    f'{convert(metric.age_error[i]):>15}'
                    f'{convert(metric.min_change[i]):>20}'
                    f'{convert(metric.age_shift[i]):>15}')

        # Set precision and order
        np.set_printoptions(precision=2)
        order = np.argsort([i.order for i in self.metrics])

        # Line list
        lines = []

        # Display header
        print(f"{'':-<155}")
        print(
            f"{'Metric':<50}{'Age':>15}{'Jack-knife Error':>20}{'Measurement Error':>20}"
            f"{'Total Error':>15}{'Minimum Change':>20}{'Offset':>15}")
        print(f"{'[Myr]':>65}{'[Myr]':>20}{'[Myr]':>20}{'[Myr]':>15}{'[%]':>20}{'[Myr]':>15}")
        print(f"{'':-<155}")

        # Display errors on age of valid
        print('Valid\n' f"{'':-<155}")
        for i in order:
            create_line(self.metrics[i], valid=True)
        print(f"{'':-<155}")

        # Display errors on age of invalid association size metrics
        print('Rejected\n' f"{'':-<155}")
        for i in order:
            create_line(self.metrics[i], valid=False)
        print(f"{'':-<155}")

    def convert_to_LaTeX_table(self, table):
        """ Converts a table or array to a LaTeX compatible format. """

        for row in table:
            print(' & '.join([str(i) for i in row], end='\n') + '\\\\')

    def create_table_observations(self):
        """ Creates a table of observations: right ascension, declination, proper motion in right
            ascension and declination, parallax and radial velocity.
        """

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
            # line += [i.get_LaTeX() for i in (velocity[0], velocity[1], velocity[2])]

            print(' & '.join([str(i) for i in line]) + '\\\\')


class Output_Group():
    """ Output methods for a group of stars. """

    def get_epoch(self, age=None, metric=None, index=None):
        """ Computes the time index of the epoch for a given association age or, association size
            metric and dimensional index.
        """

        # Index from age
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer, float or None ({} given).", type(age))
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age)
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age)

        # Index from the epoch of minimum of an association size metric
        elif metric is not None:
            metric, index = self.get_metric(metric, index)
            age = metric.age[index]
        else:
            self.series.stop(
                True, 'ValueError',
                "Either 'age' or 'metric' must not be None ({} and {} given).", age, index)

        return int(age / self.series.final_time.value * self.series.number_of_steps)

    def get_metric(self, metric, index=None):
        """ Retrieves the proprer Series.Metric instance from a string and index. """

        # Metric instance
        self.series.stop(
            type(metric) != str, 'TypeError',
            "'metric' must be a string or None ({} given).", type(metric))
        self.series.stop(
            metric not in [metric.label for metric in self.series.metrics],
            'ValueError', "'metric' must be a valid metric key ({} given).", metric)
        metric = vars(self.series)[metric]

        # Metric index
        if index is not None:
            self.series.stop(
                type(index) != int, 'TypeError',
                "'index' must be an integer or None ({} given).", type(index))
            self.series.stop(
                index > metric.value.size - 1, 'ValueError',
                "'index' is too large for this metric ({} given, {} in size).",
                index, metric.value.size)
        else:
            self.series.stop(
                metric.value.size > 1, 'ValueError',
                "No 'index' is provided (metric is {} in size).", metric.value.size)

        return (metric, index if metric.value.size > 1 else 0)

    def trajectory_xyz(
            self, title=False, labels=False, age=None, metric=None, index=None,
            forced=False, default=False, cancel=False):
        """ Draw the xyz trajectories of stars in the group. """

        # Initialize figure
        fig = plt.figure(figsize=(3.345, 3.315), facecolor='w', dpi=300)
        left, bottom = (0.133, 0.102)
        short1, short2  = (0.236, 0.2381)
        long1, long2 = (0.618, 0.6236)
        inside1, inside2 = (0.012, 0.0119)
        ax1 = fig.add_axes([left, bottom + inside2 + short2, long1, long2])
        ax2 = fig.add_axes([left + long1 + inside1, bottom + inside2 + short2, short1, long2])
        ax3 = fig.add_axes([left, bottom, long1, short2])

        # Check labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))

        # Birth epoch
        birth = self.get_epoch(age=age, metric=metric, index=index)

        # Select conversion factors from pc to kpc
        for ax, x, y in ((ax1, 1, 0), (ax2, 2, 0), (ax3, 1, 2)):
            factor_x = 1.0 if x == 2 else 1000
            factor_y = 1.0 if y == 2 else 1000

            # Plot stars' trajectories
            for star in self:
                color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 0.0)
                ax.plot(
                    star.position_xyz.T[x] / factor_x,
                    star.position_xyz.T[y] / factor_y,
                    color=color, alpha=0.6, linewidth=0.5,
                    solid_capstyle='round', zorder=0.1)

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        np.array([star.position_xyz[0,x]]) / factor_x,
                        np.array([star.position_xyz[0,y]]) / factor_y,
                        color=(0.0, 0.0, 0.0, 0.8), edgecolors=(0.0, 0.0, 0.0, 1.0),
                        alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Plot stars' birth positions
                    if birth is not None:
                        color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 1.0)
                        ax.scatter(
                            np.array([star.position_xyz[birth,x]]) / factor_x,
                            np.array([star.position_xyz[birth,y]]) / factor_y,
                            color=color + (0.8,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Show stars' names
                    if labels:
                        ax.text(
                            np.array(star.position_xyz.T[0,x]) / factor_x,
                            np.array(star.position_xyz.T[0,y]) / factor_y,
                            star.name, horizontalalignment='left', fontsize=6)

            # Plot average model star's trajectory
            if self.series.from_model:
                ax.plot(
                    self.average_model_star.position_xyz.T[x] / factor_x,
                    self.average_model_star.position_xyz.T[y] / factor_y,
                    color=(0.0, 1.0, 0.0), alpha=0.8,
                    linewidth=0.5, solid_capstyle='round', zorder=0.3)

                # Plot average model star's birth and current positions
                for t, size, marker in ((-1, 12, '*'), (0, 6, 'o')):
                    ax.scatter(
                        np.array([self.average_model_star.position_xyz[t,x]]) / factor_x,
                        np.array([self.average_model_star.position_xyz[t,y]]) / factor_y,
                        color=(0.0, 1.0, 0.0, 0.8), edgecolors=(0.0, 1.0, 0.0, 1.0),
                        alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.3)

                # Plot model stars' trajectories
                for star in self.model_stars:
                    ax.plot(
                        star.position_xyz.T[x] / factor_x,
                        star.position_xyz.T[y] / factor_y,
                        color=(0.0, 0.0, 1.0), alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2)

                    # Plot model stars' birth and current positions
                    for t, size, marker in ((0, 12, '*'), (-1, 6, 'o')):
                        ax.scatter(
                            np.array([star.position_xyz[t,x]]) / factor_x,
                            np.array([star.position_xyz[t,y]]) / factor_y,
                            color=(0.0, 0.0, 1.0, 0.8), edgecolors=(0.0, 0.0, 1.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.2)

            # Draw vertical and horizontal lines through the Sun's position at the current epoch
            ax.axhline(0., color='#000000', linewidth=0.5, linestyle=':', alpha=0.8, zorder=0.0)
            ax.axvline(0., color='#000000', linewidth=0.5, linestyle=':', alpha=0.8, zorder=0.0)

        # Draw circles around the galactic center located at 8.122 kpc from the Sun
        for radius in range(1, 16):
            ax1.add_artist(plt.Circle(
                (0, 8.122), radius, color='#000000', fill=False,
                linewidth=0.5, linestyle=':', alpha=0.2, zorder=0.0))

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            fig.suptitle(f"$XYZ$ trajectories of stars in {self.name}", y=1.05, fontsize=8)

        # Set labels
        ax1.set_ylabel('$X$ (kpc)', fontsize=8)
        ax2.set_xlabel('$Z$ (pc)', fontsize=8)
        ax3.set_xlabel('$Y$ (kpc)', fontsize=8)
        ax3.set_ylabel('$Z$ (pc)', fontsize=8)

        # Invert y axis
        ax1.invert_xaxis()
        ax3.invert_xaxis()

        # Set limits
        ax1.set_xlim(1, -9)
        ax1.set_ylim(-1, 9)
        ax2.set_xlim(-80, 80)
        ax2.set_ylim(-1, 9)
        ax3.set_xlim(1, -9)
        ax3.set_ylim(-80, 80)

        # Set ticks
        ax1.set_xticklabels([])
        ax2.set_yticklabels([])
        for ax in (ax1, ax2, ax3):
            ax.tick_params(
                top=True, right=True, which='both',
                direction='in', width=0.5, labelsize=8)

        # Set spines
        for ax in (ax1, ax2, ax3):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        save_figure(
            self.name, f'trajectory_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)

    def trajectory_ξηζ(
            self, title=False, labels=False, age=None, metric=None, index=None,
            forced=False, default=False, cancel=False):
        """ Draws the ξηζ trajectories of stars in the group. """

        # Initialize figure
        fig = plt.figure(figsize=(3.345, 3.315), facecolor='w', dpi=300)
        left, bottom = (0.133, 0.102)
        short1, short2  = (0.236, 0.2381)
        long1, long2 = (0.618, 0.6236)
        inside1, inside2 = (0.012, 0.0119)
        ax1 = fig.add_axes([left, bottom + inside2 + short2, long1, long2])
        ax2 = fig.add_axes([left + long1 + inside1, bottom + inside2 + short2, short1, long2])
        ax3 = fig.add_axes([left, bottom, long1, short2])

        # Check labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))

        # Birth epoch
        birth = self.get_epoch(age=age, metric=metric, index=index)

        # Plot stars' trajectories
        for ax, x, y in ((ax1, 0, 1), (ax2, 2, 1), (ax3, 0, 2)):
            for star in self:
                color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 0.0)
                ax.plot(
                    star.position_ξηζ.T[x], star.position_ξηζ.T[y],
                    color=color, alpha=0.6, linewidth=0.5,
                    solid_capstyle='round', zorder=0.1)

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        np.array([star.position_ξηζ[0,x]]),
                        np.array([star.position_ξηζ[0,y]]),
                        color=(0.0, 0.0, 0.0, 0.8), edgecolors=(0.0, 0.0, 0.0, 1.0),
                        alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Plot stars' birth positions
                    if birth is not None:
                        color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 1.0)
                        ax.scatter(
                            np.array([star.position_ξηζ[birth,x]]),
                            np.array([star.position_ξηζ[birth,y]]),
                            color=color + (0.8,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Show stars' names
                    if labels:
                        ax.text(
                            np.array(star.position_ξηζ.T[0,x]),
                            np.array(star.position_ξηζ.T[0,y]),
                            star.name, horizontalalignment='left', fontsize=6)

            # Plot average model star's trajectory
            if self.series.from_model:
                ax.plot(
                    self.average_model_star.position_ξηζ.T[x],
                    self.average_model_star.position_ξηζ.T[y],
                    color=(0.0, 1.0, 0.0), alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3)

                # Plot average model star's birth and current positions
                for t, size, marker in ((-1, 12, '*'), (0, 6, 'o')):
                    ax.scatter(
                        np.array([self.average_model_star.position_ξηζ[t,x]]),
                        np.array([self.average_model_star.position_ξηζ[t,y]]),
                        color=(0.0, 1.0, 0.0, 0.8), edgecolors=(0.0, 1.0, 0.0, 1.0),
                        alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.3)

                # Plot model stars' trajectories
                for star in self.model_stars:
                    ax.plot(
                        star.position_ξηζ.T[x], star.position_ξηζ.T[y],
                        color=(0.0, 0.0, 1.0), alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2)

                    # Plot model stars' birth and current positions
                    for t, size, marker in ((0, 12, '*'), (-1, 6, 'o')):
                        ax.scatter(
                            np.array([star.position_ξηζ[t,x]]),
                            np.array([star.position_ξηζ[t,y]]),
                            color=(0.0, 0.0, 1.0, 0.8), edgecolors=(0.0, 0.0, 1.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.2)

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            fig.suptitle(
                f'$ξ^\prime η^\prime ζ^\prime$ trajectories of stars in {self.name}',
                y=1.05, fontsize=8)

        # Set labels
        ax1.set_ylabel('$η^\prime$ (pc)', fontsize=8)
        ax2.set_xlabel('$ζ^\prime$ (pc)', fontsize=8)
        ax3.set_xlabel('$ξ^\prime$ (pc)', fontsize=8)
        ax3.set_ylabel('$ζ^\prime$ (pc)', fontsize=8)

        # Set limits
        ax1.set_xlim(-225, 60)
        ax1.set_ylim(-45, 110)
        ax2.set_xlim(-40, 49)
        ax2.set_ylim(-45, 110)
        ax3.set_xlim(-225, 60)
        ax3.set_ylim(-40, 49)

        # Set ticks
        ax1.set_xticklabels([])
        ax2.set_yticklabels([])
        for ax in (ax1, ax2, ax3):
            ax.tick_params(
                top=True, right=True, which='both',
                direction='in', width=0.5, labelsize=8)

        # Set spines
        for ax in (ax1, ax2, ax3):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        save_figure(
            self.name, f'trajectory_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def trajectory_time_xyz(
            self, style, title=False, age=None, metric=None,
            forced=False, default=False, cancel=False):
        """ Draws the xyz trajectories as a function of time of stars. """

        # Check style
        self.series.stop(
            type(style) != str, 'TypeError',
            "'style' must be a string ({} given).", type(style))
        self.series.stop(
            style not in ('2x2', '1x3'), 'ValueError',
            "'style' must be either '2x2' or '1x3' ({} given).", style)

        # Initialize a 2x2 figure
        if style == '2x2':
            fig = plt.figure(facecolor='w', figsize=(7.090, 6.317), dpi=300)
            left1, bottom1 = (0.083, 0.528)
            left2, bottom2 = (0.507, 0.052)
            width, height = (0.410, 0.460)
            ax0 = fig.add_axes([left1, bottom1, width, height])
            ax1 = fig.add_axes([left2, bottom1, width, height])
            ax2 = fig.add_axes([left1, bottom2, width, height])
            ax3 = fig.add_axes([left2, bottom2, width, height])

        # Initialize a 1x3 figure
        if style == '1x3':
            fig = plt.figure(facecolor='w', figsize=(3.345, 9.134), dpi=300)
            left, bottom = (0.133, 0.0355)
            width, height = (0.866, 0.3112)
            inside = 0.01095
            ax1 = fig.add_axes([left, bottom + 2 * (height + inside), width, height])
            ax2 = fig.add_axes([left, bottom + height + inside, width, height])
            ax3 = fig.add_axes([left, bottom, width, height])

        # Birth epoch
        birth = [self.get_epoch(age=age, metric=metric, index=index) for index in range(3)]

        # Plot stars' trajectories
        for ax, y in ((ax1, 0), (ax2, 1), (ax3, 2)):
            for star in self:
                ax.plot(
                    -self.series.time, (star.position_xyz - self.position_xyz)[:,y],
                    color = (1.0, 0.0, 0.0) if star.outlier else greys[0],
                    alpha=0.6, linewidth=0.5, solid_capstyle='round', zorder=0.1)

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        -self.series.time[0], (star.position_xyz - self.position_xyz)[0,y],
                        color=(0.0, 0.0, 0.0, 0.8), edgecolors=(0.0, 0.0, 0.0, 1.0),
                        alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Plot stars' birth positions
                    if birth is not None:
                        color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 1.0)
                        ax.scatter(
                            -self.series.time[birth[y]],
                            (star.position_xyz - self.position_xyz)[birth[y],y],
                            color=color + (0.8,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

            # Show vectical dashed line
            ax.axvline(
                x=-self.series.time[birth[y]], color=greys[0],
                linewidth=0.5, linestyle='--', zorder=0.0)

            # Plot average model star's trajectory
            if self.series.from_model:
                position_xyz = np.mean(
                    [star.position_xyz[:,y] for star in self.model_stars], axis=0)
                average_model_star_position_xyz = (
                    self.average_model_star.position_xyz[:,y] - position_xyz[::-1])
                ax.plot(
                    self.average_model_star.time,
                    average_model_star_position_xyz,
                    color=(0.0, 1.0, 0.0), alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3)

                # Plot average model star's birth and current positions
                for t, x, size, marker in ((-1, -1, 10, '*'), (0, 0, 6, 'o')):
                    ax.scatter(
                        np.array([self.average_model_star.time[t]]),
                        np.array([average_model_star_position_xyz[x]]),
                        color=(0.0, 1.0, 0.0, 0.8), edgecolors=(0.0, 1.0, 0.0, 1.0),
                        alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.3)

                # Plot model stars' trajectories
                for star in self.model_stars:
                    model_star_position_xyz = star.position_xyz[:,y] - position_xyz
                    ax.plot(
                        -star.time[::-1], model_star_position_xyz,
                        color=(0.0, 0.0, 1.0), alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2)

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 10, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            -np.array([star.time[t]]),
                            np.array([model_star_position_xyz[x]]),
                            color=(0.0, 0.0, 1.0, 0.8), edgecolors=(0.0, 0.0, 1.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.2)

        # Plot stars' average trajectory
        if style == '2x2':
            for y, label, linestyle in ((0,'$<X>$', '-'), (1,'$<Y>$', '--'), (2, '$<Z>$', ':')):
                ax0.plot(
                    -self.series.time, self.position_xyz[:,y] / 1000,
                    label=label, color=(0.0, 0.0, 0.0), alpha=0.8, linestyle=linestyle,
                    linewidth=1.0, solid_capstyle='round', zorder=0.1)

                # Plot stars' average current positions
                if self.series.from_data:
                    ax0.scatter(
                        -self.series.time[0], self.position_xyz[0,y] / 1000,
                        color=(0.0, 0.0, 0.0, 0.8), edgecolors=(0.0, 0.0, 0.0, 1.0),
                        alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Plot stars' average birth positions
                    if birth is not None:
                        color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 1.0)
                        ax0.scatter(
                            -self.series.time[birth[y]], self.position_xyz[birth[y],y] / 1000,
                            color=color + (0.8,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                # Plot average model star's trajectory
                if self.series.from_model:
                    average_model_star_position_xyz = self.average_model_star.position_xyz[:,y] / 1000
                    ax0.plot(
                        self.average_model_star.time,
                        average_model_star_position_xyz,
                        color=(0.0, 1.0, 0.0), alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', zorder=0.3)

                    # Plot average model star's birth and current positions
                    for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                        ax0.scatter(
                            np.array([self.average_model_star.time[t]]),
                            np.array([average_model_star_position_xyz[x]]),
                            color=(0.0, 1.0, 0.0, 0.8), edgecolors=(0.0, 1.0, 0.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.3)
                            # s=6, marker=m, color=greens[y + 2], zorder=0.3)

                    # Plot model stars' trajectories
                    model_star_position_xyz = np.mean(
                        [star.position_xyz[:,y] for star in self.model_stars], axis=0) / 1000
                    ax0.plot(
                        -self.model_stars[0].time[::-1], model_star_position_xyz,
                        color=(0.0, 0.0, 1.0), alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', zorder=0.2)

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax0.scatter(
                            -np.array([self.model_stars[0].time[t]]),
                            np.array([model_star_position_xyz[x]]),
                            color=(0.0, 0.0, 1.0, 0.8), edgecolors=(0.0, 0.0, 1.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.2)
                            # s=6, marker=m, color=blues[y + 2], zorder=0.2)

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if style == '2x2':
                fig.suptitle(
                    f'$XYZ$ trajectories of stars in {self.name} as a function of time',
                    y=1.025, fontsize=8)
            if style == '1x3':
                fig.suptitle(
                    f'$XYZ$ trajectories of stars in {self.name}\nas a function of time',
                    y=1.03, fontsize=8)

        # Set legend
        if style == '2x2':
            legend = ax0.legend(loc=4, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.8))
            legend.get_frame().set_edgecolor((0.0, 0.0, 0.0, 1.0))
            legend.get_frame().set_linewidth(0.5)

        # Set labels
        ax3.set_xlabel('Epoch (Myr)', fontsize=8)
        ax1.set_ylabel('$X - <X>$ (pc)', fontsize=8)
        ax2.set_ylabel('$Y - <Y>$ (pc)', fontsize=8)
        ax3.set_ylabel('$Z - <Z>$ (pc)', fontsize=8)
        if style == '2x2':
            ax2.set_xlabel('Epoch (Myr)', fontsize=8)
            ax0.set_ylabel('$<XYZ>$ (kpc)', fontsize=8)
            ax1.yaxis.set_label_position('right')
            ax3.yaxis.set_label_position('right')

        # Set limits
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            ax.set_xlim(-np.max(self.series.time), -(np.min(self.series.time) - 1))

        # Set ticks
        if style == '1x3':
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
        if style == '2x2':
            ax1.yaxis.tick_right()
            ax3.yaxis.tick_right()
            ax0.set_xticklabels([])
            ax1.set_xticklabels([])
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            ax.tick_params(
                top=True, right=True, which='both',
                direction='in', width=0.5, labelsize=8)

        # Set spines
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        save_figure(
            self.name, f"trajectory_time_xyz_{style}_{self.name}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def trajectory_time_ξηζ(
            self, style, title=False, age=None, metric=None,
            forced=False, default=False, cancel=False):
        """ Draws the ξηζ trajectories as a function of time of stars. """

        # Check style
        self.series.stop(
            type(style) != str, 'TypeError',
            "'style' must be a string ({} given).", type(style))
        self.series.stop(
            style not in ('2x2', '1x3'), 'ValueError',
            "'style' must be either '2x2' or '1x3' ({} given).", style)

        # Initialize a 2x2 figure
        if style == '2x2':
            fig = plt.figure(facecolor='w', figsize=(7.090, 6.317), dpi=300)
            left1, bottom1 = (0.083, 0.528)
            left2, bottom2 = (0.507, 0.052)
            width, height = (0.410, 0.460)
            ax0 = fig.add_axes([left1, bottom1, width, height])
            ax1 = fig.add_axes([left2, bottom1, width, height])
            ax2 = fig.add_axes([left1, bottom2, width, height])
            ax3 = fig.add_axes([left2, bottom2, width, height])

        # Initialize a 1x3 figure
        if style == '1x3':
            fig = plt.figure(facecolor='w', figsize=(3.345, 9.134), dpi=300)
            left, bottom = (0.133, 0.0355)
            width, height = (0.866, 0.3112)
            inside = 0.01095
            ax1 = fig.add_axes([left, bottom + 2 * (height + inside), width, height])
            ax2 = fig.add_axes([left, bottom + height + inside, width, height])
            ax3 = fig.add_axes([left, bottom, width, height])

        # Birth epoch
        birth = [self.get_epoch(age=age, metric=metric, index=index) for index in range(3)]

        # Plot stars' trajectories
        for ax, y in ((ax1, 0), (ax2, 1), (ax3, 2)):
            for star in self:
                ax.plot(
                    -self.series.time, (star.position_ξηζ - self.position_ξηζ)[:,y],
                    color = (1.0, 0.0, 0.0) if star.outlier else greys[0],
                    alpha=0.6, linewidth=0.5, solid_capstyle='round', zorder=0.1)

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        -self.series.time[0], (star.position_ξηζ - self.position_ξηζ)[0,y],
                        color=(0.0, 0.0, 0.0, 0.8), edgecolors=(0.0, 0.0, 0.0, 1.0),
                        alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Plot stars' birth positions
                    if birth is not None:
                        color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 1.0)
                        ax.scatter(
                            -self.series.time[birth[y]],
                            (star.position_ξηζ - self.position_ξηζ)[birth[y],y],
                            color=color + (0.8,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

            # Show vectical dashed line
            ax.axvline(
                x=-self.series.time[birth[y]], color=greys[0],
                linewidth=0.5, linestyle='--', zorder=0.0)

            # Plot average model star's trajectory
            if self.series.from_model:
                position_ξηζ = np.mean(
                    [star.position_ξηζ[:,y] for star in self.model_stars], axis=0)
                average_model_star_position_ξηζ = (
                    self.average_model_star.position_ξηζ[:,y] - position_ξηζ[::-1])
                ax.plot(
                    self.average_model_star.time,
                    average_model_star_position_ξηζ,
                    color=(0.0, 1.0, 0.0), alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3)

                # Plot average model star's birth and current positions
                for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                    ax.scatter(
                        np.array([self.average_model_star.time[t]]),
                        np.array([average_model_star_position_ξηζ[x]]),
                        color=(0.0, 1.0, 0.0, 0.8), edgecolors=(0.0, 1.0, 0.0, 1.0),
                        alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.3)

                # Plot model stars' trajectories
                for star in self.model_stars:
                    model_star_position_ξηζ = star.position_ξηζ[:,y] - position_ξηζ
                    ax.plot(
                        -star.time[::-1], model_star_position_ξηζ,
                        color=(0.0, 0.0, 1.0), alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2)

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            -np.array([star.time[t]]),
                            np.array([model_star_position_ξηζ[x]]),
                            color=(0.0, 0.0, 1.0, 0.8), edgecolors=(0.0, 0.0, 1.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.2)

        # Plot stars' average trajectories
        if style == '2x2':
            for y, label, linestyle in (
                    (0, '$<ξ^\prime>$', '-'), (1, '$<η^\prime>$', '--'), (2, '$<ζ^\prime>$', ':')):
                ax0.plot(
                    -self.series.time, self.position_ξηζ[:,y],
                    label=label, color=(0.0, 0.0, 0.0), alpha=0.8, linestyle=linestyle,
                    linewidth=1.0, solid_capstyle='round', zorder=0.1)

                # Plot stars' average current positions
                if self.series.from_data:
                    ax0.scatter(
                        -self.series.time[0], self.position_ξηζ[0,y],
                        color=(0.0, 0.0, 0.0, 0.8), edgecolors=(0.0, 0.0, 0.0, 1.0),
                        alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                    # Plot stars' average birth positions
                    if birth is not None:
                        color = (1.0, 0.0, 0.0) if star.outlier else (0.0, 0.0, 1.0)
                        ax0.scatter(
                            -self.series.time[birth[y]], self.position_ξηζ[birth[y],y],
                            color=color + (0.8,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.5, zorder=0.2)

                # Plot average model star's trajectory
                if self.series.from_model:
                    average_model_star_position_ξηζ = self.average_model_star.position_ξηζ[:,y]
                    ax0.plot(
                        self.average_model_star.time,
                        average_model_star_position_ξηζ,
                        color=(0.0, 1.0, 0.0), alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', zorder=0.3)

                    # Plot average model star's birth and current positions
                    for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                        ax0.scatter(
                            np.array([self.average_model_star.time[t]]),
                            np.array([average_model_star_position_ξηζ[x]]),
                            color=(0.0, 1.0, 0.0, 0.8), edgecolors=(0.0, 1.0, 0.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.3)
                            # s=6, marker=m, color=greens[y + 2], zorder=0.3)

                    # Plot model stars' trajectories
                    model_star_position_ξηζ = np.mean(
                        [star.position_ξηζ[:,y] for star in self.model_stars], axis=0)
                    ax0.plot(
                        -self.model_stars[0].time[::-1], model_star_position_ξηζ,
                        color=(0.0, 0.0, 1.0), alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', zorder=0.2)

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax0.scatter(
                            -np.array([self.model_stars[0].time[t]]),
                            np.array([model_star_position_ξηζ[x]]),
                            color=(0.0, 0.0, 1.0, 0.8), edgecolors=(0.0, 0.0, 1.0, 1.0),
                            alpha=None, s=size, marker=marker, linewidths=0.5, zorder=0.2)
                            # s=6, marker=m, color=blues[y + 2], zorder=0.2)

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if style == '2x2':
                fig.suptitle(
                    f'$ξ^\prime η^\prime ζ^\prime$ trajectories of stars in {self.name} '
                    'as a function of time', y=1.025, fontsize=8)
            if style == '1x3':
                fig.suptitle(
                    f'$ξ^\prime η^\prime ζ^\prime$ trajectories of stars in {self.name}\n'
                    'as a function of time', y=1.03, fontsize=8)

        # Set legend
        if style == '2x2':
            legend = ax0.legend(loc=4, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.8))
            legend.get_frame().set_edgecolor((0.0, 0.0, 0.0, 1.0))
            legend.get_frame().set_linewidth(0.5)

        # Set labels
        ax3.set_xlabel('Epoch (Myr)', fontsize=8)
        ax1.set_ylabel('$ξ^\prime - <ξ^\prime>$ (pc)', fontsize=8)
        ax2.set_ylabel('$η^\prime - <η^\prime>$ (pc)', fontsize=8)
        ax3.set_ylabel('$ζ^\prime - <ζ^\prime>$ (pc)', fontsize=8)
        if style == '2x2':
            ax2.set_xlabel('Epoch (Myr)', fontsize=8)
            ax0.set_ylabel('$<ξ^\prime η^\prime ζ^\prime>$ (pc)', fontsize=8)
            ax1.yaxis.set_label_position('right')
            ax3.yaxis.set_label_position('right')

        # Set limits
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            ax.set_xlim(-np.max(self.series.time), -(np.min(self.series.time) - 1))
        ax2.set_ylim(-70, 70)

        # Set ticks
        if style == '1x3':
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
        if style == '2x2':
            ax1.yaxis.tick_right()
            ax3.yaxis.tick_right()
            ax0.set_xticklabels([])
            ax1.set_xticklabels([])
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            ax.tick_params(
                top=True, right=True, which='both',
                direction='in', width=0.5, labelsize=8)

        # Set spines
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        save_figure(
            self.name, f'trajectory_time_ξηζ_{style}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_map(self, labels=False, title=False, forced=False, default=False, cancel=False):
        """ Creates a Mollweide projection of a traceback. For this function to work, uvw
            velocities must not compensated for the sun velocity and computing xyz positions.
        """

        # Initialize figure
        fig = plt.figure(figsize=(6.66, 3.33), facecolor='w', dpi=300)
        ax = fig.add_subplot(111, projection="mollweide")

        # Compute coordinates
        from Traceback.coordinate import galactic_xyz_equatorial_rδα
        positions = np.array([[
            galactic_xyz_equatorial_rδα(*star.position_xyz[step])[0]
            for step in range(self.series.number_of_steps)] for star in self])
        alphas = np.vectorize(lambda α: α - (2 * np.pi if α > np.pi else 0.0))(positions[:,:,2])
        deltas = positions[:,:,1]

        # Plot trajectories
        for star in range(len(self)):
            color = '#2635ff' if not self[star].outlier else '#ff2634'

            # Identify discontinuties
            discontinuties = (
                np.abs(alphas[star, 1:] - alphas[star, :-1]) > 3 * np.pi / 2).nonzero()[0] + 1

            # Create individual segments
            segments = []
            lower_limit = 0
            for upper_limit in discontinuties:
                segments.append(range(lower_limit, upper_limit))
                lower_limit = upper_limit
            segments.append(np.arange(lower_limit, alphas.shape[1]))

            # Plot individual segments
            for i in segments:
                ax.plot(alphas[star, i], deltas[star, i], color=color, lw=1, zorder=0.2)

            # Labels
            if labels:
                ax.text(
                    alphas[star, 0] + 0.1, deltas[star, 0] + 0.1, star + 1,
                    horizontalalignment='left', fontsize=6, zorder=0.2)

        # Plot current-day positions
        ax.scatter(alphas[:,0], deltas[:,0], marker='.', color='#000000', zorder=0.3)

        # Show proper motion arrows
        for star in self.series.data:
            ax.arrow(
                star.position.values[2] - (2 * np.pi if star.position.values[2] > np.pi else 0.0),
                star.position.values[1], -star.velocity.values[2]/4, -star.velocity.values[1]/4,
                head_width=0.03, head_length=0.03, color='#000000', zorder=0.4)

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title('Mollweide projection of tracebacks', fontsize=8)

        # Format axis
        ax.grid(zorder=1)

        # Save figure
        save_figure(
            self.name, f'Mollweide_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_scatter(
            self, i, j, step=None, age=None, errors=False, labels=False, mst=False,
            title=False, forced=False, default=False, cancel=False):
        """ Creates a 2D scatter plot of star positions in i and j at a given 'step' or 'age' in
            Myr. If 'age' doesn't match a step, the closest step is used instead. 'age' overrules
            'steps' if both are given. 'labels' adds the stars' name and 'mst' adds the minimun
            spanning tree branches.
        """

        # Initialize axis
        axis = {'x': 0, 'y': 1, 'z': 2}
        keys = tuple(axis.keys())

        # Check if X and Y axes are valid, and initialize axis
        self.series.stop(
            type(i) != str, 'TypeError',
            "X axis 'i' must be a string ({} given).", type(i))
        self.series.stop(
            i.lower() not in keys, 'ValueError',
            "X axis 'i' must be an axis key ('x', 'y' or 'z', {} given).", i)
        i = axis[i.lower()]
        self.series.stop(
            type(j) != str, 'TypeError',
            "Y axis 'j' must be a string ({} given).", type(j))
        self.series.stop(
            j.lower() not in keys, 'ValueError',
            "Y axis 'j' must be an axis key ('x', 'y' or 'z', {} given).", j)
        j = axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step)
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step)
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age)
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age)

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Initialize figure
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
        ax = fig.add_subplot(111)

        # Plot xyz positions
        ax.scatter(
            [star.position_xyz[step, i] for star in self.sample],
            [star.position_xyz[step, j] for star in self.sample], marker='o', color='0.0')

        # Plot error bars
        self.series.stop(
            type(errors) != bool, 'TypeError',
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

        # Show star labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.sample:
                ax.text(star.position_xyz[step, i] + 1, star.position_xyz[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=6)

        # Create branches
        self.series.stop(
            type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (branch.start.position[step, i], branch.end.position[step, i]),
                    (branch.start.position[step, j], branch.end.position[step, j]), color='#0000ff')

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                "{} and {} positions of stars in β Pictoris at {} Myr "
                "wihtout outliers.\n".format(keys[i].upper(), keys[j].upper(), age), fontsize=8)

        # Set labels
        ax.set_xlabel(f'${keys[i].lower()}$ (pc)')
        ax.set_ylabel(f'${keys[j].lower()}$ (pc)')

        # Save figure
        save_figure(
            self.name,
            f'2D_Scatter_{self.name}_{keys[i].upper()}{keys[j].upper()}_at_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_3D_scatter(
            self, step=None, age=None, errors=False, labels=False, mst=False,
            title=False, forced=False, default=False, cancel=False):
        """ Creates a 3D scatter plot of star positions in x, y and z at a given 'step' or 'age'
            in Myr. If 'age' doesn't match a step, the closest step is used instead. 'age'
            overrules 'step' if both are given. 'labels' adds the stars' name and 'mst' adds the
            minimum spanning tree branches.
        """

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step)
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step)
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age)
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age)

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(step * self.series.timestep.value, 2)

        # Initialize figure
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Plot xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.sample],
            [star.relative_position_xyz[step, 1] for star in self.sample],
            [star.relative_position_xyz[step, 2] for star in self.sample],
            marker='o', color='0.0')

        # Plot error bars
        self.series.stop(
            type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.sample:
                position = star.relative_position_xyz[step]
                error = star.relative_position_xyz_error[step]

                # X axis error bars
                ax.plot(
                    (position[0] - error[0], position[0] + error[0]),
                    (position[1], position[1]), (position[2], position[2]),
                    color='0.1', linewidth=0.7)

                # Y axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1] - error[1], position[1] + error[1]),
                    (position[2], position[2]), color='0.1', linewidth=0.7)

                # Z axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1], position[1]),
                    (position[2] - error[2], position[2] + error[2]), color='0.1', linewidth=0.7)

        # Show star labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.sample:
                ax.text(
                    star.position_xyz[step, 0] + 2, star.position_xyz[step, 1] + 2,
                    star.position_xyz[step, 2] + 2, star.name,
                    horizontalalignment='left', fontsize=6)

        # Create branches
        self.series.stop(
            type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                ax.plot((
                    branch.start.relative_position_xyz[step, 0],
                    branch.end.relative_position_xyz[step, 0]), (
                    branch.start.relative_position_xyz[step, 1],
                    branch.end.relative_position_xyz[step, 1]), (
                    branch.start.relative_position_xyz[step, 2],
                    branch.end.relative_position_xyz[step, 2]), color='#0000ff')

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                f'Minimum spanning tree of stars in β Pictoris at {age:.1f} Myr\n', fontsize=8)

        # Set labels
        ax.set_xlabel('\n $x$ (pc)')
        ax.set_ylabel('\n $y$ (pc)')
        ax.set_zlabel('\n $z$ (pc)')

        # Save figure
        save_figure(
            self.name, f'3D_Scatter_{self.name}_at_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_and_3D_scatter(
            self, ages, title=False, forced=False, default=False, cancel=False):
        """ Creates a three 4-axis columns of xy, xz and yz 2D scatters, and a 3D scatter at three
            ages definied by a list or tuple.
        """

        # Check if ages is valid
        self.series.stop(
            type(ages) not in (tuple, list), 'TypeError',
            "'ages' must a tuple or a list ({} given).", type(ages))
        self.series.stop(
            len(ages) != 3, 'ValueError', "'ages' must be have a length of 3 ({} given).", ages)

        # Initialize figure
        self.fig = plt.figure(figsize=(9, 11.15), facecolor='w', dpi=300)

        # Create axes
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

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                '$X-Y$, $X-Z$, $Y-Z$ and 3D scatters at '
                f'{ages[0]:.1f}, {ages[1]:.1f} and {ages[2]:.1f}Myr.\n', fontsize=8)

        # Save figure
        save_figure(
            self.name, '2D_Scatter_{}_{}_{}_{}_Myr.pdf'.format(self.name, *ages),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_axis(self, i, j, step=None, age=None, index=1, left=0, bottom=0):
        """ Creates a 2D axis. """

        # Initialize axis
        axis = {'x': 0, 'y': 1, 'z': 2}
        keys = tuple(axis.keys())

        # Check if X and Y axes are valid, and initialize axis
        self.series.stop(
            type(i) != str, 'TypeError',
            "X axis 'i' must be a string ({} given).", type(i))
        self.series.stop(
            i.lower() not in keys, 'ValueError',
            "X axis 'i' must be an axis key ('x', 'y' or 'z', {} given).", i)
        i = axis[i.lower()]
        self.series.stop(
            type(j) != str, 'TypeError',
            "Y axis 'j' must be a string ({} given).", type(j))
        self.series.stop(
            j.lower() not in keys, 'ValueError',
            "Y axis 'j' must be an axis key ('x', 'y' or 'z', {} given).", j)
        j = axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step)
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step)
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age)
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age)

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Initialize axis
        ax = self.fig.add_subplot(4, 3, index, position=[left, bottom, 0.255, 0.20])

        # Plot stars' xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, i] for star in self.sample],
            [star.relative_position_xyz[step, j] for star in self.sample],
            marker='o', s=8, color='#000000')

        # Plot outliers' xyz relative positions
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

        # Set labels
        ax.set_xlabel(f'${keys[i].upper()}$ (pc)')
        ax.set_ylabel(f'${keys[j].upper()}$ (pc)', labelpad=-12.)

        # Set limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

    def create_3D_axis(self, step=None, age=None, index=1, left=0, bottom=0):
        """ Creates a 3D axis. """

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step)
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step)
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age)
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age)

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(step * self.series.timestep.value, 2)

        # Initialize figure
        ax = self.fig.add_subplot(
            4, 3, index, projection='3d', position=[left, bottom, 0.29, 0.215])

        # Plot stars' xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.sample],
            [star.relative_position_xyz[step, 1] for star in self.sample],
            [star.relative_position_xyz[step, 2] for star in self.sample],
            marker='o', color='0.0')

        # Plot outliers' xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.outliers],
            [star.relative_position_xyz[step, 1] for star in self.outliers],
            [star.relative_position_xyz[step, 2] for star in self.outliers],
            marker='o', color='#ff2634')

        # Create branches
        for branch in self.mst[step]:
            ax.plot((
                branch.start.relative_position_xyz[step, 0],
                branch.end.relative_position_xyz[step, 0]), (
                branch.start.relative_position_xyz[step, 1],
                branch.end.relative_position_xyz[step, 1]), (
                branch.start.relative_position_xyz[step, 2],
                branch.end.relative_position_xyz[step, 2]), color='#2635ff')

        # Set labels
        ax.set_xlabel('$X$ (pc)')
        ax.set_ylabel('$Y$ (pc)')
        ax.set_zlabel('$Z$ (pc)')

        # Set limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)

        # Set view
        ax.view_init(azim=45)

    def create_cross_covariances_scatter(
            self, i, j, step=None, age=None, errors=False, labels=False,
            title=False, forced=False, default=False, cancel=False):
        """ Creates a cross covariance scatter of star positions and velocities in i and j at a
            given 'step' or 'age' in Myr. If 'age' doesn't match a step, the closest step is used
            instead. 'age' overrules 'steps' if both are given. 'labels' adds the stars' name
            and 'mst' adds the minimum spanning tree branches.
        """

        # Initialize axis
        position_axis = {'x': 0, 'y': 1, 'z': 2}
        velocity_axis = {'u': 0, 'v': 1, 'w': 2}
        position_keys = tuple(position_axis.keys())
        velocity_keys = tuple(velocity_axis.keys())

        # Check if position and velocity axes are valid, and initialization
        self.series.stop(
            type(i) != str, 'TypeError',
            "Position axis 'i' must be a string ({} given).", type(i))
        self.series.stop(
            i.lower() not in position_keys, 'ValueError',
            "Position axis 'i' must be an postion axis key ('x', 'y' or 'z', {} given).", i)
        i = position_axis[i.lower()]
        self.series.stop(
            type(j) != str, 'TypeError',
            "Velocity axis 'j' must be a string ({} given).", type(j))
        self.series.stop(
            j.lower() not in velocity_keys, 'ValueError',
            "Velocity axis 'j' must be an postion axis key ('u', 'v' or 'w', {} given).", j)
        j = velocity_axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step)
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step)
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age)
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age)

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Initialize figure
        fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
        ax = fig.add_subplot(111)

        # Plot xyz positions
        ax.scatter(
            [star.position_xyz[step, i] for star in self.sample],
            [star.velocity_xyz[step, j] for star in self.sample], marker='o', color='0.0')

        # Plot error bars
        self.series.stop(
            type(errors) != bool, 'TypeError',
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

        # Show star labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.sample:
                ax.text(star.position_xyz[step, i] + 1, star.velocity_xyz[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=6)

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                "{} and {} covariance of stars in β Pictoris at {} Myr wihtout "
                "outliers.\n".format(position_keys[i].upper(), velocity_keys[j].upper(), age),
                fontsize=8)

        # Set labels
        ax.set_xlabel(f'{position_keys[i].upper()} (pc)')
        ax.set_ylabel(f'{velocity_keys[j].upper()} (pc/Myr)')

        # Save figure
        save_figure(
            self.name, f'covariances_scatter_{self.name}_'
            f'{position_keys[i].upper()}-{velocity_keys[j].upper()}.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_age_distribution(
            self, title=False, metric=None, index=None,
            forced=False, default=False, cancel=False):
        """ Creates a plot of the distribution of jack-knife Monte Carlo ages computed in a
            group.
        """

        # Initialize figure
        fig = plt.figure(figsize=(3.345, 3.401), facecolor='w', dpi=300)
        ax = fig.add_axes([0.104, 0.096, 0.895, 0.880])

        # Retrieve ages
        metric, index = self.get_metric(metric, index)
        metric_name = metric.label
        ages = metric.ages
        if ages.ndim == 2:
            ages = ages[self.number]
        elif ages.ndim == 3:
            ages = ages[self.number,:,index]

        # Plot uncorrected histogram and gaussian curve
        x = np.linspace(8, 36, 1000)
        μ = metric.age[index]
        σ = metric.age_int_error[index]
        gauss = np.exp(-0.5 * ((x - μ) / σ)**2) / np.sqrt(2 * np.pi) / σ
        i, = (gauss > 0.001).nonzero()
        ax.plot(
            x[i], gauss[i], color=blues[3], zorder=0.8,
            label='$\\xi^\\prime$ variance', linewidth=1.0)
        ax.hist(
            ages, bins=np.linspace(12, 32, 81), density=True,
            color=blues[3], alpha=0.3, zorder=0.8)
        ax.vlines(
            μ, ymin=0.0, ymax=np.max(gauss), color=blues[3],
            linestyle='--', zorder=0.8, linewidth=1.0)

        # Plot corrected histogram and gaussian curve
        x = np.linspace(8, 36, 1000)
        μ = metric.age_ajusted[index]
        σ = (metric.age_int_error[index]**2 + 1.56**2)**0.5
        gauss = np.exp(-0.5 * ((x - μ) / σ)**2) / np.sqrt(2 * np.pi) / σ
        i, = (gauss > 0.001).nonzero()
        ax.plot(
            x[i], gauss[i], color=blues[2], zorder=0.9,
            label='Corrected $\\xi^\\prime$ variance', linewidth=1.0)
        ages = (ages - metric.age[index]) * (σ / metric.age_int_error[index]) + μ
        ax.hist(
            ages, bins=np.linspace(12, 32, 81), density=True,
            color=blues[2], alpha=0.3, zorder=0.9)
        ax.fill_between(
            x[i], np.zeros_like(x[i]), gauss[i], color=blues[2],
            alpha=0.3, linewidth=0., zorder=0.3)
        ax.vlines(
            μ, ymin=0.0, ymax=np.max(gauss), color=blues[2],
            linestyle='--', zorder=0.9, linewidth=1.0)

        # Plot gaussian curve from Miret-Roig et al. (2020)
        μ = 18.5
        σ1, σ2 = 2.4, 2.0
        x1, x2 = np.arange(μ - 10, μ, 0.01), np.arange(μ, μ + 10, 0.01)
        gauss1 = np.exp(-0.5 * ((x1 - μ) / σ1)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        gauss2 = np.exp(-0.5 * ((x2 - μ) / σ2)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        x = np.concatenate((x1, x2))
        gauss = np.concatenate((gauss1, gauss2))
        i, = (gauss > 0.001).nonzero()
        ax.plot(
            x[i], gauss[i], color=greens[3], zorder=0.75,
            label='Miret-Roig et al. (2020)', linewidth=1.0)
        ax.fill_between(
            x[i], np.zeros_like(x[i]), gauss[i], color=greens[1],
            alpha=0.3, linewidth=0., zorder=0.7)
        ax.vlines(
            μ, ymin=0.0, ymax=np.max(gauss), color=greens[1],
            linestyle='--', zorder=0.75, linewidth=1.0)

        # Plot gaussian curve from Crundall et al. (2019)
        μ = 18.3
        σ1, σ2 = 1.2, 1.3
        x1, x2 = np.arange(μ - 10, μ, 0.01), np.arange(μ, μ + 10, 0.01)
        gauss1 = np.exp(-0.5 * ((x1 - μ) / σ1)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        gauss2 = np.exp(-0.5 * ((x2 - μ) / σ2)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        x = np.concatenate((x1, x2))
        gauss = np.concatenate((gauss1, gauss2))
        i, = (gauss > 0.001).nonzero()
        ax.plot(
            x[i], gauss[i], color=greens[2], zorder=0.6,
            label='Crundall et al. (2019)', linewidth=1.0)
        ax.fill_between(
            x[i], np.zeros_like(x[i]), gauss[i], color=greens[2],
            alpha=0.3, linewidth=0., zorder=0.6)
        ax.vlines(
            μ, ymin=0.0, ymax=np.max(gauss), color=greens[2],
            linestyle='--', zorder=0.6, linewidth=1.0)

        # Show a shaded area for LDB and isochrone ages
        LDB_range = np.array([20, 26])
        ax.fill_between(
            LDB_range, 0, 1, color='0.5', alpha=0.1,  linewidth=0.,
            transform=ax.get_xaxis_transform(), zorder=0.5)

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                f'Distribution of {self.series.jackknife_number} jack-knife Monte Carlo,\n'
                f'Average age: ({metric.age[0]:.1f} '
                f'± {metric.age_int_error[0]:.1F}) Myr\n', fontsize=8)

        # Set legend
        legend = ax.legend(loc=1, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.8))
        legend.get_frame().set_edgecolor((0.0, 0.0, 0.0, 1.0))
        legend.get_frame().set_linewidth(0.5)

        # Set labels
        ax.set_xlabel('Age (Myr)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

        # Set limits
        ax.set_xlim(13, 29)

        # Set ticks
        ax.set_xticks([14., 16., 18., 20., 22., 24., 26., 28.])
        ax.tick_params(
            top=True, right=True, which='both',
            direction='in', width=0.5, labelsize=8)

        # Set spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Save figure
        save_figure(
            self.name, f'age_distribution_jackknife_{self.name}_{metric_name}.pdf',
            tight=False, forced=forced, default=default, cancel=cancel)
        # plt.show()

def create_histogram(
        ages, initial_scatter, number_of_stars, number_of_groups, age,
        title=False, forced=False, default=False, cancel=False):
    """ Creates an histogram of ages computed by multiple tracebacks. """

    # Check if ages are valid
    stop(
        type(ages) not in (tuple, list), 'TypeError',
        "'ages' must either must be a tuple or list ({} given)", type(ages))
    for age in ages:
        stop(
            type(age) not in (int, float), 'TypeError',
            "All 'ages' must be an integer or float ({} given).", type(age))
        stop(
            age < 0, 'ValueError',
            "All 'ages' must be greater than or equal to 0.0 ({} given).", type(age))

    # Check if initial scatter is valid
    stop(
        type(initial_scatter) not in (int, float), 'TypeError',
        "'initial_scatter' must be an integer or float ({} given).", type(initial_scatter))
    stop(
        initial_scatter < 0, 'ValueError',
        "'initial_scatter' must be greater than or equal to 0.0 ({} given).", type(initial_scatter))

    # Check if number_of_stars is valid
    stop(
        type(number_of_stars) not in (int, float), 'TypeError',
        "'number_of_stars' must an integer or float ({} given).", type(number_of_stars))
    stop(
        number_of_stars % 1 != 0, 'ValueError',
        "'number_of_stars' must be convertible to an integer ({} given).", number_of_stars)
    number_of_stars = int(number_of_stars)

    # Check if number_of_groups is valid
    stop(
        type(number_of_groups) not in (int, float), 'TypeError',
        "'number_of_groups' must an integer or float ({} given).", type(number_of_groups))
    stop(
        number_of_groups % 1 != 0, 'ValueError',
        "'number_of_groups' must be convertible to an integer ({} given).", number_of_groups)
    number_of_groups = int(number_of_groups)

    # Check if age is valid
    stop(
        type(age) not in (int, float), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(
        age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Initialize figure
    fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
    ax = fig.add_subplot(111)

    # Plot histogram
    hist, bin_edges = np.histogram(ages, density=True)
    ax.hist(ages, bins='auto', density=True) # bins=np.arange(21.975, 26.025, 0.05)

    # Set title
    stop(
        type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title(
            "Distribution of ages ({} groups, {} Myr, {} stars,\ninitial scatter = "
            "{} pc, {})".format(number_of_groups, age, number_of_stars, initial_scatter,
                'calculated age = ({} ± {}) Myr'.format(
                    np.round(np.average(ages), 3), np.round(np.std(ages), 3))),
            fontsize=8)

    # Set labels
    ax.set_xlabel('Age (Myr)')
    ax.set_ylabel('Number of groups')

    # Save figure
    save_figure(
        self.name, f'Distribution of ages for {number_of_groups} groups, {age:.1f}Myr, '
        f'{number_of_stars} stars, initial scatter = {initial_scatter}pc.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_color_mesh(
        initial_scatter, number_of_stars, errors, age, number_of_groups, method,
        title=False, forced=False, default=False, cancel=False):
    """ Creates a color mesh of errors over the initial scatter and number_of_stars.
        !!! Créer une fonction pour passer d'un array Numpy de shape (n, 3) à un !!!
        !!! color mesh + smoothing, genre create_color_mesh(x, y, z, smoothing). !!!
    """

    # Check if initial scatter is valid
    stop(
        type(initial_scatter) not in (tuple, list, np.ndarray),
        "'initial_scatter' must either must be a tuple or list ({} given)", type(initial_scatter))
    for scatter in np.array(initial_scatter).flatten():
        stop(
            type(scatter) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'initial_scatter' must be an integer or float ({} given).", type(scatter))
        stop(
            age < 0, 'ValueError',
            "All 'initial_scatter' must be greater than or equal to 0.0 ({} given).", type(scatter))

    # Check if number_of_stars is valid
    stop(
        type(number_of_stars) not in (tuple, list, np.ndarray),
        "'number_of_stars' must either must be a tuple or list ({} given)", type(errors))
    for star in np.array(number_of_stars).flatten():
        stop(
            type(star) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'initial_scatter' must be an integer or float ({} given).", type(star))
        stop(
            star < 0, 'ValueError',
            "All 'initial_scatter' must be greater than or equal to 0.0 ({} given).", type(star))
        stop(
            star % 1 != 0, 'ValueError',
            "All 'number_of_stars' must be convertible to an integer ({} given).", star)

    # Check if errors are valid
    stop(
        type(errors) not in (tuple, list, np.ndarray),
        "'errors' must either must be a tuple or list ({} given)", type(errors))
    for error in np.array(errors).flatten():
        stop(
            type(error) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'errors' must be an integer or float ({} given).", type(error))
        stop(
            error < 0, 'ValueError',
            "All 'errors' must be greater than or equal to 0.0 ({} given).", type(error))

    # Check if age is valid
    stop(
        type(age) not in (int, float, np.int64, np.float64), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(
        age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Check if number_of_groups is valid
    stop(
        type(number_of_groups) not in (int, float, np.int64, np.float64), 'TypeError',
        "'number_of_groups' must an integer or float ({} given).", type(number_of_groups))
    stop(
        number_of_groups % 1 != 0, 'ValueError',
        "'number_of_groups' must be convertible to an integer ({} given).", number_of_groups)
    number_of_groups = int(number_of_groups)

    # Check if method is valid
    stop(type(method) != str, 'TypeError', "'method' must a string ({} given).", type(method))

    # Initialize figure
    fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
    ax = fig.add_subplot(111)

    # Plot mesh
    x, y = np.meshgrid(initial_scatter, number_of_stars)
    grid_x, grid_y = np.mgrid[0:20:81j, 20:100:81j]
    grid_z = griddata(
        np.array([(i, j) for i in initial_scatter for j in number_of_stars]),
        errors.T.flatten(), (grid_x, grid_y), method='linear')
    ax.pcolormesh(grid_x, grid_y, grid_z, cmap=plt.cm.PuBu_r, vmin=0, vmax=6)
    fig.colorbar(
        mappable=plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 6.0), cmap=plt.cm.PuBu_r),
        ax=ax, ticks=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], format='%0.1f')

    # Set title
    stop(
        type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title(
            'Scatter on age (Myr) over the initial scatter (pc)\n'
            f'and the number of stars ({number_of_groups} groups, {age:.1f}Myr)', fontsize=8)

    # Set labels
    ax.set_xlabel('Initial scatter (pc)')
    ax.set_ylabel('Number of stars')

    # Set ticks
    ax.set_xticks([0., 5., 10., 15., 20.])
    ax.set_yticks([20., 40., 60., 80., 100.])

    # Save figure
    save_figure(
        self.name, f'Scatter on age ({age:.1f}Myr, {method}).pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def plot_age_error(title=False, forced=False, default=False, cancel=False):
    """ Creates a plot of ages obtained for diffrent measurement errors on radial velocity
        and radial velocity shifts due to gravitationnal redshift.
    """

    # Initialize figure
    fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
    ax = fig.add_subplot(111)

    # Plot + 0.0 km/s points
    ax.errorbar(
        np.array([
            0.0, 0.25, 0.50, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0]),
        np.array([
            23.824, 23.506, 22.548, 21.238, 19.454, 17.639,
            16.008, 14.202, 12.670, 11.266, 9.7320, 8.8740, 8.044]),
        yerr=np.array([
            0.376, 0.517, 0.850, 1.062, 1.204, 1.383,
            1.534, 1.612, 1.544, 1.579, 1.576, 1.538, 1.504]),
        fmt='o', color='0.0', ms=6.0, elinewidth=1.0,
        label='$\\Delta v_{r,grav}$ = 0,0 km/s')

    # Plot + 0.5 km/s points
    ax.errorbar(
        np.array([
            0.0, 0.25, 0.50, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.00, 2.25, 2.5, 2.75, 3.0]),
        np.array([
            19.858, 19.655, 19.116, 19.292, 17.246, 15.988,
            14.749, 13.577, 12.379, 11.222, 10.229, 9.2100, 8.446]),
        yerr=np.array([
            0.376, 0.425, 0.641, 0.773, 0.992, 1.136,
            1.129, 1.251, 1.338, 1.331, 1.272, 1.345, 1.323]),
        fmt='D', color='0.33', ms=6.0, elinewidth=1.0,
        label='$\\Delta v_{r,grav}$ = 0,5 km/s')

    # Plot + 1.0 km/s points
    ax.errorbar(
        np.array([
            0.0, 0.25, 0.50, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.00, 2.25, 2.5, 2.75, 3.0]),
        np.array([
            16.870, 16.743, 16.404, 15.884, 15.26, 14.522,
            13.529, 12.619, 11.751, 10.847, 9.982, 9.3530, 8.461]),
        yerr=np.array([
            0.379, 0.453, 0.583, 0.685, 0.864, 0.930,
            0.951, 1.032, 1.147, 1.035, 1.142, 1.187, 1.149]),
        fmt='^', color='0.67', ms=6.0, elinewidth=1.0,
        label='$\\Delta v_{r,grav}$ = 1,0 km/s')

    # Plot β Pictoris typical error line
    ax.axvline(x=1.0105, ymin=0.0, ymax = 25.0, linewidth=1.0, color='#000000', ls='dashed')
    ax.text(
        1.15, 6.95, 'Erreur de mesure\nsur $v_r$ typique des\nmembres de $\\beta\\,$PMG',
        horizontalalignment='left', fontsize=14)

    # Set title
    stop(
        type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title(
            "Measured age of a simulation of 1000 24 Myr-old groups\n"
            "over the measurement error on RV (other errors typical of Gaia EDR3)\n", fontsize=8)

    # Set legend
    ax.legend(loc=1, fontsize=8)

    # Set labels
    ax.set_xlabel('Error on radial velocity (km/s)', fontsize=8)
    ax.set_ylabel('Age (Myr)', fontsize=8)

    # Set limits
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(6, 24.5)

    # Set ticks
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
    ax.tick_params(
        top=True, right=True, which='both',
        direction='in', width=0.5, labelsize=8)

    # Save figure
    save_figure(
        self.name, f'Errors_rv_shift_plot_{self.name}.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_minimum_error_plots(title=False, forced=False, default=False, cancel=False):
    """ Creates a plot of the error on the age of minimal scatter as a function of the error
        on the uvw velocity.
    """

    # Initialize figure
    fig = plt.figure(figsize=(3.33, 3.33), facecolor='w', dpi=300)
    ax = fig.add_subplot(111)

    # Plot ages as a function of errors
    errors = np.array([
        0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1., 1.5, 2., 2.5, 3.,
        3.5, 4., 4.5, 5., 6., 7., 8., 9., 10., 12., 14., 16., 18., 20.])
    ages = np.array([
        24.001, 23.966, 23.901, 23.74, 23.525, 22.224, 20.301, 18.113, 15.977,
        11.293, 7.9950, 5.8030, 4.358, 3.3640, 2.6650, 2.2040, 1.7560, 1.2570,
        0.9330, 0.7350, 0.5800, 0.488, 0.3460, 0.2620, 0.1920, 0.1600, 0.1340])
    ax.plot(errors, ages, '.-', color='0.0', linewidth=1.0)

    # Set title
    stop(
        type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title('Impact of UVW velocity on the age of minimal scatter.', fontsize=8)

    # Set labels
    ax.set_xlabel('Error on UVW velocity (km/s)', fontsize=8)
    ax.set_ylabel('Age at minimal XYZ scatter (Myr)', fontsize=8)

    # Set ticks
    ax.tick_params(
        top=True, right=True, which='both',
        direction='in', width=0.5, labelsize=8)

    # Save figure
    save_figure(
        self.name, f'Minimum_error_plot_{self.name}.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()
