# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
output.py: Defines functions to create data output such as plots of association size metrics
over time, 2D and 3D scatters at a given time, histograms, color mesh, etc.
"""

import numpy as np
from os import path
from cycler import cycler
from matplotlib import pyplot as plt, ticker as tkr
from colorsys import hls_to_rgb
from scipy.interpolate import griddata, interp1d
from .collection import *
from .coordinate import *

# Set pyplot rc parameters
plt.rc('font', serif='Latin Modern Math', family='serif', size='8')
plt.rc('mathtext', fontset='custom', it='Latin Modern Roman:italic', rm='Latin Modern Roman:roman')
plt.rc('lines', markersize=4)
plt.rc('pdf', fonttype=42)

# Set ticks label with commas instead of dots for French language publications
# ax.xaxis.set_major_formatter(format_ticks)
# ax.yaxis.set_major_formatter(format_ticks)
format_ticks = tkr.FuncFormatter(lambda x, pos: str(round(float(x), 1)))

# Set colors
class colors():
    """Defines a set of RGB color and grey tones."""

    # RGB color tones
    vars().update(
        {
            name: color for name, color in zip(
                (
                    'red',   'orange', 'yellow',  'chartreuse',
                    'green', 'lime',   'cyan',    'azure',
                    'blue',  'indigo', 'magenta', 'pink'
                ),
                tuple(
                    tuple(
                        tuple(
                            np.round(hls_to_rgb(hue / 360, luma, 1.0), 3)
                        ) for luma in np.arange(0.05, 1.0, 0.05)
                    ) for hue in np.arange(0, 360, 30)
                )
            )
        }
    )

    # Grey tones
    black = (0.0, 0.0, 0.0)
    grey = tuple((luma, luma, luma) for luma in np.arange(0.05, 1.0, 0.05))
    white = (1.0, 1.0, 1.0)

    # Metric colors
    metric = (green[3], green[6], green[9], green[12], azure[3], azure[6], azure[9], azure[12])

    # Color cycle
    cycle = cycler(color=(azure[6], pink[6], chartreuse[6], indigo[9], orange[9], lime[9]))

class Output_Series():
    """Output methods for a series of groups."""

    def save_figure(
        self, name, file_path=None, extension='pdf', tight=True,
        forced=False, default=False, cancel=False
    ):
        """Saves figure with or without tight layout and some padding."""

        # Check 'tight' argument
        self.stop(
            type(forced) != bool, 'TypeError',
            "'tight' must be a boolean ({} given).", type(tight)
        )

        # Save figure
        def save(file_path, tight):
            if tight:
                plt.savefig(file_path, bbox_inches='tight', pad_inches=0.01)
            else:
                plt.savefig(file_path)

        # Choose behavior
        self.choose(
            name, extension, save, tight, file_path=file_path,
            cancel=cancel, forced=forced, default=default
        )

    def save_table(
        self, name, lines, header=None, file_path=None, extension='txt',
        forced=False, default=False, cancel=False
    ):
        """Saves a table to a CSV file for a given header and data."""

        # Save table
        def save(file_path, lines, header):
            with open(file_path, 'w') as output_file:
                if header is not None:
                    output_file.write(header + '\n')
                output_file.writelines([line + '\n' for line in lines])

        # Choose behavior
        self.choose(
            name, extension, save, lines, header, file_path=file_path,
            cancel=cancel, forced=forced, default=default
        )

    def choose(
        self, name, extension, save, *save_args, file_path=None,
        forced=False, default=False, cancel=False
    ):
        """
        Checks whether a path already exists and asks for user input if it does. The base path
        is assumed to be the output directory. Also, if the path does not have an extension, a
        an extension is added.
        """

        # Get file path
        file_path = self.get_file_path(name, extension, file_path=file_path)

        # Check if a file already exists
        if path.exists(file_path):
            choice = None
            self.stop(
                type(forced) != bool, 'TypeError',
                "'forced' must be a boolean ({} given).", type(forced)
            )
            if not forced:
                self.stop(
                    type(default) != bool, 'TypeError',
                    "'default' must be a default ({} given).", type(default)
                )
                if not default:
                    self.stop(
                        type(cancel) != bool, 'TypeError',
                        "'cancel' must be a boolean ({} given).", type(cancel)
                    )
                    if not cancel:

                        # User input
                        while choice == None:
                            choice = input(
                                "A file already exists at '{}'. Do you wish to overwrite (Y), "
                                "keep both (K) or cancel (N)? ".format(file_path)
                            ).lower()

                            # Loop over question if input could not be interpreted
                            if choice not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                                print("Could not understand '{}'.".format(choice))
                                choice = None

                    # Cancel save
                    if cancel or choice in ('n', 'no'):

                        # Logging
                        self.log(
                            "'{}': file not saved because a file already exists at '{}'.",
                            name, file_path
                        )

                # Set default name and save figure
                if default or choice in ('k', 'keep'):
                    from .tools import default_name
                    file_path = default_name(file_path)
                    save(file_path, *save_args)

                    # Logging
                    self.log("'{}': file name changed and file saved at '{}'.", name, file_path)

            # Delete existing file and save figure
            if forced or choice in ('y', 'yes'):
                from os import remove
                remove(file_path)
                save(file_path, *save_args)

                # Logging
                self.log("'{}': existing file located at '{}' deleted and replaced.", name, file_path)

        # Save figure
        else:
            save(file_path, *save_args)

            # Logging
            self.log("'{}': file saved at '{}'.", name, file_path)

    def get_file_path(self, name, extension, file_path=None):
        """Returns a proper file path given a name, an extension and, optionnally, a filepath."""

        # file_path parameter
        file_path = file_path if file_path is not None else self.output(create=True) + '/'

        # Check if file_path parameter is a string, which must be done before the directory call
        self.stop(
            type(file_path) != str, 'TypeError',
            "'file_path' must be a string ({} given).", type(file_path)
        )

        # file_path redefined as the absolute path, default name and directory creation
        file_path = path.join(
            directory(self.output(), path.dirname(file_path), 'file_path', create=True),
            path.basename(file_path) if path.basename(file_path) != '' else f'{name}.{extension}'
        )

        # Check if there's an extension and add an extension, if needed
        if path.splitext(file_path)[-1] != f'.{extension}':
            file_path += f'.{extension}'

        return file_path

    def create_metrics_table(
        self, save=False, show=False, machine=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a table of the association size metrics. If 'save' if True, the table is
        saved and if 'show' is True, the table is displayed. If 'machine' is True, then a
        machine-readable table, without units in the header and '.csv' extension instead of a
        '.txt', is created. The machine-readable table also has an additional column 'status'
        to indicate whether a metric is valid or rejected, whereas the non-machine-readable
        table uses side heads.
        """

        # Returns a converted, rounded string for display
        def convert(value):

            return str(np.round(value, 2)).replace('[', '').replace(']', '')

        # Creates a single metric line
        def create_line(metric, valid=True):
            valid = np.copy(metric.valid) if valid else np.invert(metric.valid)

            # Create a machine-readable line
            if machine:
                return [
                    f'{metric.name[i]},'
                    f'{metric.latex_long[i]},'
                    f"{'Valid' if metric.valid[i] else 'Rejected'},"
                    f'{str(metric.age[i])},'
                    f'{str(metric.age_int_error[i])},'
                    f'{str(metric.age_ext_error[i])},'
                    f'{str(metric.age_error[i])},'
                    f'{str(metric.min_change[i])},'
                    f'{str(metric.age_shift[i])}' for i in np.arange(valid.size)
                ]

            # Create a human-readable line
            else:
                return [
                    f'{metric.name[i]:<50}'
                    f'{convert(metric.age[i]):>15}'
                    f'{convert(metric.age_int_error[i]):>20}'
                    f'{convert(metric.age_ext_error[i]):>20}'
                    f'{convert(metric.age_error[i]):>15}'
                    f'{convert(metric.min_change[i]):>20}'
                    f'{convert(metric.age_shift[i]):>15}'
                    for i in filter(lambda i: valid[i], np.arange(valid.size))
                    ]

        # Check save, show and machine
        self.stop(
            type(save) != bool, 'TypeError',
            "'save' must be a boolean ({} given).", type(save)
        )
        self.stop(
            type(show) != bool, 'TypeError',
            "'show' must be a boolean ({} given).", type(show)
        )
        self.stop(
            type(machine) != bool, 'TypeError',
            "'machine' must be a boolean ({} given).", type(machine)
        )

        # Set precision and order
        np.set_printoptions(precision=2)
        order = np.argsort([i.order for i in self.metrics])

        # Create header
        if machine:
            lines = [
                'Metric,LaTeX_name,Status,Age,Jackknife_error,'
                'Measurement_error,Total_error,Minimum_change,Offset'
            ]

            # Create lines
            for i in order:
                lines += create_line(self.metrics[i], valid=True)

        # Create header
        else:
            lines = [
                f"{'':-<155}",
                f"{'Association size metric':<50}{'Age':>15}{'Jackknife Error':>20}"
                f"{'Measurement Error':>20}{'Total Error':>15}{'Minimum Change':>20}{'Offset':>15}",
                f"{'[Myr]':>65}{'[Myr]':>20}{'[Myr]':>20}{'[Myr]':>15}{'[%]':>20}{'[Myr]':>15}",
                f"{'':-<155}"
            ]

            # Create lines of valid association size metrics
            lines.append('Valid\n' f"{'':-<155}")
            for i in order:
                lines += create_line(self.metrics[i], valid=True)
            lines.append(f"{'':-<155}")

            # Create lines of rejected association size metrics
            lines.append('Rejected\n' f"{'':-<155}")
            for i in order:
                lines += create_line(self.metrics[i], valid=False)
            lines.append(f"{'':-<155}")

        # Show table
        if show:
            for line in lines:
                print(line)

        # Save table
        if save:
            self.save_table(
                self.name, lines, file_path=f'metrics_{self.name}',
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

    def initialize_figure_metric(self):
        """Initializes a figure and an axis to plot association size metrics."""

        # Initialize figure
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor=colors.white, dpi=300)
        ax = fig.add_axes([0.103, 0.103, 0.895, 0.895])

        return fig, ax

    def check_robust_sklearn_metric(self, robust, sklearn):
        """
        Checks if 'robust' and 'sklearn' arguments are valid to select to proper association
        size metrics.
        """

        # Check 'robust' and 'sklearn' arguments
        self.stop(
            type(robust) != bool, 'TypeError',
            "'robust' must be a boolean ({} given).", type(robust)
        )
        self.stop(
            type(sklearn) != bool, 'TypeError',
            "'sklearn' must be a boolean ({} given).", type(sklearn)
        )
        self.stop(
            robust and sklearn, 'ValueError',
            "'robust' and 'sklearn' cannot both be True."
        )

    def plot_metric(self, ax, metric, index, color, linestyle, zorder=0.5, secondary=False):
        """
        Plots the association size metric's value over time on a given axis along with an
        enveloppe to display the uncertainty. The display is further customized with the
        'linestyle' and 'color' parameters. If 'secondary' is True, secondary lines are
        displayed as well.
        """

        # Check metric status
        if metric.status:

            # Extrapolate one point in time, value and value error arrays
            time = np.insert(self.time, 0, 1.0)
            value = np.insert(
                metric.value.T[index], 0,
                interp1d(self.time, metric.value.T[index], fill_value='extrapolate')(1.0)
            )
            value_error = np.insert(
                metric.value_error.T[index], 0,
                interp1d(self.time, metric.value_error.T[index], fill_value='extrapolate')(1.0)
            )

            # Plot the value of the metric over time
            ax.plot(
                time, value, label=(
                    f'{metric.latex_short[index]} : ({metric.age[index]:.1f}'
                    f' ± {metric.age_error[index]:.1f}) Myr'
                ),
                color=color, alpha=1.0, linewidth=1.0, linestyle=linestyle,
                solid_capstyle='round', dash_capstyle='round', zorder=zorder
            )

            # Plot an enveloppe to display the uncertainty
            ax.fill_between(
                time, value - value_error, value + value_error,
                color=color, alpha=0.15, linewidth=0.0, zorder=zorder - 0.5
            )

            # Plot secondary lines
            self.stop(
                type(secondary) != bool, 'TypeError',
                "'secondary' must be a boolean ({} given).", type(secondary)
            )
            if secondary:
                values = metric.values.reshape((
                    metric.values.shape[0] * metric.values.shape[1],
                    metric.values.shape[2], metric.values.shape[3])
                )
                for i in np.unique(
                    np.round(np.linspace(0, self.number_of_groups * self.number_of_iterations - 1, 20))
                ):
                    ax.plot(
                        self.time, values[int(i),:,index],
                        color=color, alpha=0.6, linewidth=0.5,
                        linestyle=linestyle, zorder=zorder - 0.25
                    )
        # Logging
        else:
            self.log(
                "Could not plot '{}' metric for '{}' series. It was not computed.",
                str(metric.name[index]), self.name, display=True
            )

    def set_title_metric(self, ax, title, metric):
        """Sets a title for association size metrics plots if 'title' is True."""

        self.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            if self.from_data:
                ax.set_title(
                    "{} of {}\n over {:.1f} Myr with a {:.1f} km/s radial "
                    "velocity correction\n".format(
                        metric, self.name, self.duration.value,
                        self.rv_shift.to('km/s').value
                    ), fontsize=8
                )
            elif self.from_model:
                ax.set_title(
                    "Average {} of {} simulated associations over {:.1f} Myr\n"
                    "with kinematics similar to {} and a {:.1f} km/s radial velocity "
                    "bias\n".format(
                        metric, self.number_of_groups, self.duration.value,
                        self.name, self.rv_shift.to('km/s').value
                    ), fontsize=8
                )

    def set_axis_metric(self, ax, hide_x=False, hide_y=False):
        """Sets the parameters of an axis and its figure to plot association size metrics."""

        # Set legend
        legend = ax.legend(loc=2, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor(colors.white + (0.8,))
        legend.get_frame().set_edgecolor(colors.black)
        legend.get_frame().set_linewidth(0.5)

        # Set labels
        ax.set_xlabel('Epoch (Myr)', fontsize=8)
        ax.set_ylabel('Association size (pc)', fontsize=8)

        # Set limits
        ax.set_xlim(self.final_time.value + 1, self.initial_time.value + 1)
        # ax.set_xlim(self.final_time.value, self.initial_time.value + 1)
        # ax.set_ylim(-1., 39.)

        # Set ticks
        # ax.set_xticks([0., -5., -10., -15., -20., -25., -30., -35., -40, -45, -50])
        # ax.set_yticks([0.,  5.,  10.,  15.,  20.,  25.,  30.,  35.])
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
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of x-x, y-y and z-z covariances, and the determinant and the trace of
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
        self.plot_metric(ax, covariances_xyz_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, covariances_xyz_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax, covariances_xyz, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, covariances_xyz, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, covariances_xyz, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$XYZ$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} covariances"
        )

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'covariances_xyz_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_covariances_ξηζ_plot(
        self, robust=False, sklearn=False, title=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of ξ'-ξ', η'-η' and ζ'-ζ' covariances, and the determinant and the trace
        of the covariances matrix. If either 'robust' or 'sklearn' is True, the robust or sklearn
        covariances matrix is used. Otherwise, the empirical covariances matrix is used.
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
        self.plot_metric(ax, covariances_ξηζ_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, covariances_ξηζ_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax, covariances_ξηζ, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, covariances_ξηζ, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, covariances_ξηζ, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$ξ^\prime η^\prime ζ^\prime$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} covariances"
        )

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'covariances_ξηζ_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_cross_covariances_xyz_plot(
        self, robust=False, sklearn=False, title=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of x-u, y-v and z-w cross covariances, and the determinant and the trace
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
        self.plot_metric(ax, cross_covariances_xyz_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, cross_covariances_xyz_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax, cross_covariances_xyz, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, cross_covariances_xyz, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, cross_covariances_xyz, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$X-U$, $Y-V$ and $Z-W$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} cross covariances"
        )

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'Cross_covariances_xyz_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_cross_covariances_ξηζ_plot(
        self, robust=False, sklearn=False, title=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of ξ'-vξ', η'-vη' and ζ'-vζ' cross covariances, and the determinant and
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
        self.plot_metric(ax, cross_covariances_ξηζ_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, cross_covariances_ξηζ_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax, cross_covariances_ξηζ, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, cross_covariances_ξηζ, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, cross_covariances_ξηζ, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(
            ax, title, '$ξ^\prime-\dot{ξ}^\prime$, '
            '$η^\prime-\dot{η}^\prime$ and $ζ^\prime-\dot{ζ}^\prime$'
            f"{' robust' if robust else ' sklearn' if sklearn else ''} cross covariances"
        )

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'Cross_covariances_ξηζ_{self.name}'
            f"{'_robust' if robust else '_sklearn' if sklearn else ''}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_mad_xyz_plot(self, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the total xyz median absolute deviation (MAD), and x, y and z
        components of the MAD.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot total xyz MAD, x MAD, y and z MAD
        self.plot_metric(ax, self.mad_xyz_total, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, self.mad_xyz, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, self.mad_xyz, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, self.mad_xyz, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax, title, '$XYZ$ MAD')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'MAD_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_mad_ξηζ_plot(self, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the total ξηζ median absolute deviation (MAD), and ξ', η' and ζ'
        components of the MAD.
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot total ξ'η'ζ' MAD, ξ' MAD, η' MAD and ζ' MAD
        self.plot_metric(ax, self.mad_ξηζ_total, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, self.mad_ξηζ, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, self.mad_ξηζ, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, self.mad_ξηζ, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax, title, '$ξ^\prime η^\prime ζ^\prime$ MAD')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'MAD_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_xyz_mst_plot(self, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the mean branch length (both empirical and robust) and median absolute
        deviation of the xyz minimum spanning tree (MST).
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot association size metrics
        self.plot_metric(ax, self.mst_xyz_mean, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, self.mst_xyz_mean_robust, 0, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, self.mst_xyz_mad, 0, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax, title, '$XYZ$ MST')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'MST_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_ξηζ_mst_plot(self, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the mean branch length (both empirical and robust) and median absolute
        deviation of the ξηζ minimum spanning tree (MST).
        """

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot association size metrics
        self.plot_metric(ax, self.mst_ξηζ_mean, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, self.mst_ξηζ_mean_robust, 0, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, self.mst_ξηζ_mad, 0, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax, title, '$ξ^\prime η^\prime ζ^\prime$ MST')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)

        # Save figure
        self.save_figure(
            self.name, f'MST_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_mahalanobis_plot(self, title=False, forced=False, default=False, cancel=False):
        """Creates a plot of the xyz and ξηζ Mahalanobis distance mean and median."""

        # Initialize figure
        fig, ax = self.initialize_figure_metric()

        # Plot total xyz MAD, x MAD, y and z MAD
        self.plot_metric(ax, self.mahalanobis_xyz_mean, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, self.mahalanobis_xyz_median, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, self.mahalanobis_ξηζ_mean, 0, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, self.mahalanobis_ξηζ_median, 0, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax, title, 'Mahalanobis distance')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax)
        ax.set_ylim(-0.1, 2.9)

        # Save figure
        self.save_figure(
            self.name, f'Mahalanobis_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_covariances_mad_ξηζ_plot(
        self, title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of ξ'-ξ', η'-η' and ζ'-ζ' covariances, and the determinant and the trace
        of the covariances matrix, and  the total ξηζ median absolute deviation (MAD), and ξ',
        η' and ζ' components of the MAD.
        """

        # Initialize figure
        self.check_traceback()
        fig = plt.figure(figsize=(3.345, 6.520), facecolor=colors.white, dpi=300)
        ax1 = fig.add_axes([0.100, 0.526, 0.899, 0.461])
        ax2 = fig.add_axes([0.100, 0.050, 0.899, 0.461])

        # Plot total ξ'η'ζ' covariances matrix determinant and trace,
        # and ξ'-ξ', η'-η' and ζ'-ζ' covariances
        self.plot_metric(ax1, self.covariances_ξηζ_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax1, self.covariances_ξηζ_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax1, self.covariances_ξηζ, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax1, self.covariances_ξηζ, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax1, self.covariances_ξηζ, 2, colors.metric[7], ':', 0.5)

        # Plot total ξ'η'ζ' MAD, ξ' MAD, η' MAD and ζ' MAD
        self.plot_metric(ax2, self.mad_ξηζ_total, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax2, self.mad_ξηζ, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax2, self.mad_ξηζ, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax2, self.mad_ξηζ, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax1, title, '$ξ^\prime η^\prime ζ^\prime$ covariances and MAD')

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax1, hide_x=True)
        self.set_axis_metric(ax2)

        # Save figure
        self.save_figure(
            self.name, f'covariances_MAD_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_det_mad_mst_cross_covariances_xyz_plots(
        self, other, title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of xyz determinant of the covariances matrix, xyz total median absolute
        deviation, mean branch length of the xyz minimum spanning tree, and x-u, y-v and z-w
        cross covariances.
        """

        # Check if 'other' is valid
        self.stop(
            type(other) != type(self), 'TypeError',
            "'other' must be a Series object ({} given).", type(other)
        )

        # Initialize figure
        self.check_traceback()
        other.check_traceback()
        fig, (ax0, ax1) = plt.subplots(
            ncols=2, constrained_layout=True, figsize=(6.66, 3.33), facecolor=colors.white, dpi=300)

        # Plot association size metrics (self)
        self.plot_metric(ax0, self.covariances_xyz_matrix_det, 0, colors.metric[4], '-', 0.8)
        self.plot_metric(ax0, self.mad_xyz_total, 0, colors.metric[5], '--', 0.7)
        self.plot_metric(ax0, self.mst_xyz_mean, 0, colors.metric[6], '-.', 0.6)
        self.plot_metric(ax0, self.mst_xyz_mad, 0, colors.metric[7], ':', 0.5)
        self.plot_metric(ax1, self.cross_covariances_xyz, 0, colors.metric[4], '-', 0.8)
        self.plot_metric(ax1, self.cross_covariances_xyz, 1, colors.metric[5], '--', 0.7)
        self.plot_metric(ax1, self.cross_covariances_xyz, 2, colors.metric[6], ':', 0.6)

        # Plot association size metrics (other)
        other.plot_metric(ax0, other.covariances_xyz_matrix_det, 0, colors.metric[0], '-', 0.8)
        other.plot_metric(ax0, other.mad_xyz_total, 0, colors.metric[1], '--', 0.7)
        other.plot_metric(ax0, other.mst_xyz_mean, 0, colors.metric[2], '-.', 0.6)
        other.plot_metric(ax0, other.mst_xyz_mad, 0, colors.metric[3], ':', 0.5)
        other.plot_metric(ax1, other.cross_covariances_xyz, 0, colors.metric[0], '-', 0.8)
        other.plot_metric(ax1, other.cross_covariances_xyz, 1, colors.metric[1], '--', 0.7)
        other.plot_metric(ax1, other.cross_covariances_xyz, 2, colors.metric[2], ':', 0.6)

        # Set title
        self.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            if self.from_data:
                fig.suptitle(
                    '$XYZ$ covariances, MAD, MST and cross covariances of {}\n and {} over '
                    '{:.1f} Myr with a {:.1f} km/s radial velocity correction\n'.format(
                        self.name, other.name, self.duration.value,
                        self.rv_shift.to('km/s').value
                    ), fontsize=8
                )
            elif self.from_model:
                fig.suptitle(
                    'Average $XYZ$ covariances, MAD, MST and cross covariances of {}'
                    'simulated associations over {:.1f} Myr\n with kinematics similar to'
                    "{} and {}, and a {:.1f} km/s radial velocity bias\n".format(
                        self.number_of_groups, self.duration.value,
                        self.name, other.name, self.rv_shift.to('km/s').value
                    ), fontsize=8
                )

        # Set legend, limits, labels and axes
        self.set_axis_metric(ax0)
        self.set_axis_metric(ax1)

        # Save figure
        self.save_figure(
            self.name, f'covariances_MAD_MST_Cross_covariannces_xyz_{self.name}_{other.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_age_distribution(
        self, title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of the distribution of ages computed in a series, including the effects
        of measurement errors and the jackknife Monte Carlo.
        """

        # Initialize figure
        self.check_traceback()
        fig = plt.figure(figsize=(3.33, 3.33), facecolor=colors.white, dpi=300)
        ax = fig.add_subplot(111, frame_on=True)

        # Plot histogram
        ages = [group.covariances_xyz_matrix_det.age[0] for group in self]
        ax.hist(
            ages, bins=np.linspace(16, 24, 33), density=True,
            color=colors.azure[8], alpha=0.7, label='metric'
        )
        # bins=np.arange(21.975, 26.025, 0.05)

        # Set title
        self.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            ax.set_title(
                f'Distribution of {self.number_of_groups} moving groups age,\n'
                f'Average age: ({self.covariances_xyz_matrix_det.age[0]:.2f} '
                f'± {self.covariances_xyz_matrix_det.age_error[0]:.2f}) Myr\n', fontsize=8
            )

        # Set labels
        ax.set_xlabel('Age (Myr)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

        # Set ticks
        ax.tick_params(top=True, right=True, which='both', direction='in', width=0.5, labelsize=8)

        # Set spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Save figure
        self.save_figure(
            self.name, f'age_distribution_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

class Output_Group():
    """Output methods for a group of stars."""

    def create_time_kinematics_table(
        self, save=False, show=False, machine=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a table of the group average kinematics over time. If 'save' if True, the table
        is saved and if 'show' is True, the table is displayed. If 'machine' is True, then a
        machine-readable table, without units in the header and '.csv' extension instead of a
        '.txt', is created. The machine-readable table also has an additional column 'status'
        to indicate whether a metric is valid or rejected, whereas the non-machine-readable
        table uses side heads.
        """

        # Check save, show and machine
        self.series.stop(
            type(save) != bool, 'TypeError',
            "'save' must be a boolean ({} given).", type(save)
        )
        self.series.stop(
            type(show) != bool, 'TypeError',
            "'show' must be a boolean ({} given).", type(show)
        )
        self.series.stop(
            type(machine) != bool, 'TypeError',
            "'machine' must be a boolean ({} given).", type(machine)
        )

        # Create header
        if machine:
            lines = [
                'Time,X,Y,Z,U,V,W,xi,eta,zeta,v_xi,v_eta,v_zeta'
            ]

            # Create lines
            for t in np.arange(self.series.time.size):
                lines.append(
                    (
                        f'{self.series.time[t]},'
                        f'{self.position_xyz[t,0]},'
                        f'{self.position_xyz[t,1]},'
                        f'{self.position_xyz[t,2]},'
                        f'{self.velocity_xyz[t,0]},'
                        f'{self.velocity_xyz[t,1]},'
                        f'{self.velocity_xyz[t,2]},'
                        f'{self.position_ξηζ[t,0]},'
                        f'{self.position_ξηζ[t,1]},'
                        f'{self.position_ξηζ[t,2]},'
                        f'{self.velocity_ξηζ[t,0]},'
                        f'{self.velocity_ξηζ[t,1]},'
                        f'{self.velocity_ξηζ[t,2]}'
                    )
                )

        # Create header
        else:
            lines = [
                f"{'':-<152}",
                f"{'Time':<8}{'X':>12}{'Y':>12}{'Z':>12}{'U':>12}{'V':>12}{'W':>12}"
                f"{'ξ':>12}{'η':>12}{'ζ':>12}{'vξ':>12}{'vη':>12}{'vζ':>12}",
                f"{'[Myr]':<8}{'[pc]':>12}{'[pc]':>12}{'[pc]':>12}"
                f"{'[pc/Myr]':>12}{'[pc/Myr]':>12}{'[pc/Myr]':>12}"
                f"{'[pc]':>12}{'[pc]':>12}{'[pc]':>12}"
                f"{'[pc/Myr]':>12}{'[pc/Myr]':>12}{'[pc/Myr]':>12}",
                f"{'':-<152}"
            ]

            # Create lines
            for t in np.arange(self.series.time.size):
                lines.append(
                    (
                        f'{self.series.time[t]:<8.1f}'
                        f'{self.position_xyz[t,0]:>12.2f}'
                        f'{self.position_xyz[t,1]:>12.2f}'
                        f'{self.position_xyz[t,2]:>12.2f}'
                        f'{self.velocity_xyz[t,0]:>12.2f}'
                        f'{self.velocity_xyz[t,1]:>12.2f}'
                        f'{self.velocity_xyz[t,2]:>12.2f}'
                        f'{self.position_ξηζ[t,0]:>12.2f}'
                        f'{self.position_ξηζ[t,1]:>12.2f}'
                        f'{self.position_ξηζ[t,2]:>12.2f}'
                        f'{self.velocity_ξηζ[t,0]:>12.2f}'
                        f'{self.velocity_ξηζ[t,1]:>12.2f}'
                        f'{self.velocity_ξηζ[t,2]:>12.2f}'
                    )
                )

            # Create footer
            lines.append(f"{'':-<152}")

        # Show table
        if show:
            for line in lines:
                print(line)

        # Save table
        if save:
            self.series.save_table(
                self.name, lines, file_path=f'kinematics_time_{self.name}',
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

    def create_kinematics_table(
        self, save=False, show=False, machine=False, age=None,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a table of the 6D kinematics (XYZ Galactic positions and UVW space velocities)
        at the given age of all members in the group. If 'save' if True, the table is
        saved and if 'show' is True, the table is displayed. If 'machine' is True, then a
        machine-readable table, with separate columns for values and errors, no units in the
        header and '.csv' extension instead of a '.txt', is created.
        """

        # Retrieve the epoch index
        epoch_index = self.get_epoch(age=age)[0]

        # Retrieve xyz positions and uvw velocities and convert units
        def get_position_velocity_xyz(star):
            position_xyz = Quantity(
                star.position_xyz[epoch_index], 'pc', star.position_xyz_error
            )
            velocity_xyz = Quantity(
                star.velocity_xyz[epoch_index], 'pc/Myr', star.velocity_xyz_error
            ).to('km/s')

            return position_xyz, velocity_xyz

        # Check save, show and machine
        self.series.stop(
            type(save) != bool, 'TypeError',
            "'save' must be a boolean ({} given).", type(save)
        )
        self.series.stop(
            type(show) != bool, 'TypeError',
            "'show' must be a boolean ({} given).", type(show)
        )
        self.series.stop(
            type(machine) != bool, 'TypeError',
            "'machine' must be a boolean ({} given).", type(machine)
        )

        # Create header
        if machine:
            lines = ['Designation,X,eX,Y,eY,Z,eZ,U,eU,V,eV,W,eW']

            # Create lines
            for star in self:
                position_xyz, velocity_xyz = get_position_velocity_xyz(star)
                lines.append(
                    ','.join(
                        [star.name] + [
                            str(float(i)) for i in [
                                position_xyz.values[0], position_xyz.errors[0],
                                position_xyz.values[1], position_xyz.errors[1],
                                position_xyz.values[2], position_xyz.errors[2],
                                velocity_xyz.values[0], velocity_xyz.errors[0],
                                velocity_xyz.values[1], velocity_xyz.errors[1],
                                velocity_xyz.values[2], velocity_xyz.errors[2]
                            ]
                        ]
                    )
                )

        # Create header
        else:
            lines = [
                f"{'':-<155}",
                f"{'Designation':<35}{'X':>20}{'Y':>20}{'Z':>20}{'U':>20}{'V':>20}{'W':>20}",
                f"{'[pc]':>55}{'[pc]':>20}{'[pc]':>20}{'[km/s]':>20}{'[km/s]':>20}{'[km/s]':>20}",
                f"{'':-<155}"
            ]

            # Create lines
            for star in self:
                position_xyz, velocity_xyz = get_position_velocity_xyz(star)
                x = f'{position_xyz.values[0]:.2f} ± {position_xyz.errors[0]:.2f}'
                y = f'{position_xyz.values[1]:.2f} ± {position_xyz.errors[1]:.2f}'
                z = f'{position_xyz.values[2]:.2f} ± {position_xyz.errors[2]:.2f}'
                u = f'{velocity_xyz.values[0]:.2f} ± {velocity_xyz.errors[0]:.2f}'
                v = f'{velocity_xyz.values[1]:.2f} ± {velocity_xyz.errors[1]:.2f}'
                w = f'{velocity_xyz.values[2]:.2f} ± {velocity_xyz.errors[2]:.2f}'
                lines.append(f'{star.name:<35}{x:>20}{y:>20}{z:>20}{u:>20}{v:>20}{w:>20}')

            # Creater footer
            lines.append(f"{'':-<155}")

        # Show table
        if show:
            for line in lines:
                print(line)

        # Save table
        if save:
            self.series.save_table(
                self.name, lines, file_path=f'kinematics_{self.name}_{age}Myr',
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

    def get_epoch(self, age=None, metric=None, index=None):
        """
        Computes the time index of the epoch for a given association age or, association size
        metric and dimensional index. Return the birth index, age and age error.
        """

        # Index from age
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer, float or None ({} given).", type(age)
            )
            self.series.stop(
                age < np.min(self.series.time), 'ValueError',
                "'age' must be younger the oldest time ({} Myr given, earliest time: {} Myr).",
                age, np.min(self.series.time)
            )
            self.series.stop(
                age > np.max(self.series.time), 'ValueError',
                "'age' must be older the latest time ({} Myr given, latest time: {} Myr).",
                age, np.max(self.series.time)
            )
            return np.argmin(np.abs(age - self.series.time)), age, None

        # Index from the epoch of minimum of an association size metric
        elif metric is not None:
            metric, index = self.get_metric(metric, index)
            if metric.status:
                age = metric.age[index]
                return np.argmin(np.abs(age - self.series.time)), age, metric.age_error[index]

            else:
                self.series.log(
                    "Could not use '{}' metric for '{}' group. It was not computed.",
                    str(metric.name[index]), self.name, display=True
                )
                return None, None, None

        # No birth index, age or age error
        else:
            return (None, None, None)

    def get_metric(self, metric, index=None):
        """Retrieves the proprer Series.Metric instance from a string and index."""

        # Metric instance
        self.series.stop(
            type(metric) != str, 'TypeError',
            "'metric' must be a string or None ({} given).", type(metric)
        )
        self.series.stop(
            metric not in [metric.label for metric in self.series.metrics],
            'ValueError', "'metric' must be a valid metric key ({} given).", metric
        )
        metric = vars(self.series)[metric]

        # Metric index
        if index is not None:
            self.series.stop(
                type(index) != int, 'TypeError',
                "'index' must be an integer or None ({} given).", type(index)
            )
            self.series.stop(
                index > metric.value.size - 1, 'ValueError',
                "'index' is too large for this metric ({} given, {} in size).",
                index, metric.value.size
            )
        else:
            self.series.stop(
                metric.value.size > 1, 'ValueError',
                "No 'index' is provided (metric is {} in size).", metric.value.size
            )

        return metric, index if metric.value.size > 1 else 0

    def trajectory_xyz(
        self, title=False, labels=False, age=None, metric=None, index=None,
        forced=False, default=False, cancel=False
    ):
        """Draw the xyz trajectories of stars in the group."""

        # Initialize figure
        fig = plt.figure(figsize=(3.345, 3.315), facecolor=colors.white, dpi=300)
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
            "'labels' must be a boolean ({} given).", type(labels)
        )

        # Birth index, age and age error
        birth_index, age, age_error = self.get_epoch(age=age, metric=metric, index=index)

        # Select conversion factors from pc to kpc
        for ax, x, y in ((ax1, 1, 0), (ax2, 2, 0), (ax3, 1, 2)):
            factor_x = 1.0 if x == 2 else 1000.0
            factor_y = 1.0 if y == 2 else 1000.0

            # Plot stars' trajectories
            for star in self:
                color = colors.red[6] if star.outlier else colors.black
                ax.plot(
                    star.position_xyz.T[x] / factor_x,
                    star.position_xyz.T[y] / factor_y,
                    color=color, alpha=0.6, linewidth=0.5,
                    solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        np.array([star.position_xyz[0,x]]) / factor_x,
                        np.array([star.position_xyz[0,y]]) / factor_y,
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth positions
                    if birth_index is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax.scatter(
                            np.array([star.position_xyz[birth_index,x]]) / factor_x,
                            np.array([star.position_xyz[birth_index,y]]) / factor_y,
                            color=color + (0.4,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                    # Show stars' names
                    if labels:
                        ax.text(
                            np.array(star.position_xyz.T[0,x]) / factor_x,
                            np.array(star.position_xyz.T[0,y]) / factor_y,
                            star.name, horizontalalignment='left', fontsize=6
                        )

            # Plot the average model star's trajectory
            if self.series.from_model:
                ax.plot(
                    self.average_model_star.position_xyz.T[x] / factor_x,
                    self.average_model_star.position_xyz.T[y] / factor_y,
                    color=colors.green[6], alpha=0.8,
                    linewidth=0.5, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current positions
                for t, size, marker in ((-1, 12, '*'), (0, 6, 'o')):
                    ax.scatter(
                        np.array([self.average_model_star.position_xyz[t,x]]) / factor_x,
                        np.array([self.average_model_star.position_xyz[t,y]]) / factor_y,
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Plot model stars' trajectories
                for star in self.model_stars:
                    ax.plot(
                        star.position_xyz.T[x] / factor_x,
                        star.position_xyz.T[y] / factor_y,
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, size, marker in ((0, 12, '*'), (-1, 6, 'o')):
                        ax.scatter(
                            np.array([star.position_xyz[t,x]]) / factor_x,
                            np.array([star.position_xyz[t,y]]) / factor_y,
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

            # Draw vertical and horizontal lines through the Sun's position at the current epoch
            ax.axhline(0., color=colors.black, alpha=0.8, linewidth=0.5, linestyle=':', zorder=0.0)
            ax.axvline(0., color=colors.black, alpha=0.8, linewidth=0.5, linestyle=':', zorder=0.0)

        # Draw circles around the galactic center located at 8.122 kpc from the Sun
        for radius in range(1, 16):
            ax1.add_artist(
                plt.Circle(
                    (0, 8.122), radius, color=colors.grey[17], fill=False,
                    linewidth=0.5, linestyle=':', zorder=0.0
                )
            )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
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
                direction='in', width=0.5, labelsize=8
            )

        # Set spines
        for ax in (ax1, ax2, ax3):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        self.series.save_figure(
            self.name, f'trajectory_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def trajectory_ξηζ(
        self, title=False, labels=False, age=None, metric=None, index=None,
        forced=False, default=False, cancel=False
    ):
        """Draws the ξηζ trajectories of stars in the group."""

        # Initialize figure
        fig = plt.figure(figsize=(3.345, 3.315), facecolor=colors.white, dpi=300)
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
            "'labels' must be a boolean ({} given).", type(labels)
        )

        # Birth index, age and age error
        birth_index, age, age_error = self.get_epoch(age=age, metric=metric, index=index)

        # Plot stars' trajectories
        for ax, x, y in ((ax1, 0, 1), (ax2, 2, 1), (ax3, 0, 2)):
            for star in self:
                color = colors.red[6] if star.outlier else colors.black
                ax.plot(
                    star.position_ξηζ.T[x], star.position_ξηζ.T[y],
                    color=color, alpha=0.6, linewidth=0.5,
                    solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        np.array([star.position_ξηζ[0,x]]),
                        np.array([star.position_ξηζ[0,y]]),
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth positions
                    if birth_index is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        color = colors.green[6] if star.name == 'HR 8799' else colors.blue[6]
                        ax.scatter(
                            np.array([star.position_ξηζ[birth_index,x]]),
                            np.array([star.position_ξηζ[birth_index,y]]),
                            color=color + (0.4,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                    # Show stars' names
                    if labels:
                        ax.text(
                            np.array(star.position_ξηζ.T[0,x]),
                            np.array(star.position_ξηζ.T[0,y]),
                            star.name, horizontalalignment='left', fontsize=6
                        )

            # Plot the average model star's trajectory
            if self.series.from_model:
                ax.plot(
                    self.average_model_star.position_ξηζ.T[x],
                    self.average_model_star.position_ξηζ.T[y],
                    color=colors.green[6], alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current positions
                for t, size, marker in ((-1, 12, '*'), (0, 6, 'o')):
                    ax.scatter(
                        np.array([self.average_model_star.position_ξηζ[t,x]]),
                        np.array([self.average_model_star.position_ξηζ[t,y]]),
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Plot model stars' trajectories
                for star in self.model_stars:
                    ax.plot(
                        star.position_ξηζ.T[x], star.position_ξηζ.T[y],
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, size, marker in ((0, 12, '*'), (-1, 6, 'o')):
                        ax.scatter(
                            np.array([star.position_ξηζ[t,x]]),
                            np.array([star.position_ξηζ[t,y]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            fig.suptitle(
                f'$ξ^\prime η^\prime ζ^\prime$ trajectories of stars in {self.name}',
                y=1.05, fontsize=8
            )

        # Set labels
        ax1.set_ylabel('$η^\prime$ (pc)', fontsize=8)
        ax2.set_xlabel('$ζ^\prime$ (pc)', fontsize=8)
        ax3.set_xlabel('$ξ^\prime$ (pc)', fontsize=8)
        ax3.set_ylabel('$ζ^\prime$ (pc)', fontsize=8)

        # Set limits
        # ax1.set_xlim(-225, 60)
        # ax1.set_ylim(-45, 110)
        # ax2.set_xlim(-40, 49)
        # ax2.set_ylim(-45, 110)
        # ax3.set_xlim(-225, 60)
        # ax3.set_ylim(-40, 49)

        # Set ticks
        ax1.set_xticklabels([])
        ax2.set_yticklabels([])
        for ax in (ax1, ax2, ax3):
            ax.tick_params(
                top=True, right=True, which='both',
                direction='in', width=0.5, labelsize=8
            )

        # Set spines
        for ax in (ax1, ax2, ax3):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        self.series.save_figure(
            self.name, f'trajectory_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def trajectory_time_xyz(
        self, style, title=False, age=None, metric=None,
        forced=False, default=False, cancel=False
    ):
        """Draws the xyz trajectories as a function of time of stars."""

        # Check style
        self.series.stop(
            type(style) != str, 'TypeError',
            "'style' must be a string ({} given).", type(style)
        )
        self.series.stop(
            style not in ('2x2', '1x3'), 'ValueError',
            "'style' must be either '2x2' or '1x3' ({} given).", style
        )

        # Initialize a 2x2 figure
        if style == '2x2':
            fig = plt.figure(facecolor=colors.white, figsize=(7.090, 6.317), dpi=300)
            left1, bottom1 = (0.083, 0.528)
            left2, bottom2 = (0.507, 0.052)
            width, height = (0.410, 0.460)
            ax0 = fig.add_axes([left1, bottom1, width, height])
            ax1 = fig.add_axes([left2, bottom1, width, height])
            ax2 = fig.add_axes([left1, bottom2, width, height])
            ax3 = fig.add_axes([left2, bottom2, width, height])

        # Initialize a 1x3 figure
        if style == '1x3':
            fig = plt.figure(facecolor=colors.white, figsize=(3.345, 9.134), dpi=300)
            left, bottom = (0.133, 0.0355)
            width, height = (0.866, 0.3112)
            inside = 0.01095
            ax1 = fig.add_axes([left, bottom + 2 * (height + inside), width, height])
            ax2 = fig.add_axes([left, bottom + height + inside, width, height])
            ax3 = fig.add_axes([left, bottom, width, height])

        # Birth index, age and age error
        birth_index, age, age_error  = tuple(
            zip(*[self.get_epoch(age=age, metric=metric, index=index) for index in range(3)])
        )

        # Plot stars' trajectories
        for ax, y in ((ax1, 0), (ax2, 1), (ax3, 2)):
            for star in self:
                ax.plot(
                    self.series.time, (star.position_xyz - self.position_xyz)[:,y],
                    color = colors.red[6] if star.outlier else colors.black, alpha=0.6,
                    linewidth=0.5, solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        self.series.time[0], (star.position_xyz - self.position_xyz)[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black, alpha=None,
                        s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth positions
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax.scatter(
                            self.series.time[birth_index[y]],
                            (star.position_xyz - self.position_xyz)[birth_index[y],y],
                            color=color + (0.4,), edgecolors=color, alpha=None,
                            s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

            # Show vectical dashed line
            if birth_index[y] is not None:
                ax.axvline(
                    x=self.series.time[birth_index[y]], color=colors.black,
                    linewidth=0.5, linestyle='--', zorder=0.1
                )

                # Show a grey shaded area
                if age_error[y] is not None:
                    ax.fill_between(
                        np.array([age[y] - age_error[y], age[y] + age_error[y]]), 0, 1,
                        transform=ax.get_xaxis_transform(), color=colors.grey[9],
                        alpha=0.1, linewidth=0.0, zorder=0.1
                    )

            # Plot the average model star's trajectory
            if self.series.from_model:
                position_xyz = np.mean(
                    [star.position_xyz[:,y] for star in self.model_stars], axis=0
                )
                average_model_star_position_xyz = (
                    self.average_model_star.position_xyz[:,y] - position_xyz[::-1]
                )
                ax.plot(
                    self.series.model_time,
                    average_model_star_position_xyz,
                    color=colors.green[6], alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current positions
                for t, x, size, marker in ((-1, -1, 10, '*'), (0, 0, 6, 'o')):
                    ax.scatter(
                        np.array([self.series.model_time[t]]),
                        np.array([average_model_star_position_xyz[x]]),
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Plot model stars' trajectories
                for star in self.model_stars:
                    model_star_position_xyz = star.position_xyz[:,y] - position_xyz
                    ax.plot(
                        self.series.model_time[::-1], model_star_position_xyz,
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 10, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_position_xyz[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

        # Plot stars' average trajectory
        if style == '2x2':
            for y, label, linestyle in ((0,'$<X>$', '-'), (1,'$<Y>$', '--'), (2, '$<Z>$', ':')):
                ax0.plot(
                    self.series.time, self.position_xyz[:,y] / 1000,
                    label=label, color=colors.black, alpha=0.8, linestyle=linestyle,
                    linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.1
                )

                # Plot stars' average current positions
                if self.series.from_data:
                    ax0.scatter(
                        self.series.time[0], self.position_xyz[0,y] / 1000,
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' average birth positions
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax0.scatter(
                            self.series.time[birth_index[y]],
                            self.position_xyz[birth_index[y],y] / 1000,
                            color=color + (0.4,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                # Plot the average model star's trajectory
                if self.series.from_model:
                    average_model_star_position_xyz = self.average_model_star.position_xyz[:,y] / 1000
                    ax0.plot(
                        self.series.model_time,
                        average_model_star_position_xyz,
                        color=colors.green[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.3
                    )

                    # Plot the average model star's birth and current positions
                    for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                        ax0.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([average_model_star_position_xyz[x]]),
                            color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                        )

                    # Plot model stars' trajectories
                    model_star_position_xyz = np.mean(
                        [star.position_xyz[:,y] for star in self.model_stars], axis=0
                    ) / 1000
                    ax0.plot(
                        self.series.model_time[::-1], model_star_position_xyz,
                        color=colors.blue[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax0.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_position_xyz[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            if style == '2x2':
                fig.suptitle(
                    f'$XYZ$ trajectories of stars in {self.name} as a function of time',
                    y=1.025, fontsize=8
                )
            if style == '1x3':
                fig.suptitle(
                    f'$XYZ$ trajectories of stars in {self.name}\nas a function of time',
                    y=1.03, fontsize=8
                )

        # Set legend
        if style == '2x2':
            legend = ax0.legend(loc=4, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor(colors.white + (0.8,))
            legend.get_frame().set_edgecolor(colors.black)
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
            ax.set_xlim(np.min(self.series.time), np.max(self.series.time) + 1)

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
                direction='in', width=0.5, labelsize=8
            )

        # Set spines
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        self.series.save_figure(
            self.name, f"trajectory_time_xyz_{style}_{self.name}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def trajectory_time_ξηζ(
        self, style, title=False, age=None, metric=None,
        forced=False, default=False, cancel=False
    ):
        """Draws the ξηζ trajectories as a function of time of stars."""

        # Check style
        self.series.stop(
            type(style) != str, 'TypeError',
            "'style' must be a string ({} given).", type(style)
        )
        self.series.stop(
            style not in ('2x2', '1x3'), 'ValueError',
            "'style' must be either '2x2' or '1x3' ({} given).", style
        )

        # Initialize a 2x2 figure
        if style == '2x2':
            fig = plt.figure(facecolor=colors.white, figsize=(7.090, 6.317), dpi=300)
            left1, bottom1 = (0.083, 0.528)
            left2, bottom2 = (0.507, 0.052)
            width, height = (0.410, 0.460)
            ax0 = fig.add_axes([left1, bottom1, width, height])
            ax1 = fig.add_axes([left2, bottom1, width, height])
            ax2 = fig.add_axes([left1, bottom2, width, height])
            ax3 = fig.add_axes([left2, bottom2, width, height])

        # Initialize a 1x3 figure
        if style == '1x3':
            # fig = plt.figure(facecolor=colors.white, figsize=(3.345, 9.134), dpi=300)
            # left, bottom = (0.133, 0.0355)
            # width, height = (0.866, 0.3112)
            # inside = 0.01095

            fig = plt.figure(facecolor=colors.white, figsize=(3.345, 7.5), dpi=300)
            left, bottom = (0.133, 0.0432)
            width, height = (0.866, 0.3064)
            inside = 0.0133

            ax1 = fig.add_axes([left, bottom + 2 * (height + inside), width, height])
            ax2 = fig.add_axes([left, bottom + height + inside, width, height])
            ax3 = fig.add_axes([left, bottom, width, height])

        # Birth index, age and age error
        birth_index, age, age_error  = tuple(
            zip(*[self.get_epoch(age=age, metric=metric, index=index) for index in range(3)])
        )

        # Plot stars' trajectories
        for ax, y in ((ax1, 0), (ax2, 1), (ax3, 2)):
            for star in self:
                ax.plot(
                    self.series.time, (star.position_ξηζ - self.position_ξηζ)[:,y],
                    color = colors.red[6] if star.outlier else colors.black,
                    alpha=0.6, linewidth=0.5, solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        self.series.time[0], (star.position_ξηζ - self.position_ξηζ)[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth positions
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax.scatter(
                            self.series.time[birth_index[y]],
                            (star.position_ξηζ - self.position_ξηζ)[birth_index[y],y],
                            color=color + (0.4,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

            # Show vectical dashed line
            if birth_index[y] is not None:
                ax.axvline(
                    x=self.series.time[birth_index[y]], color=colors.black,
                    linewidth=0.5, linestyle='--', zorder=0.1
                )

                # Show a grey shaded area
                if age_error[y] is not None:
                    ax.fill_between(
                        np.array([age[y] - age_error[y], age[y] + age_error[y]]), 0, 1,
                        transform=ax.get_xaxis_transform(), color=colors.grey[9],
                        alpha=0.1, linewidth=0.0, zorder=0.1
                    )

            # Plot the average model star's trajectory
            if self.series.from_model:
                position_ξηζ = np.mean(
                    [star.position_ξηζ[:,y] for star in self.model_stars], axis=0
                )
                average_model_star_position_ξηζ = (
                    self.average_model_star.position_ξηζ[:,y] - position_ξηζ[::-1]
                )
                ax.plot(
                    self.series.model_time,
                    average_model_star_position_ξηζ,
                    color=colors.green[6], alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current positions
                for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                    ax.scatter(
                        np.array([self.series.model_time[t]]),
                        np.array([average_model_star_position_ξηζ[x]]),
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Plot model stars' trajectories
                for star in self.model_stars:
                    model_star_position_ξηζ = star.position_ξηζ[:,y] - position_ξηζ
                    ax.plot(
                        self.series.model_time[::-1], model_star_position_ξηζ,
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_position_ξηζ[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

        # Plot stars' average trajectories
        if style == '2x2':
            for y, label, linestyle in (
                (0, '$<ξ^\prime>$', '-'), (1, '$<η^\prime>$', '--'), (2, '$<ζ^\prime>$', ':')
            ):
                ax0.plot(
                    self.series.time, self.position_ξηζ[:,y],
                    label=label, color=colors.black, alpha=0.8, linestyle=linestyle,
                    linewidth=1.0, solid_capstyle='round', zorder=0.1
                )

                # Plot stars' average current positions
                if self.series.from_data:
                    ax0.scatter(
                        self.series.time[0], self.position_ξηζ[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' average birth positions
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax0.scatter(
                            self.series.time[birth_index[y]],
                            self.position_ξηζ[birth_index[y],y],
                            color=color + (0.4,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                # Plot the average model star's trajectory
                if self.series.from_model:
                    average_model_star_position_ξηζ = self.average_model_star.position_ξηζ[:,y]
                    ax0.plot(
                        self.series.model_time,
                        average_model_star_position_ξηζ,
                        color=colors.green[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', zorder=0.3
                    )

                    # Plot the average model star's birth and current positions
                    for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                        ax0.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([average_model_star_position_ξηζ[x]]),
                            color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                        )

                    # Plot model stars' trajectories
                    model_star_position_ξηζ = np.mean(
                        [star.position_ξηζ[:,y] for star in self.model_stars], axis=0
                    )
                    ax0.plot(
                        self.series.model_time[::-1], model_star_position_ξηζ,
                        color=colors.blue[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax0.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_position_ξηζ[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            if style == '2x2':
                fig.suptitle(
                    f'$ξ^\prime η^\prime ζ^\prime$ trajectories of stars in {self.name} '
                    'as a function of time', y=1.025, fontsize=8
                )
            if style == '1x3':
                fig.suptitle(
                    f'$ξ^\prime η^\prime ζ^\prime$ trajectories of stars in {self.name}\n'
                    'as a function of time', y=1.03, fontsize=8
                )

        # Set legend
        if style == '2x2':
            legend = ax0.legend(loc=4, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor(colors.white + (0.8,))
            legend.get_frame().set_edgecolor(colors.black)
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
            ax.set_xlim(np.min(self.series.time), np.max(self.series.time) + 1)
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
                direction='in', width=0.5, labelsize=8
            )

        # Set spines
        for ax in (ax1, ax2, ax3) + ((ax0,) if style == '2x2' else ()):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # Save figure
        self.series.save_figure(
            self.name, f'trajectory_time_ξηζ_{style}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_map(self, labels=False, title=False, forced=False, default=False, cancel=False):
        """
        Creates a Mollweide projection of a traceback. For this function to work, uvw
        velocities must not compensated for the sun velocity and computing xyz positions.
        """

        # Initialize figure
        fig = plt.figure(figsize=(6.66, 3.33), facecolor=colors.white, dpi=300)
        ax = fig.add_subplot(111, projection="mollweide")

        # Compute coordinates
        from .coordinate import galactic_xyz_equatorial_rδα
        positions = np.array(
            [
                [
                    galactic_xyz_equatorial_rδα(*star.position_xyz[step])[0]
                    for step in range(self.series.number_of_steps)
                ] for star in self
            ]
        )
        alphas = np.vectorize(lambda α: α - (2 * np.pi if α > np.pi else 0.0))(positions[:,:,2])
        deltas = positions[:,:,1]

        # Plot trajectories
        for star in range(len(self)):
            color = colors.blue[6] if not self[star].outlier else colors.red[6]

            # Identify discontinuties
            discontinuties = (
                np.abs(alphas[star, 1:] - alphas[star, :-1]) > 3 * np.pi / 2
            ).nonzero()[0] + 1

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
                    horizontalalignment='left', fontsize=6, zorder=0.2
                )

        # Plot current-day positions
        ax.scatter(alphas[:,0], deltas[:,0], marker='.', color=colors.black, zorder=0.3)

        # Show proper motion arrows
        for star in self.series.data:
            ax.arrow(
                star.position.values[2] - (2 * np.pi if star.position.values[2] > np.pi else 0.0),
                star.position.values[1], -star.velocity.values[2]/4, -star.velocity.values[1]/4,
                head_width=0.03, head_length=0.03, color=colors.black, zorder=0.4
            )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            ax.set_title('Mollweide projection of tracebacks', fontsize=8)

        # Format axis
        ax.grid(zorder=1)

        # Save figure
        self.series.save_figure(
            self.name, f'Mollweide_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_2D_scatter(
        self, i, j, step=None, age=None, errors=False, labels=False, mst=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a 2D scatter plot of star positions in i and j at a given 'step' or 'age' in
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
            "X axis 'i' must be a string ({} given).", type(i)
        )
        self.series.stop(
            i.lower() not in keys, 'ValueError',
            "X axis 'i' must be an axis key ('x', 'y' or 'z', {} given).", i
        )
        i = axis[i.lower()]
        self.series.stop(
            type(j) != str, 'TypeError',
            "Y axis 'j' must be a string ({} given).", type(j)
        )
        self.series.stop(
            j.lower() not in keys, 'ValueError',
            "Y axis 'j' must be an axis key ('x', 'y' or 'z', {} given).", j
        )
        j = axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step)
            )
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step
            )
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step
            )
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step
            )
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age)
            )
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age
            )
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age
            )

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Initialize figure
        fig = plt.figure(figsize=(3.33, 3.33), facecolor=colors.white, dpi=300)
        ax = fig.add_subplot(111)

        # Plot xyz positions
        ax.scatter(
            [star.position_xyz[step, i] for star in self.sample],
            [star.position_xyz[step, j] for star in self.sample],
            marker='o', color=colors.black
        )

        # Plot error bars
        self.series.stop(
            type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors)
        )
        if errors:
            for star in self.sample:
                position = star.position_xyz[step]
                error = star.position_xyz_error[step]

                # Horizontal error bars
                ax.plot(
                    (position[i] - error[i], position[i] + error[i]),
                    (position[j], position[j]),
                    color=colors.grey[1], linewidth=0.5
                )

                # Vertical error bars
                ax.plot(
                    (position[i], position[i]),
                    (position[j] - error[j], position[j] + error[j]),
                    color=colors.grey[1], linewidth=0.5
                )

        # Show star labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels)
        )
        if labels:
            for star in self.sample:
                ax.text(
                    star.position_xyz[step, i] + 1, star.position_xyz[step, j] + 1,
                    star.name, horizontalalignment='left', fontsize=6
                )

        # Create branches
        self.series.stop(
            type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst)
        )
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (branch.start.position[step, i], branch.end.position[step, i]),
                    (branch.start.position[step, j], branch.end.position[step, j]),
                    color=colors.blue[6]
                )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            ax.set_title(
                "{} and {} positions of stars in β Pictoris at {} Myr "
                "wihtout outliers.\n".format(keys[i].upper(), keys[j].upper(), age), fontsize=8
            )

        # Set labels
        ax.set_xlabel(f'${keys[i].lower()}$ (pc)')
        ax.set_ylabel(f'${keys[j].lower()}$ (pc)')

        # Save figure
        self.series.save_figure(
            self.name,
            f'2D_Scatter_{self.name}_{keys[i].upper()}{keys[j].upper()}_at_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_3D_scatter(
        self, step=None, age=None, errors=False, labels=False, mst=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a 3D scatter plot of star positions in x, y and z at a given 'step' or 'age'
        in Myr. If 'age' doesn't match a step, the closest step is used instead. 'age'
        overrules 'step' if both are given. 'labels' adds the stars' name and 'mst' adds the
        minimum spanning tree branches.
        """

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step)
            )
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step
            )
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step
            )
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step
            )
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age)
            )
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age
            )
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age
            )

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(step * self.series.timestep.value, 2)

        # Initialize figure
        fig = plt.figure(figsize=(3.33, 3.33), facecolor=colors.white, dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Plot xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.sample],
            [star.relative_position_xyz[step, 1] for star in self.sample],
            [star.relative_position_xyz[step, 2] for star in self.sample],
            marker='o', color=colors.black)

        # Plot error bars
        self.series.stop(
            type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors)
        )
        if errors:
            for star in self.sample:
                position = star.relative_position_xyz[step]
                error = star.relative_position_xyz_error[step]

                # X axis error bars
                ax.plot(
                    (position[0] - error[0], position[0] + error[0]),
                    (position[1], position[1]), (position[2], position[2]),
                    color=colors.grey[1], linewidth=0.5
                )

                # Y axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1] - error[1], position[1] + error[1]),
                    (position[2], position[2]),
                    color=colors.grey[1], linewidth=0.5
                )

                # Z axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1], position[1]),
                    (position[2] - error[2], position[2] + error[2]),
                    color=colors.grey[1], linewidth=0.5
                )

        # Show star labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels)
        )
        if labels:
            for star in self.sample:
                ax.text(
                    star.position_xyz[step, 0] + 2, star.position_xyz[step, 1] + 2,
                    star.position_xyz[step, 2] + 2, star.name,
                    horizontalalignment='left', fontsize=6
                )

        # Create branches
        self.series.stop(
            type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst)
        )
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (
                        branch.start.relative_position_xyz[step, 0],
                        branch.end.relative_position_xyz[step, 0]
                    ), (
                        branch.start.relative_position_xyz[step, 1],
                        branch.end.relative_position_xyz[step, 1]
                    ), (
                        branch.start.relative_position_xyz[step, 2],
                        branch.end.relative_position_xyz[step, 2]
                    ), color=colors.blue[6]
                )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            ax.set_title(
                f'Minimum spanning tree of stars in β Pictoris at {age:.1f} Myr\n', fontsize=8)

        # Set labels
        ax.set_xlabel('\n $x$ (pc)')
        ax.set_ylabel('\n $y$ (pc)')
        ax.set_zlabel('\n $z$ (pc)')

        # Save figure
        self.series.save_figure(
            self.name, f'3D_Scatter_{self.name}_at_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_2D_and_3D_scatter(
        self, ages, title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a three 4-axis columns of xy, xz and yz 2D scatters, and a 3D scatter at three
        ages definied by a list or tuple.
        """

        # Check if ages is valid
        self.series.stop(
            type(ages) not in (tuple, list), 'TypeError',
            "'ages' must a tuple or a list ({} given).", type(ages))
        self.series.stop(
            len(ages) != 3, 'ValueError', "'ages' must be have a length of 3 ({} given).", ages)

        # Initialize figure
        self.fig = plt.figure(figsize=(9, 11.15), facecolor=colors.white, dpi=300)

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
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            self.fig.suptitle(
                '$X-Y$, $X-Z$, $Y-Z$ and 3D scatters at '
                f'{ages[0]:.1f}, {ages[1]:.1f} and {ages[2]:.1f}Myr.\n', fontsize=8
            )

        # Save figure
        self.series.save_figure(
            self.name, '2D_Scatter_{}_{}_{}_{}_Myr.pdf'.format(self.name, *ages),
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_2D_axis(self, i, j, step=None, age=None, index=1, left=0, bottom=0):
        """Creates a 2D axis."""

        # Initialize axis
        axis = {'x': 0, 'y': 1, 'z': 2}
        keys = tuple(axis.keys())

        # Check if X and Y axes are valid, and initialize axis
        self.series.stop(
            type(i) != str, 'TypeError',
            "X axis 'i' must be a string ({} given).", type(i)
        )
        self.series.stop(
            i.lower() not in keys, 'ValueError',
            "X axis 'i' must be an axis key ('x', 'y' or 'z', {} given).", i
        )
        i = axis[i.lower()]
        self.series.stop(
            type(j) != str, 'TypeError',
            "Y axis 'j' must be a string ({} given).", type(j)
        )
        self.series.stop(
            j.lower() not in keys, 'ValueError',
            "Y axis 'j' must be an axis key ('x', 'y' or 'z', {} given).", j
        )
        j = axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step)
            )
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step
            )
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step
            )
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step
            )
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age)
            )
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age
            )
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age
            )

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
            color=colors.black, s=8, marker='o'
        )

        # Plot outliers' xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, i] for star in self.outliers],
            [star.relative_position_xyz[step, j] for star in self.outliers],
            color=colors.red[6], s=8, marker='o'
        )

        # Error bars
        for star in self:
            position = star.relative_position_xyz[step]
            error = star.position_xyz_error[step]
            color = colors.black if not star.outlier else colors.red[6]

            # Horizontal error bars
            ax.plot(
                (position[i] - error[i], position[i] + error[i]),
                (position[j], position[j]),
                color=color, linewidth=0.5
            )

            # Vertical error bars
            ax.plot(
                (position[i], position[i]),
                (position[j] - error[j], position[j] + error[j]),
                color=color, linewidth=0.5
            )

        # Set labels
        ax.set_xlabel(f'${keys[i].upper()}$ (pc)')
        ax.set_ylabel(f'${keys[j].upper()}$ (pc)', labelpad=-12.)

        # Set limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

    def create_3D_axis(self, step=None, age=None, index=1, left=0, bottom=0):
        """Creates a 3D axis."""

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step)
            )
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step
            )
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step
            )
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step
            )
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age)
            )
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age
            )
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age
            )

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(step * self.series.timestep.value, 2)

        # Initialize figure
        ax = self.fig.add_subplot(
            4, 3, index, projection='3d', position=[left, bottom, 0.29, 0.215]
        )

        # Plot stars' xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.sample],
            [star.relative_position_xyz[step, 1] for star in self.sample],
            [star.relative_position_xyz[step, 2] for star in self.sample],
            color=colors.black, marker='o'
        )

        # Plot outliers' xyz relative positions
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.outliers],
            [star.relative_position_xyz[step, 1] for star in self.outliers],
            [star.relative_position_xyz[step, 2] for star in self.outliers],
            color=colors.red[6], marker='o'
        )

        # Create branches
        for branch in self.mst[step]:
            ax.plot(
                (
                    branch.start.relative_position_xyz[step, 0],
                    branch.end.relative_position_xyz[step, 0]
                ), (
                    branch.start.relative_position_xyz[step, 1],
                    branch.end.relative_position_xyz[step, 1]
                ), (
                    branch.start.relative_position_xyz[step, 2],
                    branch.end.relative_position_xyz[step, 2]
                ), colors.blue[6]
            )

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
        title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a cross covariance scatter of star positions and velocities in i and j at a
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
            "Position axis 'i' must be a string ({} given).", type(i)
        )
        self.series.stop(
            i.lower() not in position_keys, 'ValueError',
            "Position axis 'i' must be an postion axis key ('x', 'y' or 'z', {} given).", i
        )
        i = position_axis[i.lower()]
        self.series.stop(
            type(j) != str, 'TypeError',
            "Velocity axis 'j' must be a string ({} given).", type(j)
        )
        self.series.stop(
            j.lower() not in velocity_keys, 'ValueError',
            "Velocity axis 'j' must be an postion axis key ('u', 'v' or 'w', {} given).", j
        )
        j = velocity_axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(
                type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step)
            )
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step
            )
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0.0 ({} given).", step
            )
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({}, {} given).",
                self.series.number_of_steps, step
            )
        if age is not None:
            self.series.stop(
                type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age)
            )
            self.series.stop(
                age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", age
            )
            self.series.stop(
                age > self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({} Myr, {} Myr given).",
                self.series.final_time.value, age
            )

        # Compute step or age
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Initialize figure
        fig = plt.figure(figsize=(3.33, 3.33), facecolor=colors.white, dpi=300)
        ax = fig.add_subplot(111)

        # Plot xyz positions
        ax.scatter(
            [star.position_xyz[step, i] for star in self.sample],
            [star.velocity_xyz[step, j] for star in self.sample],
            marker='o', color=colors.black
        )

        # Plot error bars
        self.series.stop(
            type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors)
        )
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
                    color=colors.grey[1], linewidth=0.5
                )

                # Velocity (vertical) error bars
                ax.plot(
                    (position[i], position[i]),
                    (velocity[j] - velocity_error[j], velocity[j] + velocity_error[j]),
                    color=colors.grey[1], linewidth=0.5
                )

        # Show star labels
        self.series.stop(
            type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels)
        )
        if labels:
            for star in self.sample:
                ax.text(
                    star.position_xyz[step, i] + 1, star.velocity_xyz[step, j] + 1,
                    star.name, horizontalalignment='left', fontsize=6
                )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            ax.set_title(
                "{} and {} covariance of stars in β Pictoris at {} Myr wihtout "
                "outliers.\n".format(position_keys[i].upper(), velocity_keys[j].upper(), age),
                fontsize=8
            )

        # Set labels
        ax.set_xlabel(f'{position_keys[i].upper()} (pc)')
        ax.set_ylabel(f'{velocity_keys[j].upper()} (pc/Myr)')

        # Save figure
        self.series.save_figure(
            self.name, f'covariances_scatter_{self.name}_'
            f'{position_keys[i].upper()}-{velocity_keys[j].upper()}.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_age_distribution(
        self, title=False, metric=None, index=None,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of the distribution of jackknife Monte Carlo ages computed in a
        group.
        """

        # Initialize figure
        fig = plt.figure(figsize=(3.345, 3.401), facecolor=colors.white, dpi=300)
        ax = fig.add_axes([0.104, 0.096, 0.895, 0.880])

        # Retrieve ages
        metric, index = self.get_metric(metric, index)
        if metric.status:
            metric_name = metric.label
            ages = metric.ages
            if ages.ndim == 2:
                ages = ages[self.number]
            elif ages.ndim == 3:
                ages = ages[self.number,:,index]

            # Plot uncorrected histogram and gaussian curve
            if False:
                x = np.linspace(8, 36, 1000)
                μ = metric.age[index]
                σ = metric.age_int_error[index]
                gauss = np.exp(-0.5 * ((x - μ) / σ)**2) / np.sqrt(2 * np.pi) / σ
                i, = (gauss > 0.001).nonzero()
                ax.plot(
                    x[i], gauss[i], label='$\\xi^\\prime$ variance',
                    color=colors.cyan[6], alpha=1.0, linewidth=1.0, zorder=0.8
                )
                ax.hist(
                    ages, bins=np.linspace(12, 32, 81), density=True,
                    color=colors.cyan[6], alpha=0.15, zorder=0.8
                )
                ax.vlines(
                    μ, ymin=0.0, ymax=np.max(gauss), color=colors.cyan[6],
                    alpha=0.8, linewidth=0.5, linestyle='--', zorder=0.8
                )

            # Plot corrected histogram and gaussian curve
            x = np.linspace(8, 36, 1000)
            μ = metric.age_ajusted[index]
            μ = 20.4
            σ = 2.5
            gauss = np.exp(-0.5 * ((x - μ) / σ)**2) / np.sqrt(2 * np.pi) / σ
            i, = (gauss > 0.001).nonzero()
            ax.plot(
                x[i], gauss[i], label='Corrected $\\xi^\\prime$ variance',
                color=colors.lime[5], alpha=1.0, linewidth=1.0, zorder=0.9
            )
            ages = (ages - metric.age[index]) * (σ / metric.age_int_error[index]) + μ
            ax.hist(
                ages, bins=np.linspace(12, 32, 81), density=True,
                color=colors.lime[6], alpha=0.3, zorder=0.6
            )
            # ax.fill_between(
            #     x[i], np.zeros_like(x[i]), gauss[i], color=colors.lime[6],
            #     alpha=0.15, linewidth=0., zorder=0.6
            # )
            ax.vlines(
                μ, ymin=0.0, ymax=np.max(gauss), color=colors.lime[6],
                alpha=0.8, linewidth=0.5, linestyle='--', zorder=0.9
            )

        # Logging
        else:
            self.series.log(
                "Could not use '{}' metric for '{}' group. It was not computed.",
                str(metric.name[index]), self.name, display=True
            )

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
            x[i], gauss[i], label='Miret-Roig et al. (2020)',
            color=colors.orange[6], alpha=1.0, linewidth=1.0, zorder=0.8
        )
        ax.fill_between(
            x[i], np.zeros_like(x[i]), gauss[i], color=colors.orange[6],
            alpha=0.15, linewidth=0., zorder=0.5
        )
        ax.vlines(
            μ, ymin=0.0, ymax=np.max(gauss), color=colors.orange[6],
            alpha=0.8, linewidth=0.5, linestyle='--', zorder=0.8
        )

        # Plot gaussian curve from Crundall et al. (2019)
        μ = 17.7
        σ1, σ2 = 1.2, 1.2
        x1, x2 = np.arange(μ - 10, μ, 0.01), np.arange(μ, μ + 10, 0.01)
        gauss1 = np.exp(-0.5 * ((x1 - μ) / σ1)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        gauss2 = np.exp(-0.5 * ((x2 - μ) / σ2)**2) / np.sqrt(2 * np.pi) / np.mean((σ1, σ2))
        x = np.concatenate((x1, x2))
        gauss = np.concatenate((gauss1, gauss2))
        i, = (gauss > 0.001).nonzero()
        ax.plot(
            x[i], gauss[i], label='Crundall et al. (2019)',
            color=colors.azure[6], alpha=1.0, linewidth=1.0, zorder=0.7
        )
        ax.fill_between(
            x[i], np.zeros_like(x[i]), gauss[i], color=colors.azure[6],
            linewidth=0.0, alpha=0.15, zorder=0.4
        )
        ax.vlines(
            μ, ymin=0.0, ymax=np.max(gauss), color=colors.azure[6],
            alpha=0.8, linewidth=0.5, linestyle='--', zorder=0.7
        )

        # Show a shaded area for LDB and isochrone ages
        LDB_range = np.array([20, 26])
        ax.fill_between(
            LDB_range, 0, 1, transform=ax.get_xaxis_transform(),
            color=colors.grey[9], alpha=0.1, linewidth=0.0, zorder=0.1
        )

        # Set title
        self.series.stop(
            type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title)
        )
        if title:
            ax.set_title(
                f'Distribution of {self.series.number_of_iterations} jackknife Monte Carlo' + (
                    f',\nAverage age: ({metric.age[0]:.1f} '
                    f'± {metric.age_int_error[0]:.1F}) Myr\n'
                ) if metric.status else '', fontsize=8
            )

        # Set legend
        legend = ax.legend(loc=1, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor(colors.white + (0.8,))
        legend.get_frame().set_edgecolor(colors.black)
        legend.get_frame().set_linewidth(0.5)

        # Set labels
        ax.set_xlabel('Age (Myr)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

        # Set limits
        ax.set_xlim(13, 27)
        ax.set_ylim(0.0, 0.7)

        # Set ticks
        ax.set_xticks([14., 16., 18., 20., 22., 24., 26.])
        ax.tick_params(
            top=True, right=True, which='both',
            direction='in', width=0.5, labelsize=8
        )

        # Set spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Save figure
        self.series.save_figure(
            self.name, f'age_distribution_jackknife_{self.name}_{metric_name}.pdf',
            tight=False, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

class Output_Star():
    """Ouput methods for a star."""

    def create_time_kinematics_table(
        self, save=False, show=False, machine=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a table of the star 6D kinematics over time. If 'save' if True, the table is
        saved and if 'show' is True, the table is displayed. If 'machine' is True, then a
        machine-readable table, without units in the header and '.csv' extension instead of a
        '.txt', is created. The machine-readable table also has an additional column 'status'
        to indicate whether a metric is valid or rejected, whereas the non-machine-readable
        table uses side heads.
        """

        # Check save, show and machine
        self.group.series.stop(
            type(save) != bool, 'TypeError',
            "'save' must be a boolean ({} given).", type(save)
        )
        self.group.series.stop(
            type(show) != bool, 'TypeError',
            "'show' must be a boolean ({} given).", type(show)
        )
        self.group.series.stop(
            type(machine) != bool, 'TypeError',
            "'machine' must be a boolean ({} given).", type(machine)
        )

        # Create header
        if machine:
            lines = [
                'Time,X,Y,Z,U,V,W,xi,eta,zeta,v_xi,v_eta,v_zeta'
            ]

            # Create lines
            for t in np.arange(self.group.series.time.size):
                lines.append(
                    (
                        f'{self.group.series.time[t]},'
                        f'{self.position_xyz[t,0]},'
                        f'{self.position_xyz[t,1]},'
                        f'{self.position_xyz[t,2]},'
                        f'{self.velocity_xyz[t,0]},'
                        f'{self.velocity_xyz[t,1]},'
                        f'{self.velocity_xyz[t,2]},'
                        f'{self.position_ξηζ[t,0]},'
                        f'{self.position_ξηζ[t,1]},'
                        f'{self.position_ξηζ[t,2]},'
                        f'{self.velocity_ξηζ[t,0]},'
                        f'{self.velocity_ξηζ[t,1]},'
                        f'{self.velocity_ξηζ[t,2]}'
                    )
                )

        # Create header
        else:
            lines = [
                f"{'':-<152}",
                f"{'Time':<8}{'X':>12}{'Y':>12}{'Z':>12}{'U':>12}{'V':>12}{'W':>12}"
                f"{'ξ':>12}{'η':>12}{'ζ':>12}{'vξ':>12}{'vη':>12}{'vζ':>12}",
                f"{'[Myr]':<8}{'[pc]':>12}{'[pc]':>12}{'[pc]':>12}"
                f"{'[pc/Myr]':>12}{'[pc/Myr]':>12}{'[pc/Myr]':>12}"
                f"{'[pc]':>12}{'[pc]':>12}{'[pc]':>12}"
                f"{'[pc/Myr]':>12}{'[pc/Myr]':>12}{'[pc/Myr]':>12}",
                f"{'':-<152}"
            ]

            # Create lines
            for t in np.arange(self.group.series.time.size):
                lines.append(
                    (
                        f'{self.group.series.time[t]:<8.1f}'
                        f'{self.position_xyz[t,0]:>12.2f}'
                        f'{self.position_xyz[t,1]:>12.2f}'
                        f'{self.position_xyz[t,2]:>12.2f}'
                        f'{self.velocity_xyz[t,0]:>12.2f}'
                        f'{self.velocity_xyz[t,1]:>12.2f}'
                        f'{self.velocity_xyz[t,2]:>12.2f}'
                        f'{self.position_ξηζ[t,0]:>12.2f}'
                        f'{self.position_ξηζ[t,1]:>12.2f}'
                        f'{self.position_ξηζ[t,2]:>12.2f}'
                        f'{self.velocity_ξηζ[t,0]:>12.2f}'
                        f'{self.velocity_ξηζ[t,1]:>12.2f}'
                        f'{self.velocity_ξηζ[t,2]:>12.2f}'
                    )
                )

            # Create footer
            lines.append(f"{'':-<152}")

        # Show table
        if show:
            for line in lines:
                print(line)

        # Save table
        if save:
            self.group.series.save_table(
                self.name, lines, file_path=f'kinematics_time_{self.name}',
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )