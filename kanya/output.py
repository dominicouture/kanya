# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
output.py: Defines functions to create data output such as plots of association size metrics
over time, 2D and 3D scatters at a given time, histograms, color mesh, etc.
"""

from matplotlib import pyplot as plt, ticker as tkr
from scipy.interpolate import interp1d
from colorsys import hls_to_rgb
from cycler import cycler
from .collection import *
from .coordinate import *
from copy import deepcopy

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
    """Output methods of a series of groups."""

    def save_table(
        self, name, lines, header=None, extension='txt', file_type=None,
        output_dir=None, forced=False, default=False, cancel=False
    ):
        """Saves a table to a CSV file for a given header and data."""

        # Check the types of lines and header
        self.check_type(lines, 'lines', ('tuple', 'list'))
        self.check_type(header, 'header', ('string', 'None'))

        # Save table
        def save(output_path, lines, header):
            with open(output_path, 'w') as output_file:
                if header is not None:
                    output_file.write(header + '\n')
                output_file.writelines([line + '\n' for line in lines])

        # Choose behavior
        self.choose(
            name, save, lines, header, extension=extension, file_type=file_type,
            output_dir=output_dir, cancel=cancel, forced=forced, default=default
        )

    def save_figure(
        self, name, extension='pdf', file_type=None, output_dir=None,
        tight=False, forced=False, default=False, cancel=False
    ):
        """Saves figure with or without tight layout and some padding."""

        # Check the type of tight
        self.check_type(tight, 'tight', 'boolean')

        # Save figure
        def save(output_path, tight):
            if tight:
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01)
            else:
                plt.savefig(output_path)

        # Choose behavior
        self.choose(
            name, save, tight, extension=extension, file_type=file_type,
            output_dir=output_dir, cancel=cancel, forced=forced, default=default
        )

    def choose(
        self, name, save, *save_args, extension=None, file_type=None,
        output_dir=None, forced=False, default=False, cancel=False
    ):
        """
        Checks whether a path already exists and asks for user input if it does. The base path
        is assumed to be the output directory. Also, if the path does not have an extension, a
        an extension is added.
        """

        # Get file path
        file_path = self.get_output_path(name, extension, file_type=file_type, output_dir=output_dir)

        # Check if a file already exists
        if path.exists(file_path):
            choice = None
            self.check_type(forced, 'forced', 'boolean')
            if not forced:
                self.check_type(default, 'default', 'boolean')
                if not default:
                    self.check_type(cancel, 'cancel', 'boolean')
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
                    file_path = get_default_filename(file_path)
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

    def get_output_path(self, name, extension=None, file_type=None, output_dir=None):
        """
        Returns a proper file path given a name, an extension and, optionnally, a file path
        relative to the output directory.
        """

        # Check the type of output_dir
        self.check_type(output_dir, 'output_dir', ('string', 'None'))

        # Set output_dir parameter, if needed
        if 'output_dir' not in vars(self).keys():
            self.output_dir = self.set_path(
                self.config.output_dir, 'output_dir',
                dir_only=True, check=False, create=False
            )

        # Set output_path
        output_path = self.set_path(
            self.output_dir if output_dir is None else output_dir, 'output_path',
            name=name, extension=extension, file_type=file_type,
            full_path=True, check=False, create=True
        )

        return output_path

    def set_figure_metric(self):
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

        # Check the types of robust and sklearn
        self.check_type(robust, 'robust', 'boolean')
        self.check_type(sklearn, 'sklearn', 'boolean')
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
                    f'{metric.latex_short[index]} : {metric.age[index]:.1f}'
                    f' ± {metric.age_error[index]:.1f}  Myr'
                ).replace('-', '–'),
                color=color, alpha=1.0, linewidth=1.0, linestyle=linestyle,
                solid_capstyle='round', dash_capstyle='round', zorder=zorder
            )

            # Plot an enveloppe to display the uncertainty
            ax.fill_between(
                time, value - value_error, value + value_error,
                color=color, alpha=0.15, linewidth=0.0, zorder=zorder - 0.5
            )

            # Check the type of secondary
            self.check_type(secondary, 'secondary', 'boolean')

            # Plot secondary lines
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

        # Check the type of title
        self.check_type(title, 'title', 'boolean')

        # Title from data
        if title:
            if self.from_data:
                ax.set_title(
                    "{} of {}\n over {:.1f} Myr with a {:.1f} km/s radial "
                    "velocity correction\n".format(
                        metric, self.name, self.duration.value,
                        self.rv_shift.to('km/s').value
                    ), fontsize=8
                )

            # Title from a model
            elif self.from_model:
                ax.set_title(
                    "Average {} of {} modeled associations over {:.1f} Myr\n"
                    "with kinematics similar to {} and a {:.1f} km/s radial velocity "
                    "bias\n".format(
                        metric, self.number_of_groups, self.duration.value,
                        self.name, self.rv_shift.to('km/s').value
                    ), fontsize=8
                )

    def set_axis_metric(self, ax, hide_x=False, hide_y=False, units_y='pc'):
        """Sets the parameters of an axis to plot association size metrics."""

        # Set legend
        legend = ax.legend(loc=2, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor(colors.white + (0.8,))
        legend.get_frame().set_edgecolor(colors.black)
        legend.get_frame().set_linewidth(0.5)

        # Set labels
        ax.set_xlabel('Epoch (Myr)', fontsize=8)
        ax.set_ylabel(f'Association size ({units_y})', fontsize=8)

        # Set limits
        ax.set_xlim(self.final_time.value + 1, self.initial_time.value + 1)
        # ax.set_xlim(self.final_time.value, self.initial_time.value + 1)
        # ax.set_ylim(-1., 39.)

        # Set ticks
        # ax.set_xticks([0., -5., -10., -15., -20., -25., -30., -35., -40, -45, -50])
        # ax.set_yticks([0.,  5.,  10.,  15.,  20.,  25.,  30.,  35.])
        ax.tick_params(top=True, right=True, which='both', direction='in', width=0.5, labelsize=8)

        # Set spines
        ax.spines[:].set_linewidth(0.5)

        # Hide labels and tick labels, if needed
        if hide_x:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        if hide_y:
            ax.set_ylabel('')
            ax.set_yticklabels([])

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

        # Check the types of save, show and machine
        self.check_type(save, 'save', 'boolean')
        self.check_type(show, 'show', 'boolean')
        self.check_type(machine, 'machine', 'boolean')

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
                f'metrics_{self.name}', lines,
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

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
        fig, ax = self.set_figure_metric()

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
            f'covariances_xyz_{self.name}'
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
        fig, ax = self.set_figure_metric()

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
            f'covariances_ξηζ_{self.name}'
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
        fig, ax = self.set_figure_metric()

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
        self.set_axis_metric(ax, units_y='pc$\:$Myr$^{-1/2}$')

        # Save figure
        self.save_figure(
            f'cross_covariances_xyz_{self.name}'
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
        fig, ax = self.set_figure_metric()

        # Check 'robust' and 'sklearn' arguments
        self.check_robust_sklearn_metric(robust, sklearn)

        # Select empirical, robust or sklearn association size metrics
        if sklearn:
            cross_covariances_ξηζ_matrix_det = self.cross_covariances_ξηζ_matrix_det
            cross_covariances_ξηζ_matrix_trace = self.cross_covariances_ξηζ_matrix_trace
            cross_covariances_ξηζ = self.cross_covariances_ξηζ
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
        self.set_axis_metric(ax, units_y='pc$\:$Myr$^{-1/2}$')

        # Save figure
        self.save_figure(
            f'cross_covariances_ξηζ_{self.name}'
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
        fig, ax = self.set_figure_metric()

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
            f'mad_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_mad_ξηζ_plot(self, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the total ξηζ median absolute deviation (MAD), and ξ', η' and ζ'
        components of the MAD.
        """

        # Initialize figure
        fig, ax = self.set_figure_metric()

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
            f'mad_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_mst_xyz_plot(self, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the mean branch length (both empirical and robust) and median absolute
        deviation of the xyz minimum spanning tree (MST).
        """

        # Initialize figure
        fig, ax = self.set_figure_metric()

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
            f'mst_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_mst_ξηζ_plot(self, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the mean branch length (both empirical and robust) and median absolute
        deviation of the ξηζ minimum spanning tree (MST).
        """

        # Initialize figure
        fig, ax = self.set_figure_metric()

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
            f'mst_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_mahalanobis_plot(self, title=False, forced=False, default=False, cancel=False):
        """Creates a plot of the xyz and ξηζ Mahalanobis distance mean and median."""

        # Initialize figure
        fig, ax = self.set_figure_metric()

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
            f'mahalanobis_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_covariances_mad_ξηζ_plot(
        self, title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of ξ'-ξ', η'-η' and ζ'-ζ' covariances, and the determinant and the trace
        of the total ξ'η'ζ' covariances matrix, and the total ξηζ median absolute deviation (MAD),
        and ξ', η' and ζ' components of the MAD.
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
            f'covariances_mad_ξηζ_{self.name}.pdf',
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

        # Check the type of other
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

        # Check the type of title
        self.check_type(title, 'title', 'boolean')

        # Title from data
        if title:
            if self.from_data:
                fig.suptitle(
                    '$XYZ$ covariances, MAD, MST and cross covariances of {}\n and {} over '
                    '{:.1f} Myr with a {:.1f} km/s radial velocity correction\n'.format(
                        self.name, other.name, self.duration.value,
                        self.rv_shift.to('km/s').value
                    ), fontsize=8
                )

            # Title from a model
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
            f'covariances_mad_mst_cross_covariannces_xyz_{self.name}_{other.name}.pdf',
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

        # Check the type of title
        self.check_type(title, 'title', 'boolean')

        # Set title
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
        ax.spines[:].set_linewidth(0.5)

        # Save figure
        self.save_figure(
            f'age_distribution_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

class Output_Group():
    """Output methods of a group of stars."""

    def set_figure(self, style, *options):
        """
        Initializes a figure with multiple axes. The axes are created with at correct position
        and size. Ticks and labels are repositioned and set invisible accordingly.
        """

        # Check traceback
        self.series.check_traceback()

        # Check the type and value of style
        self.series.check_type(style, 'style', 'string')
        self.series.stop(
            style not in ('2+1', '2x2', '1x3'), 'ValueError',
            "'style' must be either '2+1', '2x2' or 1x3' ({} given).", style
        )

        # Initialize a 2+1 figure
        if style == '2+1':
            fig = plt.figure(figsize=(3.345, 3.315), facecolor=colors.white, dpi=300)

            # Set dimensions
            left, bottom = (0.133, 0.102)
            short1, short2  = (0.236, 0.2381)
            long1, long2 = (0.618, 0.6236)
            inside1, inside2 = (0.012, 0.0119)

            # Create axes
            ax0 = fig.add_axes([left, bottom + inside2 + short2, long1, long2])
            ax1 = fig.add_axes([left + long1 + inside1, bottom + inside2 + short2, short1, long2])
            ax2 = fig.add_axes([left, bottom, long1, short2])
            ax3 = None

            # Set axes
            ax0.tick_params(labelbottom=False)
            ax1.tick_params(labelleft=False)

            # Set title
            fig.joint = ''
            fig.offset = 1.05

        # Initialize a 2x2 figure
        if style == '2x2':
            fig = plt.figure(facecolor=colors.white, figsize=(7.090, 6.317), dpi=300)

            # Set dimensions
            left1, bottom1 = (0.083, 0.528)
            left2, bottom2 = (0.507, 0.052)
            width, height = (0.410, 0.460)

            # Create axes
            ax0 = fig.add_axes([left1, bottom1, width, height], zorder=0.5)
            ax1 = fig.add_axes([left2, bottom1, width, height], zorder=0.5)
            ax2 = fig.add_axes([left1, bottom2, width, height], zorder=0.5)
            ax3 = fig.add_axes(
                [left2, bottom2, width, height],
                projection='3d' if '3d' in options else None, zorder=0.4
            )

            # Set axes
            ax1.yaxis.set_label_position('right')
            ax1.yaxis.tick_right()
            if 'hide_x' in options:
                ax0.tick_params(labelbottom=False)
                ax1.tick_params(labelbottom=False)
            if '3d' in options:
                self.set_axis_3d(ax3)
            else:
                ax3.yaxis.set_label_position('right')
                ax3.yaxis.tick_right()

            # Set title
            fig.joint = ' '
            fig.offset = 1.025

        # Initialize a 1x3 figure
        if style == '1x3':
            # fig = plt.figure(facecolor=colors.white, figsize=(3.345, 7.5), dpi=300)
            fig = plt.figure(facecolor=colors.white, figsize=(3.345, 9.134), dpi=300)

            # Set dimensions
            if 'hide_x' in options:
                left, bottom = (0.133, 0.0355)
                width, height = (0.866, 0.3112)
                inside = 0.01095
                # left, bottom = (0.133, 0.0432)
                # width, height = (0.866, 0.3064)
                # inside = 0.0133
            else:
                left, bottom = (0.2022, 0.0355)
                width, height = (0.7968, 0.2918)
                inside = 0.0400

            # Create axes
            ax0 = fig.add_axes([left, bottom + 2 * (height + inside), width, height])
            ax1 = fig.add_axes([left, bottom + height + inside, width, height])
            ax2 = fig.add_axes([left, bottom, width, height])
            ax3 = None

            # Set axes
            if 'hide_x' in options:
                ax0.tick_params(labelbottom=False)
                ax1.tick_params(labelbottom=False)

            # Set title
            fig.joint = '\n'
            fig.offset = 1.03

        # Set ticks and spines
        for ax in filter(lambda ax: ax is not None, (ax0, ax1, ax2, ax3)):
            self.set_axis_2d(ax)

        return fig, ax0, ax1, ax2, ax3

    def set_title(self, fig, title, line_1, line_2=''):
        """Sets a title for the trajectory figure, if 'title' is True."""

        # Check the type of title
        self.series.check_type(title, 'title', 'boolean')

        # Set title
        if title:
            fig.suptitle(
                f'{line_1} of stars in {self.name}'
                f"{fig.joint if line_2 != '' else ''}{line_2}",
                y=fig.y_offset, fontsize=8
            )

    def set_axis_2d(self, ax):
        """Sets the basic parameter of a 2d axis."""

        # Set ticks
        ax.tick_params(
            top=True, right=True, which='both',
            direction='in', width=0.5, labelsize=8
        )

        # Set spines
        ax.spines[:].set_linewidth(0.5)

    def set_axis_3d(self, ax):
        """Sets the parameters of a 3d axis."""

        # Set ticks
        for iax in (ax.xaxis, ax.yaxis, ax.zaxis):
            iax._axinfo['tick']['outward_factor'] = 0.5
            iax._axinfo['tick']['inward_factor'] = 0.0
            iax._axinfo['tick']['linewidth'] = {True: 0.5, False: 0.0}

            # Set grids
            iax._axinfo['grid']['color'] = colors.grey[17]
            iax._axinfo['grid']['linewidth'] = 0.5

            # Set panes
            iax.pane._alpha = 1.0
            iax.pane.fill = False
            iax.pane.set_linewidth(0.5)
            iax.pane.set_edgecolor(colors.grey[17])

            # Set lines
            iax.line.set_linewidth(0.5)

        # Set view
        # ax.view_init(azim=45)

    def set_legend(self, ax):
        """Sets the parameters of an axis."""

        legend = ax.legend(loc=4, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor(colors.white + (0.8,))
        legend.get_frame().set_edgecolor(colors.black)
        legend.get_frame().set_linewidth(0.5)

    def get_epoch(self, age=None, metric=None, index=None):
        """
        Computes the time index of the epoch for a given association age or, association size
        metric and dimensional index. Return the birth index, age and age error.
        """

        # Index from age
        if age is not None:
            self.series.check_type(age, 'age', ('integer', 'float', 'None'))
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
        self.series.check_type(metric, 'metric', ('string', 'None'))
        self.series.stop(
            metric not in [metric.label for metric in self.series.metrics], 'ValueError',
            "'metric' must be a valid metric key ({} given).", metric
        )
        metric = vars(self.series)[metric]


        # If the metric has a size of 1, index is ignored
        if metric.value.shape[-1] == 1:
            index = 0

        # Metric index
        else:
            self.series.check_type(index, 'index', ('integer', 'None'))
            self.series.stop(
                metric.value.shape[-1] > 1 and index is None, 'ValueError',
                "No 'index' is provided (metric is {} in size).", metric.value.shape[-1]
            )
            self.series.stop(
                index > metric.value.shape[-1] - 1 and metric.value.shape[-1] > 1, 'ValueError',
                "'index' is too large for this metric ({} given, {} in size).",
                index, metric.value.shape[-1]
            )

        return metric, index

    def get_step_age(self, step, age):
        """
        Checks the types and values of step and step, and computes the appropriate values of
        step and age. If both are not None, the priority is given the age.
        """

        # Check the type and value of step
        if step is not None:
            self.series.check_type(step, 'step', ('integer', 'float'))
            self.series.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step
            )
            self.series.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0 ({} given).", step
            )
            self.series.stop(
                step >= self.series.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({} and {} given).",
                self.series.number_of_steps, step
            )

        # Check the type and value of age
        if age is not None:
            self.series.check_type(age, 'age', ('integer', 'float'))
            self.series.stop(
                age > self.series.initial_time.value, 'ValueError',
                "'age' must older than the initial time  ({:.1f} Myr and {:.1f} Myr given).",
                age, self.series.initial_time.value
            )
            self.series.stop(
                age < self.series.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({:.1f} Myr and {:.1f} Myr given).",
                age, self.series.final_time.value
            )

        # Check if both step and age are None
        self.series.stop(
            step is None and age is None, 'ValueError',
            "'step' and 'age' cannot both be None."
        )

        # Compute step and age
        if age is not None:
            step = int(round((self.series.initial_time.value - age) / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(self.series.initial_time.value - step * self.series.timestep.value, 2)

        return step, age

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

        # Check the types of save, show and machine
        self.series.check_type(save, 'save', 'boolean')
        self.series.check_type(show, 'show', 'boolean')
        self.series.check_type(machine, 'machine', 'boolean')

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
                f'kinematics_{self.name}_{age}Myr', lines,
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

    def create_kinematics_time_table(
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

        # Check the types of save, show and machine
        self.series.check_type(save, 'save', 'boolean')
        self.series.check_type(show, 'show', 'boolean')
        self.series.check_type(machine, 'machine', 'boolean')

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
                f'kinematics_time_{self.name}', lines,
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

    def plot_trajectory(self, ax0, ax1, ax2, coord, age, metric, index, labels):
        """Draws the trajectory of stars in the group."""

        # Check the type and value of coord
        self.series.check_type(coord, 'coord', 'string')
        self.series.stop(
            coord not in ('xyz', 'ξηζ'), 'ValueError',
            "'coord' can only take as value 'xyz' or 'ξηζ' ({} given).", coord
        )

        # Check the type of labels
        self.series.check_type(labels, 'labels', 'boolean')

        # Birth index, age and age error
        birth_index, age, age_error = self.get_epoch(age=age, metric=metric, index=index)

        # Select axes
        if coord == 'xyz':
            i, j = (1, 0)
        if coord == 'ξηζ':
            i, j = (0, 1)

        # Select conversion factors from pc to kpc
        for ax, x, y in ((ax0, i, j), (ax1, 2, j), (ax2, i, 2)):
            if coord == 'xyz':
                factors = np.array([1 if y == 2 else 1000, 1 if x == 2 else 1000, 1])

            # Select coordinates
            for star in self:
                if coord == 'xyz':
                    position = star.position_xyz / factors
                if coord == 'ξηζ':
                    position = star.position_ξηζ

                # Plot stars' trajectories
                color = colors.red[6] if star.outlier else colors.black
                ax.plot(
                    position.T[x], position.T[y],
                    color=color, alpha=0.6, linewidth=0.5,
                    solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        np.array([position[0,x]]), np.array([position[0,y]]),
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth positions
                    if birth_index is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax.scatter(
                            np.array([position[birth_index,x]]),
                            np.array([position[birth_index,y]]),
                            color=color + (0.4,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                    # Show stars' names
                    if labels:
                        ax.text(
                            np.array(position.T[0,x]), np.array(position.T[0,y]),
                            star.name, horizontalalignment='left', fontsize=6
                        )

            # Select coordinates
            if self.series.from_model:
                if coord == 'xyz':
                    average_model_star_position = self.average_model_star.position_xyz / factors
                if coord == 'ξηζ':
                    average_model_star_position = self.average_model_star.position_ξηζ

                # Plot the average model star's trajectory
                ax.plot(
                    average_model_star_position.T[x],
                    average_model_star_position.T[y],
                    color=colors.green[6], alpha=0.8,
                    linewidth=0.5, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current positions
                for t, size, marker in ((-1, 12, '*'), (0, 6, 'o')):
                    ax.scatter(
                        np.array([average_model_star_position[t,x]]),
                        np.array([average_model_star_position[t,y]]),
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Select coordinates
                for star in self.model_stars:
                    if coord == 'xyz':
                        position = star.position_xyz / factors
                    if coord == 'ξηζ':
                        position = star.position_ξηζ

                    # Plot model stars' trajectories
                    ax.plot(
                        star.position_xyz.T[x], star.position_xyz.T[y],
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, size, marker in ((0, 12, '*'), (-1, 6, 'o')):
                        ax.scatter(
                            np.array([star.position_xyz[t,x]]),
                            np.array([star.position_xyz[t,y]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

    def create_trajectory_xyz(
        self, age=None, metric=None, index=None, labels=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """Draws the xyz trajectories of stars in the group."""

        # Initialize figure
        fig, ax0, ax1, ax2, ax3 = self.set_figure('2+1')

        # Plot xyz trajectories
        self.plot_trajectory(ax0, ax1, ax2, 'xyz', age, metric, index, labels)

        # Draw vertical and horizontal lines through the Sun's position at the current epoch
        for ax in (ax0, ax1, ax2):
            ax.axhline(0., color=colors.black, alpha=0.8, linewidth=0.5, linestyle=':', zorder=0.0)
            ax.axvline(0., color=colors.black, alpha=0.8, linewidth=0.5, linestyle=':', zorder=0.0)

        # Draw circles around the galactic center located at 8.122 kpc from the Sun
        for radius in range(1, 16):
            ax0.add_artist(
                plt.Circle(
                    (0, 8.122), radius, color=colors.grey[17],
                    fill=False, linewidth=0.5, linestyle=':', zorder=0.0
                )
            )

        # Set title
        self.set_title(fig, title, '$XYZ$ trajectories')

        # Set labels
        ax0.set_ylabel('$X$ (kpc)', fontsize=8)
        ax1.set_xlabel('$Z$ (pc)', fontsize=8)
        ax2.set_xlabel('$Y$ (kpc)', fontsize=8)
        ax2.set_ylabel('$Z$ (pc)', fontsize=8)

        # Set limits
        ax0.set_xlim(-9, 1)
        ax0.set_ylim(-1, 9)
        ax1.set_xlim(-80, 80)
        ax1.set_ylim(-1, 9)
        ax2.set_xlim(-9, 1)
        ax2.set_ylim(-80, 80)

        # Invert y axis
        ax0.invert_xaxis()
        ax2.invert_xaxis()

        # Save figure
        self.series.save_figure(
            f'trajectory_xyz_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_trajectory_ξηζ(
        self, age=None, metric=None, index=None, labels=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """Draws the ξηζ trajectories of stars in the group."""

        # Initialize figure
        fig, ax0, ax1, ax2, ax3 = self.set_figure('2+1')

        # Plot xyz trajectories
        self.plot_trajectory(ax0, ax1, ax2, 'ξηζ', age, metric, index, labels)

        # Set title
        self.set_title(fig, title, '$ξ^\prime η^\prime ζ^\prime$ trajectories')

        # Set labels
        ax0.set_ylabel('$η^\prime$ (pc)', fontsize=8)
        ax1.set_xlabel('$ζ^\prime$ (pc)', fontsize=8)
        ax2.set_xlabel('$ξ^\prime$ (pc)', fontsize=8)
        ax2.set_ylabel('$ζ^\prime$ (pc)', fontsize=8)

        # Set limits
        # ax0.set_xlim(-225, 60)
        # ax0.set_ylim(-45, 110)
        # ax1.set_xlim(-40, 49)
        # ax1.set_ylim(-45, 110)
        # ax2.set_xlim(-225, 60)
        # ax2.set_ylim(-40, 49)

        # Save figure
        self.series.save_figure(
            f'trajectory_ξηζ_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def plot_position_time(self, ax0, ax1, ax2, ax3, coord, style, age, metric):
        """Draws the trajectory of stars in the group over time."""

        # Check the type and value of coord
        self.series.check_type(coord, 'coord', 'string')
        self.series.stop(
            coord not in ('xyz', 'ξηζ'), 'ValueError',
            "'coord' can only take as value 'xyz' or 'ξηζ' ({} given).", coord
        )

        # Birth index, age and age error
        birth_index, age, age_error  = tuple(
            zip(*[self.get_epoch(age=age, metric=metric, index=index) for index in range(3)])
        )

        # Select coordinates
        for ax, y in ((ax0, 0), (ax1, 1), (ax2, 2)):
            for star in self:
                if coord == 'xyz':
                    position = star.relative_position_xyz
                if coord == 'ξηζ':
                    position = star.relative_position_ξηζ

                # Plot stars' trajectories
                ax.plot(
                    self.series.time, position[:,y],
                    color = colors.red[6] if star.outlier else colors.black, alpha=0.6,
                    linewidth=0.5, solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current positions
                if self.series.from_data:
                    ax.scatter(
                        self.series.time[0], position[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black, alpha=None,
                        s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth positions
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax.scatter(
                            self.series.time[birth_index[y]], position[birth_index[y],y],
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

            # Select coordinates
            if self.series.from_model:
                if coord == 'xyz':
                    position = np.mean(
                        [star.position_xyz[:,y] for star in self.model_stars], axis=0
                    )
                    average_model_star_position = (
                        self.average_model_star.position_xyz[:,y] - position[::-1]
                    )
                if coord == 'ξηζ':
                    position = np.mean(
                        [star.position_ξηζ[:,y] for star in self.model_stars], axis=0
                    )
                    average_model_star_position = (
                        self.average_model_star.position_ξηζ[:,y] - position[::-1]
                    )

                # Plot the average model star's trajectory
                ax.plot(
                    self.series.model_time,
                    average_model_star_position,
                    color=colors.green[6], alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current positions
                for t, x, size, marker in ((-1, -1, 10, '*'), (0, 0, 6, 'o')):
                    ax.scatter(
                        np.array([self.series.model_time[t]]),
                        np.array([average_model_star_position[x]]),
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Select coordinates
                for star in self.model_stars:
                    if coord == 'xyz':
                        model_star_position = star.position_xyz[:,y] - position
                    if coord == 'ξηζ':
                        model_star_position = star.position_ξηζ[:,y] - position

                    # Plot model stars' trajectories
                    ax.plot(
                        self.series.model_time[::-1], model_star_position,
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 10, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_position[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

        # Select coordinates
        if style == '2x2':
            if coord == 'xyz':
                position = self.position_xyz / 1000
                label_0, label_1, label_2 = ('$<X>$', '$<Y>$', '$<Z>$')
            if coord == 'ξηζ':
                position = self.position_ξηζ
                label_0, label_1, label_2 = ('$<ξ^\prime>$', '$<η^\prime>$', '$<ζ^\prime>$')

            # Plot stars' average trajectory
            for y, label, linestyle in ((0, label_0, '-'), (1, label_1, '--'), (2, label_2, ':')):
                ax3.plot(
                    self.series.time, position[:,y],
                    label=label, color=colors.black, alpha=0.8, linestyle=linestyle,
                    linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.1
                )

                # Plot stars' average current positions
                if self.series.from_data:
                    ax3.scatter(
                        self.series.time[0], position[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' average birth positions
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax3.scatter(
                            self.series.time[birth_index[y]],
                            position[birth_index[y],y],
                            color=color + (0.4,), edgecolors=color + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                # Select coordinates
                if self.series.from_model:
                    if coord == 'xyz':
                        average_model_star_position = self.average_model_star.position_xyz[:,y] / 1000
                    if coord == 'ξηζ':
                        average_model_star_position = self.average_model_star.position_ξηζ[:,y]

                    # Plot the average model star's trajectory
                    ax3.plot(
                        self.series.model_time, average_model_star_position,
                        color=colors.green[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.3
                    )

                    # Plot the average model star's birth and current positions
                    for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                        ax3.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([average_model_star_position[x]]),
                            color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                        )

                    # Select coordinates
                    if coord == 'xyz':
                        model_star_position = np.mean(
                            [star.position_xyz[:,y] for star in self.model_stars], axis=0
                        ) / 1000
                    if coord == 'ξηζ':
                        model_star_position = np.mean(
                            [star.position_ξηζ[:,y] for star in self.model_stars], axis=0
                        )

                    # Plot model stars' trajectories
                    ax3.plot(
                        self.series.model_time[::-1], model_star_position,
                        color=colors.blue[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current positions
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax3.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_position[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

    def create_position_xyz_plot(
        self, style, age=None, metric=None, title=False,
        forced=False, default=False, cancel=False
    ):
        """Draws the xyz trajectories of stars as a function of time."""

        # Initialize figure
        fig, ax0, ax1, ax2, ax3 = self.set_figure(style, 'hide_x')

        # Plot xyz trajectories over time
        self.plot_position_time(ax0, ax1, ax2, ax3, 'xyz', style, age, metric)

        # Set title
        self.set_title(fig, title, '$XYZ$ positions', 'as a function of time')

        # Set legend
        if style == '2x2':
            self.set_legend(ax3)

        # Set labels
        ax2.set_xlabel('Epoch (Myr)', fontsize=8)
        ax0.set_ylabel('$X - <X>$ (pc)', fontsize=8)
        ax1.set_ylabel('$Y - <Y>$ (pc)', fontsize=8)
        ax2.set_ylabel('$Z - <Z>$ (pc)', fontsize=8)
        if style == '2x2':
            ax3.set_xlabel('Epoch (Myr)', fontsize=8)
            ax3.set_ylabel('$<XYZ>$ (kpc)', fontsize=8)

        # Set limits
        for ax in (ax0, ax1, ax2) + ((ax3,) if style == '2x2' else ()):
            ax.set_xlim(np.min(self.series.time), np.max(self.series.time) + 1)

        # Save figure
        self.series.save_figure(
            f"position_xyz_time_{style}_{self.name}.pdf",
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_position_ξηζ_plot(
        self, style, age=None, metric=None, title=False,
        forced=False, default=False, cancel=False
    ):
        """Draws the ξηζ trajectories of stars as a function of time."""

        # Initialize figure
        fig, ax0, ax1, ax2, ax3 = self.set_figure(style, 'hide_x')

        # Plot ξηζ trajectories over time
        self.plot_position_time(ax0, ax1, ax2, ax3, 'ξηζ', style, age, metric)

        # Set title
        self.set_title(fig, title, '$ξ^\prime η^\prime ζ^\prime$ positions', 'as a function of time')

        # Set legend
        if style == '2x2':
            self.set_legend(ax3)

        # Set labels
        ax2.set_xlabel('Epoch (Myr)', fontsize=8)
        ax0.set_ylabel('$ξ^\prime - <ξ^\prime>$ (pc)', fontsize=8)
        ax1.set_ylabel('$η^\prime - <η^\prime>$ (pc)', fontsize=8)
        ax2.set_ylabel('$ζ^\prime - <ζ^\prime>$ (pc)', fontsize=8)
        if style == '2x2':
            ax3.set_xlabel('Epoch (Myr)', fontsize=8)
            ax3.set_ylabel('$<ξ^\prime η^\prime ζ^\prime>$ (pc)', fontsize=8)

        # Set limits
        for ax in (ax0, ax1, ax2) + ((ax3,) if style == '2x2' else ()):
            ax.set_xlim(np.min(self.series.time), np.max(self.series.time) + 1)
            ax2.set_ylim(-70, 70)

        # Save figure
        self.series.save_figure(
            f'position_ξηζ_time_{style}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def plot_scatter(self, ax, axes, coord, step, errors=False, labels=False, mst=False):
        """Creates a 2d scatter of axes i versus j, or a 3d scatter at a given step."""

        # Check the type and value of axes
        self.series.check_type(axes, 'axes', 'tuple')
        self.series.stop(
            len(axes) not in (2, 3), 'ValueError',
            "'axes' must have a length of 2 or 3 ({} given).", len(axes)
        )

        # Check the types and values of elements of axes
        for i in axes:
            self.series.check_type(i, 'i value in axes', 'integer')
            self.series.stop(
                i < 0 or i > 2, 'ValueError',
                "All values in axes must 0, 1 or 2 ({} given).", i
            )

        # Check the type and value of coord
        self.series.check_type(coord, 'coord', 'string')
        self.series.stop(
            coord not in ('xyz', 'ξηζ'), 'ValueError',
            "'coord' can only take as value 'xyz' or 'ξηζ' ({} given).", coord
        )

        # Check the types of errors, labels and mst
        self.series.check_type(errors, 'errors', 'boolean')
        self.series.check_type(labels, 'labels', 'boolean')
        self.series.check_type(mst, 'mst', 'boolean')

        # Select axes
        if len(axes) == 2:
            x, y = axes
            projection = '2d'
        if len(axes) == 3:
            x, y, z = axes
            projection = '3d'

        # Select coordinates
        for star in self.sample:
            if coord == 'xyz':
                position = star.position_xyz[step]
                error = star.position_xyz_error[step].diagonal()
            if coord == 'ξηζ':
                position = star.position_ξηζ[step]
                error = star.position_ξηζ_error[step].diagonal()

            # Select color
            color = colors.red[9] if star.outlier else colors.black

            # Plot position or velocity
            if projection == '2d':
                ax.scatter(
                    position[x], position[y],
                    color=color + (0.4,), edgecolors=color,
                    alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.4
                )
            if projection == '3d':
                ax.scatter(
                    position[x], position[y], position[z],
                    color=color + (0.4,), edgecolors=color,
                    alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.4
                )

            # Plot error bars
            if errors:
                if projection == '2d':
                    ax.plot(
                        (position[x] - error[x], position[x] + error[x]),
                        (position[y], position[y]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                    ax.plot(
                        (position[x], position[x]),
                        (position[y] - error[y], position[y] + error[y]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                if projection == '3d':
                    ax.plot(
                        (position[x] - error[x], position[x] + error[x]),
                        (position[y], position[y]), (position[z], position[z]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                    ax.plot(
                        (position[x], position[x]), (position[y] - error[y], position[y] + error[y]),
                        (position[z], position[z]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                    ax.plot(
                        (position[x], position[x]), (position[y], position[y]),
                        (position[z] - error[z], position[z] + error[z]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )

            # Show labels
            if labels:
                if projection == '2d':
                    ax.text(
                        position[step, x] + 1, position[step, y] + 1, star.name,
                        color=color, horizontalalignment='left',
                        verticalaligment='top', fontsize=6, zorder=0.9
                    )
                if projection == '3d':
                    ax.text(
                        position[x] + 2, position[y] + 2, position[z] + 2, star.name,
                        color=color, horizontalalignment='left',
                        verticalaligment='top', fontsize=6, zorder=0.9
                    )

        # Select branches
        if mst:
            for branch in self.mst_xyz[step]:
                if coord == 'xyz':
                    position_start = branch.start.position_xyz[step]
                    position_end = branch.end.position_xyz[step]
                if coord == 'ξηζ':
                    position_start = branch.start.position_ξηζ[step]
                    position_end = branch.end.position_ξηζ[step]

                # Plot branches
                if projection == '2d':
                    ax.plot(
                        (position_start[x], position_end[x]),
                        (position_start[y], position_end[y]),
                        color=colors.blue[6], alpha=0.6, linestyle='-',
                        linewidth=0.5, solid_capstyle='round', zorder=0.4
                    )
                if projection == '3d':
                    ax.plot(
                        (position_start[x], position_end[x]),
                        (position_start[y], position_end[y]),
                        (position_start[z], position_end[z]),
                        color=colors.blue[6], alpha=0.6, linestyle='-',
                        linewidth=0.5, solid_capstyle='round', zorder=0.4
                    )

        # Set labels
        axis_labels = {
            'xyz': ('$X$', '$Y$', '$Z$'),
            'ξηζ': ('$ξ^\prime$', '$η^\prime$', '$ζ^\prime$')
        }
        if projection == '2d':
            ax.set_xlabel(f'{axis_labels[coord][x]} (pc)', fontsize=8)
            ax.set_ylabel(f'{axis_labels[coord][y]} (pc)', fontsize=8)
        if projection == '3d':
            ax.set_xlabel(f'{axis_labels[coord][x]} (pc)', fontsize=8)
            ax.set_ylabel(f'{axis_labels[coord][y]} (pc)', fontsize=8)
            ax.set_zlabel(f'{axis_labels[coord][z]} (pc)', fontsize=8)

    def create_position_xyz_scatter(
        self, style, step=None, age=None, errors=False, labels=False,
        mst=False, title=False, forced=False, default=False, cancel=False
    ):
        """
        Draws scatter plots of xyz positions of stars at a given 'step' or 'age' in Myr. If 'age'
        doesn't match a step, the closest step is used instead. 'age' overrules 'steps' if both
        are given. 'errors', adds error bars, 'labels' adds the stars' name and 'mst' adds
        the minimun spanning tree branches.
        """

        # Initialize figure
        fig, ax0, ax1, ax2, ax3 = self.set_figure(style, '3d')

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Check if the minimum spanning tree was computed
        if not self.series.mst_metrics and mst == True:
            mst = False
            self.series.log(
                "Could not display the minimum spanning tree because it was not computed.",
                level='info', display=True
            )

        # Plot 2d scatters
        for ax, axes in ((ax0, (0, 1)), (ax1, (2, 1)), (ax2, (0, 2))):
            self.plot_scatter(ax, axes, 'xyz', step, errors=errors, labels=labels, mst=mst)

        # Plot 3d scatter
        if style == '2x2':
            self.plot_scatter(ax3, (0, 1, 2), 'xyz', step, errors=errors, labels=labels, mst=mst)

        # Set title
        self.set_title(fig, title, '$XYZ$ positions', f'at {age} Myr')

        # Set limits
        if style=='1x3':
            ax.set_aspect('equal')
            ax.set_adjustable('datalim')

        # Set limits
        xlim = (30, 180)
        ylim = (-1280, -1140)
        zlim = (-85, 65)
        ax0.set_xlim(*xlim)
        ax0.set_ylim(*ylim)
        ax1.set_xlim(*zlim)
        ax1.set_ylim(*ylim)
        ax2.set_xlim(*xlim)
        ax2.set_ylim(*zlim)
        if style == '2x2':
            ax3.set_xlim(*xlim)
            ax3.set_ylim(*ylim)
            ax3.set_zlim(*zlim)

            # Set label and ticks
            ax0.set_xlabel('')
            ax0.tick_params(labelbottom=False)

        # Save figure
        self.series.save_figure(
            f'position_xyz_{style}_{self.name}_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_position_ξηζ_scatter(
        self, style, step=None, age=None, errors=False, labels=False,
        mst=False, title=False, forced=False, default=False, cancel=False
    ):
        """
        Draws scatter plots of ξηζ positions of stars at a given 'step' or 'age' in Myr. If 'age'
        doesn't match a step, the closest step is used instead. 'age' overrules 'steps' if both
        are given. 'errors', adds error bars, 'labels' adds the stars' name and 'mst' adds
        the minimun spanning tree branches.
        """

        # Initialize figure
        fig, ax0, ax1, ax2, ax3 = self.set_figure(style, '3d')

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Check if the minimum spanning tree was computed
        if not self.series.mst_metrics and mst == True:
            mst = False
            self.series.log(
                "Could not display the minimum spanning tree because it was not computed.",
                level='info', display=True
            )

        # Plot 2d scatters
        for ax, axes in ((ax0, (0, 1)), (ax1, (2, 1)), (ax2, (0, 2))):
            self.plot_scatter(ax, axes, 'ξηζ', step, errors=errors, labels=labels, mst=mst)

        # Plot 3d scatter
        if style == '2x2':
            self.plot_scatter(ax3, (0, 1, 2), 'ξηζ', step, errors=errors, labels=labels, mst=mst)

        # Set title
        self.set_title(fig, title, '$ξ^\prime η^\prime ζ^\prime$ positions', f'at {age:.1f} Myr')

        # Set limits
        # xlim = (30, 180)
        # ylim = (-1280, -1140)
        # zlim = (-85, 65)
        # ax0.set_xlim(*xlim)
        # ax0.set_ylim(*ylim)
        # ax1.set_xlim(*zlim)
        # ax1.set_ylim(*ylim)
        # ax2.set_xlim(*xlim)
        # ax2.set_ylim(*zlim)
        if style == '2x2':
            # ax3.set_xlim(*xlim)
            # ax3.set_ylim(*ylim)
            # ax3.set_zlim(*zlim)

            # Set label and ticks
            ax0.set_xlabel('')
            ax0.tick_params(labelbottom=False)

        # Save figure
        self.series.save_figure(
            f'position_ξηζ_{style}_{self.name}_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_2d_3d_scatters_xyz(
        self, ages, errors=False, labels=False, mst=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a three 4-axis columns of xy, xz and yz 2D scatters, and a 3D scatter at three
        ages definied by a list or tuple.
        """

        # Initialize figure
        self.fig = plt.figure(figsize=(9, 11.15), facecolor=colors.white, dpi=300)

        # Check the type and value of ages
        self.series.check_type(ages, 'ages', ('tuple', 'list'))
        self.series.stop(
            len(ages) != 3, 'ValueError',
            "'ages' must be have a length of 3 ({} given).", ages
        )

        # Check the types of errors, labels, mst and title
        self.series.check_type(errors, 'errors', 'boolean')
        self.series.check_type(labels, 'labels', 'boolean')
        self.series.check_type(mst, 'mst', 'boolean')
        self.series.check_type(title, 'title', 'boolean')

        # Create axes
        row1 = 0.795
        row2 = 0.545
        row3 = 0.295
        row4 = 0.035
        col1 = 0.070
        col2 = 0.398
        col3 = 0.730
        self.create_2D_axis_old('x', 'y', age=ages[0], index=1, left=col1, bottom=row1, errors=errors)
        self.create_2D_axis_old('x', 'y', age=ages[1], index=2, left=col2, bottom=row1, errors=errors)
        self.create_2D_axis_old('x', 'y', age=ages[2], index=3, left=col3, bottom=row1, errors=errors)
        self.create_2D_axis_old('x', 'z', age=ages[0], index=4, left=col1, bottom=row2, errors=errors)
        self.create_2D_axis_old('x', 'z', age=ages[1], index=5, left=col2, bottom=row2, errors=errors)
        self.create_2D_axis_old('x', 'z', age=ages[2], index=6, left=col3, bottom=row2, errors=errors)
        self.create_2D_axis_old('y', 'z', age=ages[0], index=7, left=col1, bottom=row3, errors=errors)
        self.create_2D_axis_old('y', 'z', age=ages[1], index=8, left=col2, bottom=row3, errors=errors)
        self.create_2D_axis_old('y', 'z', age=ages[2], index=9, left=col3, bottom=row3, errors=errors)
        self.create_3D_axis_old(age=ages[0], index=10, left=0.0535, bottom=row4, mst=mst)
        self.create_3D_axis_old(age=ages[1], index=11, left=0.381, bottom=row4, mst=mst)
        self.create_3D_axis_old(age=ages[2], index=12, left=0.712, bottom=row4, mst=mst)

        # Set title
        if title:
            self.fig.suptitle(
                '$X-Y$, $X-Z$, $Y-Z$ and 3D scatters at '
                f'{ages[0]:.1f}, {ages[1]:.1f} and {ages[2]:.1f}Myr.\n', fontsize=8
            )

        # Save figure
        self.series.save_figure(
            '2d_3d_scatters_xyz_{}_{}_{}_{}_Myr.pdf'.format(self.name, *ages),
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_2D_axis_old(self, i, j, step=None, age=None, index=1, left=0, bottom=0, errors=False):
        """Creates a 2D axis."""

        # Initialize axis
        ax = self.fig.add_subplot(4, 3, index, position=[left, bottom, 0.255, 0.20])

        # Initialize axes
        axes = {'x': 0, 'y': 1, 'z': 2}
        keys = tuple(axes.keys())

        # Check the type and value of i axis, and initialize axis
        self.series.check_type(i, 'i', 'string')
        self.series.stop(
            i.lower() not in keys, 'ValueError',
            "'i' must be an axis key ('x', 'y' or 'z', {} given).", i
        )
        i = axes[i.lower()]

        # Check the type and value of j axis, and initialize axis
        self.series.check_type(j, 'j', 'string')
        self.series.stop(
            j.lower() not in keys, 'ValueError',
            "'j' must be an axis key ('x', 'y' or 'z', {} given).", j
        )
        j = axes[j.lower()]

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

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
        if errors:
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

    def create_3D_axis_old(self, step=None, age=None, index=1, left=0, bottom=0, mst=False):
        """Creates a 3D axis."""

        # Initialize axis
        ax = self.fig.add_subplot(
            4, 3, index, projection='3d', position=[left, bottom, 0.29, 0.215]
        )

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

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
        if mst:
            if self.series.mst_metrics:
                for branch in self.mst_xyz[step]:
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
            else:
                self.series.log(
                    "Could not display the minimum spanning tree because it was not computed.",
                    level='info', display=True
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

    def create_cross_covariances_scatter_xyz(
        self, i, j, step=None, age=None, errors=False, labels=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """
        Creates a cross covariance scatter of star positions and velocities in i and j at a
        given 'step' or 'age' in Myr. If 'age' doesn't match a step, the closest step is used
        instead. 'age' overrules 'steps' if both are given. 'labels' adds the stars' name
        and 'mst' adds the minimum spanning tree branches.
        """

        # Initialize figure
        fig = plt.figure(figsize=(3.33, 3.33), facecolor=colors.white, dpi=300)
        ax = fig.add_subplot(111)

        # Initialize axes
        position_axes = {'x': 0, 'y': 1, 'z': 2}
        velocity_axes = {'u': 0, 'v': 1, 'w': 2}
        position_keys = tuple(position_axes.keys())
        velocity_keys = tuple(velocity_axes.keys())

        # Check the type and value of i axis, and initialize axis
        self.series.check_type(i, 'i', 'string')
        self.series.stop(
            i.lower() not in position_keys, 'ValueError',
            "'i' must be an postion axis key ('x', 'y' or 'z', {} given).", i
        )
        i = position_axes[i.lower()]

        # Check the type and value of j axis, and initialize axis
        self.series.check_type(j, 'j', 'string')
        self.series.stop(
            j.lower() not in velocity_keys, 'ValueError',
            "'j' must be an postion axis key ('u', 'v' or 'w', {} given).", j
        )
        j = velocity_axes[j.lower()]

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Check the types of errors, labels, mst and title
        self.series.check_type(errors, 'errors', 'boolean')
        self.series.check_type(labels, 'labels', 'boolean')
        self.series.check_type(title, 'title', 'boolean')

        # Plot xyz positions
        ax.scatter(
            [star.position_xyz[step, i] for star in self.sample],
            [star.velocity_xyz[step, j] for star in self.sample],
            marker='o', color=colors.black
        )

        # Plot error bars
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
        if labels:
            for star in self.sample:
                ax.text(
                    star.position_xyz[step, i] + 1, star.velocity_xyz[step, j] + 1,
                    star.name, horizontalalignment='left', fontsize=6
                )

        # Set title
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
            f'covariances_scatter_xyz_{self.name}_'
            f'{position_keys[i].upper()}-{velocity_keys[j].upper()}.pdf',
            forced=forced, default=default, cancel=cancel
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

        # Check the types of labels and title
        self.series.check_type(labels, 'labels', 'boolean')
        self.series.check_type(title, 'title', 'boolean')

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
        if title:
            ax.set_title('Mollweide projection of tracebacks', fontsize=8)

        # Format axis
        ax.grid(zorder=1)

        # Save figure
        self.series.save_figure(
            f'Mollweide_{self.name}.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def create_age_distribution(
        self, metric=None, index=None, title=False,
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

        # Check the type of title
        self.series.check_type(title, 'title', 'boolean')

        # Set title
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
        ax.spines[:].set_linewidth(0.5)

        # Save figure
        self.series.save_figure(
            f'age_distribution_jackknife_{self.name}_{metric_name}.pdf',
            tight=False, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

class Output_Star():
    """Ouput methods of a star."""

    def create_kinematics_time_table(
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

        # Check the types of save, show and machine
        self.group.series.check_type(save, 'save', 'boolean')
        self.group.series.check_type(show, 'show', 'boolean')
        self.group.series.check_type(machine, 'machine', 'boolean')

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
                f'kinematics_time_{self.name}', lines,
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )