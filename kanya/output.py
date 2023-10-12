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

    def set_figure(
        self, style, *options, width=None, height=None,
        left=None, bottom=None, right=None, top=None, styles=None
    ):
        """
        Initializes a figure with multiple axes. The axes are created with at correct position
        and size. Ticks and labels are repositioned and set invisible accordingly.
        """

        def get_dimensions(
            width, height, left, bottom, right, top, colpad, rowpad, nrow, ncol,
            ratio=1.0, adjust=None, h_align='left', v_align='bottom'
        ):
            """
            Computes the dimensions of the axes of a figure based the width and height of the figure,
            and the number of axes along the x and y dimensions. The width of the figure is supposed
            to be fixed. The ratio is the desired value of the height an axis over its width.
            """

            # Compute the width and size of figures, in inches
            ax_width = (width - left - colpad * (ncol - 1) - right) / ncol
            ax_height = (height - bottom - rowpad * (nrow - 1) - top) / nrow

            # Adjust figure width
            if adjust == 'fig_width':
                ax_width = ax_height / ratio
                width = left + ax_width * ncol + colpad * (ncol - 1) + right

            # Adjust figure height
            if adjust == 'fig_height':
                ax_height = ax_width * ratio
                height = bottom + ax_height * nrow + rowpad * (nrow - 1) + top

            # Adjust axis width
            if adjust == 'ax_width' and ax_height / ax_width < ratio:
                ax_width = ax_height / ratio

                # Horizontal alignment
                if h_align == 'right':
                    left = width - right - ax_width * ncol - colpad * (ncol - 1)
                if h_align == 'center':
                    left = (width - ax_width * ncol - colpad * (ncol - 1)) / 2
                if h_align == 'justify':
                    colpad = (width - left - ax_width * x_um - right) / (ncol - 1)

            # Adjust axis height
            if adjust == 'ax_height' and ax_height / ax_width > ratio:
                ax_height = ax_width * ratio

                # Horizontal alignment
                if v_align == 'top':
                    bottom = height - top - ax_height * nrow - rowpad * (nrow - 1)
                if v_align == 'center':
                    bottom = (height - ax_height * nrow - rowpad * (nrow - 1)) / 2
                if v_align == 'justify':
                    rowpad = (height - bottom - ax_height * y_um - top) / (nrow - 1)

            # Compute relative parameters
            left /= width
            bottom /= height
            ax_width /= width
            ax_height /= height
            colpad /= width
            rowpad /= height

            # Compute axes parameters
            axes_params = np.array(
                [
                    [
                        [
                            left + i * (ax_width + colpad),
                            bottom + j * (ax_height + rowpad),
                            ax_width, ax_height
                        ] for i in list(range(ncol))
                    ] for j in list(range(nrow))[::-1]
                ]
            )

            return width, height, axes_params

        def get_axes(fig, axes_params, remove_extra=True, **kwargs):
            """Add axes to the figure using the axes parameters."""

            # Add dimensions and create an empty array
            while axes_params.ndim != 3:
                axes_params = axes_params[None]
            axes = np.empty((axes_params.shape[:2]), dtype=object)

            # Create axes
            for i in range(axes_params.shape[0]):
                for j in range(axes_params.shape[1]):
                    axes[i, j] = fig.add_axes(axes_params[i,j], **kwargs)

            # Remove extra dimensions
            if remove_extra:
                while axes.shape[0] == 1 and axes.ndim > 1:
                    axes = axes[0]

                return axes[0] if axes.size == 1 else axes

            # Return axes with all dimensions
            else:
                return axes

        def set_axis(ax, projection):
            """Sets the basic parameter of a 2d or 3d axis."""

            # Set ticks
            if projection == '2d':
                ax.tick_params(
                    top=True, right=True, which='both',
                    direction='in', width=0.5, labelsize=8
                )

                # Set spines
                ax.spines[:].set_linewidth(0.5)

            # Set ticks
            if projection == '3d':
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
                ax.view_init(azim=45)

        # Check traceback
        self.check_traceback()

        # Check the type and value of style
        styles = styles if styles is not None else (
            '2+1',
            '1x1', '1x2', '1x3',
            '2x1', '2x2', '2x3',
            '3x1', '3x2', '3x3',
            '4x1', '4x2', '4x3'
        )
        self.check_type(style, 'style', 'string')
        self.stop(
            style not in styles, 'ValueError',
            "'style' must be {} ({} given).", enumerate_strings(*styles), style
        )

        # Number of axes in y and x
        nrow = int(style[0])
        ncol = int(style[-1])

        # Default margins, in inches
        left = 0.445 if left is None else left
        bottom = 0.335 if bottom is None else bottom
        right = 0.005 if right is None else right
        top = 0.0815 if top is None else top
        colpad = 0.100 if 'hide_y' in options else 0.5400
        rowpad = 0.100 if 'hide_x' in options else 0.3654

        # Set margins if the axes and tick labels are moved to the right of the rightmost axes
        if 'label_right' in options and ncol == 2:
            right = left
            colpad = 0.100

        # Set margins for a figure with 3d axes
        if '3d' in options:
            left = right = 0.6

        # Initialize a 2+1 figure
        if style == '2+1':
            fig = plt.figure(figsize=(3.345, 3.3115), facecolor=colors.white, dpi=300)

            # Set dimensions
            left, bottom = (0.133, 0.102)
            x_short, y_short  = (0.236, 0.2381)
            x_long, y_long = (0.618, 0.6236)
            colpad, rowpad = (0.012, 0.0119)

            # Create axes
            axes = np.empty((2,2), dtype=object)
            axes[0,0] = fig.add_axes([left, bottom + rowpad + y_short, x_long, y_long])
            axes[0,1] = fig.add_axes([left + x_long + colpad, bottom + rowpad + y_short, x_short, y_long])
            axes[1,0] = fig.add_axes([left, bottom, x_long, y_short])

            # Set axes
            axes[0,0].set_xlabel('', visible=False)
            axes[0,1].set_ylabel('', visible=False)
            axes[0,0].tick_params(labelbottom=False)
            axes[0,1].tick_params(labelleft=False)

            # Set title
            fig.joint = ''
            fig.offset = 1.05

        # Initialize a 1x1 figure
        if style == '1x1':
            width, height, axes_params = get_dimensions(
                3.3450 if width is None else width,
                3.3115 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params, remove_extra=False)

        # Initialize a 1x2 figure
        if style == '1x2':
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                3.3115 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params, remove_extra=False)

        # Initialize a 1x3 figure
        if style == '1x3':
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                2.3330 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params, remove_extra=False)

        # Initialize a 2x1 figure
        if style == '2x1':
            width, height, axes_params = get_dimensions(
                3.3450 if width is None else width,
                6.5720 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params)

        # Initialize a 2x2 figure
        if style == '2x2':
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                6.317 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            axes[0,:] = get_axes(fig, axes_params[0,:], zorder=0.5)
            axes[1,0] = get_axes(fig, axes_params[1,0], zorder=0.5)
            axes[1,1] = get_axes(
                fig, axes_params[1,1], projection='3d' if '3d' in options else None, zorder=0.4
            )

            if '3d' in options:
                set_axis(axes[1,1], '3d')
                if 'label_right' in options or 'hide_y' in options:
                    axes[1,1].view_init(azim=-45)

        # Initialize a 2x3 figure
        if style == '2x3':
            # left = 0.5300
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                4.4450 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params)

        # Initialize a 3x1 figure
        if style == '3x1':
            # width, height = (3.3450, 7.5000)
            width, height, axes_params = get_dimensions(
                3.3450 if width is None else width,
                9.1340 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='ax_width', h_align='center'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params)

        # Initialize a 3x2 figure
        if style == '3x2':
            # left = 0.5300
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                7.5023 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='ax_width', h_align='center'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            axes[:2] = get_axes(fig, axes_params[:2], zorder=0.5)
            axes[2] = get_axes(
                fig, axes_params[2], projection='3d' if '3d' in options else None, zorder=0.4
            )

            # Set 3d axes
            if '3d' in options:
                for ax in axes[2]:
                    set_axis(ax, '3d')
                if 'label_right' in options:
                    axes[2,1].view_init(azim=-45)

        # Initialize a 3x3 figure
        if style == '3x3':
            # left = 0.5300
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                7.5023 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params)

        # Initialize a 4x1 figure
        if style == '4x1':
            # width, height = (3.3450, 7.5000)
            width, height, axes_params = get_dimensions(
                3.3450 if width is None else width,
                9.1340 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='ax_width', h_align='center'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            axes[:3,0] = get_axes(fig, axes_params[:3,0], zorder=0.5)
            axes[3,0] = get_axes(
                fig, axes_params[3,0], projection='3d' if '3d' in options else None, zorder=0.4
            )

            if '3d' in options:
                set_axis(axes[3,0], '3d')

        # Initialize a 4x2 figure
        if style == '4x2':
            left = right = 0.600
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                9.0994 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            axes[:3] = get_axes(fig, axes_params[:3], zorder=0.5)
            axes[3] = get_axes(
                fig, axes_params[3], projection='3d' if '3d' in options else None, zorder=0.4
            )

            # Set 3d axes
            if '3d' in options:
                for ax in axes[3]:
                    set_axis(ax, '3d')
                if 'label_right' in options:
                    axes[3,1].view_init(azim=-45)

        # Initialize a 4x3 figure
        if style == '4x3':
            left = right = 0.600
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                9.0994 if height is None else height,
                left, bottom, right, top, colpad, rowpad,
                nrow, ncol, adjust='fig_height'
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            axes[:3] = get_axes(fig, axes_params[:3], zorder=0.5)
            axes[3] = get_axes(
                fig, axes_params[3], projection='3d' if '3d' in options else None, zorder=0.4
            )

            if '3d' in options:
                for ax in axes[3]:
                    set_axis(ax, '3d')

        # Hide axis and tick labels, if needed
        if 'hide_x' in options and nrow > 1:
            for ax in axes[:nrow - 1 - (1 if '3d' in options else 0)].flatten():
                ax.set_xlabel('', visible=False)
                ax.tick_params(labelbottom=False)
        if 'hide_y' in options and ncol > 1:
            if not (ncol == 2 and 'label_right' in options):
                for ax in axes[:nrow - (1 if style == '2x2' and '3d' in options else 0),1:].flatten():
                    ax.set_ylabel('', visible=False)
                    ax.tick_params(labelleft=False)

        # Move axes and tick labels to the right of the rightmost axes
        if 'label_right' in options and ncol == 2:
            for ax in axes[:nrow - (1 if '3d' in options else 0),1].flatten():
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()

        # Adjust specific parameters for 2x2 style:
        if style == '2x2' and '3d' in options and 'hide_x' in options:
            axes[0,0].set_xlabel('', visible=False)
            axes[0,0].tick_params(labelbottom=False)

        # Set ticks and spines
        for ax in filter(lambda ax: ax is not None, axes.flatten()):
            set_axis(ax, '2d')

        # Set title joint and offset
        if ncol == 1:
            fig.joint = '\n'
            fig.offset = 1.03
        else:
            fig.joint = ' '
            fig.offset = 1.025

        return fig, axes

    def set_legend(self, ax, loc):
        """Sets the parameters of the legend of the axis."""

        legend = ax.legend(loc=loc, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor(colors.white + (0.8,))
        legend.get_frame().set_edgecolor(colors.black)
        legend.get_frame().set_linewidth(0.5)

    def set_title(self, fig, title, line_1, line_2=''):
        """Sets a title for the figure, if 'title' is True."""

        # Check the type of title
        self.check_type(title, 'title', 'boolean')
        print(line_1, line_2)

        # Set title
        if title:
            fig.suptitle(
                f'{line_1} of stars in {self.name}'
                f"{fig.joint if line_2 != '' else ''}{line_2}",
                y=fig.offset, fontsize=8
            )

    def select_metric(self, metric, system, robust=False, sklearn=False):
        """
        Checks if 'robust' and 'sklearn' arguments are valid and select the corresponding
        association size metrics based on the system.
        """

        # Check the type and value of system
        self.check_type(system, 'system', 'string')
        self.stop(
            system not in ('xyz', 'ξηζ'), 'ValueError',
            "'system' can only take as value 'xyz' or 'ξηζ' ({} given).", system
        )

        # Check the types of robust and sklearn
        self.check_type(robust, 'robust', 'boolean')
        self.check_type(sklearn, 'sklearn', 'boolean')
        self.stop(
            robust and sklearn, 'ValueError',
            "'robust' and 'sklearn' cannot both be True."
        )
        self.stop(
            metric == 'cross_covariance' and sklearn, 'ValueError',
            "'sklearn' cross covariances are not computed."
        )

        return f'{metric}_{system}', '_sklearn' if sklearn else '_robust' if robust else ''

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

        # Set legend
        self.set_legend(ax, 2)

        # Select units
        units_y = ' (pc)'
        if metric.label[:5] == 'cross':
            units_y = ' (pc/Myr$^{1/2}$)'
        if metric.label[:11] == 'mahalanobis':
            units_y = ''

        # Set labels
        ax.set_xlabel('Epoch (Myr)', fontsize=8)
        ax.set_ylabel(f'Association size{units_y}', fontsize=8)

        # Set limits
        ax.set_xlim(self.final_time.value + 1, self.initial_time.value + 1)

        # Set ticks
        # ax.set_xticks([0., -5., -10., -15., -20., -25., -30., -35., -40, -45, -50])
        # ax.set_yticks([0.,  5.,  10.,  15.,  20.,  25.,  30.,  35.])

    def set_title_metric(self, ax, title, metric):
        """Sets a title for association size metrics plots if 'title' is True."""

        # Check the type of title
        self.check_type(title, 'title', 'boolean')
        print(f'{metric} of {self.name}')

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

    def draw_covariances(
        self, system, robust=False, sklearn=False, title=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of the covariances, and the determinant and the trace of the covariances
        matrix. If either 'robust' or 'sklearn' is True, the robust or sklearn covariances matrix
        is used. Otherwise, the empirical covariances matrix is used.
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)
        ax = axes[0,0]

        # Select association size metrics
        covariances, selection = (
            self.select_metric(
                'covariances', system, robust=robust, sklearn=sklearn
            )
        )
        covariances_matrix_det = vars(self)[f'{covariances}_matrix_det{selection}']
        covariances_matrix_trace = vars(self)[f'{covariances}_matrix_trace{selection}']
        covariances = vars(self)[f'{covariances}{selection}']

        # Plot covariance matrix determinant and trace, and covariances
        self.plot_metric(ax, covariances_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, covariances_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax, covariances, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, covariances, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, covariances, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(
            ax, title,
            f"${systems[system].latex['position']}${selection.replace('_', ' ')} covariances"
        )

        # Save figure
        self.save_figure(
            f'covariances_{system}_{self.name}{selection}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_cross_covariances(
        self, system, robust=False, sklearn=False, title=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot the cross covariances between positions and velocities, and the determinant
        and the trace of the cross covariances matrix between positions and velocities. If either
        'robust' or 'sklearn' is True, the robust or sklearn covariances matrix is used. Otherwise,
        the empirical covariances matrix is used.
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)
        ax = axes[0,0]

        # Select association size metrics
        cross_covariances, selection = (
            self.select_metric(
                'cross_covariances', system, robust=robust, sklearn=sklearn
            )
        )
        cross_covariances_matrix_det = vars(self)[f'{cross_covariances}_matrix_det{selection}']
        cross_covariances_matrix_trace = vars(self)[f'{cross_covariances}_matrix_trace{selection}']
        cross_covariances = vars(self)[f'{cross_covariances}{selection}']

        # Plot the cross covariance matrix determinant and trace, and cross covariances
        self.plot_metric(ax, cross_covariances_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, cross_covariances_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax, cross_covariances, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, cross_covariances, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, cross_covariances, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(
            ax, title,
            f"${systems[system].latex['position']}$ and "
            f"${systems[system].latex['velocity']}${selection.replace('_', ' ')} covariances"
        )

        # Save figure
        self.save_figure(
            f'cross_covariances_{system}_{self.name}{selection}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_mad(self, system, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the total median absolute deviation (MAD), the components of the MAD.
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)
        ax = axes[0,0]

        # Select association size metrics
        mad = self.select_metric('mad', system)[0]
        mad_total = vars(self)[f'{mad}_total']
        mad = vars(self)[mad]

        # Plot the total median aboslute deviation (MAD), and the components of the MAD
        self.plot_metric(ax, mad_total, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, mad, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, mad, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, mad, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax, title, f"${systems[system].latex['position']}$ MAD")

        # Save figure
        self.save_figure(
            f'mad_{system}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_mst(self, system, title=False, forced=False, default=False, cancel=False):
        """
        Creates a plot of the mean branch length (both empirical and robust) and median absolute
        deviation of the minimum spanning tree (MST).
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)
        ax = axes[0,0]

        # Select association size metrics
        mst = self.select_metric('mst', system)[0]

        # Plot the mean and median absolute deviation of the minimum spanning tree
        self.plot_metric(ax, vars(self)[f'{mst}_mean'], 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, vars(self)[f'{mst}_mean_robust'], 0, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, vars(self)[f'{mst}_mad'], 0, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(ax, title, f"${systems[system].latex['position']}$ MST")

        # Save figure
        self.save_figure(
            f'mst_{system}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_mahalanobis(self, system, title=False, forced=False, default=False, cancel=False):
        """Creates a plot of the mean and median Mahalanobis distance."""

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)
        ax = axes[0,0]

        # Select association size metrics
        mahalanobis = self.select_metric('mahalanobis', system)[0]

        # Plot mean and median Mahalanobis distance
        self.plot_metric(ax, vars(self)[f'{mahalanobis}_mean'], 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, vars(self)[f'{mahalanobis}_median'], 0, colors.metric[6], '--', 0.7)

        # Set title
        self.set_title_metric(
            ax, title, f"${systems[system].latex['position']}$ Mahalanobis distance"
        )

        # Set labels and limits
        ax.set_ylabel(ax.get_ylabel(), labelpad=3)
        ax.set_ylim(-0.1, 2.9)

        # Save figure
        self.save_figure(
            f'mahalanobis_{system}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_covariances_mad(
        self, system, robust=False, sklearn=False, title=False,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a plot of the covariances, and the determinant and the trace of the covariances
        matrix, the total median absolute deviation (MAD), and the components of the MAD.
        """

        # Initialize figure
        fig, axes = self.set_figure('2x1', 'hide_x', left=0.3430)
        ax1, ax2 = (axes[0,0], axes[1,0])

        # Select association size metrics (covariances)
        covariances, selection = (
            self.select_metric(
                'covariances', system, robust=robust, sklearn=sklearn
            )
        )
        covariances_matrix_det = vars(self)[f'{covariances}_matrix_det{selection}']
        covariances_matrix_trace = vars(self)[f'{covariances}_matrix_trace{selection}']
        covariances = vars(self)[f'{covariances}{selection}']

        # Select association size metrics (covariances)
        mad = self.select_metric('mad', system)[0]
        mad_total = vars(self)[f'{mad}_total']
        mad = vars(self)[mad]

        # Plot covariance matrix determinant and trace, and covariances
        self.plot_metric(ax1, covariances_matrix_det, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax1, covariances_matrix_trace, 0, colors.metric[1], '--', 0.6)
        self.plot_metric(ax1, covariances, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax1, covariances, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax1, covariances, 2, colors.metric[7], ':', 0.5)

        # Plot the total median aboslute deviation (MAD), and the components of the MAD
        self.plot_metric(ax2, mad_total, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax2, mad, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax2, mad, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax2, mad, 2, colors.metric[7], ':', 0.5)

        # Set title
        self.set_title_metric(
            ax1, title, f"${systems[system].latex['position']}$ covariances and MAD"
        )

        # Save figure
        self.save_figure(
            f'covariances_mad_{system}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_det_mad_mst_cross_covariances_xyz(
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
            ncols=2, constrained_layout=True, figsize=(6.66, 3.33), facecolor=colors.white, dpi=300
        )

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

        # Set legend
        for ax in (ax0, ax1):
            self.set_legend(ax)

            # Set labels
            ax.set_xlabel('Epoch (Myr)', fontsize=8)
            ax.set_ylabel(f'Association size ({units_y})', fontsize=8)

            # Set limits
            ax.set_xlim(self.final_time.value + 1, self.initial_time.value + 1)

            # Set ticks
            ax.tick_params(
                top=True, right=True, which='both', direction='in', width=0.5, labelsize=8
            )

            # Set spines
            ax.spines[:].set_linewidth(0.5)

        # Save figure
        self.save_figure(
            f'covariances_mad_mst_cross_covariannces_xyz_{self.name}_{other.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_age_distribution(self, title=False, forced=False, default=False, cancel=False):
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

class Output_Group():
    """Output methods of a group of stars."""

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

    def plot_trajectory(self, ax, x, y, coord, system, age, metric, index, labels):
        """Draws the trajectory of stars in the group."""

        # Check the type and value of x and y
        self.series.check_type(x, 'x', 'integer')
        self.series.stop(
            x < 0 or x > 2, 'ValueError',
            "'x' must be 0, 1, or 2 ({} given).", x
        )
        self.series.check_type(y, 'y', 'integer')
        self.series.stop(
            y < 0 or y > 2, 'ValueError',
            "'y' must be 0, 1, or 2 ({} given).", y
        )

        # Check the type and value of coord
        self.series.check_type(coord, 'coord', 'string')
        self.series.stop(
            coord not in ('position', 'velocity'),
            'ValueError', "'coord' must be 'position' or 'velocity' ({} given).", coord
        )

        # Check the type and value of system
        self.series.check_type(system, 'system', 'string')
        self.series.stop(
            system not in ('xyz', 'ξηζ'), 'ValueError',
            "'system' can only take as value 'xyz' or 'ξηζ' ({} given).", system
        )

        # Check the type of labels
        self.series.check_type(labels, 'labels', 'boolean')

        # Birth index, age and age error
        birth_index, age, age_error = self.get_epoch(age=age, metric=metric, index=index)

        # Select conversion factors from pc to kpc
        factors = np.ones(3)
        if coord == 'position' and system == 'xyz':
            factors = np.array([1000, 1000, 1])

        # Select coordinates
        for star in self:
            value = vars(star)[f'{coord}_{system}'] / factors

            # Plot stars' values
            color = colors.red[6] if star.outlier else colors.black
            ax.plot(
                value.T[x], value.T[y],
                color=color, alpha=0.6, linewidth=0.5,
                solid_capstyle='round', zorder=0.1
            )

            # Plot stars' current values
            if self.series.from_data:
                ax.scatter(
                    np.array([value[0,x]]), np.array([value[0,y]]),
                    color=colors.black + (0.4,), edgecolors=colors.black,
                    alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                )

                # Plot stars' birth values
                if birth_index is not None:
                    color = colors.red[6] if star.outlier else colors.blue[6]
                    ax.scatter(
                        np.array([value[birth_index,x]]),
                        np.array([value[birth_index,y]]),
                        color=color + (0.4,), edgecolors=color + (1.0,),
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                # Show stars' names
                if labels:
                    ax.text(
                        np.array(value.T[0,x]), np.array(value.T[0,y]),
                        star.name, horizontalalignment='left', fontsize=6
                    )

        # Select coordinates
        if self.series.from_model:
            average_model_star_value = vars(self.average_model_star)[f'{coord}_{system}'] / factors

            # Plot the average model star's trajectory
            ax.plot(
                average_model_star_value.T[x],
                average_model_star_value.T[y],
                color=colors.green[6], alpha=0.8,
                linewidth=0.5, solid_capstyle='round', zorder=0.3
            )

            # Plot the average model star's birth and current values
            for t, size, marker in ((-1, 12, '*'), (0, 6, 'o')):
                ax.scatter(
                    np.array([average_model_star_value[t,x]]),
                    np.array([average_model_star_value[t,y]]),
                    color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                    alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                )

            # Select coordinates
            for star in self.model_stars:
                value = vars(star)[f'{coord}_{system}'] / factors

                # Plot model stars' trajectories
                ax.plot(
                    value.T[x], value.T[y],
                    color=colors.blue[6], alpha=0.6,
                    linewidth=0.5, solid_capstyle='round', zorder=0.2
                )

                # Plot model stars' birth and current values
                for t, size, marker in ((0, 12, '*'), (-1, 6, 'o')):
                    ax.scatter(
                        np.array([value[t,x]]),
                        np.array([value[t,y]]),
                        color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                    )

        # Set labels
        axis_unit = vars(systems[system])[coord][x].unit.label
        if coord == 'position' and system == 'xyz' and x != 2:
            axis_unit = 'kpc'
        ax.set_xlabel(f'${vars(systems[system])[coord][x].latex}$ ({axis_unit})', fontsize=8)
        axis_unit = vars(systems[system])[coord][y].unit.label
        if coord == 'position' and system == 'xyz' and y != 2:
            axis_unit = 'kpc'
        ax.set_ylabel(f'${vars(systems[system])[coord][y].latex}$ ({axis_unit})', fontsize=8)

    def draw_trajectory(
        self, coord, system, age=None, metric=None, index=None, labels=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """Draws the trajectories of stars in the group."""

        # Initialize figure
        fig, axes = self.series.set_figure('2+1')

        # Plot trajectories
        if system == 'xyz':
            i, j = zip(*((1, 0), (2, 0), (1, 2)))
        if system == 'ξηζ':
            i, j = zip(*((0, 1), (2, 1), (0, 2)))
        for ax, x, y in zip(axes.flatten(), i, j):
            self.plot_trajectory(ax, x, y, coord, system, age, metric, index, labels)

        # Set title
        self.series.set_title(fig, title, (f'${systems[system].latex[coord]}$ trajectories'))

        # Draw vertical and horizontal lines through the Sun's position at the current epoch
        if coord == 'position' and system == 'xyz':
            for ax, x, y in zip(axes.flatten(), i, j):
                ax.axhline(
                    0.0, color=colors.black, alpha=0.8,
                    linewidth=0.5, linestyle=':', zorder=0.2
                )
                ax.axvline(
                    0.0, color=colors.black, alpha=0.8,
                    linewidth=0.5, linestyle=':', zorder=0.2
                )

            # Draw circles around the galactic center located at 8.122 kpc from the Sun
            for radius in range(1, 16):
                axes[0,0].add_artist(
                    plt.Circle(
                        (0, 8.122), radius, color=colors.grey[17],
                        fill=False, linewidth=0.5, linestyle=':', zorder=0.1
                    )
                )

            # Set limits
            axes[0,0].set_xlim(-9, 1)
            axes[0,0].set_ylim(-1, 9)
            axes[0,1].set_xlim(-80, 80)
            axes[0,1].set_ylim(-1, 9)
            axes[1,0].set_xlim(-9, 1)
            axes[1,0].set_ylim(-80, 80)

        # Invert y axis
        if system == 'xyz':
            axes[0,0].invert_xaxis()
            axes[1,0].invert_xaxis()

        # Set limits
        # if coord == 'position' and system == 'ξηζ':
        #     axes[0,0].set_xlim(-225, 60)
        #     axes[0,0].set_ylim(-45, 110)
        #     axes[0,1].set_xlim(-40, 49)
        #     axes[0,1].set_ylim(-45, 110)
        #     axes[1,0].set_xlim(-225, 60)
        #     axes[1,0].set_ylim(-40, 49)

        # Save figure
        self.series.save_figure(
            f'trajectory_{coord}_{system}_{self.name}.pdf',
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def plot_time(self, ax, y, coord, system, age, metric):
        """Draws the y coordinate of stars in the group over time."""

        # Check the type and value of y
        self.series.check_type(y, 'y', ('integer', 'None'))
        if y is not None:
            self.series.stop(
                y < 0 or y > 2, 'ValueError',
                "'y' must be 0, 1, 2 or None ({} given).", y
            )

        # Check the type and value of coord
        self.series.check_type(coord, 'coord', 'string')
        self.series.stop(
            coord not in ('position', 'velocity'),
            'ValueError', "'coord' must be 'position' or 'velocity' ({} given).", coord
        )

        # Check the type and value of system
        self.series.check_type(system, 'system', 'string')
        self.series.stop(
            system not in ('xyz', 'ξηζ'), 'ValueError',
            "'system' can only take as value 'xyz' or 'ξηζ' ({} given).", system
        )

        # Birth index, age and age error
        birth_index, age, age_error  = tuple(
            zip(*[self.get_epoch(age=age, metric=metric, index=index) for index in range(3)])
        )

        # Select coordinates
        if y in (0, 1, 2):
            for star in self:
                value = vars(star)[f'relative_{coord}_{system}']

                # Plot stars' trajectories
                ax.plot(
                    self.series.time, value[:,y],
                    color = colors.red[6] if star.outlier else colors.black, alpha=0.6,
                    linewidth=0.5, solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current values
                if self.series.from_data:
                    ax.scatter(
                        self.series.time[0], value[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black, alpha=None,
                        s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth values
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax.scatter(
                            self.series.time[birth_index[y]], value[birth_index[y],y],
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
                average_value = np.mean(
                    [vars(star)[f'{coord}_{system}'][:,y] for star in self.model_stars], axis=0
                )
                average_model_star_value = (
                    vars(self.average_model_star)[f'{coord}_{system}'][:,y] - average_value[::-1]
                )

                # Plot the average model star's value
                ax.plot(
                    self.series.model_time,
                    average_model_star_value,
                    color=colors.green[6], alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current values
                for t, x, size, marker in ((-1, -1, 10, '*'), (0, 0, 6, 'o')):
                    ax.scatter(
                        np.array([self.series.model_time[t]]),
                        np.array([average_model_star_value[x]]),
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Select coordinates
                for star in self.model_stars:
                    model_star_value = vars(star)[f'{coord}_{system}'][:,y] - average_value

                    # Plot model stars' values
                    ax.plot(
                        self.series.model_time[::-1], model_star_value,
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current values
                    for t, x, size, marker in ((-1, 0, 10, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_value[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

            # Set label
            ax.set_ylabel(
                f'${vars(systems[system])[coord][y].latex} - <{vars(systems[system])[coord][y].latex}>$ '
                f'({vars(systems[system])[coord][y].unit.label})', fontsize=8
            )

        # Select conversion factor from pc to kpc
        if y is None:
            factor = 1000 if coord == 'position' and system == 'xyz' else 1

            # Select coordinates
            value = vars(self)[f'{coord}_{system}'] / factor

            # Plot stars' average values
            for y, linestyle in ((0, '-'), (1, '--'), (2, ':')):
                ax.plot(
                    self.series.time, value[:,y],
                    label=f'$<{vars(systems[system])[coord][y].latex}>$', color=colors.black,
                    alpha=0.8, linestyle=linestyle, linewidth=1.0, solid_capstyle='round',
                    dash_capstyle='round', zorder=0.1
                )

                # Plot stars' average current values
                if self.series.from_data:
                    ax.scatter(
                        self.series.time[0], value[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' average birth values
                    if birth_index[y] is not None:
                        ax.scatter(
                            self.series.time[birth_index[y]],
                            value[birth_index[y],y],
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6] + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                # Select coordinates
                if self.series.from_model:
                    average_model_star_value = (
                        vars(self.average_model_star)[f'{coord}_{system}'][:,y]
                    ) / factor

                    # Plot the average model star's value
                    ax.plot(
                        self.series.model_time, average_model_star_value,
                        color=colors.green[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.3
                    )

                    # Plot the average model star's birth and current values
                    for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                        ax.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([average_model_star_value[x]]),
                            color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                        )

                    # Select coordinates
                    model_star_value = np.mean(
                        [vars(star)[f'{coord}_{system}'][:,y] for star in self.model_stars], axis=0
                    ) / factor

                    # Plot model stars' values
                    ax.plot(
                        self.series.model_time[::-1], model_star_value,
                        color=colors.blue[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current values
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            np.array([self.series.model_time[t]]),
                            np.array([model_star_value[x]]),
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6],
                            alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.2
                        )

            # Set label
            axis_unit = vars(systems[system])[coord][0].unit.label
            if coord == 'position' and system == 'xyz':
                axis_unit = 'kpc'
            ax.set_ylabel(f'$<{systems[system].latex[coord]}>$ ({axis_unit})', fontsize=8)

            # Set legend
            self.series.set_legend(ax, 4)

        # Set label
        ax.set_xlabel('Epoch (Myr)', fontsize=8)

        # Set limits
        ax.set_xlim(np.min(self.series.time), np.max(self.series.time) + 1)

    def draw_time(
        self, coord, system, style, age=None, metric=None, title=False,
        forced=False, default=False, cancel=False
    ):
        """Draws the positions or velocities of stars over time."""

        # Initialize figure
        fig, axes = self.series.set_figure(style, 'hide_x', 'label_right', styles=('1x3', '2x2', '3x1'))

        # Plot xyz position
        for y in range(3):
            self.plot_time(axes.flatten()[y], y, coord, system, age, metric)

        # Plot the average xyz position
        if style == '2x2':
            self.plot_time(axes[1,1], None, coord, system, age, metric)

        self.series.set_title(
            fig, title, (
                f'${systems[system].latex[coord]}$ {coord}s'.replace('citys', 'cities')
            ), 'over time'
        )

        # Save figure
        self.series.save_figure(
            f"{coord}_{system}_{style}_{self.name}_time.pdf",
            tight=title, forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def plot_scatter(
        self, ax, axes, coords, system, step, errors=False,
        labels=False, mst=False, relative=False
    ):
        """Creates a 2d or 3d scatter of positions or velocities, at a given step."""

        # Check the type and value of axes
        self.series.check_type(axes, 'axes', 'tuple')
        self.series.stop(
            len(axes) not in (2, 3), 'ValueError',
            "'axes' must have a length of 2 or 3 ({} given).", len(axes)
        )
        for i in axes:
            self.series.check_type(i, 'i value in axes', 'integer')
            self.series.stop(
                i < 0 or i > 2, 'ValueError',
                "All values in axes must 0, 1 or 2 ({} given).", i
            )

        # Select projection
        if len(axes) == 2:
            projection = '2d'
        if len(axes) == 3:
            projection = '3d'

        # Check the type and value of coords
        self.series.check_type(coords, 'coords', ('tuple', 'list', 'string'))
        coords = [coords] if type(coords) not in (tuple, list) else coords
        self.series.stop(
            len(coords) not in (1, 2), 'ValueError',
            "'coords' must have a length of 1 or 2 ({} given).", len(coords)
        )
        for coord in coords:
            self.series.check_type(coord, 'coord', 'string')
            self.series.stop(
                coord not in ('position', 'velocity'),
                'ValueError', "'coord' must be 'position' or 'velocity' ({} given).",
                coord
            )

        # Adjust the length of coords, if needed
        if len(coords) == 1 and projection == '2d':
            coords *= 2
        if projection == '3d':
            self.series.stop(
                len(coords) == 2, 'ValueError', "'coords' must have a length of 1 "
                "if the projection is '3d' ({} given).", len(coords)
            )
            coords *= 3

        # Check the type and value of system
        self.series.check_type(system, 'system', 'string')
        self.series.stop(
            system not in ('xyz', 'ξηζ'), 'ValueError',
            "'system' can only take as value 'xyz' or 'ξηζ' ({} given).", system
        )

        # Check the types of errors, labels, mst, relative and view
        self.series.check_type(errors, 'errors', 'boolean')
        self.series.check_type(labels, 'labels', 'boolean')
        self.series.check_type(mst, 'mst', 'boolean')
        self.series.check_type(relative, 'relative', 'boolean')

        # Check if the minimum spanning tree can be displayed
        self.series.stop(
            mst and coords[0] != coords[1], 'ValueError', 'The minimum spanning tree cannot be '
            'displayed is coordinates do not match ({} and {} given).', coords[0], coords[1]
        )
        self.series.stop(
            mst and not self.series.mst_metrics, 'ValueError',
            'The minimum spanning tree cannot be displayed if it was not computed.'
        )

        # Value and error coordinates
        value_coords = [f'{coords[i]}_{system}' for i in range(len(coords))]
        error_coords = [f'{coords[i]}_{system}_error' for i in range(len(coords))]

        # Select coordinates
        for star in self.sample:
            value = [
                vars(star)[('relative_' if relative else '') + value_coords[i]][step, axes[i]]
                for i in range(len(axes))
            ]
            error = [
                vars(star)[error_coords[i]][step].diagonal()[axes[i]] for i in range(len(axes))
            ]

            # Select color
            color = colors.red[9] if star.outlier else colors.black

            # Plot value
            if projection == '2d':
                ax.scatter(
                    value[0], value[1],
                    color=color + (0.4,), edgecolors=color,
                    alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.4
                )
            if projection == '3d':
                ax.scatter(
                    value[0], value[1], value[2],
                    color=color + (0.4,), edgecolors=color,
                    alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.4
                )

            # Plot error bars
            if errors:
                if projection == '2d':
                    ax.plot(
                        (value[0] - error[0], value[0] + error[0]),
                        (value[1], value[1]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                    ax.plot(
                        (value[0], value[0]),
                        (value[1] - error[1], value[1] + error[1]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                if projection == '3d':
                    ax.plot(
                        (value[0] - error[0], value[0] + error[0]),
                        (value[1], value[1]), (value[2], value[2]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                    ax.plot(
                        (value[0], value[0]), (value[1] - error[1], value[1] + error[1]),
                        (value[2], value[2]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )
                    ax.plot(
                        (value[0], value[0]), (value[1], value[1]),
                        (value[2] - error[1], value[2] + error[1]),
                        color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                    )

            # Show labels
            if labels:
                if projection == '2d':
                    ax.text(
                        value[0] + 1, value[1] + 1, star.name,
                        color=color, horizontalalignment='left',
                        verticalaligment='top', fontsize=6, zorder=0.9
                    )
                if projection == '3d':
                    ax.text(
                        value[0] + 2, value[1] + 2, value[2] + 2, star.name,
                        color=color, horizontalalignment='left',
                        verticalaligment='top', fontsize=6, zorder=0.9
                    )


        # Select branches
        if mst:
            for branch in self.mst_xyz[step]:
                value_start = [
                    vars(branch.start)[
                        ('relative_' if relative else '') + value_coords[i]
                    ][step, axes[i]] for i in range(len(axes))
                ]
                value_end = [
                    vars(branch.end)[
                        ('relative_' if relative else '') + value_coords[i]
                    ][step, axes[i]] for i in range(len(axes))
                ]

                # Plot branches
                if projection == '2d':
                    ax.plot(
                        (value_start[0], value_end[0]),
                        (value_start[1], value_end[1]),
                        color=colors.blue[6], alpha=0.6, linestyle='-',
                        linewidth=0.5, solid_capstyle='round', zorder=0.4
                    )
                if projection == '3d':
                    ax.plot(
                        (value_start[0], value_end[0]),
                        (value_start[1], value_end[1]),
                        (value_start[2], value_end[2]),
                        color=colors.blue[6], alpha=0.6, linestyle='-',
                        linewidth=0.5, solid_capstyle='round', zorder=0.4
                    )

        # Set labels
        ax.set_xlabel(
            f'${vars(systems[system])[coords[0]][axes[0]].latex}$ '
            f'({vars(systems[system])[coords[0]][axes[0]].unit.label})',
            fontsize=8, #labelpad=16
        )
        ax.set_ylabel(
            f'${vars(systems[system])[coords[1]][axes[1]].latex}$ '
            f'({vars(systems[system])[coords[1]][axes[1]].unit.label})',
            fontsize=8, #labelpad=16
        )
        if projection == '3d':
            ax.set_zlabel(
                f'${vars(systems[system])[coords[2]][axes[2]].latex}$ '
                f'({vars(systems[system])[coords[2]][axes[2]].unit.label})',
                fontsize=8, #labelpad=20
            )

    def draw_scatter(
        self, coord, system, style, step=None, age=None, errors=False, labels=False,
        mst=False, title=False, forced=False, default=False, cancel=False
    ):
        """
        Draws scatter plots of positions or velocities of stars at a given 'step' or 'age' in Myr.
        If 'age' doesn't match a step, the closest step is used instead. 'age' overrules 'steps'
        if both are given. 'errors', adds error bars, 'labels' adds the stars' names and 'mst' adds
        the minimun spanning tree branches.
        """

        # Initialize figure
        fig, axes = self.series.set_figure(
            style, 'label_right', 'hide_x' if style == '2x2' else '',
            '3d' if style in ('2x2', '4x1') else '', styles=('1x3', '2x2', '3x1', '4x1')
        )

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Plot positions or velocites (2d)
        i, j = zip(*((0, 1), (2, 1), (0, 2)))
        for ax, x, y in zip(axes.flatten()[:3], i, j):
            self.plot_scatter(
                ax, (x, y), coord, system, step,
                errors=errors, labels=labels, mst=mst
            )

        # Plot positions or velocites (3d)
        if style in ('2x2', '4x1'):
            i, j = (1,1) if style == '2x2' else (3,0)
            self.plot_scatter(
                axes[i,j], (0, 1, 2), coord, system, step,
                errors=errors, labels=labels, mst=mst
            )

        # Set title
        self.series.set_title(
            fig, title, f'${systems[system].latex[coord]}$ {coord}s'.replace('citys', 'cities'),
            f'at {age:.1f} Myr'
        )

        # Set limits
        if coord == 'position' and system == 'xyz':
            xlim = (30, 180)
            ylim = (-1280, -1140)
            zlim = (-85, 65)
            if style == '2x2':
                axes[0,0].set_xlim(*xlim)
                axes[0,0].set_ylim(*ylim)
                axes[0,1].set_xlim(*zlim)
                axes[0,1].set_ylim(*ylim)
                axes[1,0].set_xlim(*xlim)
                axes[1,0].set_ylim(*zlim)
                axes[1,1].set_xlim(*xlim)
                axes[1,1].set_ylim(*ylim)
                axes[1,1].set_zlim(*zlim)

        # Set limits
        # if style in ('3x1', '1x3', '4x1'):
        #     for ax in axes.flatten():
        #         ax.set_aspect('equal')
        #         ax.set_adjustable('datalim')

        # Save figure
        self.series.save_figure(
            f'{coord}_{system}_{style}_{self.name}_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_cross_scatter(
        self, system, step=None, age=None, errors=False, labels=False,
        title=False, forced=False, default=False, cancel=False
    ):
        """
        Draws cross scatter plots of positions and velocities, at a given 'step' or 'age' in Myr.
        If 'age' doesn't match a step, the closest step is used instead. 'age' overrules 'steps'
        if both are given. 'errors', adds error bars and 'labels' adds the stars' names.
        """

        # Initialize figure
        fig, axes = self.series.set_figure('3x3', 'hide_x', 'hide_y')

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Plot ξηζ position and ξηζ velocity
        for x, y in [(x, y) for y in range(3) for x in range(3)]:
            self.plot_scatter(
                axes[x, y], (x, y), ('position', 'velocity'), system,
                step, errors=errors, labels=labels
            )

        # Set title
        self.series.set_title(
            fig, title, (
                f"${systems[system].latex['position']}$ positions and "
                f"${systems[system].latex['velocity']}$ velocities"
            ), f'at {age:.1f} Myr'
        )

        # Save figure
        self.series.save_figure(
            f'position_velocity_{system}_{self.name}_{age:.1f}Myr.pdf',
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_time_scatter(
        self, coord, system, style, steps=None, ages=None, errors=False, labels=False,
        mst=False, title=False, forced=False, default=False, cancel=False
    ):
        """
        Draws scatter plots of positions or velocities of stars for 3 'step' or 'age' in Myr. If
        an 'age' doesn't match a step, the closest step is used instead. 'age' overrules 'steps'
        if both are given. 'errors', adds error bars, 'labels' adds the stars' names and 'mst'
        adds the minimun spanning tree branches.
        """

        # Initialize figure
        fig, axes = self.series.set_figure(
            style, 'hide_y', '3d' if style in ('4x2', '4x3') else '',
            'label_right', styles=('3x2', '3x3', '4x2', '4x3')
        )

        # Set the number of timesteps
        number_of_steps = int(style[-1])

        # Check the type and value of steps
        self.series.check_type(steps, 'steps', ('tuple', 'list', 'None'))
        if steps is None:
            steps = number_of_steps * [None]
        self.series.stop(
            len(steps) != number_of_steps, 'ValueError',
            "'steps' must be have a length of {} with the style '{}' ({} given).",
            number_of_steps, style, len(steps)
        )

        # Check the type and value of ages
        self.series.check_type(ages, 'ages', ('tuple', 'list', 'None'))
        if ages is None:
            ages = number_of_steps * [None]
        self.series.stop(
            len(ages) != number_of_steps, 'ValueError',
            "'ages' must be have a length of {} with the style '{}' ({} given).",
            number_of_steps, style, len(ages)
        )

        # Compute the values of step and age
        steps, ages = zip(*[self.get_step_age(step, age) for step, age in zip(steps, ages)])
        ages_str = [f'{age:.1f}' for age in ages]

        # Plot positions or velocities (2d)
        for i in range(3):
            x, y = ((0, 1), (2, 1), (0, 2))[i]
            for j in range(number_of_steps):
                self.plot_scatter(
                    axes[i,j], (x, y), coord, system, steps[j],
                    errors=errors, labels=labels, mst=mst, relative=True
                )

        # Plot positions or velocities (3d)
        if style in ('4x2', '4x3'):
            for ax, step in zip(axes[3], steps):
                self.plot_scatter(
                    ax, (0, 1, 2), coord, system, step,
                    errors=errors, labels=labels, mst=mst, relative=True
                )

        # Set title
        self.series.set_title(
            fig, title, f'${systems[system].latex[coord]}$ {coord}s at'.replace('citys', 'cities'),
            f"{enumerate_strings(*ages_str, conjunction='and')} Myr"
        )

        # Set limits
        limits = (-100, 100) if coord == 'position' else (-6, 6) if coord == 'velocity' else None
        for ax in axes.flatten():
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
        if style in ('4x2', '4x3'):
            for ax in axes[3]:
                ax.set_zlim(*limits)
            # for ax in axes[3,1:]:
            #     ax.set_ylabel('', visible=False)
            #     ax.tick_params(labelleft=False)

        # Save figure
        self.series.save_figure(
            f"{coord}_{system}_{style}_{self.name}_{'_'.join(ages_str)}Myr.pdf",
            forced=forced, default=default, cancel=cancel
        )
        # plt.show()

    def draw_map(self, labels=False, title=False, forced=False, default=False, cancel=False):
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

    def draw_age_distribution(
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

    def create_kinematics_table(
        self, save=False, show=False, machine=False, age=None,
        forced=False, default=False, cancel=False
    ):
        """
        Creates a table of the 6D kinematics (xyz Galactic positions and uvw space velocities)
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