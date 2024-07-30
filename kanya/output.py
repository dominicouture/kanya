# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""output.py: Defines methods used to create figures and tables."""

from matplotlib import pyplot as plt, ticker as tkr
from scipy.interpolate import interp1d
from colorsys import hls_to_rgb
from cycler import cycler
from .collection import *
from .coordinate import *

# Set pyplot rc parameters
plt.rc('font', serif='Latin Modern Math', family='serif', size='8')
plt.rc('mathtext', fontset='custom', it='Latin Modern Roman:italic', rm='Latin Modern Roman:roman')
plt.rc('lines', markersize=4)
plt.rc('pdf', fonttype=42)

# Set ticks label with commas instead of dots for French language publications
# format_ticks = tkr.FuncFormatter(lambda x, pos: str(round(float(x), 1)))
# ax.xaxis.set_major_formatter(format_ticks)
# ax.yaxis.set_major_formatter(format_ticks)

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

class Output():
    """Output method for a groups."""

    def save_output(
        self, file_name, save, *save_args, extension=None, file_type=None,
        output_dir=None, logging=True, forced=None, default=None, cancel=None
    ):
        """
        Creates a proper output path given a file name, an extension and, optionnally, a file path
        relative to the output directory.Checks whether a path already exists and asks for user
        input if it does. The base path is assumed to be the current directory. Also, if the path
        does not have an extension, a an extension is added. The output is saved to the output
        path using the save function.
        """

        # Check the type of output_dir
        self.check_type(output_dir, 'output_dir', ('string', 'None'))
        self.check_type(logging, 'logging', 'boolean')

        # Set output_dir parameter, if needed
        if 'output_dir' not in vars(self).keys():
            self.output_dir = self.set_path(
                self.config.output_dir, 'output_dir',
                dir_only=True, check=False, create=False
            )

        # Set output_path
        output_path = self.set_path(
            self.output_dir if output_dir is None else output_dir, 'output_path',
            name=file_name, extension=extension, file_type=file_type,
            full_path=True, check=False, create=True
        )

        # Choose behaviour if a file already exists
        if path.exists(output_path):
            forced, default, cancel = self.choose(
                f"An output file already exists at '{output_path}'.", 3, forced, default, cancel
            )

            # Delete existing file and save output
            if forced:
                remove(output_path)
                save(output_path, *save_args)

                # Logging
                self.log(
                    "Existing output file located at '{}' deleted and replaced.",
                    output_path, logging=logging
                )

            # Set default file name and save output
            if default:
                old_filename = path.basename(output_path)
                output_path = get_default_filename(output_path)
                save(output_path, *save_args)

                # Logging
                self.log(
                    "Output file renamed from '{}' to '{}', and output saved at '{}'.",
                    old_filename, path.basename(output_path), output_path, logging=logging
                )

            # Cancel save
            if cancel:
                self.log(
                    "Output was not saved because an output file already exists at '{}'.",
                    output_path, logging=logging
                )

        # Save output
        else:
            save(output_path, *save_args)

            # Logging
            self.log("Output saved at '{}'.", output_path, logging=logging)

    def save_figure(
        self, file_name, fig, extension='pdf', file_type=None, output_dir=None, tight=False,
        show=False, close=True, logging=True, forced=None, default=None, cancel=None
    ):
        """Saves figure with or without tight layout and some padding."""

        # Check the type of tight, show, close and logging
        self.check_type(tight, 'tight', 'boolean')
        self.check_type(show, 'show', 'boolean')
        self.check_type(close, 'close', 'boolean')
        self.check_type(logging, 'logging', 'boolean')

        # Logging
        self.log('{} figure created.', fig.line_log, display=logging, logging=logging)

        # Save figure
        def save(output_path, tight):
            if tight:
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01)
            else:
                plt.savefig(output_path)

        # Save output
        self.save_output(
            file_name, save, tight, extension=extension, file_type=file_type,
            output_dir=output_dir, cancel=cancel, forced=forced, default=default
        )

        # Show figure
        if show:
            plt.show()

        # Close figure
        if close:
            plt.close(fig)

    def save_table(
        self, file_name, line_log, lines, header=None, extension='txt', file_type=None,
        output_dir=None, logging=True, forced=None, default=None, cancel=None
    ):
        """Saves a table to a CSV file for a given header and data."""

        # Check the type of lines, header and logging
        self.check_type(lines, 'lines', ('tuple', 'list'))
        self.check_type(header, 'header', ('string', 'None'))
        self.check_type(logging, 'logging', 'boolean')

        # Logging
        self.log('{} table created.', line_log, display=logging, logging=logging)

        # Save table
        def save(output_path, lines, header):
            with open(output_path, 'w') as output_file:
                if header is not None:
                    output_file.write(header + '\n')
                output_file.writelines([line + '\n' for line in lines])

        # Choose behavior
        self.save_output(
            file_name, save, lines, header, extension=extension, file_type=file_type,
            output_dir=output_dir, cancel=cancel, forced=forced, default=default
        )

    def set_figure(
        self, style, *options, width=None, height=None, left=None, bottom=None, right=None,
        top=None, colpad=None, rowpad=None, ratio=None, adjust=None, v_align=None, h_align=None,
        styles=None
    ):
        """
        Initializes a figure with multiple axes. The axes are created with at the correct position
        and with the correct size. Ticks and labels are repositioned and set invisible accordingly.
        """

        def get_dimensions(
            width, height, left, bottom, right, top, colpad, rowpad, nrow, ncol,
            ratio, adjust=None, h_align='left', v_align='bottom'
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
                    colpad = (width - left - ax_width * ncol - right) / (ncol - 1)

            # Adjust axis height
            if adjust == 'ax_height' and ax_height / ax_width > ratio:
                ax_height = ax_width * ratio

                # Horizontal alignment
                if v_align == 'top':
                    bottom = height - top - ax_height * nrow - rowpad * (nrow - 1)
                if v_align == 'center':
                    bottom = (height - ax_height * nrow - rowpad * (nrow - 1)) / 2
                if v_align == 'justify':
                    rowpad = (height - bottom - ax_height * nrow - top) / (nrow - 1)

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
                    top=True, right=True, bottom=True, left=True,
                    which='both', direction='in', width=0.5, labelsize=8
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

                    # Set spines
                    iax.line.set_linewidth(0.5)

                # Set view
                ax.view_init(azim=45)

        # Check traceback
        self.check_traceback()

        # Check the type and value of style
        styles = styles if styles is not None else (
            '1x1', '1x2', '1x3',
            '2x1', '2x2', '2x3',
            '3x1', '3x2', '3x3',
            '4x1', '4x2', '4x3',
            '4x4', '6x6', '2+1',
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
        colpad = 0.100 if 'hide_y' in options else 0.5400 if colpad is None else colpad
        rowpad = 0.100 if 'hide_x' in options else 0.3654 if rowpad is None else rowpad
        ratio = 1.0 if ratio is None else ratio

        # Set margins if the axes and tick labels are moved to the right of the rightmost axes
        if 'label_right' in options and ncol == 2:
            right = left
            colpad = 0.100

        # Set margins for a figure with 3d axes
        if '3d' in options and style not in ('3x3', '6x6'):
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
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust, h_align=h_align
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(
                fig, axes_params, remove_extra=False,
                projection='mollweide' if 'mollweide' in options else None
            )

        # Initialize a 1x2 figure
        if style == '1x2':
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                3.3115 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params, remove_extra=False)

        # Initialize a 1x3 figure
        if style == '1x3':
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                2.3330 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params, remove_extra=False)

        # Initialize a 2x1 figure
        if style == '2x1':
            width, height, axes_params = get_dimensions(
                3.3450 if width is None else width,
                6.5720 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = get_axes(fig, axes_params)

        # Initialize a 2x2 figure
        if style == '2x2':
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                6.317 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
            )

            # Add padding for 3d axis
            if '3d' in options:
                axes_params[0,1,1] += 1.5 * rowpad / height

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            axes[:,0] = get_axes(fig, axes_params[:,0], zorder=0.5)
            axes[1,1] = get_axes(fig, axes_params[1,1], zorder=0.5)
            if 'corner' not in options:
                axes[0,1] = get_axes(fig, axes_params[0,1], zorder=0.5)

            if '3d' in options:
                axes[0,1] = get_axes(fig, axes_params[0,1], projection='3d', zorder=0.4)
                set_axis(axes[0,1], '3d')
                if 'label_right' in options or 'hide_y' in options:
                    axes[0,1].view_init(azim=-45)

        # Initialize a 2x3 figure
        if style == '2x3':
            # left = 0.5300
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                4.4450 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
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
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='ax_width' if adjust is None else adjust,
                h_align='right' if adjust is None else h_align
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
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='ax_width' if adjust is None else adjust,
                h_align='center' if adjust is None else h_align
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
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            if 'corner' not in options:
                axes = get_axes(fig, axes_params)
            else:
                axes = np.empty((axes_params.shape[:2]), dtype=object)
                for x, y in filter(
                    lambda i: i[0] <= i[1], [(x, y) for y in range(3) for x in range(3)]
                ):
                    axes[y, x] = get_axes(fig, axes_params[y, x])

                # Set 3d axes
                if '3d' in options:
                    axes_params[0,2,0] -= colpad / width
                    axes[0,2] = get_axes(fig, axes_params[0,2], projection='3d', zorder=0.4)
                    set_axis(axes[0,2], '3d')

        # Initialize a 4x1 figure
        if style == '4x1':
            # width, height = (3.3450, 7.5000)
            width, height, axes_params = get_dimensions(
                3.3450 if width is None else width,
                9.1340 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='ax_width' if adjust is None else adjust,
                h_align='center' if adjust is None else h_align
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
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
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
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
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
                for ax in axes[3,1:]:
                    ax.set_zlabel('', visible=False)

        # Initialize a 4x4 figure
        if style == '4x4':
            # left = 0.5300
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                7.5023 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            for x, y in filter(
                lambda i: not (i[0] == 3 and i[1] == 0),
                [(x, y) for y in range(4) for x in range(4)]
            ):
                axes[y, x] = get_axes(fig, axes_params[y, x])

        # Initialize a 6x6 figure
        if style == '6x6':
            width, height, axes_params = get_dimensions(
                7.0900 if width is None else width,
                7.5023 if height is None else height,
                left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
                adjust='fig_height' if adjust is None else adjust
            )

            # Create figure and axes
            fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
            axes = np.empty((axes_params.shape[:2]), dtype=object)
            for x, y in filter(
                lambda i: i[0] <= i[1], [(x, y) for y in range(6) for x in range(6)]
            ):
                axes[y, x] = get_axes(fig, axes_params[y, x])

            # Set 3d axes
            if '3d' in options:
                axes_params[1,2,2:] += (colpad / width / 2 + axes_params[1,2,2:] * 0.9)
                axes_params[1,2,0] += (colpad / width * 1.5 + axes_params[0,0,2] / 2)
                axes_params[1,2,1] += rowpad / height * 3
                axes_params[1,2,0] -= colpad / width / 2
                axes_params[3,4,2:] += (rowpad / height / 2 + axes_params[3,4,2:] * 0.9)
                axes_params[3,4,1] += (rowpad / height * 2 + axes_params[0,0,3] / 2)
                axes[1,2] = get_axes(fig, axes_params[1,2], projection='3d', zorder=0.5)
                axes[3,4] = get_axes(fig, axes_params[3,4], projection='3d', zorder=0.4)
                set_axis(axes[1,2], '3d')
                set_axis(axes[3,4], '3d')

        # Hide axis and tick labels, if needed
        if 'hide_x' in options and nrow > 1:
            for ax in filter(
                lambda ax: ax is not None,
                axes[:nrow - 1 - (1 if '3d' in options and style != '6x6' else 0)].flatten()
            ):
                if not (style in ('3x3', '6x6') and ax.name == '3d'):
                    ax.set_xlabel('', visible=False)
                    ax.tick_params(labelbottom=False)
        if 'hide_y' in options and ncol > 1:
            if not (ncol == 2 and 'label_right' in options):
                for ax in filter(
                    lambda ax: ax is not None,
                    axes[:nrow - (1 if style == '2x2' and '3d' in options else 0),1:].flatten()
                ):
                    if not (style in ('3x3', '6x6') and ax.name == '3d'):
                        ax.set_ylabel('', visible=False)
                        ax.tick_params(labelleft=False)

        # Move axes and tick labels to the right of the rightmost axes
        if 'label_right' in options and ncol == 2:
            if style == '2x2' and '3d' in options:
                right_axes = [axes[1,1]]
            else:
                right_axes = axes[:nrow - (1 if '3d' in options else 0),1].flatten()
            for ax in filter(lambda ax: ax is not None, right_axes):
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

    def set_title(self, title, fig, get_title, *args, **kwargs):
        """Sets a title for the figure, if 'title' is True."""

        # Check the type of title
        self.check_type(title, 'title', 'boolean')

        # Create title for figure and logging
        fig.line_title, fig.line_log = get_title(fig, *args, **kwargs)

        # Set title
        if title:
            fig.suptitle(fig.line_title, y=fig.offset, fontsize=8)

    def check_system(self, system):
        """
        Check the type and value of system. Returns the value of the short label if the long
        label is provided.
        """

        # Check the type and value of system
        self.check_type(system, 'system', 'string')
        systems = ('cartesian', 'xyz', 'curvilinear', 'ξηζ')
        self.stop(
            system not in systems, 'ValueError',
            "'system' can only take as value {} ({} given).", enumerate_strings(*systems), system
        )

        # Select the value of the short label if the long label is provided
        if system == 'cartesian':
            system = 'xyz'
        if system == 'curvilinear':
            system = 'ξηζ'

        return system

    def select_metric(self, metric, system, robust=False, sklearn=False):
        """
        Checks if 'robust' and 'sklearn' arguments are valid and select the corresponding
        association size metrics based on the system.
        """

        # Check the type and value of system
        system = self.check_system(system)

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
                "Could not plot '{}' metric for '{}' group. It was not computed.",
                str(metric.name[index]), self.name, display=True
            )

        # Set legend
        self.set_legend(ax, 2)

        # Select units
        units = f' ({metric.units[index]})' if metric.units[index] != '' else ''

        # Set labels
        ax.set_xlabel('Epoch (Myr)', fontsize=8)
        ax.set_ylabel(f'Association size{units}', fontsize=8)

        # Set limits
        ax.set_xlim(self.final_time.value + 1, self.initial_time.value + 1)

    def get_metric(self, metric, index=None):
        """Retrieves the proprer Metric instance from a string and index."""

        # Metric instance
        self.check_type(metric, 'metric', ('string', 'None'))
        self.stop(
            metric not in [metric.label for metric in self.metrics], 'ValueError',
            "'metric' must be a valid metric key ({} given).", metric
        )
        metric = vars(self)[metric]

        # If the metric has a size of 1, index is ignored
        if metric.status:
            if metric.value.shape[-1] == 1:
                index = 0

            # Metric index
            else:
                self.check_type(index, 'index', ('integer', 'None'))
                self.stop(
                    metric.value.shape[-1] > 1 and index is None, 'ValueError',
                    "No 'index' is provided (metric is {} in size).", metric.value.shape[-1]
                )
                self.stop(
                    index > metric.value.shape[-1] - 1 and metric.value.shape[-1] > 1, 'ValueError',
                    "'index' is too large for this metric ({} given, {} in size).",
                    index, metric.value.shape[-1]
                )

        return metric, index

    def get_title_metric(self, fig, line_1, system):
        """Sets a title for association size metrics plots if 'title' is True."""

        # Set line 1
        line_1_title = line_1.format(
            system=f"${systems[system].latex['position']}$",
            position=f"${systems[system].latex['position']}$",
            velocity=f"${systems[system].latex['velocity']}$",
        )
        line_1_log = line_1.format(
            system=f"{systems[system].labels['position']}",
            position=f"{systems[system].labels['position']}",
            velocity=f"{systems[system].labels['velocity']}",
        )

        # Create title from data
        if self.from_data:
            line_2 = f'of {self.name} over {self.duration.value:.1f} Myr'

            return f'{line_1_title}\n{line_2}', f'{line_1_log} {line_2}'

        # Create title from a model
        if self.from_model:
            line_2 = (
                f"of {self.number_of_groups} association{'s' if self.number_of_groups > 1 else ''} "
                f'modeled after {self.name} over {self.duration.value:.1f} Myr'
            )

            return f'Average {line_1_title}\n{line_2}', f'Average {line_1_log} {line_2}'

    def plot_covariances(self, ax, system, robust, sklearn):
        """Plots the determinant, the trace and the diagonal terms of the covariances matrix."""

        # Select association size metrics
        covariances, selection = (
            self.select_metric('covariances', system, robust=robust, sklearn=sklearn)
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

        return selection

    def draw_covariances(
        self, system, robust=False, sklearn=False,
        title=False, forced=None, default=None, cancel=None
    ):
        """
        Creates a plot of the determinant, the trace and the diagonal terms of the covariances
        matrix. If either 'robust' or 'sklearn' is True, the robust or sklearn covariances matrix
        is used. Otherwise, the empirical covariances matrix is used.
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)

        # Plot covariance matrix determinant and trace, and covariances
        selection = self.plot_covariances(axes[0,0], system, robust, sklearn)

        # Set title
        self.set_title(
            title, fig, self.get_title_metric,
            '{system}'f"{selection.replace('_', ' ')} covariances", system
        )

        # Save figure
        self.save_figure(
            f'covariances_{system}_{self.name}{selection}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def plot_cross_covariances(self, ax, system, robust, sklearn):
        """
        Plots the determinant, the trace and the diagonal terms of the cross covariances matrix
        between positions and velocities.
        """

        # Select association size metrics
        cross_covariances, selection = (
            self.select_metric('cross_covariances', system, robust=robust, sklearn=sklearn)
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

        return selection

    def draw_cross_covariances(
        self, system, robust=False, sklearn=False,
        title=False, forced=None, default=None, cancel=None
    ):
        """
        Creates a plot the determinant, the trace and the diagonal terms of the cross covariances
        matrix between positions and velocities. If either 'robust' or 'sklearn' is True, the robust
        or sklearn covariances matrix is used. Otherwise, the empirical covariances matrix is used.
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)

        # Plot the cross covariance matrix determinant and trace, and cross covariances
        selection = self.plot_cross_covariances(axes[0,0], system, robust, sklearn)

        # Set title
        self.set_title(
            title, fig, self.get_title_metric,
            '{position} and {velocity}'f"{selection.replace('_', ' ')} cross covariances", system
        )

        # Save figure
        self.save_figure(
            f'cross_covariances_{system}_{self.name}{selection}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def plot_mad(self, ax, system):
        """Plots the total median absolute deviation (MAD), and the components of the MAD."""

        # Select association size metrics
        mad = self.select_metric('mad', system)[0]
        mad_total = vars(self)[f'{mad}_total']
        mad = vars(self)[mad]

        # Plot the total median aboslute deviation (MAD), and the components of the MAD
        self.plot_metric(ax, mad_total, 0, colors.metric[0], '-', 0.8)
        self.plot_metric(ax, mad, 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, mad, 1, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, mad, 2, colors.metric[7], ':', 0.5)

    def draw_mad(self, system, title=False, forced=None, default=None, cancel=None):
        """
        Creates a plot of the total median absolute deviation (MAD), the components of the MAD.
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)

        # Plot the total median aboslute deviation (MAD), and the components of the MAD
        self.plot_mad(axes[0,0], system)

        # Set title
        self.set_title(
            title, fig, self.get_title_metric, '{system} median absolute deviation', system
        )

        # Save figure
        self.save_figure(
            f'mad_{system}_{self.name}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def plot_tree(self, ax, system):
        """
        Plots the mean, robust mean, and median absolute deviation full and minimum spanning tree
        (MST) branch lengths.
        """

        # Select association size metrics
        mst = self.select_metric('mst', system)[0]
        branches = self.select_metric('branches', system)[0]

        # Plot the mean, robust mean, and median absolute deviation full tree branch lengths
        self.plot_metric(ax, vars(self)[f'{branches}_mean'], 0, colors.metric[0], '-', 0.9)
        self.plot_metric(ax, vars(self)[f'{branches}_mean_robust'], 0, colors.metric[1], '--', 0.7)
        self.plot_metric(ax, vars(self)[f'{branches}_mad'], 0, colors.metric[2], ':', 0.5)

        # Plot the mean, robust mean, and median absolute deviation minimum spanning tree
        # branch lengths
        self.plot_metric(ax, vars(self)[f'{mst}_mean'], 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, vars(self)[f'{mst}_mean_robust'], 0, colors.metric[6], '--', 0.7)
        self.plot_metric(ax, vars(self)[f'{mst}_mad'], 0, colors.metric[7], ':', 0.5)

    def draw_tree(self, system, title=False, forced=None, default=None, cancel=None):
        """
        Creates a plot the mean, robust mean, and median absolute deviation full and minimum
        spanning tree (MST) branch lengths.
        """

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)

        # Plot the mean, robust mean, and median absolute deviation full and minimum spanning
        # tree branch lengths
        self.plot_tree(axes[0,0], system)

        # Set title
        self.set_title(
            title, fig, self.get_title_metric, '{system} full and minimum spanning tree', system
        )

        # Save figure
        self.save_figure(
            f'mst_{system}_{self.name}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def draw_mahalanobis(self, system, title=False, forced=None, default=None, cancel=None):
        """Creates a plot of the mean and median Mahalanobis distance."""

        # Initialize figure
        fig, axes = self.set_figure('1x1', left=0.3430)
        ax = axes[0,0]

        # Select association size metrics
        mahalanobis = self.select_metric('mahalanobis', system)[0]

        # Plot mean and median Mahalanobis distance
        self.plot_metric(ax, vars(self)[f'{mahalanobis}_mean'], 0, colors.metric[5], '-', 0.9)
        self.plot_metric(ax, vars(self)[f'{mahalanobis}_median'], 0, colors.metric[6], '--', 0.7)

        # Set labels and limits
        ax.set_ylabel(ax.get_ylabel(), labelpad=3)
        ax.set_ylim(-0.1, 2.9)

        # Set title
        self.set_title(
            title, fig, self.get_title_metric, '{system} Mahalanobis distance', system
        )

        # Save figure
        self.save_figure(
            f'mahalanobis_{system}_{self.name}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def draw_covariances_mad(
        self, system, style, robust=False, sklearn=False,
        title=False, forced=None, default=None, cancel=None
    ):
        """
        Creates a plot of the determinant, the trace and the diagonal terms of the covariances
        matrix, and the total median absolute deviation (MAD), the components of the MAD. If
        either 'robust' or 'sklearn' is True, the robust or sklearn covariances matrix is used.
        Otherwise, the empirical covariances matrix is used.
        """

        # Initialize figure
        fig, axes = self.set_figure(
            style, 'hide_x', left=0.3430, colpad=0.4430, styles=('2x1', '1x2')
        )

        # Plot covariance matrix determinant and trace, and covariances
        selection = self.plot_covariances(axes[0,0], system, robust, sklearn)
        axes[0,0].set_yticks([0, 3, 6, 9, 12, 15, 18, 21])

        # Plot the total median aboslute deviation (MAD), and the components of the MAD
        self.plot_mad(axes[1 if style == '2x1' else 0, 1 if style == '1x2' else 0], system)

        # Set title
        self.set_title(
            title, fig, self.get_title_metric,
            '{system}'f"{selection.replace('_', ' ')} covariances and median absolution deviation",
            system
        )

        # Save figure
        self.save_figure(
            f'covariances_mad_{system}_{style}_{self.name}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def draw_covariances_cross_mad_tree(
        self, system, robust=False, sklearn=False,
        title=False, forced=None, default=None, cancel=None
    ):
        """
        Creates plots of the determinant, the trace and the diagonal terms of the covariances
        matrix, the determinant, the trace and the diagonal terms of the cross covariances matrix
        between positions and velocities, the total median absolute deviation (MAD), the components
        of the MAD, and the mean, robust mean, and median absolute deviation full and minimum
        spanning tree (MST) branch lengths.  If either 'robust' or 'sklearn' is True, the robust
        or sklearn covariances and cross covariance matrices are used. Otherwise, the empirical
        covariances and cross covariance matrices are used.
        """

        # Initialize figure
        fig, axes = self.set_figure('2x2', 'hide_x', 'label_right')

        # Plot covariance matrix determinant and t race, and covariances
        selection = self.plot_covariances(axes[0,0], system, robust, sklearn)

        # Plot the cross covariance matrix determinant and trace, and cross covariances
        self.plot_cross_covariances(axes[0,1], system, robust, sklearn=False)

        # Plot the total median aboslute deviation (MAD), and the components of the MAD
        self.plot_mad(axes[1,0], system)

        # Plot the mean, robust mean, and median absolute deviation full and minimum spanning
        # tree branch lengths
        self.plot_tree(axes[1,1], system)

        # Set title
        self.set_title(
            title, fig, self.get_title_metric,
            '{system}'f"{selection.replace('_', ' ')} covariances, cross covariances, median "
            "absolution deviation, and full and minimum spanning trees", system
        )

        # Save figure
        self.save_figure(
            f'covariances_cross_mad_mst_{system}_{self.name}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def plot_histogram(
        self, ax, values, number_of_bins, fit=None, limits=None, value=None, error=None,
        label=None, orientation='vertical', set_lim=True, error_lines=True, quick_fit=False,
        curve_color=colors.azure[6], hist_color=colors.azure[9], line_color=colors.black,
    ):
        """
        Plots a histogram of the values along with a normal or skew normal fit, a dash vertical
        line representing the average value (normal fit) or mode (skew normal fit) of the fit,and
        dotted vertical lines representing the 1σ error range. If value is None, then a shaded
        area under the curve is drawn instead.
        """

        # Check the type and value of number_of_bins
        self.check_type(number_of_bins, 'number_of_bins', 'integer')
        self.stop(
            number_of_bins < 1, 'ValueError',
            "'number_of_bins' must be greater than 0 ({} given).", number_of_bins
        )

        # Check the type and value of fit
        self.check_type(fit, 'fit', ('string', 'None'))
        self.stop(
            fit not in ('normal', 'skewnormal', None),
            'ValueError', "'fit' must be 'normal', 'skewnormal' or None ({} given).", fit
        )

        # Check the value of value and error, if values is None
        self.stop(
            values is None and (value is None or error is None),
            'ValueError', "'value' and 'error' cannot be None if values is None "
            "({} and {} given).", value, error
        )

        # Set value
        value = np.mean(values) if value is None and values is not None else value

        # Set errors
        error = np.atleast_1d(np.std(values) if error is None and values is not None else error)
        errors = np.repeat(error, 2) if error.size == 1 else error

        # Set limits
        limits = (
            (np.min(values), np.max(values)) if limits is None and values is not None else
            (value - 4 * errors[0], value + 4 * errors[1]) if limits is None else limits
        )

        # Set space
        space = np.linspace(*limits, 500)

        # Fit a normal curve
        if fit == 'normal':
            from scipy.stats import norm

            # Compute probabiilty density function
            pdf = norm.pdf(space, value, errors[0])

            # Show value and error
            if label is not None:
                print(f'{label}: {value:.3f} ± {errors[0]:.3f}')

        # Fit a skew normal curve
        if fit == 'skewnormal':
            from scipy.optimize import minimize
            from scipy.stats import skewnorm
            from scipy.special import erf

            def get_skewnormal(skew, loc, scale, mode='max'):
                """
                Computes the probability density function, mode and asymetric standard deviations
                of a skewnormal distribution.
                """

                # Compute density functions
                pdf = skewnorm.pdf(space, skew, loc, scale)
                cdf = skewnorm.cdf(space, skew, loc, scale)

                # Compute mode estimate
                if mode == 'estimate':
                    α, ξ, ω = skew, loc, scale
                    δ = α / np.sqrt(1 + α**2)
                    γ_1 = (4 - np.pi) / 2 * (δ * np.sqrt(2/np.pi)**3) / (1 - 2 * δ**2 / np.pi)**1.5
                    m_0 = (
                        np.sqrt(2 / np.pi) * δ - γ_1 * np.sqrt(1 - 2 / np.pi * δ**2) / 2 -
                        np.sign(α) / 2 * np.exp(-2 * np.pi / np.abs(α))
                    )
                    mode = ξ + m_0 * ω

                # Compute mode from the maximum
                if mode == 'max':
                    mode = space[np.argmax(pdf)]

                # Compute errors
                mode_cdf = cdf[np.argmin(np.abs(mode - space))]
                prob = (
                    np.repeat(mode_cdf, 2) + np.array([-1, 1]) *
                    erf(1 / np.sqrt(2)) / 2 * np.array([mode_cdf / 0.5, (1 - mode_cdf) / 0.5])
                )
                errors = np.abs(
                    space[
                        np.argmin(np.abs(np.repeat(cdf[None], 2, axis=0).T - prob), axis=0)
                    ] - mode
                )

                return pdf, mode, errors

            def Χ2(parameters):
                """Computes the chi squarred value of a skewnormal fit."""

                fit_errors = get_skewnormal(parameters[0], value, parameters[1])[2]
                return np.sum((fit_errors - errors)**2)

            # Fit a skewnormal curve based on value and errors
            if values is None:

                # Fit two gaussian curves
                if quick_fit:
                    top = np.argmin(np.abs(space - value))
                    pdf = np.exp(
                        -0.5 * np.concatenate(
                            ((space[:top] - value) / errors[0], (space[top:] - value) / errors[1])
                        )**2
                    ) / np.sqrt(2 * np.pi) / np.mean(errors)

                # Fit the skew and scale
                else:
                    sign = np.sign(errors[1] - errors[0])
                    bound = sign * (10 * (errors[1] - errors[0]) + 1)
                    initial_parameters = np.array([bound / 2, np.mean(errors)])
                    bounds = (
                        (bound if sign < 0 else 0, bound if sign > 0 else 0),
                        (0, 3 * np.mean(errors))
                    )
                    skew, scale = minimize(
                        Χ2, initial_parameters, method='Nelder-Mead', bounds=bounds
                    ).x

                    # Recenter the mode on the value
                    pdf = skewnorm.pdf(space, skew, value, scale)
                    pdf = skewnorm.pdf(space - value + space[np.argmax(pdf)], skew, value, scale)

            # Fit a skewnormal curve based on values
            else:
                skew, loc, scale = skewnorm.fit(values)
                pdf, value, errors = get_skewnormal(skew, loc, scale)

            # Show value and error
            if label is not None:
                print(f'{label}: {value:.3f} -{errors[0]:.3f} +{errors[1]:.3f}')

        # Plot probability density function
        if fit in ('normal', 'skewnormal'):
            i = pdf > 0.0001 * np.max(pdf)
            ax.plot(
                *(space[i], pdf[i])[::1 if orientation == 'vertical' else -1],
                color=curve_color, alpha=1.0, label=label, linewidth=1.0, linestyle='-',
                solid_capstyle='round', dash_capstyle='round', zorder=0.8
            )

            # Draw value and error range
            (ax.axvline if orientation == 'vertical' else ax.axhline)(
                value, color=line_color, alpha=0.8, linewidth=0.5, linestyle='--', zorder=0.9
            )
            if error_lines:
                for sign, i in zip((-1, 1), (0, 1)):
                    (ax.axvline if orientation == 'vertical' else ax.axhline)(
                        value + sign * errors[i], color=line_color, alpha=0.8,
                        linewidth=0.5, linestyle=':', zorder=0.9
                    )

            # Fill the area under the curve
            if values is None:
                i = pdf > 0.0001 * np.max(pdf)
                ax.fill_between(
                    space[i], np.zeros_like(space[i]), pdf[i],
                    color=hist_color, alpha=0.15, linewidth=0., zorder=0.5
                )

        # Plot histogram
        if values is not None:
            ax.hist(
                values, bins=np.linspace(*limits, number_of_bins + 1),
                density=True, color=hist_color, alpha=0.3, zorder=0.7,
                orientation=orientation
            )

        # Set limits
        if set_lim:
            (ax.set_xlim if orientation == 'vertical' else ax.set_ylim)(*limits)
        (ax.set_ylim if orientation == 'vertical' else ax.set_xlim)(0.0, auto=True)

        return value, errors

    def draw_age_distribution(
        self, metric, index=None, fit=None, number_of_bins=60, adjusted=False,
        title=False, forced=None, default=None, cancel=None
    ):
        """Draws a plot of the age distribution of a group for a given metric."""

        # Initialize figure
        fig, axes = self.set_figure('1x1')

        # Retrieve metric
        metric, index = self.get_metric(metric, index)

        # Adjust ages, if needed
        if metric.status:
            ages = metric.ages[:,:,index].flatten()
            value = metric.age_adjusted[index] if adjusted else metric.age[index]
            ages += (value - np.mean(ages) if adjusted else 0.0)

            # Plot histogram
            self.plot_histogram(
                axes[0,0], ages, number_of_bins, fit=fit, value=value,
                error=metric.age_error[index], limits=(np.min(ages), np.max((np.max(ages), 1.0))),
                label=f"{'Corrected ' if adjusted else ''}{metric.name[index]}"
            )

        # Logging
        else:
            self.log(
                "Could not use '{}' metric for '{}' group. It was not computed.",
                metric.name[index], self.name, display=True
            )

        # Plot results from Miret-Roig et al. (2020) and Crundall et al. (2019)
        self.plot_MiretRoig2020_Crundall2019(axes[0,0], mr2020=False, cr2019=False, ldb=False)

        # Set title
        self.set_title(
            title, fig, lambda fig, title: (title, title.replace('\n', ' ')),
            f'Age distribution of {self.number_of_groups} groups of {self.name},\n'
            f'using the {metric.name[index]} as association size metric'
        )

        # Set legend
        self.set_legend(axes[0,0], 2)

        # Set labels
        axes[0,0].set_xlabel('Age (Myr)', fontsize=8)
        axes[0,0].set_ylabel('Density', fontsize=8)

        # Set limits
        axes[0,0].set_xlim(-40, 1)
        axes[0,0].set_ylim(0, 0.17)

        # Save figure
        self.save_figure(
            f"age_distribution_{self.name}_{'corrected_' if adjusted else ''}"
            f"{metric.name[index].replace(' ', '_')}.pdf", fig, tight=title,
            forced=forced, default=default, cancel=cancel
        )

    def plot_MiretRoig2020_Crundall2019(self, ax, mr2020=True, cr2019=True, ldb=True):
        """
        Plots histogram curves of the kinematic age of the Beta Pictoris Moving group using results
        from Miret-Roig et al. (2020) and Crundall et al. (2019). A dark shaded area showing the
        range of ages from lithium depletion boundary and isochrones methods is also plotted.
        """

        # Plot curve from Miret-Roig et al. (2020)
        if mr2020:
            self.plot_histogram(
                ax, None, 1, fit='skewnormal', value=-18.5, error=(2.0, 2.4),
                label='Miret-Roig et al. (2020)', set_lim=False, error_lines=False,
                curve_color=colors.orange[6], hist_color=colors.orange[9],
                line_color=colors.orange[6]
            )

        # Plot curve from Crundall et al. (2019)
        if cr2019:
            self.plot_histogram(
                ax, None, 1, fit='normal', value=-17.7, error=1.2,
                label='Crundall et al. (2019)', set_lim=False, error_lines=False,
                curve_color=colors.azure[6], hist_color=colors.azure[9], line_color=colors.azure[6]
            )

        # Show a shaded area for LDB and isochrone ages
        if ldb:
            ylim = ax.get_ylim()
            LDB_range = np.array([-20, -26])
            ax.fill_between(
                LDB_range, 0, 1, transform=ax.get_xaxis_transform(),
                color=colors.grey[9], alpha=0.1, linewidth=0.0, zorder=0.1
            )
            ax.set_ylim(*ylim)

    def create_metrics_table(
        self, save=False, show=False, machine=False,
        forced=None, default=None, cancel=None
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
                f'metrics_{self.name}', f'Metrics of {self.name}', lines,
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
            self.check_type(age, 'age', ('integer', 'float', 'None'))
            self.stop(
                age < np.min(self.time), 'ValueError',
                "'age' must be younger the oldest time ({} Myr given, earliest time: {} Myr).",
                age, np.min(self.time)
            )
            self.stop(
                age > np.max(self.time), 'ValueError',
                "'age' must be older the latest time ({} Myr given, latest time: {} Myr).",
                age, np.max(self.time)
            )
            return np.argmin(np.abs(age - self.time)), age, None

        # Index from the epoch of minimum of an association size metric
        elif metric is not None:
            metric, index = self.get_metric(metric, index)
            if metric.status:
                age = metric.age[index]
                return np.argmin(np.abs(age - self.time)), age, metric.age_error[index]

            else:
                self.log(
                    "Could not use '{}' metric for '{}' group. It was not computed.",
                    str(metric.name[index]), self.name, display=True
                )
                return None, None, None

        # No birth index, age or age error
        else:
            return None, None, None

    def get_step_age(self, step, age):
        """
        Checks the types and values of step and step, and computes the appropriate values of
        step and age. If 'age' doesn't match a step, the closest step is used instead. 'age'
        overrules 'steps' if both are given.
        """

        # Check the type and value of step
        if step is not None:
            self.check_type(step, 'step', ('integer', 'float'))
            self.stop(
                step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step
            )
            self.stop(
                step < 0, 'ValueError',
                "'step' must be greater than or equal to 0 ({} given).", step
            )
            self.stop(
                step >= self.number_of_steps, 'ValueError',
                "'step' must be lower than the number of steps ({} and {} given).",
                self.number_of_steps, step
            )

        # Check the type and value of age
        if age is not None:
            self.check_type(age, 'age', ('integer', 'float'))
            self.stop(
                age > self.initial_time.value, 'ValueError',
                "'age' must older than the initial time  ({:.1f} Myr and {:.1f} Myr given).",
                age, self.initial_time.value
            )
            self.stop(
                age < self.final_time.value, 'ValueError',
                "'age' must be younger than the final time ({:.1f} Myr and {:.1f} Myr given).",
                age, self.final_time.value
            )

        # Check if both step and age are None
        self.stop(
            step is None and age is None, 'ValueError',
            "'step' and 'age' cannot both be None."
        )

        # Compute step and age
        if age is not None:
            step = int(round((self.initial_time.value - age) / self.timestep.value))
            age = round(self.time[step], 2)
        else:
            step = int(step)
            age = round(self.initial_time.value - step * self.timestep.value, 2)

        return step, age

    def get_metric(self, metric, index=None):
        """Retrieves the proprer Metric instance from a string and index."""

        # Metric instance
        self.check_type(metric, 'metric', ('string', 'None'))
        self.stop(
            metric not in [metric.metric.label for metric in self.metrics], 'ValueError',
            "'metric' must be a valid metric key ({} given).", metric
        )
        metric = vars(self)[metric]

        # If the metric has a size of 1, index is ignored
        if metric.value.shape[-1] == 1:
            index = 0

        # Metric index
        else:
            self.check_type(index, 'index', ('integer', 'None'))
            self.stop(
                metric.value.shape[-1] > 1 and index is None, 'ValueError',
                "No 'index' is provided (metric is {} in size).", metric.value.shape[-1]
            )
            self.stop(
                index > metric.value.shape[-1] - 1 and metric.value.shape[-1] > 1, 'ValueError',
                "'index' is too large for this metric ({} given, {} in size).",
                index, metric.value.shape[-1]
            )

        return metric, index

    def check_coord(self, coord):
        """Checks the type and value of a coordinate."""

        self.check_type(coord, 'coord', 'string')
        coords = ('position', 'velocity')
        self.stop(
            coord not in coords,
            'ValueError', "'coord' must be {} ({} given).", enumerate_strings(*coords), coord
        )

    def check_axis(self, axis, label):
        """Checks the type and value of an axis."""

        self.check_type(axis, label, 'integer')
        self.stop(
            axis < 0 or axis > 2, 'ValueError',
            "'{}' must be 0, 1, or 2 ({} given).", label, axis
        )

    def get_title_coord(self, fig, coord, system, line_1, line_2=''):
        """Creates a title for position or velocity coordinates."""

        # Create title for figure
        return (
            (line_1 + ' of stars in {name}{joint}' + line_2).format(
                system=f'${systems[system].latex[coord]}$', coord=coord,
                position=f"${systems[system].latex['position']}$",
                velocity=f"${systems[system].latex['velocity']}$",
                name=self.name, joint=fig.joint if line_2 != '' else ''
            ).replace('ys ', 'ies '),

            # Create title for logging
            (line_1 + ' of stars in {name}{joint}' + line_2).format(
                system=systems[system].labels[coord], coord=coord,
                position=f"{systems[system].labels['position']}",
                velocity=f"{systems[system].labels['velocity']}",
                name=self.name, joint=' ' if line_2 != '' else ''
            ).replace('ys ', 'ies ')
        )

    def get_limits(self, coord, system, step, relative=False):
        """Finds the limits of positions or velocities, at a given step."""

        # Check the type and value of coord
        self.check_coord(coord)

        # Check the type and value of system
        system = self.check_system(system)

        # Check the type and value of relative
        self.check_type(relative, 'relative', 'boolean')

        # Find limits, increased by 25%
        scale_factor = 1.25
        values = np.array(
            [
                [
                    vars(star)[f"{'relative_' if relative else ''}{coord}_{system}"][step]
                    for star in group.sample
                ] for group in self
            ]
        ).reshape((-1, 3)).T
        values_range = np.repeat(np.max((np.max(values, axis=1) - np.min(values, axis=1))), 3)
        limits = (
            np.array([-values_range, values_range]) * scale_factor +
            np.max(values, axis=1) + np.min(values, axis=1)
        ).T  / 2

        return values, limits

    def plot_trajectory(self, ax, x, y, coord, system, age, metric, index, labels):
        """Draws the trajectory of stars in the group."""

        # Check the type and value of x and y
        self.check_axis(x, 'x')
        self.check_axis(y, 'y')

        # Check the type and value of coord
        self.check_coord(coord)

        # Check the type and value of system
        system = self.check_system(system)

        # Check the type of labels
        self.check_type(labels, 'labels', 'boolean')

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
            if self.from_data:
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
                        color=color + (0.4,), edgecolors=color,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                # Show stars' names
                if labels:
                    ax.text(
                        np.array(value.T[0,x]), np.array(value.T[0,y]), star.name,
                        color=colors.black, horizontalalignment='left', fontsize=6
                    )

        # Select coordinates
        if self.from_model:
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
        title=False, forced=None, default=None, cancel=None
    ):
        """Draws the trajectories in position or velocity of stars."""

        # Initialize figure
        fig, axes = self.set_figure('2+1')

        # Plot trajectories
        if system == 'xyz':
            i, j = zip(*((1, 0), (2, 0), (1, 2)))
        if system == 'ξηζ':
            i, j = zip(*((0, 1), (2, 1), (0, 2)))
        for ax, x, y in zip(axes.flatten(), i, j):
            self.plot_trajectory(ax, x, y, coord, system, age, metric, index, labels)

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
                        fill=False, linewidth=0.5, linestyle=':', zorder=0.05
                    )
                )

            # Set limits
            # axes[0,0].set_xlim(-9, 1)
            # axes[0,0].set_ylim(-1, 9)
            # axes[0,1].set_xlim(-80, 80)
            # axes[0,1].set_ylim(-1, 9)
            # axes[1,0].set_xlim(-9, 1)
            # axes[1,0].set_ylim(-80, 80)

            # axes[0,0].set_xlim(-9, 1)
            # axes[0,0].set_ylim(-1, 9)
            # axes[0,1].set_xlim(-150, 130)
            # axes[0,1].set_ylim(-1, 9)
            # axes[1,0].set_xlim(-9, 1)
            # axes[1,0].set_ylim(-150, 130)
            axes[1,0].set_ylabel(axes[1,0].get_ylabel(), labelpad=-1)

        # Invert y axis
        if system == 'xyz':
            axes[0,0].invert_xaxis()
            axes[1,0].invert_xaxis()

        # Set limits
        if coord == 'position' and system == 'ξηζ':
            axes[0,0].set_ylabel(axes[0,0].get_ylabel(), labelpad=-1)
            # axes[1,0].set_ylabel(axes[1,0].get_ylabel(), labelpad=-0.1)
        #     axes[0,0].set_xlim(-225, 60)
        #     axes[0,0].set_ylim(-45, 110)
        #     axes[0,1].set_xlim(-40, 49)
        #     axes[0,1].set_ylim(-45, 110)
        #     axes[1,0].set_xlim(-225, 60)
        #     axes[1,0].set_ylim(-40, 49)

        # Set title
        self.set_title(
            title, fig, self.get_title_coord,
            coord, system, '{system} trajectories'
        )


        # Save figure
        self.save_figure(
            f'trajectory_{coord}_{system}_{self.name}.pdf', fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def plot_time(self, ax, y, coord, system, age, metric):
        """Draws the y coordinate of stars in the group over time."""

        # Check the type and value of y
        if y is not None:
            self.check_axis(y, 'y')

        # Check the type and value of coord
        self.check_coord(coord)

        # Check the type and value of system
        system = self.check_system(system)

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
                    self.time, value[:,y],
                    color = colors.red[6] if star.outlier else colors.black, alpha=0.6,
                    linewidth=0.5, solid_capstyle='round', zorder=0.1
                )

                # Plot stars' current values
                if self.from_data:
                    ax.scatter(
                        self.time[0], value[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black, alpha=None,
                        s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' birth values
                    if birth_index[y] is not None:
                        color = colors.red[6] if star.outlier else colors.blue[6]
                        ax.scatter(
                            self.time[birth_index[y]], value[birth_index[y],y],
                            color=color + (0.4,), edgecolors=color, alpha=None,
                            s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

            # Show vectical dashed line
            if birth_index[y] is not None:
                ax.axvline(
                    x=self.time[birth_index[y]], color=colors.black,
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
            if self.from_model:
                average_value = np.mean(
                    [vars(star)[f'{coord}_{system}'][:,y] for star in self.model_stars], axis=0
                )
                average_model_star_value = (
                    vars(self.average_model_star)[f'{coord}_{system}'][:,y] - average_value[::-1]
                )

                # Plot the average model star's value
                ax.plot(
                    self.model_time,
                    average_model_star_value,
                    color=colors.green[6], alpha=0.8,
                    linewidth=1.0, solid_capstyle='round', zorder=0.3
                )

                # Plot the average model star's birth and current values
                for t, x, size, marker in ((-1, -1, 10, '*'), (0, 0, 6, 'o')):
                    ax.scatter(
                        np.array([self.model_time[t]]),
                        np.array([average_model_star_value[x]]),
                        color=colors.green[6] + (0.4,), edgecolors=colors.green[6],
                        alpha=None, s=size, marker=marker, linewidths=0.25, zorder=0.3
                    )

                # Select coordinates
                for star in self.model_stars:
                    model_star_value = vars(star)[f'{coord}_{system}'][:,y] - average_value

                    # Plot model stars' values
                    ax.plot(
                        self.model_time[::-1], model_star_value,
                        color=colors.blue[6], alpha=0.6,
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current values
                    for t, x, size, marker in ((-1, 0, 10, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            np.array([self.model_time[t]]),
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
                    self.time, value[:,y],
                    label=f'$<{vars(systems[system])[coord][y].latex}>$', color=colors.black,
                    alpha=0.8, linestyle=linestyle, linewidth=1.0, solid_capstyle='round',
                    dash_capstyle='round', zorder=0.1
                )

                # Plot stars' average current values
                if self.from_data:
                    ax.scatter(
                        self.time[0], value[0,y],
                        color=colors.black + (0.4,), edgecolors=colors.black,
                        alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                    )

                    # Plot stars' average birth values
                    if birth_index[y] is not None:
                        ax.scatter(
                            self.time[birth_index[y]],
                            value[birth_index[y],y],
                            color=colors.blue[6] + (0.4,), edgecolors=colors.blue[6] + (1.0,),
                            alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.2
                        )

                # Select coordinates
                if self.from_model:
                    average_model_star_value = (
                        vars(self.average_model_star)[f'{coord}_{system}'][:,y]
                    ) / factor

                    # Plot the average model star's value
                    ax.plot(
                        self.model_time, average_model_star_value,
                        color=colors.green[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.3
                    )

                    # Plot the average model star's birth and current values
                    for t, x, size, marker in ((-1, -1, 12, '*'), (0, 0, 6, 'o')):
                        ax.scatter(
                            np.array([self.model_time[t]]),
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
                        self.model_time[::-1], model_star_value,
                        color=colors.blue[6], alpha=0.8, linestyle=linestyle,
                        linewidth=1.0, solid_capstyle='round', dash_capstyle='round', zorder=0.2
                    )

                    # Plot model stars' birth and current values
                    for t, x, size, marker in ((-1, 0, 12, '*'), (0, -1, 6, 'o')):
                        ax.scatter(
                            np.array([self.model_time[t]]),
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
            self.set_legend(ax, 4)

        # Set label
        ax.set_xlabel('Epoch (Myr)', fontsize=8)

        # Set limits
        ax.set_xlim(np.min(self.time), np.max(self.time) + 1)

    def draw_time(
        self, coord, system, style, age=None, metric=None,
        title=False, forced=None, default=None, cancel=None
    ):
        """Draws the positions or velocities of stars over time."""

        # Initialize figure
        fig, axes = self.set_figure(
            style, 'hide_x', 'label_right', adjust='height', left=0.428,
            h_align='right' if style == '3x1' else None,# ratio=0.8,
            styles=('1x3', '2x2', '3x1')
        )

        # Plot xyz position
        for y in range(3):
            if style == '2x2':
                self.plot_time((axes[0,0], axes[1,0], axes[1,1])[y], y, coord, system, age, metric)
            else:
                self.plot_time(axes.flatten()[y], y, coord, system, age, metric)

        # Plot the average xyz position
        if style == '2x2':
            self.plot_time(axes[0,1], None, coord, system, age, metric)

        # Set title
        self.set_title(
            title, fig, self.get_title_coord, coord, system,
            '{system} {coord}s', line_2=f'over time'
        )

        # Save figure
        self.save_figure(
            f"{coord}_{system}_{style}_{self.name}_time.pdf", fig,
            tight=title, forced=forced, default=default, cancel=cancel
        )

    def plot_scatter(
        self, ax, axes, coords, system, step, errors=False, labels=False,
        mst=False, ellipses=True, relative=False, values=None, limits=None
    ):
        """
        Creates a 2d or 3d scatter of positions or velocities, at a given step. 'errors', adds
        error  bars, 'labels' adds the stars' names, 'mst' adds the minimun spanning tree branches,
        'ellipses' adds 1σ and 2σ probability ellipses, and 'relative' cause the use of relative
        values instead absolute values.
        """

        def bivariate_gauss_to_ellipsoid(cov_matrix):
            """
            Computes the semi-major (a) and semi-minor (b) axes, and the rotation angle (θ) of the
            ellipsoid based on the covariance matrix at 1σ and 2σ.
            """

            from scipy.special import erf

            # Compute inverse matrix
            inv_matrix = np.linalg.inv(cov_matrix)

            # Compute θ, with θ = π / 2 if the covariance matrix is diagonal
            if inv_matrix[0, 1] == 0.0:
                θ = np.pi / 2.
            else:
                g = (inv_matrix[0, 0] - inv_matrix[1, 1]) / inv_matrix[0, 1]
                θ = np.arctan((np.sqrt(g**2 + 4) - g) / 2)

            # Computre ρ_u and ρ_v
            ρ_u = np.abs(
                inv_matrix[0, 0] * np.cos(θ)**2 +
                inv_matrix[1, 1] * np.sin(θ)**2 +
                inv_matrix[0, 1] * np.sin(θ) * np.cos(θ) * 2
            )
            ρ_v = np.abs(
                inv_matrix[0, 0] * np.sin(θ)**2 +
                inv_matrix[1, 1] * np.cos(θ)**2 -
                inv_matrix[0, 1] * np.sin(θ) * np.cos(θ) * 2
            )

            # Computre the 2D 1σ and 2σ semi-major and semi-minor axes
            prob = np.array([erf(1 / np.sqrt(2)), erf(2 / np.sqrt(2))])
            r = np.sqrt(-np.log(1. - prob)) * 1.36
            a = r / np.sqrt(ρ_u)
            b = r / np.sqrt(ρ_v)

            return a, b, θ

        def singular_value_decomposition(cov_matrix):
            """
            Computes the semi-major (a) and semi-minor (b) axes, and the rotation angle (θ) of the
            1σ error bar using singular value decomposition.
            """

            u, w, v = np.linalg.svd(cov_matrix)
            rot_matrix = np.linalg.inv(u)
            rot_matrix /= np.linalg.det(rot_matrix)
            θ = np.arctan(rot_matrix[0, 1] / rot_matrix[0, 0])
            a, b = np.sqrt(w) #* 1.36

            return a, b, θ

        def rotate(x, y, θ, x0, y0):
            """
            Rotates 2D vectors of coordinates x and y by an angle θ, and translate the result to
            x0 and y0 coordinates.
            """

            x2 = x * np.cos(θ) - y * np.sin(θ) + x0
            y2 = x * np.sin(θ) + y * np.cos(θ) + y0

            return np.array([x2, y2])

        # Check the types and values of axes
        self.check_type(axes, 'axes', 'tuple')
        ndim = len(axes)
        self.stop(
            ndim not in (2, 3), 'ValueError',
            "'axes' must have a length of 2 or 3 ({} given).", ndim
        )
        for i in axes:
            self.check_axis(i, 'i')

        # Select projection
        if ndim == 2:
            projection = '2d'
        if ndim == 3:
            projection = '3d'

        # Check the types and values of coords
        self.check_type(coords, 'coords', ('tuple', 'list', 'string'))
        coords = [coords] if type(coords) not in (tuple, list) else coords
        self.stop(
            len(coords) not in (1, 2), 'ValueError',
            "'coords' must have a length of 1 or 2 ({} given).", len(coords)
        )
        for coord in coords:
            self.check_coord(coord)

        # Adjust the length of coords, if needed
        if len(coords) == 1 and projection == '2d':
            coords *= 2
        if projection == '3d':
            self.stop(
                len(coords) == 2, 'ValueError', "'coords' must have a length of 1 "
                "if the projection is '3d' ({} given).", len(coords)
            )
            coords *= 3

        # Check the type and value of system
        system = self.check_system(system)

        # Check the types of errors, labels, mst, relative and ellipses
        self.check_type(errors, 'errors', 'boolean')
        self.check_type(labels, 'labels', 'boolean')
        self.check_type(mst, 'mst', 'boolean')
        self.check_type(relative, 'relative', 'boolean')
        self.check_type(ellipses, 'ellipses', 'boolean')

        # Value and error coordinates
        value_coords = [
            f"{'relative_' if relative else ''}{coords[i]}_{system}" for i in range(ndim)
        ]
        error_coords = [f'{coords[i]}_{system}_error' for i in range(ndim)]

        # Select coordinates
        for star in self.sample:
            value = np.array([vars(star)[value_coords[i]][step, axes[i]] for i in range(ndim)])
            error = np.array(
                [np.diag(vars(star)[error_coords[i]][step])[axes[i]] for i in range(ndim)]
            )**0.5

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

            # Error bars
            if errors:
                if projection == '2d':

                    # Rotated error bars for 2d correlation scatters, only if errors make sense
                    if (
                        self.from_data and self.number == 0
                        and value_coords[0] == value_coords[1]
                    ):
                        all_values = np.array(
                            [
                                [
                                    vars(group[star.index - 1])[value_coords[i]][step, axes[i]]
                                    for i in range(ndim)
                                ] for group in self[1:]
                            ]
                        )

                        # Compute rotated horizontal and vertical error bars
                        a, b, θ = singular_value_decomposition(np.cov(all_values.T))
                        error_a = rotate(np.array([-a, a]), np.zeros(2), θ, *value)
                        error_b = rotate(np.zeros(2), np.array([-b, b]), θ, *value)

                        # Plot error bars
                        ax.plot(
                            error_a[0], error_a[1], color=color, alpha=0.6,
                            linewidth=0.25, linestyle='-', zorder=0.3
                        )
                        ax.plot(
                            error_b[0], error_b[1], color=color, alpha=0.6,
                            linewidth=0.25, linestyle='-', zorder=0.3
                        )

                    # Non-rotated error bar for 2d cross correlation scatters
                    else:
                        ax.plot(
                            (value[0] - error[0], value[0] + error[0]), (value[1], value[1]),
                            color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                        )
                        ax.plot(
                            (value[0], value[0]), (value[1] - error[1], value[1] + error[1]),
                            color=color, alpha=0.6, linewidth=0.25, linestyle='-', zorder=0.3
                        )

                # Non-rotated error bar for 3d scatters
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
                        value[0] + 2, value[1] + 3, star.name,
                        color=color, horizontalalignment='left',
                        verticalalignment='top', fontsize=4, zorder=0.9
                    )
                if projection == '3d':
                    ax.text(
                        value[0] + 2, value[1] + 3, value[2] + 2, star.name,
                        color=color, horizontalalignment='left',
                        verticalalignment='top', fontsize=4, zorder=0.9
                    )

        # Find minimum spanning tree branches, if needed
        if mst and len(set(coords)) == 1:
            mst = f'mst_{coords[0]}_{system}'
            if mst not in vars(self).keys():
                self.get_tree_branches()
            mst = vars(self)[mst]

            # Select branches
            for branch in mst[step]:
                value_start = [
                    vars(branch.start)[value_coords[i]][step, axes[i]] for i in range(ndim)
                ]
                value_end = [
                    vars(branch.end)[value_coords[i]][step, axes[i]] for i in range(ndim)
                ]

                # Plot branches
                if projection == '2d':
                    ax.plot(
                        (value_start[0], value_end[0]),
                        (value_start[1], value_end[1]),
                        color=colors.blue[6], alpha=0.6, linestyle='-',
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )
                if projection == '3d':
                    ax.plot(
                        (value_start[0], value_end[0]),
                        (value_start[1], value_end[1]),
                        (value_start[2], value_end[2]),
                        color=colors.blue[6], alpha=0.6, linestyle='-',
                        linewidth=0.5, solid_capstyle='round', zorder=0.2
                    )

        # Draw values
        if projection == '2d':
            if values is None:
                values = [vars(self)[value_coords[i]][step, axes[i]] for i in range(ndim)]
            ax.axvline(
                values[0], color=colors.black, alpha=0.8,
                linewidth=0.5, linestyle='--', zorder=0.9
            )
            ax.axhline(
                values[1], color=colors.black, alpha=0.8,
                linewidth=0.5, linestyle='--', zorder=0.9
            )

        # Draw ellipses
        if ellipses:
            all_values = np.array(
                [
                    [vars(star)[value_coords[i]][step, axes[i]] for i in range(ndim)]
                    for star in self.sample
                ]
            )

            # Compute semi-major and semi-minor axes, and angle
            a, b, θ = bivariate_gauss_to_ellipsoid(np.cov(all_values.T))

            # Compute, rotate and translate ellipses coordinates
            t = np.linspace(0, 2 * np.pi, 200)
            for i in range(a.size):
                x, y = a[i] * np.cos(t), b[i] * np.sin(t)
                x, y = rotate(x, y, θ, np.mean(all_values[:,0]), np.mean(all_values[:,1]))

                # Plot 1σ and 2σ probabilities
                ax.fill(x, y, facecolor=colors.azure[9], alpha=0.3, linestyle='None', zorder=0.1)

        # Set limits
        if limits is not None:
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            if projection == '3d':
                ax.set_zlim(*limits[2])

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
        self, coord, system, style, step=None, age=None, axis3d=True, errors=False, labels=False,
        mst=False, ellipses=False, title=False, forced=None, default=None, cancel=None
    ):
        """
        Draws 2d and 3d scatter plots of positions or velocities of stars at a given 'step' or
        'age' in Myr.
        """

        # Check the type of axis3d and set style, if needed
        self.check_type(axis3d, 'axis3d', 'boolean')
        if axis3d and style not in ('2x2', '3x3', '4x1'):
            axis3d = False
        if not axis3d and style == '4x1':
            style = '3x1'

        # Initialize figure
        fig, axes = self.set_figure(
            style, 'label_right', 'hide_x' if style in ('2x2', '2x3', '3x3') else '',
            'hide_y' if style == '3x3' else '', 'corner' if style in ('2x2', '3x3') else '',
            '3d' if axis3d else '', styles=('1x3', '2x2', '2x3', '3x1', '3x2', '3x3', '4x1')
        )

        # Select axes in accordance to style
        if style in ('1x3', '3x1', '4x1'):
            ax0, ax1, ax2 = axes.flatten()[:3]
        if style == '2x3':
            ax0, ax1, ax2 = axes[1]
            ax4, ax5, ax6 = axes[0]
        if style == '3x2':
            ax0, ax1, ax2 = axes[:,0]
            ax4, ax5, ax6 = axes[:,1]
        if style == '2x2':
            ax0, ax1, ax2, ax3 = axes[0,0], axes[1,0], axes[1,1], axes[0,1]
        if style == '3x3':
            ax0, ax1, ax2, ax3 = axes[1,0], axes[2,0], axes[2,1], axes[0,2]
            ax4, ax5, ax6 = np.diag(axes)
            ax3 = axes[0,2]
        if style in ('4x1'):
            ax3 = axes[3,0]

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Find limits
        values, limits = self.get_limits(coord, system, step)

        # Plot positions and velocities (1d)
        if style in ('2x3', '3x2', '3x3'):
            for ax, i in zip((ax4, ax5, ax6), range(3)):
                ax.value = self.plot_histogram(
                    ax, values[i], 20, fit='skewnormal', limits=limits[i],
                    label=vars(systems[system])[coord][i].name,
                    orientation='vertical' if style in ('2x3', '3x3') else 'horizontal'
                )[0]

                # Set labels and ticks
                ax.tick_params(labelleft=False, labelbottom=False, labelright=False)

        # Plot positions or velocites (2d)
        if style in ('2x3', '3x2'):
            i, j = zip(*((0, 1), (1, 2), (2, 0)))
        else:
            i, j = zip(*((0, 1), (0, 2), (1, 2)))
        for ax, x, y in zip((ax0, ax1, ax2), i, j):
            if style == '2x3':
                values = (axes[0,x].value, axes[0,y].value)
            elif style == '3x2':
                x, y = y, x
                values = (axes[x,1].value, axes[y,1].value)
            elif style == '3x3':
                values = (axes[x,x].value, axes[y,y].value)
            else:
                values = None
            self.plot_scatter(
                ax, (x, y), coord, system, step, errors=errors, labels=labels,
                mst=mst, ellipses=ellipses, values=values, limits=(limits[x], limits[y])
            )

        # Plot positions or velocites (3d)
        if axis3d:
            self.plot_scatter(
                ax3, (0, 1, 2), coord, system, step, errors=errors,
                labels=labels, mst=mst, ellipses=False, limits=limits
            )

        # Set title
        self.set_title(
            title, fig, self.get_title_coord, coord, system,
            '{system} {coord}s', line_2=f'at {age:.1f} Myr'
        )

        # Save figure
        self.save_figure(
            f'{coord}_{system}_{style}_{self.name}_{age:.1f}Myr.pdf', fig,
            forced=forced, default=default, cancel=cancel
        )

    def draw_cross_scatter(
        self, system, style, step=None, age=None, errors=False, labels=False,
        ellipses=False, title=False, forced=None, default=None, cancel=None
    ):
        """
        Draws 2s cross scatter plots of positions and velocities of stars, at a given 'step' or
        'age' in Myr.
        """

        # Initialize figure
        fig, axes = self.set_figure(style, 'hide_x', 'hide_y', styles=('3x3', '4x4'))

        # Select axes in accordance to style
        if style == '3x3':
            axes_scatter = axes
        if style == '4x4':
            axes_distribution = np.concatenate((axes[0,:3], axes[1:,3]))
            axes_scatter = axes[1:,:3]

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Find values and limits
        values, limits = {}, {}
        values['position'], limits['position'] = self.get_limits('position', system, step)
        values['velocity'], limits['velocity'] = self.get_limits('velocity', system, step)

        # Plot positions and velocities (1d)
        if style in ('4x4',):
            for i in range(6):
                j = i - (0 if i < 3 else 3)
                coord = 'position' if i < 3 else 'velocity'
                axes_distribution[i].value = self.plot_histogram(
                    axes_distribution[i], values[coord][j], 20, fit='skewnormal',
                    limits=limits[coord][j], label=vars(systems[system])[coord][j].name,
                    orientation='vertical' if i < 3 else 'horizontal'
                )[0]

                # Set labels and ticks
                axes_distribution[i].tick_params(labelleft=False, labelbottom=False)

        # Plot positions and velocities
        for x, y in [(x, y) for y in range(3) for x in range(3)]:
            values = (axes[0, x].value, axes[y + 1, 3].value) if style == '4x4' else None
            self.plot_scatter(
                axes_scatter[y, x], (x, y), ('position', 'velocity'), system, step,
                errors=errors, labels=labels, ellipses=ellipses, values=values,
                limits=(limits['position'][x], limits['velocity'][y])
            )

        # Set title
        self.set_title(
            title, fig, self.get_title_coord, 'position', system,
            '{position} positions and {velocity} velocities', line_2=f'at {age:.1f} Myr'
        )

        # Save figure
        self.save_figure(
            f'position_velocity_cross_{system}_{style}_{self.name}_{age:.1f}Myr.pdf', fig,
            forced=forced, default=default, cancel=cancel
        )

    def draw_time_scatter(
        self, coord, system, style, steps=None, ages=None, errors=False, labels=False,
        mst=False, ellipses=True, title=False, forced=None, default=None, cancel=None
    ):
        """
        Draws 2d and 3d scatter plots of positions or velocities of stars, for 2 or 3 'step' or
        'age' in Myr.
        """

        # Initialize figure
        fig, axes = self.set_figure(
            style, 'hide_y', '3d' if style in ('4x2', '4x3') else '',
            'label_right', styles=('3x2', '3x3', '4x2', '4x3')
        )

        # Set the number of timesteps
        number_of_steps = int(style[-1])

        # Check the type and value of steps
        self.check_type(steps, 'steps', ('tuple', 'list', 'None'))
        if steps is None:
            steps = number_of_steps * [None]
        self.stop(
            len(steps) != number_of_steps, 'ValueError',
            "'steps' must be have a length of {} with the style '{}' ({} given).",
            number_of_steps, style, len(steps)
        )

        # Check the type and value of ages
        self.check_type(ages, 'ages', ('tuple', 'list', 'None'))
        if ages is None:
            ages = number_of_steps * [None]
        self.stop(
            len(ages) != number_of_steps, 'ValueError',
            "'ages' must be have a length of {} with the style '{}' ({} given).",
            number_of_steps, style, len(ages)
        )

        # Compute the values of step and age
        steps, ages = zip(*[self.get_step_age(step, age) for step, age in zip(steps, ages)])

        # Find limits
        limits = np.array(
            [self.get_limits(coord, system, step, relative=True)[1] for step in steps]
        )
        limits = limits[np.argmax(limits[:,0,1] - limits[:,0,0]),:,:]

        # Plot positions or velocities (2d)
        for i in range(3):
            x, y = ((0, 1), (2, 1), (0, 2))[i]
            for j in range(number_of_steps):
                self.plot_scatter(
                    axes[i, j], (x, y), coord, system, steps[j], errors=errors, labels=labels,
                    mst=mst, ellipses=ellipses, relative=True, limits=(limits[x], limits[y])
                )

        # Plot positions or velocities (3d)
        if style in ('4x2', '4x3'):
            for ax, step in zip(axes[3], steps):
                self.plot_scatter(
                    ax, (0, 1, 2), coord, system, step, errors=errors, labels=labels,
                    mst=mst, ellipses=False, relative=True, limits=limits
                )

        # Set title
        ages_str = [f'{age:.1f}' for age in ages]
        self.set_title(
            title, fig, self.get_title_coord, coord, system, '{system} {coord}s',
            line_2=f"at {enumerate_strings(*ages_str, conjunction='and')} Myr"
        )

        # Save figure
        self.save_figure(
            f"{coord}_{system}_{style}_{self.name}_{'_'.join(ages_str)}Myr.pdf", fig,
            forced=forced, default=default, cancel=cancel
        )

    def draw_corner_scatter(
        self, system, step=None, age=None, axis3d=True, errors=False, labels=False, mst=False,
        ellipses=False, flip=False, title=False, forced=None, default=None, cancel=None
    ):
        """
        Draws corner scatter plots of positions and velocities, at a given 'step' or 'age' in Myr,
        including 1d distributions, and 2d and 3d scatter and cross scatter plots.
        """

        # Check the type of axis3d and flip
        self.check_type(axis3d, 'axis3d', 'boolean')
        self.check_type(flip, 'flip', 'boolean')

        # Initialize figure
        fig, axes = self.set_figure(
            '6x6', 'hide_x', 'hide_y', '3d' if axis3d else '', bottom=0.345
        )

        # Compute the values of step and age
        step, age = self.get_step_age(step, age)

        # Find values and limits
        values, limits = {}, {}
        values['position'], limits['position'] = self.get_limits('position', system, step)
        values['velocity'], limits['velocity'] = self.get_limits('velocity', system, step)

        # Plot positions and velocities (1d)
        for i in range(6):
            j = i - (0 if i < 3 else 3)
            coord = 'position' if i < 3 else 'velocity'
            axes[i,i].value = self.plot_histogram(
                axes[i, i], values[coord][j], 20, fit='skewnormal',
                limits=limits[coord][j], label=vars(systems[system])[coord][j].name,
                orientation='vertical' if i < 3 or not flip else 'horizontal'
            )[0]

        # Plot positions and velocities (2d)
        for x, y in filter(lambda i: i[0] < i[1], [(x, y) for y in range(6) for x in range(6)]):
            i, j = x - (0 if x < 3 else 3), y - (0 if y < 3 else 3)
            coord_x = 'position' if x < 3 else 'velocity'
            coord_y = 'position' if y < 3 else 'velocity'
            self.plot_scatter(
                axes[y, x], (i, j), (coord_x, coord_y), system, step, errors=errors,
                labels=labels, ellipses=ellipses, values=(axes[x, x].value, axes[y, y].value),
                limits=(limits[coord_x][i], limits[coord_y][j])
            )

        # Plot positions and velocites (3d)
        if axis3d:
            self.plot_scatter(
                axes[1,2], (0, 1, 2), 'position', system, step, errors=False,
                labels=labels, mst=mst, ellipses=False, limits=limits['position']
            )
            self.plot_scatter(
                axes[3,4], (0, 1, 2), 'velocity', system, step, errors=False,
                labels=labels, mst=mst, ellipses=False, limits=limits['velocity']
            )

        # Set labels and ticks
        axes[0, 0].tick_params(labelleft=False)
        if flip:
            axes[-1, -1].tick_params(labelbottom=False)
        else:
            axes[-1, -1].set_xlabel(axes[-1, -2].get_ylabel())

        # Set title
        self.set_title(
            title, fig, self.get_title_coord, 'position', system,
            '{position} positions and {velocity} velocities', line_2=f'at {age:.1f} Myr'
        )

        # Save figure
        self.save_figure(
            f'position_velocity_corner_{system}_{self.name}_{age:.1f}Myr.pdf', fig,
            forced=forced, default=default, cancel=cancel
        )

    def draw_age_distribution_group(
        self, metric, index=None, fit=None, number_of_bins=120,
        title=False, forced=None, default=None, cancel=None
    ):
        """Draws a plot of the age distribution of a group for a given metric."""

        # Initialize figure
        fig, axes = self.set_figure('1x1')

        # Retrieve metric
        metric, index = self.get_metric(metric, index)

        # Plot histogram
        if metric.metric.status:
            ages = metric.ages[:,index]
            self.plot_histogram(
                axes[0,0], ages, number_of_bins, fit=fit, value=metric.age[index],
                error=metric.age_int_error[index], limits=(np.min(ages), max((np.max(ages), 1.0))),
                label=metric.metric.name[index], error_lines=False, curve_color=colors.lime[4],
                hist_color=colors.lime[7], line_color=colors.lime[4]
            )

        # Logging
        else:
            self.log(
                "Could not use '{}' metric for '{}' group. It was not computed.",
                metric.name[index], self.name, display=True
            )

        # Plot results from Miret-Roig et al. (2020) and Crundall et al. (2019)
        # self.plot_MiretRoig2020_Crundall2019(axes[0,0])

        # Set title
        self.set_title(
            title, fig, lambda fig, title: (title, title.replace('\n', ' ')),
            f'Age distribution of {self.number_of_iterations} iterations of {self.name},\n'
            f'using the {metric.metric.name[index]} as association size metric'
        )

        # Set legend
        self.set_legend(axes[0,0], 2)

        # Set labels
        axes[0,0].set_xlabel('Age (Myr)', fontsize=8)
        axes[0,0].set_ylabel('Density', fontsize=8)

        # Set limits
        axes[0,0].set_xlim(-50, -15)
        axes[0,0].set_ylim(0, 0.49)

        # Save figure
        self.save_figure(
            f"age_distribution_{self.name}_{metric.metric.name[index].replace(' ', '_')}.pdf",
            fig, tight=False, forced=forced, default=default, cancel=cancel
        )

    def draw_map(
        self, age=None, metric=None, index=None, labels=False,
        title=False, forced=None, default=None, cancel=None
    ):
        """
        Creates a Mollweide projection of a traceback. For this function to work, uvw velocities
        must not compensated for the sun velocity when computing xyz positions. Otherwise, the
        vantage point doesn't move and the trajectories don't match the observed proper motion.
        Coordinate.sun_velocity = np.zeros(3).
        """

        # Initialize figure
        fig, axes = self.set_figure(
            '1x1', 'mollweide', width=7.0900, bottom=0.0815,
            ratio=0.50, adjust='ax_width', h_align='center'
        )

        # Check the type of labels
        self.check_type(labels, 'labels', 'boolean')

        # Birth index, age and age error
        if age is not None or metric is not None:
            birth_index, age, age_error = self.get_epoch(age=age, metric=metric, index=index)

        # Compute Sun's orbit
        sun = self.Star(
            self, name='sun', time=self.time,
            position_xyz=np.zeros(3), velocity_xyz=np.zeros(3),
        )

        # Compute coordinates
        for star in self:
            positions_πδα = position_xyz_to_πδα(*(star.position_xyz - sun.position_xyz).T).T
            alphas = positions_πδα[:,2] - 2 * np.pi * (positions_πδα[:,2] > np.pi)
            deltas = positions_πδα[:,1]

            # Identify discontinuties
            discontinuties = (
                np.abs(alphas[1:] - alphas[:-1]) > 3 * np.pi / 2
            ).nonzero()[0] + 1

            # Find individual segments
            segments = []
            lower_limit = 0
            for upper_limit in discontinuties:
                segments.append(range(lower_limit, upper_limit))
                lower_limit = upper_limit
            segments.append(np.arange(lower_limit, alphas.shape[0]))

            # Plot individual segments
            color = colors.red[6] if star.outlier else colors.black
            for i in segments:
                axes[0,0].plot(
                    alphas[i], deltas[i], color = color, alpha=0.6,
                    linewidth=0.5, solid_capstyle='round', zorder=0.6
                )

            # Plot current-day position
            axes[0,0].scatter(
                alphas[0], deltas[0], color=colors.black + (0.4,), edgecolors=colors.black,
                alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.7
            )

            # Plot birth position
            if age is not None or metric is not None:
                color = colors.red[6] if star.outlier else colors.blue[6]
                axes[0,0].scatter(
                    alphas[birth_index], deltas[birth_index], color=color + (0.4,),
                    edgecolors=color, alpha=None, s=6, marker='o', linewidths=0.25, zorder=0.7
                )

            # Show labels
            if labels:
                axes[0,0].text(
                    alphas[0] + 0.2, deltas[0] + 0.3, star.name, color=colors.black,
                    horizontalalignment='left', verticalalignment='top', fontsize=6, zorder=0.9
                )

        # Plot proper motion arrows
        for star in self.data.sample:
            axes[0,0].arrow(
                star.position.values[2] - (2 * np.pi if star.position.values[2] > np.pi else 0.0),
                star.position.values[1], -star.velocity.values[2] / 4, -star.velocity.values[1] / 4,
                color=colors.black, linewidth=0.5, joinstyle='round',
                head_width=0.03, head_length=0.03, zorder=0.8
            )

        # Set title
        self.set_title(
            title, fig, lambda fig, title: (title, title),
            f'Mollweide projection of {self.name} over {self.duration.value:.1f} Myr'
        )

        # Set grid
        axes[0,0].tick_params(top=False, bottom=False)
        axes[0,0].grid(color=colors.grey[17], linewidth=0.5, alpha=0.0)

        # Save figure, without a grid
        self.save_figure(
            f'Mollweide_{self.name}.pdf', fig, close=False,
            forced=forced, default=default, cancel=cancel
        )

        # Copy grid objects and add them back to the figure, with a new zorders and alphas
        from copy import copy
        for tick in axes[0,0].yaxis.majorTicks + axes[0,0].xaxis.majorTicks:
            gridline = copy(tick.gridline)
            gridline.set_zorder(0.3)
            gridline.set_alpha(1.0)
            axes[0,0].add_artist(gridline)

        # Save figure, with a grid
        self.save_figure(
            f'Mollweide_{self.name}.pdf', fig, logging=False,
            forced=True, default=False, cancel=False
        )

    def create_kinematics_table(
        self, save=False, show=False, machine=False, age=None,
        forced=None, default=None, cancel=None
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
                star.position_xyz[epoch_index], 'pc',
                np.sqrt(np.diag(star.position_xyz_error[epoch_index]))
            )
            velocity_xyz = Quantity(
                star.velocity_xyz[epoch_index], 'pc/Myr',
                np.sqrt(np.diag(star.velocity_xyz_error[epoch_index]))
            ).to('km/s')

            return position_xyz, velocity_xyz

        # Check the types of save, show and machine
        self.check_type(save, 'save', 'boolean')
        self.check_type(show, 'show', 'boolean')
        self.check_type(machine, 'machine', 'boolean')

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
            self.save_table(
                f'kinematics_{self.name}_{age}Myr.csv', f'Kinematics of {self.name}', lines,
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

    def create_kinematics_time_table(
        self, save=False, show=False, machine=False,
        forced=None, default=None, cancel=None
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
        self.check_type(save, 'save', 'boolean')
        self.check_type(show, 'show', 'boolean')
        self.check_type(machine, 'machine', 'boolean')

        # Create header
        if machine:
            lines = [
                'Time,X,Y,Z,U,V,W,xi,eta,zeta,v_xi,v_eta,v_zeta'
            ]

            # Create lines
            for t in np.arange(self.time.size):
                lines.append(
                    (
                        f'{self.time[t]},'
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
            for t in np.arange(self.time.size):
                lines.append(
                    (
                        f'{self.time[t]:<8.1f}'
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
            self.save_table(
                f'kinematics_time_{self.name}', f'Kinematics over time of {self.name}', lines,
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )

    def create_kinematics_time_table_star(
        self, save=False, show=False, machine=False,
        forced=None, default=None, cancel=None
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
        self.group.check_type(save, 'save', 'boolean')
        self.group.check_type(show, 'show', 'boolean')
        self.group.check_type(machine, 'machine', 'boolean')

        # Create header
        if machine:
            lines = ['Time,X,Y,Z,U,V,W,xi,eta,zeta,v_xi,v_eta,v_zeta']

            # Create lines
            for t in np.arange(self.group.time.size):
                lines.append(
                    (
                        f'{self.group.time[t]},'
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
            for t in np.arange(self.group.time.size):
                lines.append(
                    (
                        f'{self.group.time[t]:<8.1f}'
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
            self.group.save_table(
                f'kinematics_time_{self.name}', f'Kinematics over time of {self.name}', lines,
                extension='csv' if machine else 'txt',
                forced=forced, default=default, cancel=cancel
            )