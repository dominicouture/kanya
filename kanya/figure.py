# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""figure.py: Defines functions used to create figures."""

import numpy as np
from matplotlib import pyplot as plt, ticker as tkr
from colorsys import hls_to_rgb
from cycler import cycler

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Set fonts
plt.rc('font', serif='Latin Modern Roman', family='serif', size='8')
plt.rc('mathtext', fontset='custom', it='Latin Modern Roman:italic', rm='Latin Modern Roman:roman')
# plt.rc('lines', markersize=4)
# plt.rc('pdf', fonttype=42)
# plt.rc('text', usetex=True)

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
    metric = (
        green[3], green[6], green[9], green[12], azure[3], azure[6], azure[9], azure[12],
        pink[3], pink[6], pink[9], pink[12]
    )

    # Color cycle
    cycle = cycler(color=(azure[6], pink[6], chartreuse[6], indigo[9], orange[9], lime[9]))
    colors = (azure[6], pink[6], chartreuse[6], indigo[9], orange[9], lime[9])

def set_figure(
    style, *options, width=None, height=None, left=None, bottom=None, right=None,
    top=None, colpad=None, rowpad=None, ratio=None, adjust=None, v_align=None, h_align=None,
    styles=None
):
    """
    Initializes a figure with multiple axes. The axes are created with at the correct position
    and with the correct size. Ticks and labels are repositioned and set invisible accordingly.
    """

    # Check the type and value of style
    styles = styles if styles is not None else (
        '1x1', '1x2', '1x3',
        '2x1', '2x2', '2x3',
        '3x1', '3x2', '3x3',
        '4x1', '4x2', '4x3',
        '4x4', '6x6', '2+1',
    )

    # Number of axes in y and x
    nrow = int(style[0])
    ncol = int(style[-1])

    # Set margins if the axes and tick labels are moved to the right of the rightmost axes
    if 'label_right' in options and ncol == 2:
        right = left if right is None else right
        colpad = 0.100

    # Default margins, in inches
    left = 0.445 if left is None else left
    bottom = 0.335 if bottom is None else bottom
    right = 0.005 if right is None else right
    top = 0.0815 if top is None else top
    colpad = 0.100 if 'hide_y' in options else 0.5400 if colpad is None else colpad
    rowpad = 0.100 if 'hide_x' in options else 0.3654 if rowpad is None else rowpad
    ratio = 1.0 if ratio is None else ratio

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
            6.5 if width is None else width,
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
            6.5 if width is None else width,
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

    # Hide x axis and tick labels
    if 'hide_x' in options and nrow > 1:
        corner3d = ('2x2', '3x3', '6x6')
        for ax in filter(
            lambda ax: ax is not None,
            axes[:nrow - 1 - (1 if style not in corner3d and '3d' in options else 0)].flatten()
        ):
            if not ax.name == '3d':
                ax.set_xlabel('', visible=False)
                ax.tick_params(labelbottom=False)

    # Hide y axis and tick labels
    if 'hide_y' in options and ncol > 1:
        if not (ncol == 2 and 'label_right' in options):
            for ax in filter(
                lambda ax: ax is not None,
                axes[:nrow - (1 if style == '2x2' and '3d' in options else 0),1:].flatten()
            ):
                if not ax.name == '3d':
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
    # if style == '2x2' and '3d' in options and 'hide_x' in options:
    #     axes[0,0].set_xlabel('', visible=False)
    #     axes[0,0].tick_params(labelbottom=False)

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

def get_dimensions(
    width, height, left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio,
    adjust=None, h_align='left', v_align='bottom'
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

def set_legend(ax, loc, fontsize=8):
    """Sets the parameters of the legend of the axis."""

    legend = ax.legend(loc=loc, fontsize=fontsize, fancybox=False, borderpad=0.5, borderaxespad=1.0)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor(colors.white + (0.8,))
    legend.get_frame().set_edgecolor(colors.black)
    legend.get_frame().set_linewidth(0.5)
