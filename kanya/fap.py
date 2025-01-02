# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fap.py: Figures for AstroPhysics (FAP) provides tools used to create with ease figures adapted for
several astrophysical publications. It is built upon the matplotlib package.
"""

import numpy as np
from matplotlib import pyplot as plt, ticker as tkr
from colorsys import hls_to_rgb
from cycler import cycler

# Set fonts
plt.rc('font', serif='Latin Modern Roman', family='serif', size='8')
plt.rc('mathtext', fontset='custom', it='Latin Modern Roman:italic', rm='Latin Modern Roman:roman')

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
    colors = (azure[6], pink[6], chartreuse[6], indigo[9], orange[9], lime[9])

def set_figure(
    style, *options, width=None, height=None, left=None, bottom=None, right=None,
    top=None, colpad=None, rowpad=None, ratio=None, width_ratio=None, height_ratio=None,
    adjust=None, v_align=None, h_align=None
):
    """
    Initializes a figure with multiple axes. The axes are created with at the correct position
    and with the correct size. Ticks and labels are repositioned and set invisible accordingly.
    """

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
    width_ratio = np.ones(ncol) if width_ratio is None else width_ratio
    height_ratio = np.ones(nrow) if height_ratio is None else height_ratio

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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
            left, bottom, right, top, colpad, rowpad, nrow, ncol, ratio, width_ratio, height_ratio,
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
        for ax in filter(
            lambda ax: ax is not None,
            axes[:nrow - 1 - (1 if '3d' in options and style != '6x6' else 0)].flatten()
        ):
            if not (style in ('3x3', '6x6') and ax.name == '3d'):
                ax.set_xlabel('', visible=False)
                ax.tick_params(labelbottom=False)

    # Hide y axis and tick labels
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

def get_dimensions(
    fig_width, fig_height, left, bottom, right, top, colpad, rowpad, nrow, ncol,
    ratio, width_factor, height_factor, adjust='ratio', h_align='left', v_align='bottom'
):
    """
    Computes the dimensions (in inches) of the axes of a figure based on:

        -   fig_width (float): the width of the figure
        -   fig_height (float): the height of the figure
        -   left (float): left margin of the figure
        -   bottom (float): bottom margin of the figure
        -   right (float): right margin of the figure
        -   top (float): top margin of the figure
        -   rowpad (float or array): the padding between axes along the y dimension (rows)
        -   colpad (float or array): the padding between axes along the x dimension (cols)
        -   nrow (int): the number of axes along the y dimension (rows)
        -   ncol (int): the number of axes along the x dimension (columns)
        -   ratio (float): the width to height aspect ratio of the figures
        -   width_factor (float): multiplicative factor for every axis along the x dimension (column)
        -   height_factor (float): multiplicative factor for every axis along the y dimension (row)
        -   adjust (string): the behavior chosen to scale the figure or the axes
        -   h_align (string): the horizontal alignment of the axes
        -   v_align (string): the vertical alignment of the axes

    First, axes' widths and height are computed based on the nrow, ncol, ratio, width_factor,
    and height_factor arguments. Then the axes are scaled to fit the space according to various
    behaviors:

        -   ratio: the axes are scaled and the aspect ratio changed to fit the space along the
            width and height of the figure. The width and height of the figure, and the padding
            are unchanged.
        -   fig_width: the axes are scaled to fit the space along the height of the figure and
            the width of the figure is adjusted to match. The height of the figure, and the aspect
            ratio and padding of the axes are unchanged.
        -   fig_height: the axes are scaled to fit the space along the width of the figure and
            the height of the figure is adjusted to match. The width of the figure, and the aspect
            ratio and padding of the axes are unchanged.
        -   align: the axes are scaled to fit the space along the width or height of the figure,
            depending on the aspect of the figure and the axes. The width and height of the
            figure and the aspect ratio of the axes are unchanged. The axes are aligned
            horizontally or vertically. The possible horizontal alignments are: left (default),
            right, center, and justify (which alter the padding). The possible vertical alignments
            are: bottom (default), center, top, and justify (which alther the padding).

    Returns a (nrow, ncol, 4) array were the last dimension is used to save the 4 parameters of
    every axis: left, bottom, width, and height, all relative to the either the width or height
    of the figure. The axes are arranged in such a way that matrix representation of every axis
    set of parameter matches the actual arrangement of axes on the figure (e.g., the set of
    parameters located at row 0 and column 0 (0, 0) matches the upper left figure).
    """

    # Compute the x padding
    if type(colpad) in (float, int):
        colpad = float(colpad)
        x_pad = colpad * (ncol - 1)
        colpads = np.full(ncol - 1, colpad)
    else:
        colpads = np.array(colpad)
        x_pad = np.sum(colpads)

    # Compute the y padding
    if type(rowpad) in (float, int):
        rowpad = float(rowpad)
        y_pad = rowpad * (nrow - 1)
        rowpads = np.full(nrow - 1, rowpad)
    else:
        rowpads = np.array(rowpad)
        y_pad = np.sum(rowpads)

    # Compute the axes widths and heights
    ax_widths = np.ones(ncol) * ratio * width_factor
    ax_heights = np.ones(nrow) * height_factor

    # Compute the available space along the x and y dimensions
    x_space = fig_width - left - x_pad - right
    y_space = fig_height - bottom - y_pad - top

    # Compute the sums the widths and heights of the axes
    x_sum = np.sum(ax_widths)
    y_sum = np.sum(ax_heights)

    # Adjust the ratio of the axes
    if adjust == 'ratio':

        # Scale the axes to fit the available space
        ax_widths *= x_space / x_sum
        ax_heights *= y_space / y_sum

        # Recompute the sums the widths and heights of the axes
        x_sum = np.sum(ax_widths)
        y_sum = np.sum(ax_heights)

    # Adjust the width of the figure
    if adjust == 'fig_width':

        # Scale the axes to fit the available space
        ax_widths *= y_space / y_sum
        ax_heights *= y_space / y_sum

        # Recompute the sums the widths and heights of the axes
        x_sum = np.sum(ax_widths)
        y_sum = np.sum(ax_heights)

        # Compute the width and x space of the figure
        fig_width = left + x_sum + x_pad + right
        x_space = fig_width - left - x_pad - right

    # Adjust the width of the figure
    if adjust == 'fig_height':

        # Scale the axes to fit the available space
        ax_widths *= x_space / x_sum
        ax_heights *= x_space / x_sum

        # Recompute the sums the widths and heights of the axes
        x_sum = np.sum(ax_widths)
        y_sum = np.sum(ax_heights)

        # Compute the height and y space of the figure
        fig_height = bottom + y_sum + y_pad + top
        y_space = fig_height - bottom - y_pad - top

    # Adjust the alignment of the axes
    if adjust == 'align':

        # Scale the axes to fit the available space
        scale = y_space / y_sum if (y_sum / x_sum) >= (y_space / x_space) else x_space / x_sum
        ax_widths *= scale
        ax_heights *= scale

        # Recompute the sums the widths and heights of the axes
        x_sum = np.sum(ax_widths)
        y_sum = np.sum(ax_heights)

        # Horizontal alignment
        if h_align == 'right':
            left = fig_width - right - x_sum - x_pad
        if h_align == 'center':
            left = (fig_width - x_sum - x_pad) / 2
        if h_align == 'justify':
            colpads = (fig_width - left - x_sum - right) / x_pad * colpads
            x_pad = np.sum(colpads)

        # Vertical alignment
        if v_align == 'top':
            bottom = fig_height - top - y_sum - y_pad
        if v_align == 'center':
            bottom = (fig_height - y_sum - y_pad) / 2
        if v_align == 'justify':
            rowpads = (fig_height - bottom - y_sum - top) / y_pad * rowpads
            y_pad = np.sum(rowpads)

    # Compute relative parameters
    left /= fig_width
    bottom /= fig_height
    ax_widths /= fig_width
    ax_heights /= fig_height
    colpads /= fig_width
    rowpads /= fig_height

    # Compute axes parameters
    axes_params = np.array(
        [
            [
                [
                    left + np.sum(ax_widths[:x]) + np.sum(colpads[:x]),
                    bottom + np.sum(ax_heights[:y]) + np.sum(rowpads[:y]),
                    ax_widths[x], ax_heights[y]
                ] for x in list(range(ncol))
            ] for y in list(range(nrow))[::-1]
        ]
    )

    return fig_width, fig_height, axes_params

def get_axes(fig, axes_params, nrow, ncol, hide_x, hide_y, mask=None, remove_extra=True, **kwargs):
    """
    Add axes to the figure using the axes parameters (array). A mask can be used to remove
    specified axes. Extra dimension can be removed and keywords argument can be given to the
    figure's get_axes function. Returns an array of axes with the same dimensions the first 2
    dimensions of axes_params.
    """

    # Add extra dimensions and create an empty array
    while axes_params.ndim != 3:
        axes_params = axes_params[None]

    # Create default mask
    shape = nrow, ncol = axes_params.shape[:2]
    if mask is None:
        mask = np.ones(shape, dtype=bool)

    # Create corner mask
    elif type(mask) == str and mask in ('corner0', 'corner1', 'corner2', 'corner3'):
        mask = get_corner_mask(nrow, ncol, mask)

    # Create axes
    axes = np.empty(shape, dtype=object)
    for y in range(nrow):
        for x in range(ncol):
            if mask[y, x]:
                axes[y, x] = fig.add_axes(axes_params[y, x], **kwargs)

    # Remove extra dimensions
    if remove_extra:
        while axes.shape[0] == 1 and axes.ndim > 1:
            axes = axes[0]
        axes = axes[0] if axes.size == 1 else axes

    # Hide x axis and tick labels
    if hide_x and nrow > 1:
        for ax in filter(lambda ax: ax is not None, axes[:nrow - 1].flatten()):
            ax.set_xlabel('', visible=False)
            ax.tick_params(labelbottom=False)

    # Hide y axis and tick labels
    if hide_y and ncol > 1:
        for ax in filter(lambda ax: ax is not None, axes[:nrow, 1:].flatten()):
            ax.set_ylabel('', visible=False)
            ax.tick_params(labelleft=False)

    # Set ticks and spines
    for ax in filter(lambda ax: ax is not None, axes.flatten()):
        set_axis(ax, '2d')

    return axes

def get_corner_mask(nrow, ncol, style):
    """Creates a corner mask of a given style."""

    styles = {
        'corner0': lambda x, y: y >= x,
        'corner1': lambda x, y : np.max(y) <= x + y,
        'corner2': lambda x, y: y <= x,
        'corner3': lambda x, y : np.max(y) >= x + y
    }

    x, y = np.meshgrid(range(ncol), range(nrow))

    return styles[style](x, y)

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

def set_legend(ax, loc):
    """Sets the parameters of the legend of the axis."""

    legend = ax.legend(loc=loc, fontsize=8, fancybox=False, borderpad=0.5, borderaxespad=1.0)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor(colors.white + (0.8,))
    legend.get_frame().set_edgecolor(colors.black)
    legend.get_frame().set_linewidth(0.5)

def set_figure_trajectory():
    """Set a trajectory figure, including both coordinate systems."""

    # Options
    hide_x = True
    hide_y = True
    mask = np.array([[True, True, True, True], [True, False, True, False]])

    # Figure width and height
    fig_width = 6.5
    fig_height = 3.3115

    # Margins
    left = 0.405
    bottom = 0.325
    right = 0.005
    top = 0.0815

    # Padding
    rowpad = np.array([0.1])
    colpad = np.array([0.1, 0.5400, 0.1])

    # Number of rows and columns
    nrow = 2
    ncol = 4

    # Ratios
    ratio = 1.0
    width_factor = (2.5, 1.0, 2.5, 1.0)
    height_factor = (1.0, 2.5)

    # Compute width, height, and axis parameters
    width, height, axes_params = get_dimensions(
        fig_width, fig_height, left, bottom, right, top, colpad, rowpad,
        nrow, ncol, ratio, width_factor, height_factor,
        adjust='fig_height', h_align='justify', v_align='bottom'
    )

    # Create figure and axes
    fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
    axes = get_axes(fig, axes_params, nrow, ncol, hide_x, hide_y, mask=mask)

    return fig, axes

def set_example():
    """Sets an example figure."""

    # Options
    hide_x = True
    hide_y = True
    # mask = None
    mask = np.array([[True, True, True, True], [True, False, True, False]])

    # Figure width and height
    # fig_width = 3.3450
    # fig_width = 7.0900
    fig_width = 6.5
    fig_height = 3.3115
    # fig_height = 7.0900

    # Margins
    left = 0.445
    bottom = 0.335
    right = 0.005
    top = 0.0815

    # Padding
    # rowpad = 0.100 if hide_x else 0.3654
    # colpad = 0.100 if hide_y else 0.5400
    rowpad = np.array([0.1])
    colpad = np.array([0.1, 0.5400, 0.1])

    # Number of rows and columns
    nrow = 2
    ncol = 4

    # Ratios
    ratio = 1.0
    # width_factor = np.ones(ncol)
    width_factor = (2.5, 1.0, 2.5, 1.0)
    # height_factor = np.ones(nrow)
    height_factor = (1.0, 2.5)

    # Compute width, height, and axis parameters
    width, height, axes_params = get_dimensions(
        fig_width, fig_height, left, bottom, right, top, colpad, rowpad,
        nrow, ncol, ratio, width_factor, height_factor,
        adjust='fig_height', h_align='justify', v_align='bottom'
    )

    # Create figure and axes
    fig = plt.figure(facecolor=colors.white, figsize=(width, height), dpi=300)
    axes = get_axes(fig, axes_params, nrow, ncol, hide_x, hide_y, mask=mask)

    plt.savefig('marmalade.pdf')

if __name__ == '__main__':

    set_example()