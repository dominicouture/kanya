# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" output.py: Provides the necesary functions to output data in table, graphical or video form. """

import numpy as np
from matplotlib import rcParams, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from astropy import units as un
from os import path
from series import info

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

def create_size_indicators_plot(series, secondary=False):
    """ Creates a graph of scatter, median absolute deviation, and minimum spanning tree branches
        length mean and median absolute deviation over the entire duration of the data.
    """

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # Scatter
    mean_scatter = np.mean([group.scatter for group in series], axis=0)
    mean_scatter = mean_scatter / mean_scatter[0]
    scatter_ages = [group.scatter_age for group in series]
    scatter_age = np.round(np.mean(scatter_ages), 3)
    scatter_age_error = np.round(np.std(scatter_ages), 3)
    plt.plot(series.time, mean_scatter, 'k-', linewidth=2.0,
        label='Scatter age: ({} ± {}) Myr'.format(scatter_age, scatter_age_error))

    # MAD
    mean_mad = np.mean([group.mad for group in series], axis=0)
    mean_mad = mean_mad / mean_mad[0]
    mad_ages = [group.mad_age for group in series]
    mad_age = np.round(np.mean(mad_ages), 3)
    mad_age_error = np.round(np.std(mad_ages), 3)
    plt.plot(series.time, mean_mad, '-.', color='0.2', linewidth=2.0,
        label='MAD age: ({} ± {}) Myr'.format(mad_age, mad_age_error))

    # MST Mean
    mean_mst_mean = np.mean([group.mst_mean for group in series], axis=0)
    mean_mst_mean = mean_mst_mean / mean_mst_mean[0]
    mst_mean_ages = [group.mst_mean_age for group in series]
    mst_mean_age = np.round(np.mean(mst_mean_ages), 3)
    mst_mean_age_error = np.round(np.std(mst_mean_ages), 3)
    plt.plot(series.time, mean_mst_mean, '--', color='0.4', linewidth=2.0,
        label='MST mean age: ({} ± {}) Myr'.format(mst_mean_age, mst_mean_age_error))

    # MST MAD
    mean_mst_mad = np.mean([group.mst_mad for group in series], axis=0)
    mean_mst_mad = mean_mst_mad / mean_mst_mad[0]
    mst_mad_ages = [group.mst_mad_age for group in series]
    mst_mad_age = np.round(np.mean(mst_mad_ages), 3)
    mst_mad_age_error = np.round(np.std(mst_mad_ages), 3)
    plt.plot(series.time, mean_mst_mad, 'g:', color='0.6', linewidth=2.0,
        label='MST MAD age: ({} ± {}) Myr'.format(mst_mad_age, mst_mad_age_error))

    # Secondary lines
    if secondary:
        i = 0
        plot_i = np.arange(0, len(series), 20)
        for group in series:
            if i in plot_i:
                plt.plot(series.time, group.scatter, '-', color='0.7', linewidth=0.5)
            i += 1

    # Title, legend and axis formatting
    plt.title("Average size indicators of {} moving group simulations with kinematics similar\n"
        "to β Pictoris over {} Myr and typical measurement errors of Gaia DR2\n".format(
        series.number_of_groups, series.duration))
    # plt.title("Size indicators of β-Pictoris (without outliners) over {} Myr\n"
    #     "with {} km/s redshift correction (real errors)".format(
    #         series.duration, round(series.rv_offset * (un.pc/un.Myr).to(un.km/un.s), 2)))
    plt.legend()
    plt.xlabel('Time (Myr)')
    plt.ylabel('Scatter, MAD, MST mean and MST MAD (pc)')
    plt.xlim(0, 30)
    plt.ylim(0, 2)
    # plt.xticks([14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0])
    # plt.yticks([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    # Save and show figure
    # plt.savefig(path.join(series.output_dir, '{}.pdf'.format(series.name)))
    plt.show()

def create_2D_scatter(group, i, j, step=None, age=None, errors=False, labels=False, mst=False):
    """ Creates a scatter plot of star positions in i and j at a given 'step' or 'age' in Myr.
        If 'age' doesn't match a step, the closest step is used instead. 'age' overrules 'steps'
        if both are given. 'labels' adds the stars' name and 'mst' adds the mst branches.
    """

    # Axis selection
    axis = {'x': 0, 'y': 1, 'z': 2}
    keys = tuple(axis.keys())
    i = axis[i.lower()]
    j = axis[j.lower()]

    # Step or age calculation
    if age is not None:
        step = round(age / group.series.timestep)
        age = round(group.series.time[step], 2)
    else:
        age = round(step * group.series.timestep, 2)

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': 14, 'lines.markersize': 4})
    plt.figure(figsize=(12, 8), facecolor='w')

    # Scatter
    plt.scatter(
        [star.position[step, i] for star in group],
        [star.position[step, j] for star in group], marker='o', color='0.0')

    # Errors
    if errors:
        for star in group:
            position = star.position[step]
            error = star.position_error[step]
            plt.plot(
                (position[i] - error[i], position[i] + error[i]),
                (position[j], position[j]), c='0.1', linewidth=0.7)
            plt.plot(
                (position[i], position[i]),
                (position[j] - error[j], position[j] + error[j]), c='0.1', linewidth=0.7)

    # Star labels
    if labels:
        for star in group:
            plt.text(star.position[step, i] + 1, star.position[step, j] + 1, star.name,
            horizontalalignment='left', fontsize=7)

    # Branches creation
    if mst:
        for branch in group.mst[step]:
            plt.plot(
                (branch.start.position[step, i], branch.end.position[step, i]),
                (branch.start.position[step, j], branch.end.position[step, j]), c='b')

    # Title and axis formatting
    plt.title("{} and {} positions of stars in β Pictoris at {} Myr wihtout outliers.\n".format(
        keys[i].upper(), keys[j].upper(), age))
    plt.xlabel('{} (pc)'.format(keys[i].upper()))
    plt.ylabel('{} (pc)'.format(keys[j].upper()))

    # Show figure
    plt.show()

def create_3D_scatter(group, step=None, age=None, errors=False, labels=False, mst=False):
    """ Creates a scatter plot of star positions in x, y and z at a given 'step' or 'age' in Myr.
        If 'age' doesn't match a step, the closest step is used instead. 'age' overrules 'step'
        if both are given. 'labels' adds the stars' name and 'mst' adds the mst branches.
    """

    # Step or age calculation
    if age is not None:
        step = round(age / group.series.timestep)
        age = round(group.series.time[step], 2)
    else:
        age = round(step * group.series.timestep, 2)

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14', 'lines.markersize': 4})
    fig = plt.figure(figsize=(12, 8), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')

    # Scatter
    ax.scatter(
        [star.position[step, 0] for star in group],
        [star.position[step, 1] for star in group],
        [star.position[step, 2] for star in group], marker='o', c='0.0')

    # Error bars
    if errors:
        for star in group:
            position = star.position[step]
            error = star.position_error[step]
            ax.plot(
                (position[0] - error[0], position[0] + error[0]),
                (position[1], position[1]), (position[2], position[2]), c='0.1', linewidth=0.7)
            ax.plot(
                (position[0], position[0]), (position[1] - error[1], position[1] + error[1]),
                (position[2], position[2]), c='0.1', linewidth=0.7)
            ax.plot(
                (position[0], position[0]), (position[1], position[1]),
                (position[2] - error[2], position[2] + error[2]), c='0.1', linewidth=0.7)

    # Star labels
    if labels:
        for star in group:
            ax.text(
                star.position[step, 0] + 2, star.position[step, 1] + 2, star.position[step, 2] + 2,
                star.name, horizontalalignment='left', fontsize=7)

    # Branches creation
    if mst:
        for branch in group.mst[step]:
            ax.plot(
                (branch.start.position[step, 0], branch.end.position[step, 0]),
                (branch.start.position[step, 1], branch.end.position[step, 1]),
                (branch.start.position[step, 2], branch.end.position[step, 2]), c='b')

    # Title and axis formatting
    plt.title("Minimum spanning tree of stars in β Pictoris at {} Myr.\n".format(age))
    ax.set_xlabel('\n X (pc)')
    ax.set_ylabel('\n Y (pc)')
    ax.set_zlabel('\n Z (pc)')

    # Show figure
    plt.show()

def plot_age_error():
    """ Creates a graph of ages obtained for diffrent measurement errors on RV and offset
        due to gravitationnal redshift.
    """

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # + 0.0 km/s points
    plt.errorbar(
        [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        [23.789, 23.763, 23.700, 23.174, 21.390, 15.400, 8.761],
        yerr=[0.268, 0.286, 0.356, 0.626, 0.960, 1.138, 1.290], fmt='o', color='0.0', label='+ 0.0 km/s')

    # + 0.5 km/s points
    plt.errorbar(
        [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        [20.658, 20.601, 20.571, 20.196, 19.076, 15.423, 8.804],
        yerr=[0.283, 0.288, 0.316, 0.487, 0.819, 1.143, 1.319], fmt='D', color='0.2', label='+ 0.5 km/s')

    # + 1.0 km/s points
    plt.errorbar(
        [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        [18.101, 18.087, 18.055, 17.827, 16.988, 14.376, 8.930],
        yerr=[0.331, 0.332, 0.342, 0.448, 0.636, 0.973, 1.066], fmt='^', color='0.4', label='+ 1.0 km/s')

    # β-Pictoris typical error line
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
    plt.show()

def create_histogram_ages(groups):
    """ Creates a histogram of ages computed by multiple tracebacks. """

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # Histogram plotting
    ages = [group.scatter_age for group in groups]
    plt.hist(ages, bins='auto') # bins=np.arange(21.975, 26.025, 0.05)

    # Title and axis formatting
    plt.title(
        'Distribution of {} moving groups age,\n'
        'without measurement errors. Average age: ({} ± {}) Myr\n'.format(
            len(groups), np.round(np.mean(ages), 3), np.round(np.std(ages), 3)))
    plt.xlabel('Age (Myr)')
    plt.ylabel('Number of groups')

    # Show figure
    plt.show()

def create_histogram(ages, initial_scatter, number_of_stars, number_of_groups, age):
    """ Creates an histogram of ages computed by multiple tracebacks. """

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
    plt.savefig(path.join(series.output_dir, '{}.pdf'.format(
        'Distribution of ages ({} groups, {} Myr, {} stars, initial scatter = {} pc)'.format(
            number_of_groups, age, number_of_stars, initial_scatter))))

def create_single_scatter_graph(groups):
    """ Creates a graph of scatter over time of a single group (0). """

    # Figure initialization
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')

    # Scatter plotting
    mean = np.mean([group.scatter for group in groups], axis=0)
    plt.plot(groups[0].time, mean, '-', color='0.0', linewidth=0.5)

    # Title and axis formatting
    plt.title('Scatter of a moving group over time\n')
    plt.xlabel('Time (Myr)')
    plt.ylabel('Scatter (pc)')
    plt.xlim(0, 30)

    # Save and show figure
    plt.savefig(path.join(series.output_dir, 'Scatter of a moving group over time.pdf'))
    plt.show()

def create_color_mesh(initial_scatter, number_of_stars, ages, age, number_of_groups):
    """ Creates a color mesh of ages over the initial scatter and number_of_stars. !!! Créer un code
        pour passer d'un array numpy de shape (n, 3) à un color mesh + un smoothing. Genre
        create_color_mesh(x, y, z, smoothing). !!!
    """

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
    plt.savefig(path.join(series.output_dir, 'Scatter on age ({} Myr).png').format(age))

def create_minimum_error_plots():
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

    # Figure show
    plt.show()
