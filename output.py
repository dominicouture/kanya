# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" output.py: Provides the necesary functions to output data in table, graph or video form.
"""

import numpy as np
from logging import basicConfig, warning, INFO
from time import strftime
from os.path import join
from matplotlib import rcParams, pyplot as plt
from scipy.interpolate import griddata
from init import *

# Configuration of the log file
basicConfig(
    filename=logs_path, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

def create_graph(x, y):
    """ Creates a graph of scatter over time.
    """
    rcParams.update({'font.family': 'serif', 'font.size': '15'})
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.plot(x, y,  'k.-')
    plt.plot(x, y, '-', color='0.5')
    plt.xlabel('Age (Myr)')
    plt.ylabel('Scatter (pc)')
    plt.title('Scatter of a moving group over time\n')
    plt.savefig(join(output_dir, 'Scatter of a moving group over time.pdf'))

def create_scatter_graph(groups, name):
    """ Creates a graph of scatter over time.
    """
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')
    i = 0
    plot_i = np.arange(0, len(groups), 20)
    for group in groups:
        if i in plot_i:
            plt.plot(group.time, group.scatter, '-', color='0.7', linewidth=0.5)
        i += 1
    mean = np.mean([group.scatter for group in groups], axis=0)
    print(groups[0].time)
    print(mean)
    plt.plot(groups[0].time, mean, 'k-', linewidth=2.0)
    ages = [group.scatter_age for group in groups]
    plt.title(
        'Scatter of {} moving groups over time\nwith measurement errors (corrected). Average age: ({} ± {}) Myr\n'.format(
            len(groups), np.round(np.mean(ages), 3), np.round(np.std(ages), 3)
        )
    )
    plt.xlabel('Time (Myr)')
    plt.ylabel('Scatter (pc)')
    plt.xticks([14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0])
    plt.yticks([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    plt.xlim(14, 34)
#    plt.savefig(join(output_dir, '{}.pdf'.format(name)))
    plt.show()

def create_scatter_graph2(groups):
    """ Creates a graph of scatter over time.
    """
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')
    mean = np.mean([group.scatter for group in groups], axis =0)
    plt.plot(groups[0].time, mean, '-', color='0.0', linewidth=0.5)
    plt.xlabel('Time (Myr)')
    plt.ylabel('Scatter (pc)')
    plt.xlim(0, 30)

def create_histogram_ages(groups):
    """ Creates an histogram of ages computed by multiple tracebacks.
    """
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')
    ages = [group.scatter_age for group in groups]
    plt.hist(ages, bins='auto') # bins=np.arange(21.975, 26.025, 0.05)
    plt.xlabel('Age (Myr)')
    plt.ylabel('Number of groups')
    plt.title(
        'Distribution of {} moving groups age,\nwithout measurement errors. Average age: ({} ± {}) Myr\n'.format(
            len(groups), np.round(np.mean(ages), 3), np.round(np.std(ages), 3)
        )
    )
    plt.show()

def create_histogram(ages, initial_scatter, number_of_stars, number_of_groups, age):
    """ Creates an histogram of ages computed by multiple tracebacks.
    """
    rcParams.update({'font.family': 'serif', 'font.size': '15'})
    plt.figure(figsize=(12, 8), facecolor='w')
    hist, bin_edges = np.histogram(ages, density=True)
    plt.hist(ages, bins='auto', density=True) # bins=np.arange(21.975, 26.025, 0.05)
    plt.xlabel('Age (Myr)')
    plt.ylabel('Number of groups')
    plt.title(
        'Distribution of ages ({} groups, {} Myr, {} stars,\ninitial scatter = {} pc, {})'.format(
            number_of_groups, age, number_of_stars, initial_scatter,
            'calculated age = ({} ± {}) Myr'.format(
                np.round(np.average(ages), 3),
                np.round(np.std(ages), 3)
            )
        )
    )
    plt.savefig(
        join(
            output_dir, '{}.pdf'.format(
                'Distribution of ages ({} groups, {} Myr, {} stars, initial scatter = {} pc)'.format(
                    number_of_groups, age, number_of_stars, initial_scatter
                )
            )
        )
    )

def create_color_mesh(initial_scatter, number_of_stars, ages, age, number_of_groups):
    """ Creates a color mesh of ages over the initial scatter and number_of_stars. Créer un code
        pour passer d'un array numpy de shape (n, 3) à un color mesh + un smoothing. Genre
        create_color_mesh(x, y, z, smoothing).
    """
    x, y = np.meshgrid(initial_scatter, number_of_stars)
    grid_x, grid_y = np.mgrid[5:25:300j, 30:200:300j]
    grid_z = griddata(
        np.array([(i, j) for i in initial_scatter for j in number_of_stars]),
        ages.T.flatten(),
        (grid_x, grid_y),
        method='linear'
    )
    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.pcolormesh(grid_x, grid_y, grid_z, cmap=plt.cm.Greens_r, vmin=0, vmax=2)
    plt.colorbar()
    plt.xlabel('Initial scatter (pc)')
    plt.ylabel('Number of stars')
    plt.xticks([5.0, 10.0, 15.0, 20.0, 25.0])
    plt.title(
        'Scatter on age (Myr) over the initial scatter (pc)\n'
        'and the number of stars ({} groups, {} Myr)'.format(number_of_groups, age)
    )
    plt.savefig(join(output_dir, 'Scatter on age ({} Myr).png').format(age))

if __name__ == '__main__':
    a5 = np.array(
        [
            [0.387, 0.787, 1.152, 1.572, 1.918],
            [0.330, 0.683, 0.994, 1.321, 1.677],
            [0.287, 0.590, 0.905, 1.227, 1.517],
            [0.267, 0.538, 0.783, 1.091, 1.350],
            [0.207, 0.409, 0.632, 0.811, 1.024],
            [0.148, 0.310, 0.426, 0.603, 0.737]
        ]
    )

    a10 = np.array(
        [
            [0.388, 0.770, 1.167, 1.570, 1.978],
            [0.341, 0.653, 1.059, 1.382, 1.608],
            [0.290, 0.583, 0.882, 1.203, 1.509],
            [0.280, 0.550, 0.804, 1.085, 1.342],
            [0.200, 0.412, 0.625, 0.811, 1.009],
            [0.148, 0.298, 0.439, 0.616, 0.714]
        ]
    )

    a24 = np.array(
        [
            [0.401, 0.780, 1.188, 1.586, 1.943],
            [0.331, 0.689, 0.983, 1.348, 1.667],
            [0.292, 0.578, 0.894, 1.175, 1.495],
            [0.270, 0.560, 0.820, 1.090, 1.351],
            [0.215, 0.410, 0.604, 0.829, 1.046],
            [0.148, 0.279, 0.436, 0.597, 0.698]
        ]
    )

    a50 = np.array(
        [
            [0.380, 0.774, 1.195, 1.514, 1.975],
            [0.337, 0.683, 0.986, 1.369, 1.667],
            [0.294, 0.617, 0.859, 1.194, 1.474],
            [0.272, 0.544, 0.806, 1.098, 1.379],
            [0.213, 0.421, 0.633, 0.815, 1.034],
            [0.144, 0.293, 0.446, 0.573, 0.698]
        ]
    )

    a100 = np.array(
        [
            [0.385, 0.786, 1.162, 1.586, 1.992],
            [0.335, 0.685, 0.972, 1.287, 1.673],
            [0.304, 0.596, 0.888, 1.197, 1.518],
            [0.262, 0.547, 0.836, 1.104, 1.370],
            [0.217, 0.421, 0.621, 0.855, 1.042],
            [0.147, 0.297, 0.432, 0.604, 0.729]
        ]
    )

    initial_scatter = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    numbers_of_star = np.array([30, 40, 50, 60, 100, 200])
    number_of_groups = 1000

#    create_color_mesh(initial_scatter, numbers_of_star, a24, 5.0, number_of_groups)
#    create_color_mesh(initial_scatter, numbers_of_star, a24, 10.0, number_of_groups)
#    create_color_mesh(initial_scatter, numbers_of_star, a24, 24.0, number_of_groups)
#    create_color_mesh(initial_scatter, numbers_of_star, a24, 50.0, number_of_groups)
#    create_color_mesh(initial_scatter, numbers_of_star, a24, 100.0, number_of_groups)
#    create_color_mesh(
#        initial_scatter, numbers_of_star, (a5+a10+a24+a50+a100)/5, 'x', number_of_groups
#    )

    errors = (0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    ages = (24.001, 23.966, 23.901, 23.74, 23.525, 22.224, 20.301, 18.113, 15.977, 11.293, 7.995, 5.803, 4.358, 3.364, 2.665, 2.204, 1.756, 1.257, 0.933, 0.735, 0.580, 0.488, 0.346, 0.262, 0.192, 0.160, 0.134)

    rcParams.update({'font.family': 'serif', 'font.size': '14'})
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.xlabel('Error on UVW velcotiy (km/s)')
    plt.ylabel('Age of minimal scatter (Myr)')
    plt.plot(errors, ages, '.-', color='0.0', linewidth=1.0)
    plt.show()
