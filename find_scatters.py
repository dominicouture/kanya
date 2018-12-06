# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Algorithm to find the original scatter of β-Pictoris.

import numpy as np
from astropy import units as un
from matplotlib import rcParams, pyplot as plt
from argparse import ArgumentParser
from traceback import format_exc
from logging import basicConfig, info, warning, INFO
from time import strftime
from os.path import join
from os import remove
from sys import exit
from config import *
from tools import *
from output import *

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

def find_scatters(
        number_of_stars: int, age: int,
        avg_position_xyz: tuple, avg_position_scatter_xyz: tuple,
        avg_velocity_xyz: tuple, avg_velocity_scatter_xyz: tuple
    ):
    """ Cumputes the scatters of a simulated group after a given age. Positions are in pc,
        velocities in pc/Myr and the age in Myr.
    """
    intial_positions_xyz = np.random.normal(
        np.full((number_of_stars, 3), avg_position_xyz),
        avg_position_scatter_xyz
    )
    final_positions_xyz = intial_positions_xyz + age * np.random.normal(
        np.full((number_of_stars, 3), avg_velocity_xyz),
        avg_velocity_scatter_xyz
    )
    final_scatter_xyz = np.std(final_positions_xyz, axis=0)
    final_scatter3 = np.prod(final_scatter_xyz)**(1/3)
#    final_scatter1 = np.linalg.norm(final_scatter_xyz)
#    final_scatter2 = np.std(np.linalg.norm(final_positions_xyz, axis=1))
#    final_scatter4 = np.sqrt( # 1 et 4 équivalentes.
#        sum(
#            [sum([x**2, y**2, z**2]) for x, y, z in final_positions_xyz - np.average(
#                final_positions_xyz, axis=0
#            )]
#        )/number_of_stars
#    )
    return (final_scatter3, final_scatter_xyz)

def create_graph(x, y, y_ans):
    """ Create a graph of the scatter with a linear regression curve.
    """
    m, b = np.polyfit(x, y, 1)
    print('{}*m + {}'.format(m, b))
    x_ans = (y_ans - b)/m
    print(x_ans, y_ans)
    rcParams.update({'font.family': 'serif', 'font.size': '15'})
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.plot(x, y,  'k.')
    plt.plot(x, m*x + b, 'k--')
    plt.xlabel('Initial Scatter (pc)')
    plt.ylabel('Scatter after 24 Myr (pc)')
    plt.title('Scatter of a group after 24 Myr')
    plt.show()

# Simulation parameters
number_of_groups = 500
number_of_stars = 42
age = 24.0
time = np.arange(1.0, 25.0, 0.1)

# Position scatter
initial_avg_position_xyz = (0.0, 0.0, 0.0)
current_avg_position_scatter_xyz = (29.3, 14.0, 9.0)
current_avg_position_scatter_x = np.prod(current_avg_position_scatter_xyz)**(1/3) # Value to find.
max_scatter_x = 25.0
initial_avg_position_scatters_x = np.arange(0, max_scatter_x, max_scatter_x/number_of_groups)
initial_avg_position_scatters_xyz = np.full(
    (number_of_groups, 3), (initial_avg_position_scatters_x).reshape((number_of_groups, 1))
)

# Velocity scatter
avg_velocity_xyz = (0.0, 0.0, 0.0)
avg_velocity_scatter_xyz = (2.2, 1.2, 1.0)
avg_velocity_scatter_xyz = np.full(
    (3,), np.prod(avg_velocity_scatter_xyz)**(1/3) * (un.km/un.s).to(un.pc/un.Myr)
)
#avg_velocity_xyz = np.full((3,), np.linalg.norm(avg_velocity) / 3**0.5) * (un.km/un.s).to(un.pc/un.Myr)
#avg_velocity_xyz = np.array(avg_velocity) * (un.km/un.s).to(un.pc/un.Myr)
#avg_velocity_scatter_xyz = np.array(avg_velocity_scatter) * (un.km/un.s).to(un.pc/un.Myr)


# Final scatter over the initial scatter
scatters_x = []
scatters_xyz = []

for initial_avg_position_scatter_xyz in initial_avg_position_scatters_xyz:
    scatter_x, scatter_xyz = find_scatters(
        number_of_stars, age, initial_avg_position_xyz, initial_avg_position_scatter_xyz,
        avg_velocity_xyz, avg_velocity_scatter_xyz
    )
    print(initial_avg_position_scatter_xyz, ' : ', scatter_x, scatter_xyz)
    scatters_x.append(scatter_x)
    scatters_xyz.append(scatter_xyz)

#y_ans = 3**0.5 * np.prod(avg_position_scatter)**(1/3)
#initial_avg_position_scatters = np.sqrt(3)*avg_position_scatters_x
create_graph(initial_avg_position_scatters_x, scatters_x, current_avg_position_scatter_x)


# Final scatter over the age of the group
scatters_x = []
scatters_xyz = []

for t in time:
    scatter_x, scatter_xyz = find_scatters(
        number_of_stars, t, initial_avg_position_xyz, (0.0, 0.0, 0.0),
        avg_velocity_xyz, avg_velocity_scatter_xyz
    )
    print(t, ' Myr : ', scatter_x, scatter_xyz)
    scatters_x.append(scatter_x)
    scatters_xyz.append(scatter_xyz)

create_graph(time, scatters_x, current_avg_position_scatter_x)
