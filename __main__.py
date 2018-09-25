# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Main file of the Traceback algorithm.

import numpy as np
from traceback import format_exc
from logging import basicConfig, info, warning, INFO
from time import strftime
from sys import argv
from config import *
from model import *
from output import *

# Configuration of the log file
basicConfig(
    filename=join(db_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

class Group:
    """ Contains the values and related methods of a local association and a list of Star objets
        that are part of that group. Data can be obtained from the database or calculated from a
        raw data file.
    """
    def __init__(self, name: str, number_of_steps: int, timestep: int, data=None, simulation=None):
        """ Initializes Star objects, time-independent values and time-dependent arrays to the
            correct shape.
        """
        # Creation of a list of Star objects
        self.stars = data
        # Time-independent parameters
        self.name = name
        self.date = strftime('%Y-%m-%d %H:%M:%S')
        self.duration = number_of_steps * timestep
        self.number_of_stars = len(self.stars)
        self.number_of_steps = number_of_steps + 1
        self.timestep = timestep
        self.time = tuple(range(0, self.timestep * self.number_of_steps, self.timestep))
        self.average_velocity = sum([star.velocity for star in self.stars])/self.number_of_stars
        self.average_velocity_error = sum([
            star.velocity_error for star in self.stars])/self.number_of_stars
        # Time-dependent paremeters
        self.barycenter = np.zeros([self.number_of_steps, 3])
        self.barycenter_error = np.zeros([self.number_of_steps, 3])
        self.dispersion = np.zeros([self.number_of_steps])
        self.dispersion_error = np.zeros([self.number_of_steps])
        self.minimum_spanning_tree = np.zeros([self.number_of_steps])
        self.minimum_spanning_tree_error = np.zeros([self.number_of_steps])
        self.minimum_spanning_tree_points = np.zeros([self.number_of_steps, 2, 3])
        # Completion of time-dependent values arrays
        for step in range(self.number_of_steps):
            self.create_step(step)

    def create_step(self, step: int):
        """ Creates a time step for the time-dependent arrays of the Group object and Star objects
            within self.stars using the time-indenpendent velocities.
        """
        # Calculation of the position
        if step != 0:
            for star in self.stars:
                star.position[step, :] = star.position[0, :] - star.velocity * self.time[step]
                star.position_error[step, :] = star.position_error[0, :] + star.velocity_error * self.time[step]
        # Calculation of the barycenter
        self.barycenter[step] = np.sum(np.array([
            star.position[step, :] for star in self.stars]), axis=0)/self.number_of_stars
        self.barycenter_error[step] = np.sum(np.array([
            star.position_error[step, :] for star in self.stars]), axis=0)/self.number_of_stars
        # Calculation of the relative position and distance from the barycenter
        for star in self.stars:
            star.relative_position[step, :] = star.position[step, :] - self.barycenter[step, :]
            star.relative_position_error[step, :] = star.position_error[step, :] + self.barycenter_error[step, :]
            star.distance[step] = np.linalg.norm(star.relative_position[step, :])
            star.distance_error[step] = star.distance[step] * np.linalg.norm(
                star.relative_position_error[step, :] / star.relative_position[step, :])
        # Calculation of the dispersion
        self.dispersion[step] = np.std(np.array([star.distance[step] for star in self.stars]))
        self.dispersion_error[step] = self.dispersion[step] * np.std(
            np.array([star.distance_error[step] / star.distance[step] for star in self.stars]))

class Star:
    """ Contains the values and related methods of a star.
    """
    def __init__(
        self, name: str, number_of_steps: int,
        position: np.ndarray, position_error: np.ndarray,
        velocity: np.ndarray, velocity_error: np.ndarray):
        """ Initializes basic parameters and arrays to the correct shape.
        """
        # Time independent parameters
        self.name = name
        self.velocity = velocity
        self.velocity_error = velocity_error
        # Time-dependent parameters
        self.position = np.pad(np.array(
            [position]), ((0, number_of_steps - 1), (0, 0)), 'constant', constant_values=0)
        self.position_error = np.pad(np.array(
            [position_error]), ((0, number_of_steps - 1), (0, 0)), 'constant', constant_values=0)
        self.relative_position = np.zeros([number_of_steps, 3])
        self.relative_position_error = np.zeros([number_of_steps, 3])
        self.distance = np.zeros([number_of_steps])
        self.distance_error = np.zeros([number_of_steps])

def create_random_sample(number_of_stars, number_of_steps, timestep):
    """ This function create an artificial sample of star for a given number_of_stars and duration.
        The list of Star objects can be used as input for a Group object.
    """
    stars = []
    for star in range(1, number_of_stars + 1):
        #JG. Changer uniform pour normal
        velocity = np.random.uniform(0, 2, 3)
        position = np.random.uniform(-1, 1, 3) + velocity * timestep * number_of_steps / 2
        stars.append(Star(
            's{}'.format(star), number_of_steps + 1, position, position * 0.01, velocity, velocity * 0.01))
    return stars

def create_sample_from_dict(data):
    """ This function create a list of Star objects from a list of parameters in tuples.
    """
    return [Star(
        name, number_of_steps + 1,
        np.array(value['position']), np.array(value['position_error']),
        np.array(value['velocity']), np.array(value['velocity_error'])) for name, value in data.items()
    ]

if __name__ == '__main__':
    # Initilization of the Simulation object
    stars_sample = create_random_sample(number_of_stars, number_of_steps, timestep)
    Simulation = Group('Simulation', number_of_steps, timestep, stars_sample)
    # Creation of the output files
    create_graph(Simulation.time, Simulation.dispersion)
