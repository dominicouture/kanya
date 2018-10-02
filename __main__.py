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
    def __init__(self, name: str, duration: float, number_of_steps: int, *parameters, simulation=False, data=None):
        """ Initializes Star objects and Group object from a simulated sample of stars in a local
            association, raw data in the form a Python dictionnary or from a database. This dataset
            is then moved backward in time for a given traceback duration.
        """
        # Creation of a list of Star objects
        if simulation:
            self.stars = self.stars_from_simulation(number_of_steps + 1, *parameters)
        # Time-independent parameters
        self.name = name
        self.date = strftime('%Y-%m-%d %H:%M:%S')
        self.duration = duration
        self.number_of_stars = len(self.stars)
        self.number_of_steps = number_of_steps + 1 # One more step to account for t = 0
        self.timestep = duration / number_of_steps
        self.time = tuple(np.arange(0, self.duration + self.timestep, self.timestep))
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

    def stars_from_simulation(
        self, number_of_steps: int, number_of_stars: int, age: float,
        avg_position: tuple, avg_position_error: tuple, avg_position_dispersion: tuple,
        avg_velocity: tuple, avg_velocity_error: tuple, avg_velocity_dispersion: tuple):
        """ Creates an artificial sample of star for a given number of stars and age based on
            the intial average position (XYZ) and velocity (UVW), and their respective error and
            dispersion. The sample is then moved forward in time for the given age.
        """
        # Velocity conversion factor from km/s to pc/Myr.
        vc = 1.0227120263358653
        stars = []
        for star in range(1, number_of_stars + 1):
            velocity = np.random.normal(np.array(avg_velocity)*vc, np.array(avg_velocity_dispersion)*vc, 3)
            position = np.random.normal(np.array(avg_position), np.array(avg_position_dispersion), 3) + velocity * age
            stars.append(Star(
                's{}'.format(star), number_of_steps, position, np.array(avg_position_error), velocity, np.array(avg_velocity_error)*vc))
        return stars

    def stars_from_dictionary(self, name, number_of_steps, *data):
        """ Creates a list of Star objects from a dictionnary of parameters including
            the name of the stars, their position (XYZ) and velocity (UVW), and the respective errors.
        """
        return [Star(
            name, number_of_steps,
            np.array(value['position']), np.array(value['position_error']),
            np.array(value['velocity']), np.array(value['velocity_error'])) for name, value in data.items()
        ]

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

if __name__ == '__main__':
    # Initilization of the Simulation object
    Simulation = Group(
        'Simulation', duration, number_of_steps, number_of_stars, age,
        avg_position, avg_position_error, avg_position_dispersion,
        avg_velocity, avg_velocity_error, avg_velocity_dispersion, simulation=True)
    # Creation of the output files
    create_graph(Simulation.time, Simulation.dispersion)
