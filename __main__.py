# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Main file of the Traceback algorithm

import numpy as np
from traceback import format_exc
from logging import basicConfig, info, warning, INFO
from time import strftime
from os.path import join
from os import remove
from sys import exit
import argparse
from config import *
from tools import *
from output import *

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
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
        # Creation or retrieval of the GroupModel in the database
        self.group, self.created = GroupModel.get_or_create(name=name)
        # Initialization from database if no data or simulation parameters are provided
        if not simulation and data == None:
            if self.created:
                remove(join(output_dir, '{}.db'.format(name)))
                warning('No existing data in the database with the name "{}.db".'.format(name))
                exit('No existing data in the database with the name "{}.db".'.format(name))
            else:
                self.initialize_from_database(self.group)
        # Initialization from a simulation or data
        else:
            # Initialization of a list of Star objects
            if simulation:
                self.stars = self.stars_from_simulation(number_of_steps + 1, *parameters)
            if data != None:
                self.stars = self.stars_from_data(number_of_steps + 1, data)
            # Initialization of time-independent parameters
            self.name = name
            self.date = strftime('%Y-%m-%d %H:%M:%S')
            self.duration = duration
            self.number_of_stars = len(self.stars)
            self.number_of_steps = number_of_steps + 1 # One more step to account for t = 0
            self.timestep = duration / number_of_steps
            self.time = np.arange(0, self.duration + self.timestep, self.timestep)
            self.average_velocity = sum([star.velocity for star in self.stars])/self.number_of_stars
            self.average_velocity_error = sum([
                star.velocity_error for star in self.stars])/self.number_of_stars
            # Initialization of time-dependent paremeters.
            self.barycenter = np.zeros([self.number_of_steps, 3])
            self.barycenter_error = np.zeros([self.number_of_steps, 3])
            self.dispersion = np.zeros([self.number_of_steps])
            self.dispersion_error = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_error = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_points = np.zeros([self.number_of_steps, 2, 3])
            # Completion of time-dependent values arrays.
            for step in range(self.number_of_steps):
                self.create_step(step)
            # Deletion of previous entries and creation of new entries in the database
            self.save_to_database()

    def initialize_from_database(self, group):
        """ Initializes Group object from an existing entry in the database.
        """
        # Initialization of a list of Star objects
        self.stars = []
        for star in StarModel.select().where(StarModel.group == group):
            self.stars.append(Star(star.name, None, None, None, None, None, star))
        # Initialization of time-independent parameters
        self.name = group.name
        self.data = group.date
        self.duration = group.duration
        self.number_of_stars = group.number_of_stars
        self.number_of_steps = group.number_of_steps
        self.timestep = group.timestep
        self.time = group.time
        self.average_velocity = group.average_velocity
        self.average_velocity_error = group.average_velocity_error
        # Initialization of time-dependent paremeters.
        self.barycenter = group.barycenter
        self.barycenter_error = group.barycenter_error
        self.dispersion = group.dispersion
        self.dispersion_error = group.dispersion_error
        self.minimum_spanning_tree = group.minimum_spanning_tree
        self.minimum_spanning_tree_error = group.minimum_spanning_tree_error
        self.minimum_spanning_tree_points = group.minimum_spanning_tree_points

    def stars_from_data(self, number_of_steps, data):
        """ Creates a list of Star objects from a Python dictionnary or CSV files containing the
            parameters including the name of the stars, their position (XYZ) and velocity (UVW),
            and the respective errors.
        """
        return [Star(
            name, number_of_steps,
            np.array(value['position']), np.array(value['position_error']),
            np.array(value['velocity']), np.array(value['velocity_error'])) for name, value in data.items()
        ]

    def stars_from_simulation(
        self, number_of_steps: int, number_of_stars: int, age: float,
        avg_position: tuple, avg_position_error: tuple, avg_position_dispersion: tuple,
        avg_velocity: tuple, avg_velocity_error: tuple, avg_velocity_dispersion: tuple):
        """ Creates an artificial sample of star for a given number of stars and age based on
            the intial average position (XYZ) and velocity (UVW), and their respective error and
            dispersion. The sample is then moved forward in time for the given age.
        """
        # Velocity conversion factor from km/s to pc/Myr
        vc = 1.0227120263358653
        stars = []
        for star in range(1, number_of_stars + 1):
            velocity = np.random.normal(np.array(avg_velocity)*vc, np.array(avg_velocity_dispersion)*vc, 3)
            position = np.random.normal(np.array(avg_position), np.array(avg_position_dispersion), 3) + velocity * age
            stars.append(Star(
                's{}'.format(star), number_of_steps, position, np.array(avg_position_error), velocity, np.array(avg_velocity_error)*vc))
        return stars

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

    def save_to_database(self):
        """ Saves all parameters to the database, including all Star objects within the Group object.
        """
        # Deletion of previous GroupModel and dependant StarModel database entries
        self.group.delete_instance(recursive=True)
        if self.created:
            info('New database entry "{}" added.'.format(self.name))
        else:
            info('Previous database entry "{}" deleted and replaced.'.format(self.name))
        # Creation of new GroupModel database entry
        self.group = GroupModel.create(
            # Time-independent parameters
            name=self.name,
            date=self.date,
            duration=self.duration,
            number_of_stars=self.number_of_stars,
            number_of_steps=self.number_of_steps,
            timestep=self.timestep,
            time=self.time,
            average_velocity=self.average_velocity,
            average_velocity_error=self.average_velocity_error,
            # Time-dependent parameters
            barycenter=self.barycenter,
            barycenter_error=self.barycenter_error,
            dispersion=self.dispersion,
            dispersion_error=self.dispersion_error,
            minimum_spanning_tree=self.minimum_spanning_tree,
            minimum_spanning_tree_error=self.minimum_spanning_tree_error,
            minimum_spanning_tree_points=self.minimum_spanning_tree_points
        )
        # Creation of new StarModel entries.
        for star in self.stars:
            star.save_to_database(self.group)

class Star:
    """ Contains the values and related methods of a star.
    """
    def __init__(
        self, name: str, number_of_steps: int,
        position: np.ndarray, position_error: np.ndarray,
        velocity: np.ndarray, velocity_error: np.ndarray, database_object=None):
        """ Initializes basic parameters and arrays to the correct shape.
        """
        # Initialization from database if possible
        if database_object != None:
            self.initialize_from_database(database_object)
        else:
            # Time-independent parameters
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

    def initialize_from_database(self, star):
        """ Initializes Star object from an existing instance in the database.
        """
        # Time-independent parameters
        self.name = star.name
        self.velocity = star.velocity
        self.velocity_error = star.velocity_error
        # Time-dependent parameters
        self.position = star.position
        self.position_error = star.position_error
        self.relative_position = star.relative_position
        self.relative_position_error = star.relative_position_error
        self.distance = star.distance
        self.distance_error = star.distance_error

    def save_to_database(self, group):
        """ Saves all parameters to the database and deletes previous entry.
        """
        # Creation of new StarModel entry
        StarModel.create(
            group=group,
            # Time-independent parameters
            name=self.name,
            velocity=self.velocity,
            velocity_error=self.velocity_error,
            # Time-dependent parameters
            position=self.position,
            position_error=self.position_error,
            relative_position=self.relative_position,
            relative_position_error=self.relative_position_error,
            distance=self.distance,
            distance_error=self.distance_error
        )

if __name__ == '__main__':
    # Importation of the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of the traceback used in the outputs filenames.')
    parser.add_argument('-d', '--data', help='use data parameter in the configuration file as input.', action='store_true')
    parser.add_argument('-s', '--simulation', help='run a simulation to create an input based on parameters in the configuraiton file.', action='store_true')
    args = parser.parse_args()
    # Initilization of the database
    name = args.name
    from model import *
    # Initilization of the Simulation object
    if args.data and args.simulation:
        warning('Either create the traceback "{}" from data or a simulation, not both.'.format(name))
        exit('Either create the traceback "{}" from data or a simulation, not both.'.format(name))
    elif not args.data and not args.simulation:
        info('Traceback and output from database.')
        Traceback = Group(name, duration, number_of_steps)
    elif args.data:
        info('Traceback and output from data.')
        Traceback = Group(name, duration, number_of_steps, data=data)
    elif args.simulation:
        info('Traceback and output from simulation.')
        Traceback = Group(
            name, duration, number_of_steps, number_of_stars, age,
            avg_position, avg_position_error, avg_position_dispersion,
            avg_velocity, avg_velocity_error, avg_velocity_dispersion, simulation=True)
    # Creation of the output files
    create_graph(Traceback.time, Traceback.dispersion)
