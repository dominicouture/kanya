# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Main file of the Traceback algorithm

import numpy as np
from astropy import units as un
from argparse import ArgumentParser
from traceback import format_exc
from logging import basicConfig, info, warning, INFO
from time import strftime
from os.path import join
from os import remove
from sys import exit
from tools import *
from output import *
from config import *

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

class Group:
    """ Contains the values and related methods of a local group and a list of Star objets that
        are part of it. Data can be obtained from the database or calculated from a raw data file.
    """
    def __init__(
        self, name: str, number_of_steps: int, initial_time: float, final_time: float, data, parameters):
        """ Initializes Star objects and Group object from a simulated sample of stars in a local
            association, raw data in the form a Python dictionnary or from a database. This dataset
            is then moved backward in time for a given traceback duration. Distances are in pc,
            durations in Myr and velocities in pc/Myr.
        """
#        # Creation or retrieval of the GroupModel in the database
        self.group, self.created = GroupModel.get_or_create(name=name)

        # Initialization from database if no data or simulation parameters are provided
        if data is None and parameters is None:
            pass
            if self.created:
                remove(join(output_dir, '{}.db'.format(name)))
                database_error = 'No existing database with the name "{}.db".'.format(name)
                info(database_error)

                exit('No existing data in the database with the name "{}.db".'.format(name))
            else:
                self.initialize_from_database(self.group)

        # Initialization from a simulation or data
        else:
            # Initialization of a list of Star objects
            if data is not None:
                self.stars = self.stars_from_data(number_of_steps + 1, data)
            elif parameters is not None:
                self.stars = self.stars_from_simulation(number_of_steps + 1, *parameters)
            # Initialization of time-independent parameters
            self.name = name
            self.date = strftime('%Y-%m-%d %H:%M:%S')
            self.initial_time = initial_time
            self.final_time = final_time
            self.duration = self.final_time - self.initial_time
            self.number_of_stars = len(self.stars)
            self.number_of_steps = number_of_steps + 1 # One more step to account for t = 0
            self.timestep = self.duration / number_of_steps
            self.time = np.arange(self.initial_time, self.final_time + self.timestep, self.timestep)
            self.average_velocity = sum(
                [star.velocity for star in self.stars]) / self.number_of_stars
            self.average_velocity_error = sum(
                [star.velocity_error for star in self.stars]) / self.number_of_stars
            # Initialization of time-dependent paremeters.
            self.barycenter = np.zeros([self.number_of_steps, 3])
            self.barycenter_error = np.zeros([self.number_of_steps, 3])
            self.scatter_xyz = np.zeros([self.number_of_steps, 3])
            self.scatter_xyz_error = np.zeros([self.number_of_steps, 3])
            self.scatter = np.zeros([self.number_of_steps])
            self.scatter_error = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_error = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_points = np.zeros([self.number_of_steps, 2, 3])
            self.minimum_spanning_tree_age = 0.0

            # Completion of time-dependent values arrays.
            for step in range(self.number_of_steps):
                self.create_step(step)
            self.scatter_age = self.time[np.argmin(self.scatter)]
            self.scatter_age_error = 0.0

            # Deletion of previous entries and creation of new entries in the database
            self.save_to_database()

    def initialize_from_database(self, group):
        """ Initializes Group object from an existing entry in the database.
        """
        # Initialization of a list of Star objects
        self.stars = []
        for star in StarModel.select().where(StarModel.group == group):
            self.stars.append(Star(star.name, data=star))

        # Initialization of time-independent parameters
        self.name = group.name
        self.data = group.date
        self.initial_time = group.initial_time
        self.final_time = group.final_time
        self.duration = group.duration
        self.number_of_stars = group.number_of_stars
        self.number_of_steps = group.number_of_steps
        self.timestep = group.timestep
        self.time = group.time
        self.average_velocity = group.average_velocity
        self.average_velocity_error = group.average_velocity_error
        # Initialization of time-dependent paremeters
        self.barycenter = group.barycenter
        self.barycenter_error = group.barycenter_error
        self.scatter_xyz = group.scatter_xyz
        self.scatter_xyz_error = group.scatter_xyz_error
        self.scatter = group.scatter
        self.scatter_error = group.scatter_error
        self.scatter_age = group.scatter_age
        self.scatter_age_error = group.scatter_age_error
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
                np.array(value['position']),
                np.array(value['position_error']),
                np.array(value['velocity']),
                np.array(value['velocity_error'])
            ) for name, value in data.items()
        ]

    def stars_from_simulation(
        self, number_of_steps: int, age: float, number_of_stars: int,
        avg_position: tuple, avg_position_error: tuple, avg_position_scatter: tuple,
        avg_velocity: tuple, avg_velocity_error: tuple, avg_velocity_scatter: tuple):
        """ Creates an artificial sample of star for a given number of stars and age based on
            the intial average position (XYZ) and velocity (UVW), and their respective error and
            scatter. The sample is then moved forward in time for the given age.
        """
        # Velocity conversions from km/s to pc/Myr
        avg_velocity, avg_velocity_error, avg_velocity_scatter = np.array((
            avg_velocity, avg_velocity_error, avg_velocity_scatter
        )) * (un.km/un.s).to(un.pc/un.Myr)
        # Creation of Star objects
        stars = []
        for star in range(1, number_of_stars + 1):
            velocity = np.random.normal(
                np.array(avg_velocity),
                np.array(avg_velocity_scatter)
            )
            position = velocity * age + np.random.normal(
                np.array(avg_position),
                np.array(avg_position_scatter)
            )
            stars.append(
                Star(
                    'star_{}'.format(star), number_of_steps,
                    position, np.array(avg_position_error),
                    velocity, np.array(avg_velocity_error)
                )
            )
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

        # Calculation of the scatter
        self.scatter_xyz[step] = np.std(np.array([star.position[step, :] for star in self.stars]), axis=0)
        self.scatter[step] = np.prod(self.scatter_xyz[step])**(1/3)

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
            initial_time = self.initial_time,
            final_time = self.final_time,
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
            scatter_xyz=self.scatter_xyz,
            scatter_xyz_error=self.scatter_xyz_error,
            scatter=self.scatter,
            scatter_error=self.scatter_error,
            scatter_age=self.scatter_age,
            scatter_age_error=self.scatter_age_error,
            minimum_spanning_tree=self.minimum_spanning_tree,
            minimum_spanning_tree_error=self.minimum_spanning_tree_error,
            minimum_spanning_tree_points=self.minimum_spanning_tree_points
        )

        # Creation of new StarModel entries
        for star in self.stars:
            star.save_to_database(self.group)

class Star:
    """ Contains the values and related methods of a star.
    """
    def __init__(
        self, name: str, number_of_steps=None,
        position=None, position_error=None, velocity=None, velocity_error=None, data=None):
        """ Initializes basic parameters and arrays to the correct shape. Distances are in pc and
            velocities in pc/Myr.
        """
        # Initialization from database if possible
        if data != None:
            self.initialize_from_database(data)

        else:
            # Initialization of time-independent parameters
            self.name = name
            self.velocity = velocity
            self.velocity_error = velocity_error
            # Initialization of time-dependent parameters
            self.position = np.pad(
                np.array([position]),
                ((0, number_of_steps - 1), (0, 0)), 'constant', constant_values=0
            )
            self.position_error = np.pad(
                np.array([position_error]),
                ((0, number_of_steps - 1), (0, 0)), 'constant', constant_values=0
            )
            self.relative_position = np.zeros([number_of_steps, 3])
            self.relative_position_error = np.zeros([number_of_steps, 3])
            self.distance = np.zeros([number_of_steps])
            self.distance_error = np.zeros([number_of_steps])

    def initialize_from_database(self, star):
        """ Initializes Star object from an existing instance in the database.
        """
        # Initialization of time-independent parameters
        self.name = star.name
        self.velocity = star.velocity
        self.velocity_error = star.velocity_error
        # Initialization of time-dependent parameters
        self.position = star.position
        self.position_error = star.position_error
        self.relative_position = star.relative_position
        self.relative_position_error = star.relative_position_error
        self.distance = star.distance
        self.distance_error = star.distance_error

    def save_to_database(self, group):
        """ Saves all parameters to the database in a new StarModel entry and deletes any previous
            entry.
        """
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
    # Import of arguments
    parser = ArgumentParser()
    parser.add_argument('name', help='name of the traceback used in the outputs filenames.')
    parser.add_argument(
        '-d', '--data', action='store_true',
        help='use data parameter in the configuration file as input.'
    )
    parser.add_argument(
        '-s', '--simulation', action='store_true',
        help='run a simulation to create an input based on parameters in the configuraiton file.'
    )
    args = parser.parse_args()

    # Import of configuration parameters
    if args.data and args.simulation:
        error = 'Either create "{}" traceback from data or a simulation, not both.'.format(args.name)
        info(error)
        raise ValueError(error)
    elif not args.data and not args.simulation:
        warning('Output from database.')
        data, parameters = None, None
    elif args.data:
        warning('Traceback and output from data.')
        if 'data' not in globals().keys():
            info('No data provided for traceback.')
            raise NameError('No data provided for traceback.')
        elif data is None:
            info('Data provided for traceback is None.')
            raise ValueError('Data provided for traceback is None.')
        else:
            parameters = None
    elif args.simulation:
        warning('Traceback and output from simulation.')
        for parameter in (
            'age', 'number_of_stars',
            'avg_position', 'avg_position_error', 'avg_position_scatter',
            'avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'
        ): # Add check of the type of the arguments !!!
            if parameter not in globals().keys():
                parameter_error = '{} is missing in the configuration file'.format(parameter)
                info(parameter_error)
                raise NameError(parameter_error)
        data = None
        parameters = (
            age, number_of_stars,
            avg_position, avg_position_error, avg_position_scatter,
            avg_velocity, avg_velocity_error, avg_velocity_scatter
        )

    # Creation of the database and Traceback objects
    from model import *
    groups = []
    for group_number in range(1, number_of_groups + 1):
        group_name = '{}_{}'.format(args.name, group_number)
        print('Tracing back {}'.format(group_name.replace('_', ' ')))
        warning('Tracing back {}'.format(group_name.replace('_', ' ')))
        groups.append(
            Group(group_name, number_of_steps, initial_time, final_time, data, parameters)
        )

    # Creation of output files
    create_histogram(
        np.round([group.scatter_age for group in groups], 3),
        avg_position_scatter[0],
        number_of_stars,
        number_of_groups,
        age
    )
    create_graph(groups[0].time, groups[0].scatter)

