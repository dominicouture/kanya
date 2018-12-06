# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Main file of the Traceback algorithm

import numpy as np
from astropy import units as un
from argparse import ArgumentParser
from traceback import format_exc
from logging import basicConfig, info, warning, INFO
from matplotlib import rcParams, pyplot as plt
from time import strftime
from os.path import join
from os import remove
from scipy.optimize import curve_fit
from tools import *
from old_tools import *
from output import *
from config import *

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

class Group:
    """ Contains the values and related methods of a moving group and a list of Star objets that
        are part of it. Data can be obtained from the database or calculated from a raw data file.
    """
    # All arguments but 'name' are unnecessary, unless support for multiple config is added.
    def __init__(
        self, name: str, number_of_steps: int,
        initial_time: float, final_time: float, data, parameters):
        """ Initializes Star objects and Group object from a simulated sample of stars in a local
            association, raw data in the form a Python dictionnary or from a database. This dataset
            is then moved backward in time for a given traceback duration. Distances are in pc,
            durations in Myr and velocities in pc/Myr.
        """
#        # Creation or retrieval of the group in the database
        info('Tracing back {}'.format(name.replace('_', ' ')))
        print('Tracing back {}'.format(name.replace('_', ' ')))
        self.group, self.created = GroupModel.get_or_create(name=name)

        # Initialization from a simulation or data if possible
        if data is not None or parameters is not None:
            # Group parameters
            self.name = name
            self.date = strftime('%Y-%m-%d %H:%M:%S')
            self.initial_time = initial_time
            self.final_time = final_time
            self.duration = self.final_time - self.initial_time
            self.number_of_steps = number_of_steps + 1 # One more step to account for t = 0
            self.timestep = self.duration / number_of_steps
            self.time = np.linspace(self.initial_time, self.final_time, self.number_of_steps)
            # Stars parameters
            if data is not None:
                self.stars = self.stars_from_data(data)
            elif parameters is not None:
                self.stars = self.stars_from_simulation(*parameters)
            self.number_of_stars = len(self.stars)
            self.average_velocity = np.sum(
                np.array([star.velocity for star in self.stars]), axis=0) / self.number_of_stars
            self.average_velocity_error = np.sum(
                np.array([star.velocity_error for star in self.stars])**2, axis=0
            )**0.5 / self.number_of_stars
            self.barycenter = np.sum(
                np.array([star.position for star in self.stars]), axis=0) / self.number_of_stars
            self.barycenter_error = np.sum(
                np.array([star.position_error for star in self.stars])**2, axis=0
            )**0.5 / self.number_of_stars
            for star in self.stars:
                star.get_distance(self.barycenter, self.barycenter_error)
            # Scatter parameters
            self.get_scatter()
            self.scatter_age = self.time[np.argmin(self.scatter)]
            self.scatter_age_error = 0.0
            # Minimum spanning tree parameters
            self.minimum_spanning_tree = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_error = np.zeros([self.number_of_steps])
            self.minimum_spanning_tree_points = np.zeros([self.number_of_steps, 2, 3])
            self.minimum_spanning_tree_age = 0.0
            self.minimum_spanning_tree_age_error = 0.0

            # Deletion of previous entries and creation of new entries in the database
            self.save_to_database()

        # Initialization from database
        else:
            self.initialize_from_database(self.group)

    def initialize_from_database(self, group):
        """ Initializes Group object from an existing entry in the database.
        """
        # Group parameters
        self.name = group.name
        self.data = group.date
        self.initial_time = group.initial_time
        self.final_time = group.final_time
        self.duration = group.duration
        self.number_of_steps = group.number_of_steps
        self.timestep = group.timestep
        self.time = group.time
        # Star parameters
        self.stars = [Star(
            star.name, data=star) for star in StarModel.select().where(StarModel.group == group)]
        self.number_of_stars = group.number_of_stars
        self.average_velocity = group.average_velocity
        self.average_velocity_error = group.average_velocity_error
        self.barycenter = group.barycenter
        self.barycenter_error = group.barycenter_error
        # Scatter parameters
        self.scatter_xyz = group.scatter_xyz
        self.scatter_xyz_error = group.scatter_xyz_error
        self.scatter = group.scatter
        self.scatter_error = group.scatter_error
        self.scatter_age = group.scatter_age
        self.scatter_age_error = group.scatter_age_error
        # Minimum spanning tree parameters
        self.minimum_spanning_tree = group.minimum_spanning_tree
        self.minimum_spanning_tree_error = group.minimum_spanning_tree_error
        self.minimum_spanning_tree_points = group.minimum_spanning_tree_points
        self.minimum_spanning_tree_age = group.minimum_spanning_tree_age
        self.minimum_spanning_tree_age_error = group.minimum_spanning_tree_age_error

    def stars_from_data(self, data):
        """ Creates a list of Star objects from a Python dictionnary or CSV files containing the
            parameters including the name of the stars, their position (XYZ) and velocity (UVW),
            and the respective errors.
        """
        return [
            Star(
                name, self.number_of_steps, self.time,
                np.array(value['position']), np.array(value['position_error']),
                np.array(value['velocity']), np.array(value['velocity_error'])
            ) for name, value in data[self.name].items()
        ]

    def stars_from_simulation(
        self, number_of_stars: int, age: float,
        avg_position: tuple, avg_position_error: tuple, avg_position_scatter: tuple,
        avg_velocity: tuple, avg_velocity_error: tuple, avg_velocity_scatter: tuple):
        """ Creates an artificial sample of star for a given number of stars and age based on
            the intial average position (XYZ) and velocity (UVW), and their respective error and
            scatter. The sample is then moved forward in time for the given age.
        """
        # Velocity conversions from km/s to pc/Myr
        avg_velocity, avg_velocity_scatter = np.array((
            avg_velocity, avg_velocity_scatter))
#            avg_velocity, avg_velocity_scatter)) * (un.km/un.s).to(un.pc/un.Myr)
        # Star objects creation
        stars = []
        for star in range(1, number_of_stars + 1):
            # Picks a velocity and a position based average value and scatter
            velocity = np.random.normal(
                np.array(avg_velocity), np.array(avg_velocity_scatter))
            position = velocity * age + np.random.normal(
                np.array(avg_position), np.array(avg_position_scatter))
            # Scrambles the velocity and position based on errors in spherical coordinates
            velocity_rvμδμα = uvw_to_rvμδμα(*position, *(velocity * (un.pc/un.Myr).to(un.km/un.s)))
            position_rδα = xyz_to_rδα(*position)
            velocity_uvw = rvμδμα_to_uvw(
                *position_rδα, *np.random.normal(velocity_rvμδμα, np.array(avg_velocity_error))
            ) * (un.km/un.s).to(un.pc/un.Myr)
            position_xyz = rδα_to_xyz(
                *np.random.normal(
                    position_rδα, np.array(avg_position_error) * np.array([
                        (position_rδα[0]**2)*un.mas.to(un.arcsec),
                        un.mas.to(un.deg),
                        un.mas.to(un.deg)
                    ])
                )
            )
            stars.append(
                Star(
                    'star_{}'.format(star), self.number_of_steps, self.time,
                    np.random.normal(position, avg_position_error), np.array(avg_position_error),
                    np.random.normal(velocity, avg_velocity_error), np.array(avg_velocity_error)
                )
            )
        return stars

    def get_scatter(self):
        """ Computes the xyz and total scatter of a group and their respective error for all
            timesteps, filters stars farther than 3σ from the barycenter from the calculations and
            compensates for the drift in minimal scatter age due to measurement errors.
        """
        # Add recursive function to filters stars farther than 3σ from the barycenter here !!!
        self.scatter_xyz = np.std([star.position for star in self.stars], axis=0)
        self.scatter_xyz_error = self.barycenter_error
        self.scatter = np.sum(self.scatter_xyz**2, axis=1)**0.5 - np.sum(
            np.array(avg_position_error)**2 + np.array(avg_velocity_error)**2 \
            * np.expand_dims(self.time, axis=0).T**2, axis=1)**0.5
        self.scatter_error = np.sum(
            (self.scatter_xyz * self.scatter_xyz_error)**2, axis=1)**0.5 / self.scatter

    def save_to_database(self):
        """ Saves all parameters to the database, including all Star objects within the Group object.
            Previous entries are deleted if necessary and new entries are added.
        """
        # Previous GroupModel and StarModel entries deletion
        self.group.delete_instance(recursive=True)
        if self.created:
            info('New database entry "{}" added.'.format(self.name))
        else:
            info('Previous database entry "{}" deleted and replaced.'.format(self.name))

        # GroupModel database entrycreation
        self.group = GroupModel.create(
            # Group parameters
            name=self.name,
            date=self.date,
            initial_time = self.initial_time,
            final_time = self.final_time,
            duration=self.duration,
            number_of_steps=self.number_of_steps,
            timestep=self.timestep,
            time=self.time,
            # Star parameters
#            number_of_stars=self.number_of_stars,
            average_velocity=self.average_velocity,
            average_velocity_error=self.average_velocity_error,
            barycenter=self.barycenter,
            barycenter_error=self.barycenter_error,
            # Scatter parameters
            scatter_xyz=self.scatter_xyz,
            scatter_xyz_error=self.scatter_xyz_error,
            scatter=self.scatter,
            scatter_error=self.scatter_error,
            scatter_age=self.scatter_age,
            scatter_age_error=self.scatter_age_error,
            # Minimum spanning tree parameters
            minimum_spanning_tree=self.minimum_spanning_tree,
            minimum_spanning_tree_error=self.minimum_spanning_tree_error,
            minimum_spanning_tree_points=self.minimum_spanning_tree_points,
            minimum_spanning_tree_age=self.minimum_spanning_tree_age,
            minimum_spanning_tree_age_error=self.minimum_spanning_tree_age_error
        )

        # Creation of new StarModel entries
        for star in self.stars:
            star.save_to_database(self.group)

class Star:
    """ Contains the values and related methods of a star.
    """
    def __init__(
        self, name: str, number_of_steps=None, time=None,
        position=None, position_error=None, velocity=None, velocity_error=None, data=None):
        """ Initializes basic parameters and arrays to the correct shape. Distances are in pc and
            velocities in pc/Myr.
        """
        # Initialization from data or simulation
        if data is None:
            # Time-independent parameters
            self.name = name
            self.velocity = velocity
            self.velocity_error = velocity_error
            # Time-dependent parameters
            self.position = position - (self.velocity * np.expand_dims(time, axis=0).T)
            self.position_error = np.sum(np.array(
                [position_error, self.velocity_error * np.expand_dims(time, axis=0).T]
            )**2, axis=0)**0.5

        # Initialization from database
        else:
            self.initialize_from_database(data)

    def get_distance(self, barycenter, barycenter_error):
        """ Computes the relative position and distance from the barycenter and their respective
            errorsfor all timesteps.
        """
        # Time-dependent parameters
        self.relative_position = self.position - barycenter
        self.relative_position_error = (self.position_error**2 + barycenter_error**2)**0.5
        self.distance = np.sum(self.relative_position**2, axis=1)**0.5
        self.distance_error = np.sum(
            (self.relative_position * self.relative_position_error)**2, axis=1)**0.5 / self.distance

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
    # Arguments import
    parser = ArgumentParser(
        prog='Traceback',
        description='traces given or simulated moving groups of stars back to their origin.')
    parser.add_argument(
        '-d', '--data', action='store_true',
        help='use data parameter in the configuration file as input.')
    parser.add_argument(
        '-s', '--simulation', action='store_true',
        help='run a simulation to create an input based on parameters in the configuraiton file.')
    parser.add_argument(
        'name', action='store',
        help='name of the traceback, used for database input and output, and output filenames.')
    args = parser.parse_args()

    # Database import or creation
    from model import *

    # Configuration import
    # Add check of the type of the arguments. Change to Quantity object, default values if absent !!!
    if args.data and args.simulation:
        error = 'Either traceback "{}" from data or a simulation, not both.'.format(args.name)
        warning('ValueError: {}'.format(error))
        raise ValueError(error)
    # Check if output from database is possible
    elif not args.data and not args.simulation:
        info('Output from database.')
        group_names = [group.name for group in GroupModel.select()]
        if len(group_names) == 0:
            remove(join(output_dir, '{}.db'.format(args.name)))
            error = 'No existing data in the database with the name "{}.db".'.format(args.name)
            warning('ValueError: {}'.format(error))
            raise ValueError(error)
        data, parameters = None, None
    # Check if traceback and output from data is possible.
    elif args.data:
        info('Traceback and output from data.')
        if 'data' not in globals().keys():
            warning('NameError: No data provided for traceback.')
            raise NameError('No data provided for traceback.')
        if data is None:
            warning('ValueError: Data provided for traceback is None.')
            raise ValueError('Data provided for traceback is None.')
        parameters = None
        group_names = list(data.keys())
    # Check if traceback and output from data is possible
    elif args.simulation:
        info('Traceback and output from simulation.')
        for parameter in (
                'number_of_groups', 'number_of_stars', 'age',
                'avg_position', 'avg_position_error', 'avg_position_scatter',
                'avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'
            ):
            if parameter not in globals().keys():
                error = '{} is missing in the configuration file.'.format(parameter)
                warning('NameError: {}'.format(parameter))
                raise NameError(error)
        if type(number_of_groups) is not int:
            error = 'number_of_groups must be an integer ({} given).', type(number_of_groups)
            warning('TypeError: {}'.format(error))
            raise TypeError(error)
        if not number_of_groups > 0:
            error = 'number_of_groups must be greater than 0 ({} given).'.format(number_of_groups)
            warning('ValueError: {}'.format(error))
            raise ValueError(error)
        if type(number_of_stars) is not int:
            error = 'number_of_stars must be an integer ({} given).', type(number_of_stars)
            warning('TypeError: {}'.format(error))
            raise TypeError(error)
        if not number_of_stars > 0:
            error = 'number_of_stars must be greater than 0 ({} given).'.format(number_of_stars)
            warning('ValueError: {}'.format(error))
            raise ValueError(error)
        if type(age) not in (int, float):
            error = 'age must be an integer or float ({} given).', type(age)
            warning('TypeError: {}'.format(error))
            raise TypeError(error)
        if not age >= 0.0:
            error = 'age must be greater than 0.0 ({} given).'.format(age)
            warning('ValueError: {}'.format(error))
            raise ValueError(error)
        data = None
        parameters = (
            number_of_stars, age,
            avg_position, avg_position_error, avg_position_scatter,
            avg_velocity, avg_velocity_error, avg_velocity_scatter
        )
        group_names = ['{}_{}'.format(args.name, i) for i in range(1, number_of_groups + 1)]
    # Check configuration parameters
    for parameter in ('number_of_steps', 'initial_time', 'final_time'):
        if parameter not in globals().keys():
            error = '{} is missing in the configuration file.'.format(parameter)
            warning('NameError: {}'.format(parameter))
            raise NameError(error)
    if type(number_of_steps) is not int:
        error = 'number_of_steps must be an integer ({} given).', type(number_of_steps)
        warning('TypeError: {}'.format(error))
        raise TypeError(error)
    if not number_of_steps > 0:
        error = 'number_of_steps must be greater than 0 ({} given).'.format(number_of_steps)
        warning('ValueError: {}'.format(error))
        raise ValueError(error)
    if type(initial_time) not in (int, float):
        error = 'number_of_steps must be an integer or float ({} given).'.format(initial_time)
        warming('ValueError: {}'.format(error))
        raise TypeError(error)
    if type(final_time) not in (int, float):
        error = 'final_time must be an integer or float ({} given).'.format(final_time)
        warming('ValueError: {}'.format(error))
        raise TypeError(error)
    if not final_time > initial_time:
        error = 'final_time must be greater than initial_time ({} and {} given).'.format(
            final_time, initial_time)
        warning('ValueError: {}'.format(error))
        raise ValueError(error)

    # Traceback objects creation
    groups = []
    for name in group_names:
        groups.append(
            Group(name, number_of_steps, initial_time, final_time, data, parameters))

    # Output creation
    create_scatter_graph(groups, args.name)
