# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" __main__.py: Main file of the Traceback algorithm. Includes the definition of the Group and
    Star classes and runs the main script.
"""

import numpy as np
from astropy import units as un
from logging import basicConfig, info, warning, INFO
from time import strftime
from config import *
from init import *
from output import *
from tools import *

# Configuration of the log file
basicConfig(
    filename=logs_path, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

class Group:
    """ Contains the values and related methods of a moving group and a list of Star objets that
        are part of it. Data can be obtained from the database or calculated from a raw data file.
    """
    def __init__(
            self, name: str, number_of_steps=None,
            initial_time=None, final_time=None, data=None, parameters=None):
        """ Initializes a Group object and embedded Star objects from a simulated sample of stars
            in a moving group, raw data in the form a Python dictionnary or from a database. This
            dataset is then moved backward in time from the initial to the final time. Distances
            are in pc, durations in Myr and velocities in pc/Myr.
        """
        # Creation or retrieval of the group data in the database
        if args.from_database or args.to_database:
            self.data, self.created = GroupModel.get_or_create(name=name)

        # Initialization from database
        if args.from_database:
            GroupModel.initialize_from_database(GroupModel, self, self.data)

        # Initialization from a data or simulation
        else:
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
            self.avg_velocity = np.sum(
                np.array([star.velocity for star in self.stars]), axis=0) / self.number_of_stars
            self.avg_velocity_error = np.sum(
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
            if args.to_database:
                GroupModel.save_to_database(GroupModel, self)

    def stars_from_data(self, data):
        """ Creates a list of Star objects from a Python dictionnary or CSV files containing the
            parameters including the name of the stars, their position (XYZ) and velocity (UVW),
            and the respective errors.
        """
#        return [
#            Star(
#                name, self.number_of_steps, self.time,
#                np.array(value['velocity']), np.array(value['velocity_error']),
#                np.array(value['position']), np.array(value['position_error'])
#            ) for name, value in data[self.name].items()
#        ]
        stars = []
        i = 0
        for star in data[self.name]:
            position_rδα = [1000/star[0], star[1], star[2]]
            position_xyz, position_xyz_error = rδα_to_xyz(*position_rδα)
            velocity_uvw, velocity_uvw_error = rvμδμα_to_uvw(*position_rδα, star[3], star[4], star[5] / np.cos(star[1] * un.deg.to(un.rad)))
            stars.append(
                Star('Star_{}'.format(i), self.time, velocity_uvw * (un.km/un.s).to(un.pc/un.Myr),
                np.array(avg_velocity_error) * (un.km/un.s).to(un.pc/un.Myr),
                position_xyz, np.array(avg_position_error)))
            i += 1
        return stars

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
#            avg_velocity, avg_velocity_scatter))
            avg_velocity, avg_velocity_scatter)) * (un.km/un.s).to(un.pc/un.Myr)
        # Star objects creation
        stars = []
        for star in range(1, number_of_stars + 1):
            # Picks a velocity and a position based average value and scatter
            velocity = np.random.normal(
                np.array(avg_velocity), np.array(avg_velocity_scatter))
            position = velocity * age + np.random.normal(
                np.array(avg_position), np.array(avg_position_scatter))
            # Scrambles the velocity and position based on errors in spherical coordinates
            velocity_rvμδμα = uvw_to_rvμδμα(*position, * (velocity * (un.pc/un.Myr).to(un.km/un.s)))[0]
            position_rδα = xyz_to_rδα(*position)[0]
            velocity_uvw = rvμδμα_to_uvw(
                *position_rδα, *np.random.normal(velocity_rvμδμα, np.array(avg_velocity_error))
            )[0] * (un.km/un.s).to(un.pc/un.Myr)
            position_xyz = rδα_to_xyz(
                *np.random.normal(
                    position_rδα, np.array(avg_position_error) * np.array([
                        (position_rδα[0]**2) * un.mas.to(un.arcsec),
                        un.mas.to(un.deg), un.mas.to(un.deg)
                    ])
                )
            )[0]
            stars.append(
                Star(
                    'star_{}'.format(star), self.time,
                    np.random.normal(velocity, avg_velocity_error), np.array(avg_velocity_error),
                    np.random.normal(position, avg_position_error), np.array(avg_position_error)
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

class Star:
    """ Contains the values and related methods of a star.
    """
    def __init__(
            self, name=None, time=None, velocity=None, velocity_error=None,
            initial_position=None, initial_position_error=None):
        """ Initializes Star objects and computes its position and position error from an initial
            position at different times for a given velocity. Distances are in pc and velocities in
            pc/Myr.
        """
        # Initialization from database
        if args.from_database:
            pass

        # Initialization from data or simulation
        else:
            # Time-independent parameters
            self.name = name
            self.velocity = velocity
            self.velocity_error = velocity_error
            # Time-dependent parameters
            self.position = initial_position - (self.velocity * np.expand_dims(time, axis=0).T)
            self.position_error = np.sum(np.array(
                [initial_position_error, self.velocity_error * np.expand_dims(time, axis=0).T]
            )**2, axis=0)**0.5

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

if __name__ == '__main__':
    # Traceback
    groups = []
    for name in group_names:
        info('Tracing back {}'.format(name.replace('_', ' ')))
        print('Tracing back {}'.format(name.replace('_', ' ')))
        groups.append(
            Group(name, number_of_steps, initial_time, final_time, data, parameters))

    # Output
    create_scatter_graph(groups, args.name)
