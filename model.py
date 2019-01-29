# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" model.py: Model for the peewee database of stars and local moving groups. Time-dependent data
    is stored in an ndarray() for which axis 0 corresponds to the timestep. The database only
    records coordinates and errors in the XYZ and UVW coordinates system. The database file is
    located in the output directory defined in config.py
"""

import numpy as np
from peewee import *
from time import strftime
from json import dumps, loads
from init import Config, info

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

print('marmalade')

# Database object definition
Database = SqliteDatabase(Config.db_path)

class BaseModel(Model):
    class Meta:
        database = Database

class ArrayField(Field):
    """ Defines a database field for numpy arrays.
    """
    db_field = 'text'
    field_type = 'array'
    db_value = lambda self, array: dumps(array.tolist())
    python_value = lambda self, string: np.array(loads(string))

class GroupModel(BaseModel):
    """ Time-independent and time-depenent (ArrayField) parameters of a local association of stars.
        Distances are in pc, durations in Myr and velocities in pc/Myr.
    """
    # Group parameters
    name = CharField(verbose_name='Name', unique=True)
    series = CharField(verbose_name='Series')
    date = DateField(verbose_name='Date', default=strftime('%Y-%m-%d %H:%M:%S'))
    initial_time = FloatField(verbose_name='Initial Time', default=0.0)
    final_time = FloatField(verbose_name='Final Time', default=1.0)
    duration = FloatField(verbose_name='Duration', default=1.0)
    number_of_steps = IntegerField(verbose_name='Number of Steps', default=1)
    timestep = FloatField(verbose_name='Timestep', default=1.0)
    time = ArrayField(verbose_name='Time', default=np.array([]))
    # Star parameters
    number_of_stars = IntegerField(verbose_name='Number of Stars', default=0)
    avg_velocity = ArrayField(verbose_name='Average Velocity', default=np.zeros((1, 3)))
    avg_velocity_error = ArrayField(
        verbose_name='Average Velocity Error', default=np.zeros((1, 3)))
    barycenter = ArrayField(verbose_name='Barycenter', default=np.zeros((1, 3)))
    barycenter_error = ArrayField(verbose_name='Barycenter Error', default=np.zeros((1, 3)))
    # Scatter parameters
    scatter_xyz = ArrayField(verbose_name='Scatter XYZ', default=np.zeros((1, 3)))
    scatter_xyz_error = ArrayField(verbose_name='Scatter XYZ Error', default=np.zeros((1, 3)))
    scatter = ArrayField(verbose_name='Scatter', default=np.zeros(1))
    scatter_error = ArrayField(verbose_name='Scatter Error', default=np.zeros(1))
    scatter_age = FloatField(verbose_name='Scatter Age', default=0.0)
    scatter_age_error = FloatField(verbose_name='Scatter Age Error', default=0.0)
    # Minimum spanning tree parameters
    minimum_spanning_tree = ArrayField(verbose_name='Minimum Spanning Tree', default=np.zeros(1))
    minimum_spanning_tree_error = ArrayField(
        verbose_name='Minimum Spanning Tree Error', default=np.zeros(1))
    minimum_spanning_tree_points = ArrayField(
        verbose_name='Minimum Spanning Tree Points', default=np.zeros((1, 2, 3)))
    minimum_spanning_tree_age = FloatField(verbose_name='Minimum Spanning Tree Age', default=0.0)
    minimum_spanning_tree_age_error = FloatField(
        verbose_name='Minimum Spanning Tree Age Error', default=0.0)

    def initialize_from_database(self, group, model):
        """ Initializes Group object from an existing entry in the database.
        """
        database_values = vars(model)['_data']
        del database_values['id']
        vars(group).update(database_values)
        group.stars = [StarModel.initialize_from_database(StarModel, group, star)
            for star in StarModel.select().where(StarModel.group == model)]

    def save_to_database(self, group):
        """ Saves all parameters to the database, including all Star objects within the Group object.
            Previous entries are deleted if necessary and new entries are added.
        """
        # Previous GroupModel and StarModel entries deletion
        group.model.delete_instance(recursive=True)
        if group.created:
            info('New database entry "{}" added.'.format(group.name))
        else:
            info('Previous database entry "{}" deleted and replaced.'.format(group.name))

        # GroupModel entry creation
        group_values = vars(group).copy()
        del group_values['model'], group_values['created'], group_values['stars']
        group.model = self.create(**group_values)

        # StarModel entries creation
        for star in group.stars:
            StarModel.save_to_database(StarModel, star, group.model)

class StarModel(BaseModel):
    """ Time-independent and time-dependent (ArrayField) parameters of a star in a local group.
        Distances are in pc and velocities in pc/Myr.
    """
    group = ForeignKeyField(GroupModel)
    # Time-independent parameters
    name = CharField(verbose_name='Name')
    velocity = ArrayField(verbose_name='Velocity')
    velocity_error = ArrayField(verbose_name='Velocity Error')
    # Time-dependent parameters
    position = ArrayField(verbose_name='Position')
    position_error = ArrayField(verbose_name='Position Error')
    relative_position = ArrayField(verbose_name='Relative Position')
    relative_position_error = ArrayField(verbose_name='Relative Position Error')
    distance = ArrayField(verbose_name='Distance')
    distance_error = ArrayField(verbose_name='Distance Error')

    def initialize_from_database(self, group, model):
        """ Initializes Star object from an existing instance in the database.
        """
        from group import Star
        star = Star(group=group)
        database_values = vars(model)['_data']
        del database_values['id']
        vars(star).update(database_values)
        return star

    def save_to_database(self, star, group):
        """ Saves all parameters to the database in a new StarModel entry.
        """
        self.create(group=group, **vars(star))

# GroupModel and StarModel tables creation if they don't already exists.
GroupModel.create_table(fail_silently=True)
StarModel.create_table(fail_silently=True)
