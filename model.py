# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Model for the peewee database of stars and local associations. Time-dependent data is stored in
# an ndarray() for which axis 0 corresponds to the timestep. The database only records coordinates
# and errors in the XYZ and UVW coordinates system.

import numpy as np
from traceback import format_exc
from logging import basicConfig, warning, INFO
from time import strftime
from os.path import join
from peewee import *
from playhouse.migrate import *
from json import dumps, loads
from __main__ import args
from config import output_dir

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

# Definition of the database object
Database = SqliteDatabase(join(output_dir, '{}.db'.format(args.name)))

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
    date = DateField(verbose_name='Date', default='')
    initial_time = FloatField(verbose_name='Initial Time', default=0.0)
    final_time = FloatField(verbose_name='Final Time', default=0.0)
    duration = FloatField(verbose_name='Duration', default=0.0)
    number_of_steps = IntegerField(verbose_name='Number of Steps', default=0)
    timestep = FloatField(verbose_name='Timestep', default=0.0)
    time = ArrayField(verbose_name='Time', default=np.array([]))
    # Star parameters
    number_of_stars = IntegerField(verbose_name='Number of Stars', default=0)
    average_velocity = ArrayField(verbose_name='Average Velocity', default=np.array([]))
    average_velocity_error = ArrayField(verbose_name='Average Velocity Error', default=np.array([]))
    barycenter = ArrayField(verbose_name='Baraycenter', default=np.array([]))
    barycenter_error = ArrayField(verbose_name='Barycenter Error', default=np.array([]))
    # Scatter parameters
    scatter_xyz = ArrayField(verbose_name='Scatter XYZ', default=np.array([]))
    scatter_xyz_error = ArrayField(verbose_name='Scatter XYZ Error', default=np.array([]))
    scatter = ArrayField(verbose_name='Scatter', default=np.array([]))
    scatter_error = ArrayField(verbose_name='Scatter Error', default=np.array([]))
    scatter_age = FloatField(verbose_name='Scatter Age', default=0.0)
    scatter_age_error = FloatField(verbose_name='Scatter Age Error', default=0.0)
    # Minimum spanning tree parameters
    minimum_spanning_tree = ArrayField(verbose_name='Minimum Spanning Tree', default=np.array([]))
    minimum_spanning_tree_error = ArrayField(
        verbose_name='Minimum Spanning Tree Error', default=np.array([])
    )
    minimum_spanning_tree_points = ArrayField(
        verbose_name='Minimum Spanning Tree Points', default=np.array([])
    )
    minimum_spanning_tree_age = FloatField(verbose_name='Minimum Spanning Tree Age', default=0.0)
    minimum_spanning_tree_age_error = FloatField(
        verbose_name='Minimum Spanning Tree Age Error', default=0.0
    )

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

# Create GroupModel and StarModel tables in the database if they don't exist
GroupModel.create_table(fail_silently=True)
StarModel.create_table(fail_silently=True)
