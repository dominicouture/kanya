# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Model for the peewee database of stars and local associations. Time-dependent data is stored in
# an ndarray() for which axis 0 corresponds to the timestep.

import numpy as np
from traceback import format_exc
from logging import basicConfig, warning, INFO
from time import strftime
from os.path import join
from peewee import *
from playhouse.migrate import *
from json import dumps, loads
from __main__ import name
from config import *

# Configuration of the log file
basicConfig(
    filename=join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

# Definition of the database object
data = SqliteDatabase(join(output_dir, '{}.db'.format(name)))

class BaseModel(Model):
    class Meta:
        database = data

class ArrayField(Field):
    """ Defines a database field for numpy arrays.
    """
    db_field = 'text'
    field_type = 'array'
    db_value = lambda self, array: dumps(array.tolist())
    python_value = lambda self, string: np.array(loads(string))

class GroupModel(BaseModel):
    """ Time-independent and time-depenent (ArrayField) parameters of a local association of stars.
    """
    # Time-independent parameters
    name = CharField(verbose_name='Name', unique=True)
    date = DateField(verbose_name='Date', default='')
    duration = FloatField(verbose_name='Duration', default=0.0)
    number_of_stars = IntegerField(verbose_name='Number of Stars', default=0)
    number_of_steps = IntegerField(verbose_name='Number of Steps', default=0)
    timestep = FloatField(verbose_name='Timestep', default=0.0)
    time = ArrayField(verbose_name='Time', default=np.array([]))
    average_velocity = ArrayField(verbose_name='Average Velocity', default=np.array([]))
    average_velocity_error = ArrayField(verbose_name='Average Velocity Error', default=np.array([]))
    # Time-dependent parameters
    barycenter = ArrayField(verbose_name='Baraycenter', default=np.array([]))
    barycenter_error = ArrayField(verbose_name='Barycenter Error', default=np.array([]))
    dispersion = ArrayField(verbose_name='Dispersion', default=np.array([]))
    dispersion_error = ArrayField(verbose_name='Dispersion Error', default=np.array([]))
    minimum_spanning_tree = ArrayField(verbose_name='Minimum Spanning Tree', default=np.array([]))
    minimum_spanning_tree_error = ArrayField(verbose_name='Minimum Spanning Tree Error', default=np.array([]))
    minimum_spanning_tree_points = ArrayField(verbose_name='Minimum Spanning Tree Points', default=np.array([]))

class StarModel(BaseModel):
    """ Time-independent and time-dependent (ArrayField) parameters of a star in a local association.
    """
    # Time-independent parameters
    group = ForeignKeyField(GroupModel)
    name = CharField(verbose_name='Name', unique=True)
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
