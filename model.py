# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Model for the peewee database of stars and local associations. Time-dependent data is stored in
# an ndarray() for which axis 0 corresponds to the timestep.

from peewee import *
from playhouse.migrate import *
from os.path import join
from json import dumps, loads
from traceback import format_exc
from logging import basicConfig, warning, INFO
from time import strftime
from config import *

# Configuration of the log file
basicConfig(
    filename=join(db_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

# Definition of the database object
data = SqliteDatabase(join(db_dir, 'Database_{}'.format(strftime('%Y-%m-%d %H:%M:%S'))))

class BaseModel(Model):
    class Meta:
        database = data

class ListField(Field):
    db_field = 'text'
    db_value = lambda self, x: dumps(x)
    python_value = lambda self, x: loads(x)

class ArrayField(Field):
    db_field = 'text'
    db_value = lambda self, x: dumps(x.tolist())
    python_value = lambda self, x: loads(np.array(x))

class GroupModel(BaseModel):
    """ Time-independent and time-depenent (ArrayField) parameters of a local association of stars.
    """
    # Time-independent
    name = CharField(verbose_name='Name', unique=True)
    date = DateField(verbose_name='Date')
    duration = FloatField(verbose_name='Duration')
    number_of_stars = IntegerField(verbose_name='Number of stars')
    number_of_steps = IntegerField(verbose_name='Number of steps')
    timestep = FloatField(verbose_name='Timestep')
    time = ListField(verbose_name='Time')
    velocity = ArrayField(verbose_name='Velocity')
    velocity_error = ArrayField(verbose_name='Velocity Error')
    # Time-dependent
    barycenter = ArrayField(verbose_name='Baraycenter')
    barycenter_error = ArrayField(verbose_name='Barycenter Error')
    dispersion = ListField(verbose_name='Dispersion')
    dispersion_error = ListField(verbose_name='Dispersion Error')
    minimum_spanning_tree = ListField(verbose_name='Minimum Spanning Tree')
    minimum_spanning_tree_error = ListField(verbose_name='Minimum Spanning Tree Error')
    minimum_spanning_tree_points = ArrayField(verbose_name='Minimum Spanning Tree Points')

class StarModel(BaseModel):
    """ Time-independent and time-dependent (ArrayField) parameters of a star in a local association.
    """
    # Time-independent
    group = ForeignKeyField(GroupModel)
    name = CharField(verbose_name='Name', unique=True)
    velocity = FloatField(verbose_name='Velocity')
    velocity_error = FloatField(verbose_name='Velocity Error')
    # Time-dependent
    position = ArrayField(verbose_name='Position')
    position_error = ArrayField(verbose_name='Position Error')
    relative_position = ArrayField(verbose_name='Relative Position')
    relative_position_error = ArrayField(verbose_name='Relative Position Error')
    distance = ListField(verbose_name='Distance')
    distance_error = ListField(verbose_name='Distance Error')

# Create Patient, Study and Organ tables in the database if they don't exist
GroupModel.create_table(fail_silently=True)
StarModel.create_table(fail_silently=True)
