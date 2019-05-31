# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" model.py: Model for the peewee database of stars and moving groups. Time-dependent data
    is stored in an np.ndarray for which axis 0 is time. The database only records coordinates
    and errors in the XYZ and UVW coordinates system. The database file is located at the path
    defined by Config.db_path.
"""

import numpy as np
from peewee import *
from time import strftime
from json import dumps, loads
from group import Group
from init import Config
from series import info

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Database object definition ??? Add to Series ???
Database = SqliteDatabase(Config.db_path)

class BaseModel(Model):

    class Meta:
        database = Database

class ArrayField(Field):
    """ Defines a database field for np.ndarray objects. """

    db_field = 'text'
    field_type = 'array'
    db_value = lambda self, array: dumps(array.tolist())
    python_value = lambda self, string: np.array(loads(string))

class SeriesModel(BaseModel):
    """ Defines the fields of a series of moving groups and methods to initialize from or save
        to the database.
    """

    # Series parameters
    name = CharField(verbose_name='Name', unique=True)
    date = DateField(verbose_name='Date', default=strftime('%Y-%m-%d %H:%M:%S'))
    number_of_groups = IntegerField(verbose_name='Number of groups', default=1)
    number_of_stars = IntegerField(verbose_name='Number of stars', default=1)
    number_of_steps = IntegerField(verbose_name='Number of Steps', default=1)

    # Time parameters
    initial_time = FloatField(verbose_name='Initial Time', default=0.0)
    final_time = FloatField(verbose_name='Final Time', default=1.0)
    duration = FloatField(verbose_name='Duration', default=1.0)
    timestep = FloatField(verbose_name='Timestep', default=1.0)
    time = ArrayField(verbose_name='Time', default=np.array([]))

    def load_from_database(self, series):
        """ Initializes a Series object and embeded Group objects from an existing instance in the
            database, defined as the series.model parameter.
        """

        # Series parameters retrieval
        values = vars(series.model)['_data'].copy()
        del values['id']

        # Series object update
        vars(series).update(values)

        # Group objects creation
        for group in GroupModel.select().where(GroupModel.series == series.model):
        # for group in GroupModel.select():
            GroupModel.load_from_database(GroupModel, series, group)

    def save_to_database(self, series):
        """ Saves all parameters to the database in a new SeriesModel entry. """

        # Previous SeriesModel, GroupModel and StarModel entries deletion
        series.model.delete_instance(recursive=True)

        # Logging
        info("New database entry '{}' added.".format(series.name) if series.created else
            "Previous database entry '{}' deleted and replaced.".format(series.name))

        # SeriesModel entry creation
        series.model.create(**{key: vars(series)[key] for key in filter(
            lambda key: key in vars(self).keys(), vars(series).keys())})

        # GroupModel entries creation
        for group in series:
            GroupModel.save_to_database(GroupModel, group)

class GroupModel(BaseModel):
    """ Defines the fields of a moving group and methods to initialize from or save to the
        database.
    """

    # Group parameters
    series = ForeignKeyField(SeriesModel)
    name = CharField(verbose_name='Name')

    # Average velocity
    avg_velocity = ArrayField(verbose_name='Average Velocity', default=np.zeros((1, 3)))
    avg_velocity_error = ArrayField(
        verbose_name='Average Velocity Error', default=np.zeros((1, 3)))
    # Average position
    avg_position = ArrayField(verbose_name='Average Position', default=np.zeros((1, 3)))
    avg_position_error = ArrayField(verbose_name='Average Position Error', default=np.zeros((1, 3)))

    # Scatter
    scatter_xyz = ArrayField(verbose_name='Scatter XYZ', default=np.zeros((1, 3)))
    scatter_xyz_error = ArrayField(verbose_name='Scatter XYZ Error', default=np.zeros((1, 3)))
    scatter = ArrayField(verbose_name='Scatter', default=np.zeros(1))
    scatter_error = ArrayField(verbose_name='Scatter Error', default=np.zeros(1))
    scatter_age = FloatField(verbose_name='Scatter Age', default=0.0)
    scatter_age_error = FloatField(verbose_name='Scatter Age Error', default=0.0)

    # Median Absolute Deviation
    median_xyz = ArrayField(verbose_name='Median', default=np.zeros((1, 3)))
    mad_xyz = ArrayField(verbose_name='Median Absolute Deviation XYZ', default=np.zeros((1, 3)))
    mad = ArrayField(verbose_name='Median Absolute Deviation', default=np.zeros(1))
    mad_age = FloatField(verbose_name='Median Absolute Deviation Age', default=0.0)
    mad_age_error = FloatField(verbose_name='Median Absolute Deviation Age Error', default=0.0)

    # Minimum spanning tree parameters
    minimum_spanning_tree = ArrayField(verbose_name='Minimum Spanning Tree', default=np.zeros(1))
    minimum_spanning_tree_error = ArrayField(
        verbose_name='Minimum Spanning Tree Error', default=np.zeros(1))
    minimum_spanning_tree_points = ArrayField(
        verbose_name='Minimum Spanning Tree Points', default=np.zeros((1, 2, 3)))
    minimum_spanning_tree_age = FloatField(verbose_name='Minimum Spanning Tree Age', default=0.0)
    minimum_spanning_tree_age_error = FloatField(
        verbose_name='Minimum Spanning Tree Age Error', default=0.0)

    def load_from_database(self, series, model):
        """ Initializes a Group object from an existing entry in the database. """

        # Group parameters retrival
        values = vars(model)['_data'].copy()
        del values['id'], values['series']

        # Group object creation
        group = Group(series, **values)
        group.model = model
        series.append(group)

        # Star objects creation
        for star in StarModel.select().where(StarModel.group == model):
            StarModel.load_from_database(StarModel, group, star)

    def save_to_database(self, group):
        """ Saves all parameters to the database, including all Star objects within the Group
            object. Previous entries are deleted, if needed, and new entries are added.
        """

        # GroupModel entry creation
        values = {key: vars(group)[key] for key in filter(
            lambda key: key in vars(self).keys(), vars(group).keys())}
        del values['series']
        group.model = self.create(series=group.series.model, **values)

        # StarModel entries creation
        for star in group:
            StarModel.save_to_database(StarModel, star)

class StarModel(BaseModel):
    """ Defines the fields of a star in a moving group and methods to initialize from or save
        to the database.
    """

    # Star parameters
    group = ForeignKeyField(GroupModel)
    name = CharField(verbose_name='Name')

    # Velocity and position
    velocity = ArrayField(verbose_name='Velocity')
    velocity_error = ArrayField(verbose_name='Velocity Error')
    position = ArrayField(verbose_name='Position')
    position_error = ArrayField(verbose_name='Position Error')

    # Relative position and distance
    relative_position = ArrayField(verbose_name='Relative Position')
    relative_position_error = ArrayField(verbose_name='Relative Position Error')
    distance = ArrayField(verbose_name='Distance')
    distance_error = ArrayField(verbose_name='Distance Error')

    def load_from_database(self, group, model):
        """ Initializes Star object from an existing instance in the database. """

        # Star parameters retrieval
        values = vars(model)['_data'].copy()
        del values['id'], values['group']

        # Star object creation
        star = group.Star(group, **values)
        star.model = model
        group.append(star)

    def save_to_database(self, star):
        """ Saves all parameters to the database in a new StarModel entry. """

        # StarModel entry creation
        values = {key: vars(star)[key] for key in filter(
            lambda key: key in vars(self).keys(), vars(star).keys())}
        del values['group']
        star.model = self.create(group=star.group.model, **values)

# ??? Put this part in a function so that it can be reused if need be. ???
# GroupModel and StarModel tables creation if they don't already exists.
SeriesModel.create_table(fail_silently=True)
GroupModel.create_table(fail_silently=True)
StarModel.create_table(fail_silently=True)
