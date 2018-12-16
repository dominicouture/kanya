# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" init.py: Imports information from config.py, handles exceptions (type, value, shape), handles
    unit conversion, checks for the presence of output and logs directories and creates them if
    necessary, recursively creates a list of configurations to be executed by the main Traceback
    algorithm. This script is run first before the rest of the module.
"""

from logging import basicConfig, info, warning, INFO
from time import strftime
from os import remove, makedirs
from os.path import join, exists
from argparse import ArgumentParser
from config import *

# Creation of output directory
output_dir = output
if not exists(output_dir):
    makedirs(output_dir)
if not exists(join(output_dir, 'Logs')):
    makedirs(join(output_dir, 'Logs'))
logs_path = join(output_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S')))

# Configuration of the log file
basicConfig(
    filename=logs_path, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

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
    '-b', '--to_database', action='store_true',
    help='save the output data to a database file.')
parser.add_argument(
    'name', action='store',
    help='name of the traceback, used for database input and output, and output filenames.')
args = parser.parse_args()
args.from_database = True if not args.data and not args.simulation else False

# Configuration import
if args.data and args.simulation:
    error = 'Either traceback "{}" from data or a simulation, not both.'.format(args.name)
    warning('ValueError: {}'.format(error))
    raise ValueError(error)

# Creation or import of database if output from or to database
if args.from_database or args.to_database:
    db_created = False if exists(join(output_dir, '{}.db'.format(args.name))) else True
    from model import *

# Check if output from database is possible
if args.from_database:
    info('Output from database.')
    if db_created:
        remove(join(output_dir, '{}.db'.format(args.name)))
        error = 'No existing database with the name "{}.db".'.format(args.name)
        warning('ValueError: {}'.format(error))
        raise ValueError(error)
    group_names = [group.name for group in GroupModel.select()]
    if len(group_names) == 0:
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
            'avg_velocity', 'avg_velocity_error', 'avg_velocity_scatter'):
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
        avg_velocity, avg_velocity_error, avg_velocity_scatter)
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


def import_config():
    """ Imports configuration file into Quantity objects and convert units into default units:

        time: million year (Myr)
        position: parsec (pc)
        velocity: parsec per million year (pc/Myr)
        angle: radian
        angular velocity: (Myr^-1)

        Returns a series a configurations to be passed into a Group object using a recursive
        algorithm.
    """
    pass

class Configuration():
    """ Contains a configurations to be passed into a Group object. The initialization checks for
        errors, including TypeError and ValueError.
    """
    
    def __init():
        """ Checks individual configurations for errors, including TypeError and ValueError
        """
        pass
