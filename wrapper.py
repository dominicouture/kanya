# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" wrapper.py: Wraps __main__.py to execute the Traceback algorithm with different configuration
    files.
"""

import fileinput
import os

for age in  [5.0, 10.0, 50.0, 100.0] :
    for avg_position_scatter in [5.0, 10.0, 15.0, 20.0, 25.0]:
        avg_position_scatter = (avg_position_scatter,) * 3
        for number_of_stars in [30, 40, 50, 60, 100, 200]:
            # Modification of config.py
            config = open('config.py', 'r')
            lines = [line for line in config.readlines()]
            config.close()
            os.remove('config.py')
            config = open('config.py', 'w')
            for line in lines:
                if line.startswith('number_of_stars'):
                    config.write('number_of_stars = {} # 42\n'.format(number_of_stars))
                elif line.startswith('avg_position_scatter'):
                    config.write('avg_position_scatter = {} # (29.3, 14.0, 9.0)\n'.format(avg_position_scatter))
                elif line.startswith('age'):
                    config.write('age = {}\n'.format(age))
                elif line.startswith('initial_time'):
                    config.write('initial_time = {}\n'.format(age - 5.0))
                elif line.startswith('final_time'):
                    config.write('final_time = {}\n'.format(age + 5.0))
                else:
                    config.write(line)
            config.close()

for avg_velocity_error in  [0.05, 0.1, 0.15]:
    # Modification of config.py
    config = open('config.py', 'r')
    lines = [line for line in config.readlines()]
    config.close()
    os.remove('config.py')
    config = open('config.py', 'w')
    for line in lines:
        if line.startswith('avg_velocity_error'):
            config.write('avg_velocity_error = ({}, {}, {})\n'.format(avg_velocity_error, avg_velocity_error, avg_velocity_error))
        else:
            config.write(line)
    config.close()

    # Execution of __main__.py
    os.system('python3 . Simulation_{} -s'.format(avg_velocity_error))
