# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Provide the necesary functions to output data in table, graph or video form.

import matplotlib.pyplot as plt
from os.path import join
from traceback import format_exc
from logging import basicConfig, warning, INFO
from time import strftime
from config import *

# Configuration of the log file
basicConfig(
    filename=join(db_dir, 'Logs/Traceback_{}.log'.format(strftime('%Y-%m-%d %H:%M:%S'))),
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S -', level=INFO)

def create_graph(x, y):
    plt.plot(x, y,  '.-')
    plt.xlabel('Time')
    plt.ylabel('Standard deviation')
    plt.title('Dispersion of a local association over time (arbirtrary units)')
    plt.show()
