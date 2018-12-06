# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration of a traceback and/or simulation of a sample of stars

# Path to the database directory
output_dir = '/Users/Dominic/Fichiers/Astrophysique/Projet/Traceback/Output/'

# Time units ('Myr' by default)
time_unit = 'Myr'

# Position units ('pc' by default, XYZ or rδα)
position_units = ('pc', 'pc', 'pc') # ('pc', 'deg', 'deg')

# Velocity units ('km/s' by default, UVW or rvμδμα)
velocity_units = ('km/s', 'km/s', 'km/s') # ('km/s', 'mas/yr', 'mas/yr')

# Traceback parameters
# Number of steps of the traceback, excluding the initial step at t = 0 (integer, > 0)
number_of_steps = 200

# Initial age of the traceback (inclusive, float, Myr)
initial_time = 14.0

# Final age of the traceback (inclusive, float, > initial_time, Myr)
final_time = 34.0

# Simulation parameters, used only in simulation
# Number of groups to be simulated (integer, > 0)
number_of_groups = 10

# Number of stars in each simulated group of stars (integer, > 0)
number_of_stars = 42 # 42

# Age of the simulated groups of stars (float, ≥ 0.0, Myr)
age = 24.0

# Average position of the simulated sample of stars (pc, pc, pc)
#avg_position = (258.0, 258.0, 258.0)
avg_position = (0.0, 0.0, 0.0)

# Average position error of the simulated sample of stars (mas, mas, mas)
avg_position_error = (0.3, 0.0, 0.0)
#avg_position_error = (0.0, 0.0, 0.0)

# Average position scatter of the simulated sample of stars (pc, pc, pc)
avg_position_scatter = (2.0, 2.0, 2.0) # (29.3, 14.0, 9.0)

# Average velocity of the simulated sample of stars (pc/Myr, pc/Myr, pc/Myr)
avg_velocity = (-11.622, -11.622, -11.622) # (-10.9, -16.0, -9.0)

# Average velocity error of the simulated sample of stars (km/s, mas/yr, mas/yr)
#avg_velocity_error = (0.0, 0.5, 0.5)
avg_velocity_error = (1.5, 0.2, 0.2)

# Average velocity error of the simulated sample of stars (pc/Myr, pc/Myr, pc/Myr)
avg_velocity_scatter = (1.382, 1.382, 1.382) # (2.2, 1.2, 1.0)

# Path to CSV data file or dictionary with data. Set as None if no data is needed
data = None
