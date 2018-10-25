# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration of a traceback and/or simulation of a sample of stars

# Path to the database directory
output_dir = '/Users/Dominic/Fichiers/Astrophysique/Projet/Traceback/Output/'

# Duration of the traceback (Myr)
duration = 50.0

# Number of steps of the traceback, excluding the initial step at t = 0
number_of_steps = 50

# Number of groups to be simulated
number_of_groups = 1

# Number of stars in the simulated groups of stars
number_of_stars = 42

# Age of the simulated sample of stars (Myr)
age = 24.0

# Average position of the simulated sample of stars (pc)
avg_position = (0.0, 0.0, 0.0)

# Average position error of the simulated sample of stars (pc)
avg_position_error = (0.1, 0.1, 0.1)

# Average position dispersion of the simulated sample of stars (pc)
avg_position_dispersion = (29.0, 14.0, 9.0)

# Average velocity of the simulated sample of stars (km/s)
avg_velocity = (-10.9, -16.0, -9.0)

# Average velocity error of the simulated sample of stars (km/s)
avg_velocity_error = (0.1, 0.1, 0.1)

# Average velocity error of the simulated sample of stars (km/s)
avg_velocity_dispersion = (2.2, 1.2, 1.0)
