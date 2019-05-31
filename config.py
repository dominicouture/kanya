# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" config.py: Configuration of a traceback from a simulated sample of stars or data. A 'system'
    refers to a coordinate system (e.g. 'observables', 'spherical' or 'cartesian') and an 'axis'
    to the direction of a coordinate system axis (e.g. 'equatorial' or 'galactic'). 'units' must
    be a string convertible to a Astropy.Unit object. The default units are:

        -   Time: Myr
        -   Length: pc
        -   Speed: pc/Myr
        -   Angle: rad
        -   Angular speed: rad/Myr
"""

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Output directory relative to the directory where the Traceback Package is located (str). If left
# blank (i.e. '') or absent, output files will be created in the same directory as the Package.
output_dir.value = 'Output'

# Logs directory relative to the output directory (str). If left blank (i.e. '') or absent, logs
# files will be created in the output directory.
logs_dir.value = 'Logs'

# Path to the database file used as input or output relative the output directory (str). If left
# blank (i.e. '') or absent, the database is created in the output directory as 'name.db'. If the
# file doesn't exist, it will be created.
db_path.value = ''

# Number of groups to be simulated in the series (integer, > 0)
number_of_groups.value = 1

# Number of steps of the traceback, excluding the initial step at t = 0 (integer, > 0)
number_of_steps.value = 200

# Number of stars in each simulated group of stars (integer, > 0)
number_of_stars.value = 45

# Initial age of the traceback (float, inclusive)
initial_time.value = 0.0

# Final age of the traceback (float, > initial_time, inclusive)
final_time.value = 30.0

# Age of simulated groups of stars (float, ≥ 0.0)
age.value = 24.0

# Average position of the simulated sample of stars (tuple)
avg_position.values = (0.0, 0.0, 0.0) # (15.19443946, -4.93616248, -17.07422231)

# Average position error of the simulated sample of stars (tuple)
avg_position_error.values = (0.19846, 0.0, 0.0)
# avg_position_error.values = (0.0, 0.0, 0.0)
avg_position_error.units = 'mas'
avg_position_error.system = 'observables'
avg_position_error.axis = 'equatorial'

# Average position scatter of the simulated sample of stars (tuple)
avg_position_scatter.values = (5.0, 5.0, 5.0) # (29.3, 14.0, 9.0)

# Average velocity of the simulated sample of stars (tuple)
avg_velocity.values =  (-11.3442, -11.3442, -11.3442) # (-10.54893, -15.88653,  -8.71138)
avg_velocity.units = 'km/s'

# Average velocity error of the simulated sample of stars (tuple)
avg_velocity_error.values = (1.0112, 0.30754, 0.26432)
# avg_velocity_error.values = (0.0, 0.0, 0.0)
# avg_velocity_error.units = ('km/s', 'mas/yr', 'mas/yr')
avg_velocity_error.units = 'observables'
avg_velocity_error.system = 'observables'
avg_velocity_error.axis = 'equatorial'

# Average velocity scatter of the simulated sample of stars (tuple)
avg_velocity_scatter.values = (1.68050, 1.68050, 1.68050) # (2.21753, 1.36009, 1.57355)
avg_velocity_scatter.units = 'km/s'

# Path to CSV data file (str) or Python dictionary with the data (dict)
data.value = 'bpic_updated.csv'
data.units = 'observables'
data.system = 'observables'
data.axis = 'observables'
# data.values =  {'beta_pictoris': [
#     [    'p',        'δ',       'α',    'rv',     'μδ',     'μα'],
#     [  'mas',      'deg',     'deg',  'km/s', 'mas/yr', 'mas/yr'],
#     [25.0250, -23.107737, 1.7091324, 6.50000, -47.1210,  96.7760],
#     [27.1700, -66.753496, 4.3491739, 8.75000, -16.8740,  103.039],
#     [28.6590, -32.551949, 6.9598490, 8.80000, -47.3890,  109.817],
#     [45.7900,  15.439305, 17.855935, 4.00000, -120.000,  180.000],
#     [25.1340,  28.744731, 34.355775, 4.32000, -74.0690,  86.9240],
#     [24.3600,  30.973199, 36.872279, 4.85000, -72.0020,  79.6790],
#     [19.5280,  23.709892, 53.758697, 15.5000, -62.9400,  50.3370],
#     [33.5770, -2.4735640, 69.400543, 21.0000, -63.8330,  44.3520],
#     [47.4120, 0.03477300, 70.906710, 16.9700, -107.485,  55.1120],
#     [40.9810,  1.7831241, 74.895306, 18.0800, -95.0510,  39.2340],
#     [37.1740, -57.256761, 75.196646, 18.0700,  74.1380,  35.1980],
#     [18.7230,  15.449949, 75.205429, 18.1000, -58.8350,  18.1310],
#     [50.3080, -21.585933, 76.708196, 20.7100, -15.5740,  47.1440],
#     [37.2060, -11.901168, 81.769919, 20.6100, -49.3150,  17.0590],
#     [50.6230, -51.066517, 86.821179, 20.0000,  82.5770,  2.49300],
#     [25.4700, -72.044537, 94.617428, 10.9500,  74.2950, -7.90800],
#     [30.5570, -27.701553, 93.305428, 22.5000, -5.62300, -13.1570],
#     [51.0040, -53.907368, 154.36205, 14.8000, -4.93900, -173.099],
#     [62.9440, -64.976145, 220.62479, 6.20000, -232.614, -190.466],
#     [24.9340, -57.707982, 234.73939, 3.90000, -95.8810, -55.1970],
#     [19.7560, -53.725795, 254.33436, 0.49000, -84.0870, -10.8490],
#     [32.7800, -66.951608, 259.35604, 3.30000, -137.125, -21.2130],
#     [14.7590, -54.263778, 262.47945, -0.2000, -63.5460, -5.41300],
#     [15.2130, -50.724731, 265.45431, 2.40000, -65.8580, -2.02300],
#     [12.9670, -53.103534, 267.14060, -0.2000, -56.1320, -1.92800],
#     [20.1550, -51.649368, 270.76422, -0.0900, -86.0980,  2.34400],
#     [12.5970, -29.275979, 274.96757, -7.0000, -46.1960,  4.38100],
#     [35.2890, -64.871254, 281.36214, 2.00000, -150.182,  32.0730],
#     [19.7600, -62.230058, 282.02661, 2.36000, -80.0280,  13.0560],
#     [20.1550, -31.796796, 282.68543, -6.0000, -72.2750,  17.3750],
#     [21.2190, -50.180884, 283.27458, -4.2000, -85.2540,  16.3480],
#     [13.4720, -29.884830, 284.51737, -4.9000, -48.6250,  13.4110],
#     [20.7410, -54.538391, 290.74577, -0.2500, -81.9140,  24.5600],
#     [19.5150, -32.127417, 299.01838, -6.8100, -68.3950,  33.4490],
#     [19.9460, -26.224314, 302.27192, -5.3900, -67.3820,  40.1670],
#     [20.8500, -28.028390, 302.50045, -5.8000, -62.7000,  40.4000],
#     [23.0540, -25.948119, 308.40690, -7.6000, -74.3550,  53.4650],
#     [23.5110, -24.565250, 310.92153, -5.8000, -74.8820,  55.9340],
#     [102.830, -31.342400, 311.29109, -6.3100, -359.895,  281.424],
#     [45.1000, -22.860332, 318.53414, -6.0000, -138.900,  136.900],
#     [31.7120, -66.918836, 320.37081, 5.19000, -85.3240,  105.130],
#     [30.0540, -52.931008, 326.98108, 0.15000, -88.5740,  103.318],
#     [39.1880, -6.5556227, 337.82700, -7.9000, -75.6610,  145.605],
#     [27.2770, -71.706107, 340.70515, 7.02000, -52.4640,  94.7980],
#     [47.9420, -33.250998, 341.24241, 1.10000, -123.103,  179.904],
#     [36.5320, -12.264637, 353.12918, 1.38000, -81.8890,  139.260]
# ]}

# Radial velocity offset applied to all stars (float)
rv_offset.value = 0.0
rv_offset.unit = 'km/s'
