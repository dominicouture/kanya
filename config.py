# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py: Group configuration file. For every parameter, three components can be defined: 'value'
or 'values', 'unit' or 'units', and 'system':

  - 'value' or 'values': string, integer, float or tuple, list or dictionary, depending on the
    given parameter.
  - 'unit' or 'units': string or a list or tuple of strings convertible to an Astropy.Unit
    object. The default units are:
      - Time: Myr
      - Distance: pc
      - Velocity: pc/Myr
      - Angle: rad
      - Angular speed: rad/Myr
  - 'system': string matching the name of one of the following coordinate system:
      - spherical: π, δ, α, rv, μδ, and μαcosδ
      - cartesian: x, y, z, u, v, and w
      - cylindrical: r, θ, h, vr, vt, and vh
      - curvilinear: ξ', η', ζ', vξ', vη', and vζ'
"""

# Name of the group (str or None)
# name.value = None
# name.value = 'BPMG'
name.value = 'THA'
# name.value = 'COL_CAR'
# name.value = 'COL'
# name.value = 'CAR'
# name.value = 'IC2602'
# name.value = 'PL8'
# name.value = 'beta_pictoris_Miret-Roig'
# name.value = 'beta_pictoris_Crundall'

# Path to the directory used for loading data relative to the current directory (str or None).
# By default, if this is None of absent, data is loaded from the current directory. Directory
# paths end with '/'.
data_dir.value = '../Data/'

# Path to the directory used for saving figures, tables and other output types relative to the
# current directory (str or None). By default, if this is None of absent, outputs are saved to
# the current directory. Directory paths end with '/'.
output_dir.value = '../Output/'

# Path to the directory of path used for logging relative to the current directory (str or None).
# By default, if this is None of absent, logs are saved to a 'Logs' directory in the current
# directory. Directory paths end with '/'.
logs_path.value = '../Logs/'

# Path to the file or directory used for pickle input relative to the current directory (str
# or None). By default, if this is None of absent, pickles are loaded from a file named
# 'name.value'.group in the current directory. Directory paths end with '/'.
load_path.value = 'beta_pictoris.group'

# Path to the file or directory used for pickle output relative to the current directory (str
# or None). By default, if this is None of absent, pickles are saved to a file named
# 'name.value'.group in the current directory. Directory paths end with '/'.
save_path.value = None

# Association size metrics to be computed (boolean)
size_metrics.value = True
cov_metrics.value = True
cov_robust_metrics.value = False
cov_sklearn_metrics.value = False
mad_metrics.value = False
tree_metrics.value = False

# Number of Monte Carlo (integer, ≥ 0)
number_monte_carlo.value = 10
# number_monte_carlo.value = 0

# Number of jackknife Monte Carlo (integer, ≥ 0)
number_jackknife.value = 5
# number_jackknife.value = 15

# Fraction of stars per jackknife Monte Carlo (0 < float ≤ 1)
fraction_jackknife.value = 0.67
# fraction_jackknife.value = 0.50

# Number of steps of the traceback, including the initial step at t = 0 (integer, ≥ 1)
# number_of_steps.value = 1000
number_of_steps.value = 300
# number_of_steps.value = 172
# number_of_steps.value = 10

# Initial age of the traceback (float, inclusive)
initial_time.value = 0.0

# Final age of the traceback (float, inclusive)
final_time.value = -61.0
# final_time.value = -51.0

# Age of simulated groups of stars (float, ≠ initial_time)
age.value = -24.0

# Number of stars in each simulated group of stars (integer, ≥ 1)
number_of_stars.value = 31

# Average position of the simulated sample of stars (3 floats in tuple)
# position.value = (22.686, -4.304, -18.495) # Core sample
# position.values = (18.726, -4.500, -18.938) # Gaia DR3
# position.values = (0.0, 0.0, 0.0)

# Average position error of the simulated sample of stars (3 floats in tuple)
position_error.values = (0.0218, 0.0136, 0.0165) # Gaia DR3
# position_error.values = (0.19846, 0.0, 0.0) # Gaia DR2
# position_error.values = (0.0, 0.0, 0.0)
position_error.units = 'spherical'
position_error.system = 'spherical'

# Average position scatter of the simulated sample of stars (3 floats in tuple)
position_scatter.values = (3.0, 3.0, 3.0) # Initial XYZ scatter
# position_scatter.values = (29.3, 14.0, 9.0) # Current-day XYZ scatter

# Average velocity of the simulated sample of stars (3 floats in tuple)
velocity.values = (-10.12, -15.67, -8.75) # Core sample
# velocity.values = (-10.278, -15.726, -8.759) # Gaia DR3
# velocity.values = (-11.15, -11.15, -11.15) # Core sample
velocity.units = 'km/s'

# Average velocity error of the simulated sample of stars (3 floats in tuple)
velocity_error.values = (0.2634,  0.0196, 0.0231) # Gaia DR3
# velocity_error.values = (1.0112, 0.30754, 0.26432) # Gaia DR2
# velocity_error.values = (0.0, 0.0, 0.0)
velocity_error.units = ('km/s', 'mas/yr', 'mas/yr')
# velocity_error.units = 'spherical'
velocity_error.system = 'spherical'

# Average velocity scatter of the simulated sample of stars (3 floats in tuple)
# velocity_scatter.values = (1.66, 0.52, 0.73) # Core sample
# velocity_scatter.values = (2.21753, 1.36009, 1.57355) # Gaia DR2
# velocity_scatter.values = (0.86, 0.86, 0.86) # Core sample
velocity_scatter.values = (1.0, 1.0, 1.0) # Core sample (modified)
# velocity_scatter.values = (1.6805, 1.6805, 1.6805) # Gaia DR2
velocity_scatter.units = 'km/s'

# Path to CSV data file (str) or Python dictionary with the data (dictionary)
# data.value = 'BPMG_raw_GaiaDR3.csv'
data.value = 'THA_sample.csv'
# data.value = 'COL_CAR_sample.csv'
data.units = 'spherical'
data.system = 'spherical'
# data.values =  {
#     'beta_pictoris': [
#         [    'π',        'δ',       'α',    'rv',     'μδ',     'μα'],
#         [  'mas',      'deg',     'deg',  'km/s', 'mas/yr', 'mas/yr'],
#         [25.0250, -23.107737, 1.7091324, 6.50000, -47.1210,  96.7760],
#         [27.1700, -66.753496, 4.3491739, 8.75000, -16.8740,  103.039],
#         [28.6590, -32.551949, 6.9598490, 8.80000, -47.3890,  109.817],
#         [45.7900,  15.439305, 17.855935, 4.00000, -120.000,  180.000],
#         [25.1340,  28.744731, 34.355775, 4.32000, -74.0690,  86.9240],
#         [24.3600,  30.973199, 36.872279, 4.85000, -72.0020,  79.6790],
#         [19.5280,  23.709892, 53.758697, 15.5000, -62.9400,  50.3370],
#         [33.5770, -2.4735640, 69.400543, 21.0000, -63.8330,  44.3520],
#         [47.4120, 0.03477300, 70.906710, 16.9700, -107.485,  55.1120],
#         [40.9810,  1.7831241, 74.895306, 18.0800, -95.0510,  39.2340],
#         [37.1740, -57.256761, 75.196646, 18.0700,  74.1380,  35.1980],
#         [18.7230,  15.449949, 75.205429, 18.1000, -58.8350,  18.1310],
#         [50.3080, -21.585933, 76.708196, 20.7100, -15.5740,  47.1440],
#         [37.2060, -11.901168, 81.769919, 20.6100, -49.3150,  17.0590],
#         [50.6230, -51.066517, 86.821179, 20.0000,  82.5770,  2.49300],
#         [25.4700, -72.044537, 94.617428, 10.9500,  74.2950, -7.90800],
#         [30.5570, -27.701553, 93.305428, 22.5000, -5.62300, -13.1570],
#         [51.0040, -53.907368, 154.36205, 14.8000, -4.93900, -173.099],
#         [62.9440, -64.976145, 220.62479, 6.20000, -232.614, -190.466],
#         [24.9340, -57.707982, 234.73939, 3.90000, -95.8810, -55.1970],
#         [19.7560, -53.725795, 254.33436, 0.49000, -84.0870, -10.8490],
#         [32.7800, -66.951608, 259.35604, 3.30000, -137.125, -21.2130],
#         [14.7590, -54.263778, 262.47945, -0.2000, -63.5460, -5.41300],
#         [15.2130, -50.724731, 265.45431, 2.40000, -65.8580, -2.02300],
#         [12.9670, -53.103534, 267.14060, -0.2000, -56.1320, -1.92800],
#         [20.1550, -51.649368, 270.76422, -0.0900, -86.0980,  2.34400],
#         [12.5970, -29.275979, 274.96757, -7.0000, -46.1960,  4.38100],
#         [35.2890, -64.871254, 281.36214, 2.00000, -150.182,  32.0730],
#         [19.7600, -62.230058, 282.02661, 2.36000, -80.0280,  13.0560],
#         [20.1550, -31.796796, 282.68543, -6.0000, -72.2750,  17.3750],
#         [21.2190, -50.180884, 283.27458, -4.2000, -85.2540,  16.3480],
#         [13.4720, -29.884830, 284.51737, -4.9000, -48.6250,  13.4110],
#         [20.7410, -54.538391, 290.74577, -0.2500, -81.9140,  24.5600],
#         [19.5150, -32.127417, 299.01838, -6.8100, -68.3950,  33.4490],
#         [19.9460, -26.224314, 302.27192, -5.3900, -67.3820,  40.1670],
#         [20.8500, -28.028390, 302.50045, -5.8000, -62.7000,  40.4000],
#         [23.0540, -25.948119, 308.40690, -7.6000, -74.3550,  53.4650],
#         [23.5110, -24.565250, 310.92153, -5.8000, -74.8820,  55.9340],
#         [102.830, -31.342400, 311.29109, -6.3100, -359.895,  281.424],
#         [45.1000, -22.860332, 318.53414, -6.0000, -138.900,  136.900],
#         [31.7120, -66.918836, 320.37081, 5.19000, -85.3240,  105.130],
#         [30.0540, -52.931008, 326.98108, 0.15000, -88.5740,  103.318],
#         [39.1880, -6.5556227, 337.82700, -7.9000, -75.6610,  145.605],
#         [27.2770, -71.706107, 340.70515, 7.02000, -52.4640,  94.7980],
#         [47.9420, -33.250998, 341.24241, 1.10000, -123.103,  179.904],
#         [36.5320, -12.264637, 353.12918, 1.38000, -81.8890,  139.260]
#     ]
# }

# Data from Crundall et al. 2019
# data.values = '../Data/Crundall_BPMG_core.csv'
# data.units = ['', '', '', 'pc', 'pc', 'pc', 'km/s', 'km/s', 'km/s']
# data.system = 'cartesian'

# Data from Miret-Roig et al 2020
# data.values = '../Data/β Pictoris Moving Group - Miret-Roig et al. 2020.csv'
# data.system = 'cartesian'
# data.units = ['pc',   'pc',   'pc', 'km/s', 'km/s', 'km/s', '', '']
# data.values = {
#     'beta_pictoris_Miret-Roig': [
#         [   'X',    'Y',    'Z',    'U',    'V',    'W', 'mass', 'radius'],
#         [  'pc',   'pc',   'pc', 'km/s', 'km/s', 'km/s',     '',       ''],
#         [ 14.60, -18.59, -28.21, -10.51, -16.09,  -8.60,   0.29,     0.51],
#         [-21.28,  -6.77,  -9.83, -12.42,  -16.3,  -9.28,   0.58,     0.72],
#         [ -1.54, -21.32, -16.33, -11.17,  -16.5,  -9.09,   0.58,     0.72],
#         [ -3.43, -16.65, -10.06, -11.07, -15.79,  -9.21,   1.69,     0.52],
#         [  7.59, -33.75, -18.58,  -10.5, -16.46,  -8.71,   0.83,     0.91],
#         [ 45.37, -21.65,  -5.88,  -7.35, -15.87, -10.49,   0.29,     0.51],
#         [ 32.00, -23.67, -10.97,  -8.20, -16.62,  -9.09,   0.20,     0.42],
#         [ 30.33, -10.06,  -1.24,  -8.67, -16.51, -10.21,   0.36,     0.57],
#         [ 48.62, -18.83, -11.34,  -7.17, -16.93, -10.12,   0.29,     0.51],
#         [ 70.36, -26.64, -16.96,  -7.44, -16.81,  -9.20,   0.36,     0.57],
#         [ 54.93,   0.82,  -4.03,  -7.83, -14.52,  -8.45,   0.36,     0.57],
#         [ 49.55, -21.31, -16.18,  -8.58, -15.54,  -8.09,   0.36,     0.57],
#         [ 34.91, -13.14, -10.76, -10.23, -15.13,  -8.05,   0.20,     0.42],
#         [ 26.21, -11.84,  -9.68,  -7.78, -17.09, -10.26,   0.29,     0.51],
#         [ 77.46, -13.26, -20.69,  -6.84, -16.37,  -8.99,   0.36,     0.57],
#         [ 79.32, -14.36, -21.66,  -6.45, -16.22,  -8.95,   0.73,     0.83],
#         [ 45.01, -16.58, -18.35,  -8.52,  -15.4,  -8.28,   0.29,     0.51],
#         [ 41.30, -12.80, -21.32,  -8.89, -15.43,  -8.06,   1.20,     1.17],
#         [ 64.02,  -9.10,  -29.5,  -6.73, -16.37,  -9.76,   0.58,     0.72],
#         [ 48.13,   3.22, -18.92,  -9.47, -16.14,  -8.88,   0.20,     0.42],
#         [ 57.78,  13.52, -26.17,  -7.69, -14.99,  -9.20,   0.36,     0.57],
#         [ 52.34,   7.26, -28.47,  -7.68, -15.78,  -9.13,   0.47,     0.65],
#         [ 42.62,  12.23, -23.41,  -6.96, -14.69, -10.27,   1.26,     1.22],
#         [ 34.55,  11.39, -23.62,  -8.44, -14.85,  -9.59,   0.20,     0.42],
#         [ 21.87,  12.25, -20.26,  -9.91, -15.17,  -9.78,   0.36,     0.57],
#         [ 19.57, -18.94, -24.54, -10.25, -15.84,  -7.98,   0.73,     0.83]
#     ]
# }

# Whether to use actual or simulated measurement errors (boolean)
data_errors.value = True

# Whether to use actual or simulated rv shifts (boolean)
data_rv_shifts.value = False

# Radial velocity shift added (from model) or substracted (from data) to all stars (float)
# rv_shift.value = 0.5
# rv_shift.value = 0.2
rv_shift.value = 0.0
rv_shift.unit = 'km/s'

# Cutoff (σ of the star position) below which stars are excluded (float or None, > 0)
# cutoff.value = None
cutoff.value = 3.0

# Whether to use the full (with outliers), standard sample or the subsample (str or None)
sample.value = None
# sample.value = 'rejected'
# sample.value = 'input'
# sample.value = 'extended'
# sample.value = 'core'

# Galactic potential used for orbit integration (str or None)
# potential.value = None
# potential.value = 'MWPotential2014'
potential.value = 'Irrgang13I'

# Principal component analysis
pca.value = False

# Whether to show the timer
timer.value = False
