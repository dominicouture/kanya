Kinematic Group Traceback
=========================

This software allows one to perform three-dimensional (3D) tracebacks of stars in kinematic groups and find a kinematic age for the association. This is accomplished by finding the age that minimizes:

- 3D scatter
- Median absolute deviation (MAD)
- Minimum spanning tree (MST) total length

The algorithm uses current day positions (distance, declination and right ascension) and velocities (radial velocity, declination proper motion and right ascension proper motion) as its inputs and performs the conversion into an XYZ and UVW space. Furthermore, it will also eliminate the relatively imprecise radial velocity measurements and replace them with optimal values.

Dependencies
------------

This software uses the following non-standard Python (version 3.6.3) modules:

- numpy
- scipy
- matplotlib
- astropy
- peewee
