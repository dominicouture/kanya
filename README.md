Kinematic Group Traceback
=========================

This software allows one to perform three-dimensional (3D) tracebacks of stars in local kinematic groups and find a kinematic age for the stars in the association. This is accomplished by finding the age that minimizes:

- 3D scatter
- Median absolute deviation (MAD)
- Minimum spanning tree (MST) mean branch length
- Mininum spanning tree median absolute deviation branch length
- X-U, Y-V and Z-W covariances

The algorithm uses current day positions (distance, declination and right ascension) and velocities (radial velocity, declination proper motion and right ascension proper motion) observables in an equatorial plane as its inputs and performs the conversion into XYZ positions and UVW velocities in a galactic plane. The shift toward young ages due to measurement errors and errors in radial velocity due to the gravitational redshift are compensated.

Future developments
-------------------

In the near future, this software will also eliminate the relatively imprecise radial velocity measurements and replace them with optimal values. Also, tracebacks will be performed using galactic orbits computation.

Dependencies
------------

This software uses the following non-standard Python (version 3.6.3) modules:

- numpy
- scipy
- matplotlib
- astropy
- galpy (in the future)

A LaTeX installation is also required to create figures.
