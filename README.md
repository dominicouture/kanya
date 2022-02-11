Nearby Young Association Traceback
==================================

This Python package allows one to perform three-dimensional (3D) tracebacks of stars in nearby young associations (NYAs) and find a kinematic age. This is accomplished by finding the age that minimizes one of sevaral association size metrics.

The algorithm uses current day positions (parallax, declination and right ascension) and velocities (radial velocity, declination proper motion and right ascension proper motion) observables in an equatorial plane as its inputs and performs the conversion into XYZ positions and UVW velocities in a galactic plane. XYZ positions and UVW velocities can be used as inputs directly as well. Tracebacks are performed using galactic orbits computation. Monte Carlo method is implemented to assess the impact of measurement errors in the data and the impact of sample selection. Outliers are also eliminated. The shift towards younger ages due to measurement errors, and radial velocity offset due to the gravitational redshift and convective blueshift are account for. Finally, several functions used to create output figures and tables are defined.

Future developments
-------------------

In the future, improvements will be made to improve the efficiency, ease of use and functionality of the package.

Dependencies
------------

This software uses the following non-standard Python (version 3.7.4) packages:

- numpy 1.21.5
- scipy 1.3.1
- matplotlib 3.1.1
- astropy 4.2.1
- galpy 1.7.1
- sklearn 0.23.2
