Nearby Young Association Traceback
==================================

This Python package performs three-dimensional (3D) tracebacks of members of a nearby young association (NYAs) in order to find its kinematic age. This is accomplished by finding the epoch when several association size metrics are minimal.

This tool uses current day positions (parallax, declination and right ascension) and velocities (radial velocity, declination proper motion and right ascension proper motion) of members of an NYA as its inputs and performs the conversion into XYZ Galactic positions and UVW space velocities. XYZ positions and UVW velocities can also be used as inputs directly as well. Representative simulated samples can also be created in order to evaluate the method's bias and precision. Tracebacks are performed by computing backwards Galactic orbits. A Monte Carlo method is used to assess the impact of measurement errors in astrometric and kinematic data and the impact of sample selection. The bias towards younger ages due to measurement errors, and radial velocity shifts due to the gravitational redshift and convective blueshift are accounted for.

Kinematic outliers are eliminated from the computation of association size metrics, used to evaluate the spatial extent of the NYA along its Galactic orbit. Several metrics, based on the covariance matrices, cross covariance matrices (between positions and velocities), median absolute deviation and minimum spanning trees, are computed. A custom curvilinear ξ'η'ζ' coordinate system is used to better visualize and measure the size of the association. Finally, several functions used to create figures and tables are available.

Future developments
-------------------

In the future, improvements will be made to improve the efficiency, ease of use and functionality of the package.

Dependencies
------------

This tool uses the following non-standard Python (version 3.7.4) packages:

- numpy 1.21.5
- scipy 1.3.1
- matplotlib 3.1.1
- astropy 4.2.1
- galpy 1.7.1
- sklearn 0.23.2
