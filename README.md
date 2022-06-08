Nearby Young Association Traceback
==================================

This Python package allows one to perform three-dimensional (3D) tracebacks of stars in nearby young associations (NYAs) and find a kinematic age. This is accomplished by finding the epoch when one of several association size metrics is minimal.

This tool uses current day positions (parallax, declination and right ascension) and velocities (radial velocity, declination proper motion and right ascension proper motion), in an equatorial plane, of members of an NYA as its inputs and performs the conversion into XYZ Galactic positions and UVW space velocities, in a Galactic plane. XYZ positions and UVW velocities can also be used as inputs directly as well. Furthermore, this tool is capable of creating simulated samples in order to test the method. Tracebacks are performed by computing backwards Galactic orbits. A Monte-Carlo method is implemented to assess the impact of measurement errors in the data and the impact of sample selection. The bias towards younger ages due to measurement errors, and radial velocity biases due to the gravitational redshift and convective blueshift are accounted for.

Kinematic outliers are eliminated from the computation of association size metrics, used to evaluate the spatial extent of the NYA over time along its Galactic orbit. Several metrics, based on the covariance matrices, cross covariance matrices (between positions and velocities), median absolute deviation and minimum spanning trees, are computed. A custom curvilinear ξ'η'ζ' coordinate system is used to better visualize and measure the size of the association. Finally, several functions used to create output figures and tables are defined.

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
