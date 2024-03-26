Kinematic Age for Nearby Young Associations
===========================================

The Kanya (Kinematic Age for Nearby Young Associations) Python package performs three-dimensional traceback analysis of members of a nearby young association (NYA) and computes their kinematic age by finding the epoch when the spatial extent of the NYA was minimal.

Kanya uses current-day astrometric and kinematic data (parallax, declination and right ascension, radial velocity, declination proper motion and right ascension proper motion) of members of an NYA as its inputs and performs the conversion into XYZ Galactic positions and UVW space velocities. XYZ Galactic positions and UVW space velocities can also be used as inputs directly as well. Representative simulated samples can also be created in order to evaluate the method's accurary and precision. Tracebacks are performed by computing backwards Galactic orbits using one of several available Galactic potentials or no potential at all. A Monte Carlo method is used to assess the impact of measurement errors in astrometric and kinematic data, and the impact of sample selection on the traceback age. The bias towards younger ages due to measurement errors, and radial velocity shifts due to the gravitational redshift and convective blueshift are accounted for. For models, actual or simulated measurement errors and radial velocity shifts can be used.

Kinematic outliers are eliminated from the computation of association size metrics, used to evaluate the spatial extent of the NYA along its Galactic orbit. Several metrics, based on the spatial covariance matrices and spatial-kinematic cross-covariance matrices (between Galactic positions and space velocities), the median absolute deviation, and full and minimum spanning trees, are computed. Several methods are used to compute covariance matrices: an empirical method, a robust method and a method based on scikit-learn's robust covariance estimator. A custom curvilinear ξ'η'ζ' coordinate system is used to better visualize and measure the size of the association. Finally, several functions used to create figures and tables are available.

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
- pandas 1.3.5
