Kinematic Age for Nearby Young Associations
===========================================

The Kanya (Kinematic Age for Nearby Young Associations) Python package performs three-dimensional traceback analysis of members of a nearby young association (NYA) and computes their kinematic age by finding the epoch when the spatial extent of the NYA was minimal.

Kanya uses current-day astrometric and kinematic data (parallax, declination and right ascension, radial velocity, declination proper motion and right ascension proper motion) of members of an NYA as its inputs and performs the conversion into XYZ Galactic positions and UVW space velocities. XYZ Galactic positions and UVW space velocities can also be used as inputs directly as well. Representative simulated samples can also be created in order to evaluate the method's accurary and precision. Tracebacks are performed by integrating backwards Galactic orbits using one of several available Galactic potentials or no potential at all. A Monte Carlo method is used to assess the impact of measurement errors in astrometric and kinematic data, and the impact of sample selection on the traceback age. The bias towards younger ages due to measurement errors, and radial velocity shifts due to the gravitational redshift and convective blueshift are accounted for. For models, actual or simulated measurement errors and radial velocity shifts can be used.

Kinematic outliers are eliminated from the computation of association size metrics, used to evaluate the spatial extent of the NYA along its Galactic orbit. Several size metrics, based on the spatial covariance matrices and spatial-kinematic cross-covariance matrices (between Galactic positions and space velocities), the median absolute deviation (MAD), or the branches and the minimum spanning tree (MST), are computed. Several methods are used to compute covariance matrices: an empirical method, a robust method and a method based on scikit-learn's robust covariance estimator. The XYZ and a curvilinear ξ'η'ζ' coordinate systems can be used to visualize and measure the size of the association. Finally, several functions used to create figures and tables are available.

Future developments
-------------------

In the future, improvements will be made to improve the efficiency, ease of use and functionality of the package.

Dependencies
------------

This tool requires Python 3.12.5 and the following libraries:

- numpy 2.0.1
- scipy 1.14.0
- astropy 6.1.2
- galpy 1.10.0
- scikit-learn 1.5.1
- matplotlib 3.9.1
- pandas 2.2.2
- tqdm 4.66.5
