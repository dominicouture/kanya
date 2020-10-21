# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" output.py: Defines functions to create data output such as plots of size indicators over time,
    2D and 3D scatters at a given time, histograms, color mesh, etc.
"""

import numpy as np
from os import path
from matplotlib import pyplot as plt, lines, ticker as tkr
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import linregress
from Traceback.collection import *

__author__ = 'Dominic Couture'
__email__ = 'dominic.couture.1@umontreal.ca'

# Formats ticks label with commas instead of dots for French language publications
format_ticks = tkr.FuncFormatter(lambda x, pos: str(round(float(x), 1)).replace('.', ','))

# Sets rc parameters
plt.rc('font', serif='Latin Modern Math', family='serif', size='12')
plt.rc('mathtext', fontset='custom', it='Latin Modern Roman:italic', rm='Latin Modern Math:roman')
plt.rc('lines', markersize=4)
plt.rcParams['pdf.fonttype'] = 42

def save_figure(name, file_path=None, forced=False, default=False, cancel=False):
    """ Checks whether a path already exists and asks for user input if it does. The base path
        is assumed to be the output directory. Also, if the path does not have an extension, a
        '.pdf' extension is added.
    """

    # Padding
    padding = 0.03

    # file_path parameter
    file_path = file_path if file_path is not None else output(create=True) + '/'

    # Check if file_path parameter is a string, which must be done before the directory call
    stop(type(file_path) != str, 'TypeError', "'file_path' must be a string ({} given).",
        type(file_path))

    # file_path redefined as the absolute path, default name and directory creation
    file_path = path.join(directory(output(), path.dirname(file_path), 'file_path', create=True),
        path.basename(file_path) if path.basename(file_path) != '' else '{}.pdf'.format(name))

    # Check if there's an extension and add a '.pdf' extension, if needed
    if path.splitext(file_path)[1] == '':
        file_path += '.pdf'

    # Check if a file already exists
    if path.exists(file_path):
        choice = None
        stop(type(forced) != bool, 'TypeError',
            "'forced' must be a boolean ({} given).", type(forced))
        if not forced:
            stop(type(default) != bool, 'TypeError',
                "'default' must be a default ({} given).", type(default))
            if not default:
                stop(type(cancel) != bool, 'TypeError',
                    "'cancel' must be a boolean ({} given).", type(cancel))
                if not cancel:

                    # User input
                    while choice == None:
                        choice = input(
                            "A file already exists at '{}'. Do you wish to overwrite (Y), "
                            "keep both (K) or cancel (N)? ".format(file_path)).lower()

                        # Loop over question if input could not be interpreted
                        if choice not in ('y', 'yes', 'k', 'keep', 'n', 'no'):
                            print("Could not understand '{}'.".format(choice))
                            choice = None

                # Saving cancellation
                if cancel or choice in ('n', 'no'):

                    # Logging
                    log("'{}': figure not saved because a file already exists at '{}'.",
                        name, file_path)

            # Default name and saving
            if default or choice in ('k', 'keep'):
                from Traceback.tools import default_name
                file_path = default_name(file_path)
                plt.savefig(file_path, bbox_inches='tight', pad_inches=padding)

                # Logging
                log("'{}': file name changed and figure saved at '{}'.", name, file_path)

        # Existing file deletion and saving
        if forced or choice in ('y', 'yes'):
            from os import remove
            remove(file_path)
            plt.savefig(file_path, bbox_inches='tight', pad_inches=padding)

            # Logging
            log("'{}': existing file located at '{}' deleted and replaced.", name, file_path)

    # Saving
    else:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=padding)

        # Logging
        log("'{}': figure saved at '{}'.", name, file_path)

class Output_Series():
    """ Defines output methods from a Series object. """

    def create_scatter_mad_mst_plot(
            self, secondary=False, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of scatter, median absolute deviation, and minimum spanning tree
            branches length mean and median absolute deviation over the entire duration of the
            data.
        """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # XYZ scatter
        ax.plot(self.time, self.scatter_xyz_total.values / self.scatter_xyz_total.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='σ : ({} ± {}) Ma'.format(
                self.scatter_xyz_total.age, self.scatter_xyz_total.age_error).replace('.', ','))
        ax.fill_between(self.time,
            (self.scatter_xyz_total.values - self.scatter_xyz_total.values_error) / self.scatter_xyz_total.values[0],
            (self.scatter_xyz_total.values + self.scatter_xyz_total.values_error) / self.scatter_xyz_total.values[0], color='0.0', alpha=0.3, linewidth=0.)

        # XYZ median absolute deviation
        ax.plot(self.time, self.mad_xyz_total.values / self.mad_xyz_total.values[0],
            linestyle='-.', color='0.2', linewidth=1.5, label='MAD : ({} ± {}) Ma'.format(
                self.mad_xyz_total.age, self.mad_xyz_total.age_error).replace('.', ','))
        ax.fill_between(self.time,
            (self.mad_xyz_total.values - self.mad_xyz_total.values_error) / self.mad_xyz_total.values[0],
            (self.mad_xyz_total.values + self.mad_xyz_total.values_error) / self.mad_xyz_total.values[0], color='0.2', alpha=0.3, linewidth=0.)

        # Minimum spanning tree mean branch length
        ax.plot(self.time, self.mst_mean.values / self.mst_mean.values[0],
            linestyle='--', color='0.4', linewidth=1.5, label='MST : ({} ± {}) Ma'.format(
                self.mst_mean.age, self.mst_mean.age_error).replace('.', ','))
        ax.fill_between(self.time,
            (self.mst_mean.values - self.mst_mean.values_error) / self.mst_mean.values[0],
            (self.mst_mean.values + self.mst_mean.values_error) / self.mst_mean.values[0], color='0.4', alpha=0.3, linewidth=0.)

        # Minimum spanning tree median absolute deviation branch length
        ax.plot(self.time, self.mst_mad.values / self.mst_mad.values[0],
            linestyle=':', color='0.6', linewidth=1.5, label='MAD MST : ({} ± {}) Ma'.format(
                self.mst_mad.age, self.mst_mad.age_error).replace('.', ','))
        ax.fill_between(self.time,
            (self.mst_mad.values - self.mst_mad.values_error) / self.mst_mad.values[0],
            (self.mst_mad.values + self.mst_mad.values_error) / self.mst_mad.values[0], color='0.6', alpha=0.3, linewidth=0.)

        # Secondary lines
        self.stop(type(secondary) != bool, 'TypeError',
            "'secondary' must be a boolean ({} given).", type(secondary))
        if secondary:
            i = 0
            plot_i = np.arange(0, len(self), 20)
            for group in self:
                if i in plot_i:
                    ax.plot(self.time, group.scatter_xyz_total.values, '-', color='0.7', linewidth=0.5)
                i += 1

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("Size indicators of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("Average size indicators of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=4, fontsize=9)
        ax.yaxis.set_major_formatter(format_ticks)
        ax.set_xlabel('Temps (Ma)')
        ax.set_ylabel("Indicateurs d'âge")
        ax.set_xlim(0, self.final_time.value)
        ax.set_ylim(0.0, 1.6)

        # Save figure
        save_figure(self.name, 'Size_indicators_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_scatter_mad_xyz_plot(
            self, secondary=False, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of the XYZ scatter and median absolute deviation over the entire
            duration of the data.
        """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # Plot XYZ scatter
        ax.plot(self.time, self.scatter_xyz_total.values / self.scatter_xyz_total.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='σ : ({} ± {}) Ma'.format(
                self.scatter_xyz_total.age, self.scatter_xyz_total.age_error).replace('.', ','))

        # Plot X scatter
        ax.plot(self.time, self.scatter_xyz.values[:,0].T / self.scatter_xyz.values[0,0],
            linestyle='-.', color='0.2', linewidth=1.5, label='σ_x : ({} ± {}) Ma'.format(
                self.scatter_xyz.age[0], self.scatter_xyz.age_error[0]).replace('.', ','))

        # Plot Y scatter
        ax.plot(self.time, self.scatter_xyz.values[:,1].T / self.scatter_xyz.values[0,1],
            linestyle='--', color='0.4', linewidth=1.5, label='σ_y : ({} ± {}) Ma'.format(
                self.scatter_xyz.age[1], self.scatter_xyz.age_error[1]).replace('.', ','))

        # Plot Z scatter
        ax.plot(self.time, self.scatter_xyz.values[:,2] / self.scatter_xyz.values[0,2],
            linestyle=':', color='0.6', linewidth=1.5, label='σ_z : ({} ± {}) Ma'.format(
                self.scatter_xyz.age[2], self.scatter_xyz.age_error[2]).replace('.', ','))

        # Plot XYZ MAD
        ax.plot(self.time, self.mad_xyz_total.values / self.mad_xyz_total.values[0],
            linestyle='-', color='b', linewidth=1.5, label='MAD : ({} ± {}) Ma'.format(
                self.mad_xyz_total.age, self.mad_xyz_total.age_error).replace('.', ','))

        # Plot X MAD
        ax.plot(self.time, self.mad_xyz.values[:,0].T / self.mad_xyz.values[0,0],
            linestyle='-.', color='#2635ff', linewidth=1.5, label='MAD_X : ({} ± {}) Ma'.format(
                self.mad_xyz.age[0], self.mad_xyz.age_error[0]).replace('.', ','))

        # Plot Y MAD
        ax.plot(self.time, self.mad_xyz.values[:,1].T / self.mad_xyz.values[0,1],
            linestyle='--', color='#4d58ff', linewidth=1.5, label='MAD_Y : ({} ± {}) Ma'.format(
                self.mad_xyz.age[1], self.mad_xyz.age_error[1]).replace('.', ','))

        # Plot Z MAD
        ax.plot(self.time, self.mad_xyz.values[:,2] / self.mad_xyz.values[0,2],
            linestyle=':', color='#737cff', linewidth=1.5, label='MAD_Z : ({} ± {}) Ma'.format(
                self.mad_xyz.age[2], self.mad_xyz.age_error[2]).replace('.', ','))

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("XYZ scatter of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("Average XYZ scatter of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=4, fontsize=6)
        ax.yaxis.set_major_formatter(format_ticks)
        ax.set_xlabel('Temps (Ma)')
        ax.set_ylabel("Indicateur d'âge")
        ax.set_xlim(0, self.final_time.value)
        ax.set_ylim(0.0, 1.6)

        # Save figure
        save_figure(self.name, 'XYZ_scatter__MAD_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_scatter_mad_ξηζ_plot(
            self, secondary=False, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of the ξηζ scatter over the entire duration of the data. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # Plot ξηζ scatter
        ax.plot(self.time, self.scatter_ξηζ_total.values / self.scatter_ξηζ_total.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='σ : ({} ± {}) Ma'.format(
                self.scatter_ξηζ_total.age, self.scatter_ξηζ_total.age_error).replace('.', ','))
        ax.fill_between(self.time,
            (self.scatter_ξηζ_total.values - self.scatter_ξηζ_total.values_error) / self.scatter_ξηζ_total.values[0],
            (self.scatter_ξηζ_total.values + self.scatter_ξηζ_total.values_error) / self.scatter_ξηζ_total.values[0], color='0.0', alpha=0.3, linewidth=0.)

        # Plot r scatter
        ax.plot(self.time, self.scatter_ξηζ.values[:,0].T / self.scatter_ξηζ.values[0,0],
            linestyle='-.', color='0.2', linewidth=1.5, label='σ_ξ : ({} ± {}) Ma'.format(
                self.scatter_ξηζ.age[0], self.scatter_ξηζ.age_error[0]).replace('.', ','))
        ax.fill_between(self.time,
            (self.scatter_ξηζ.values[:,0].T - self.scatter_ξηζ.values_error[:,0].T) / self.scatter_ξηζ.values[0,0],
            (self.scatter_ξηζ.values[:,0].T + self.scatter_ξηζ.values_error[:,0].T) / self.scatter_ξηζ.values[0,0], color='0.2', alpha=0.3, linewidth=0.)

        # Plot θ scatter
        ax.plot(self.time, self.scatter_ξηζ.values[:,1].T / self.scatter_ξηζ.values[0,1],
            linestyle='--', color='0.4', linewidth=1.5, label='σ_η : ({} ± {}) Ma'.format(
                self.scatter_ξηζ.age[1], self.scatter_ξηζ.age_error[1]).replace('.', ','))
        ax.fill_between(self.time,
            (self.scatter_ξηζ.values[:,1].T - self.scatter_ξηζ.values_error[:,1].T) / self.scatter_ξηζ.values[0,1],
            (self.scatter_ξηζ.values[:,1].T + self.scatter_ξηζ.values_error[:,1].T) / self.scatter_ξηζ.values[0,1], color='0.4', alpha=0.3, linewidth=0.)

        # Plot z scatter
        ax.plot(self.time, self.scatter_ξηζ.values[:,2] / self.scatter_ξηζ.values[0,2],
            linestyle=':', color='0.6', linewidth=1.5, label='σ_ζ : ({} ± {}) Ma'.format(
                self.scatter_ξηζ.age[2], self.scatter_ξηζ.age_error[2]).replace('.', ','))
        ax.fill_between(self.time,
            (self.scatter_ξηζ.values[:,2].T - self.scatter_ξηζ.values_error[:,2].T) / self.scatter_ξηζ.values[0,2],
            (self.scatter_ξηζ.values[:,2].T + self.scatter_ξηζ.values_error[:,2].T) / self.scatter_ξηζ.values[0,2], color='0.6', alpha=0.3, linewidth=0.)

        # Plot ξηζ MAD
        ax.plot(self.time, self.mad_ξηζ_total.values / self.mad_ξηζ_total.values[0],
            linestyle='-', color='b', linewidth=1.5, label='MAD : ({} ± {}) Ma'.format(
                self.mad_ξηζ_total.age, self.mad_ξηζ_total.age_error).replace('.', ','))
        ax.fill_between(self.time,
            (self.mad_ξηζ_total.values - self.mad_ξηζ_total.values_error) / self.mad_ξηζ_total.values[0],
            (self.mad_ξηζ_total.values + self.mad_ξηζ_total.values_error) / self.mad_ξηζ_total.values[0], color='b', alpha=0.3, linewidth=0.)

        # Plot r MAD
        ax.plot(self.time, self.mad_ξηζ.values[:,0].T / self.mad_ξηζ.values[0,0],
            linestyle='-.', color='#2635ff', linewidth=1.5, label='MAD_ξ : ({} ± {}) Ma'.format(
                self.mad_ξηζ.age[0], self.mad_ξηζ.age_error[0]).replace('.', ','))
        ax.fill_between(self.time,
            (self.mad_ξηζ.values[:,0].T - self.mad_ξηζ.values_error[:,0].T) / self.mad_ξηζ.values[0,0],
            (self.mad_ξηζ.values[:,0].T + self.mad_ξηζ.values_error[:,0].T) / self.mad_ξηζ.values[0,0], color='#2635ff', alpha=0.3, linewidth=0.)

        # Plot θ MAD
        ax.plot(self.time, self.mad_ξηζ.values[:,1].T / self.mad_ξηζ.values[0,1],
            linestyle='--', color='#4d58ff', linewidth=1.5, label='MAD_η : ({} ± {}) Ma'.format(
                self.mad_ξηζ.age[1], self.mad_ξηζ.age_error[1]).replace('.', ','))
        ax.fill_between(self.time,
            (self.mad_ξηζ.values[:,1].T - self.mad_ξηζ.values_error[:,1].T) / self.mad_ξηζ.values[0,1],
            (self.mad_ξηζ.values[:,1].T + self.mad_ξηζ.values_error[:,1].T) / self.mad_ξηζ.values[0,1], color='#4d58ff', alpha=0.3, linewidth=0.)

        # Plot z MAD
        ax.plot(self.time, self.mad_ξηζ.values[:,2] / self.mad_ξηζ.values[0,2],
            linestyle=':', color='#737cff', linewidth=1.5, label='MAD_ζ : ({} ± {}) Ma'.format(
                self.mad_ξηζ.age[2], self.mad_ξηζ.age_error[2]).replace('.', ','))
        ax.fill_between(self.time,
            (self.mad_ξηζ.values[:,2].T - self.mad_ξηζ.values_error[:,2].T) / self.mad_ξηζ.values[0,2],
            (self.mad_ξηζ.values[:,2].T + self.mad_ξηζ.values_error[:,2].T) / self.mad_ξηζ.values[0,2], color='#737cff', alpha=0.3, linewidth=0.)

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("ξηζ scatter of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("Average ξηζ scatter of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=4, fontsize=6)
        ax.yaxis.set_major_formatter(format_ticks)
        ax.set_xlabel('Temps (Ma)')
        ax.set_ylabel("Indicateur d'âge")
        ax.set_xlim(0, self.final_time.value)
        ax.set_ylim(0.0, 1.6)

        # Save figure
        save_figure(self.name, 'ξηζ_scatter_MAD_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_xyz_plot(self, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of X-X, Y-Y and Z-Z covariances, and determinant. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # XYZ covariance matrix determinant
        ax.plot(self.time, self.covariances_xyz_matrix_det.values / self.covariances_xyz_matrix_det.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='Det : ({} ± {}) Ma'.format(
                self.covariances_xyz_matrix_det.age,
                self.covariances_xyz_matrix_det.age_error).replace('.', ','))

        # Covariance normalization
        covariances_xyz = self.covariances_xyz.values / self.covariances_xyz.values[0]

        # X-X covariance
        ax.plot(self.time, covariances_xyz[:,0], linestyle='-.', color='0.2', linewidth=1.5,
            label='X-X : ({} ± {}) Ma'.format(
                self.covariances_xyz.age[0], self.covariances_xyz.age_error[0]).replace('.', ','))

        # Y-Y covariance
        ax.plot(self.time, covariances_xyz[:,1], linestyle='--', color='0.4', linewidth=1.5,
            label='Y-Y : ({} ± {}) Ma'.format(
                self.covariances_xyz.age[1], self.covariances_xyz.age_error[1]).replace('.', ','))

        # Z-Z covariance
        ax.plot(self.time, covariances_xyz[:,2], linestyle=':', color='0.6', linewidth=1.5,
            label='Z-Z : ({} ± {}) Ma'.format(
                self.covariances_xyz.age[2], self.covariances_xyz.age_error[2]).replace('.', ','))

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("X-X, Y-Y and Z-Z covariances of β Pictoris (without outliners) "
                    "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                    "errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("X-X, Y-Y and Z-Z covariances of {} moving group simulations with "
                    "kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=1, fontsize=9)
        ax.yaxis.set_major_formatter(format_ticks)
        ax.set_xlabel('Temps (Ma)')
        ax.set_ylabel('Covariances')
        ax.set_xlim(0, self.final_time.value)
        ax.set_ylim(0, 1.6)

        # Save figure
        save_figure(self.name, 'Covariances_XYZ_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_covariances_ξηζ_plot(self, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of r-r, θ-θ and z-z covariances, and determinant. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # ξηζ covariance matrix determinant
        ax.plot(-self.time, self.covariances_ξηζ_matrix_det.values,
        # ax.plot(self.time, self.covariances_ξηζ_matrix_det.values / self.covariances_ξηζ_matrix_det.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='Det : ({} ± {}) Ma'.format(
                self.covariances_ξηζ_matrix_det.age,
                self.covariances_ξηζ_matrix_det.age_error).replace('.', ','))
        # ax.fill_between(self.time,
        #     (self.covariances_ξηζ_matrix_det.values - self.covariances_ξηζ_matrix_det.values_error) / self.covariances_ξηζ_matrix_det.values[0],
        #     (self.covariances_ξηζ_matrix_det.values + self.covariances_ξηζ_matrix_det.values_error) / self.covariances_ξηζ_matrix_det.values[0], color='0.0', alpha=0.3, linewidth=0.)

        # Covariance matrix trace
        ax.plot(-self.time, self.covariances_ξηζ_matrix_trace.values,
        # ax.plot(self.time, self.covariances_ξηζ_matrix_trace.values / self.covariances_ξηζ_matrix_trace.values[0],
            linestyle=':', color='g', linewidth=1.5, label='Trace : ({} ± {}) Ma'.format(
                self.covariances_ξηζ_matrix_trace.age,
                self.covariances_ξηζ_matrix_trace.age_error).replace('.', ','))

        # Covariance normalization
        covariances_ξηζ = self.covariances_ξηζ.values
        # covariances_ξηζ = self.covariances_ξηζ.values / self.covariances_ξηζ.values[0]

        # ξ-ξ covariance
        ax.plot(-self.time, covariances_ξηζ[:,0], linestyle='-.', color='0.2', linewidth=1.5,
            label='ξ-ξ : ({} ± {}) Ma'.format(
                self.covariances_ξηζ.age[0], self.covariances_ξηζ.age_error[0]).replace('.', ','))
        ax.fill_between(-self.time,
            (self.covariances_ξηζ.values[:,0].T - self.covariances_ξηζ.values_error[:,0].T) / self.covariances_ξηζ.values[0,0],
            (self.covariances_ξηζ.values[:,0].T + self.covariances_ξηζ.values_error[:,0].T) / self.covariances_ξηζ.values[0,0], color='0.2', alpha=0.3, linewidth=0.)

        # η-η covariance
        ax.plot(-self.time, covariances_ξηζ[:,1], linestyle='--', color='0.4', linewidth=1.5,
            label='η-η : ({} ± {}) Ma'.format(
                self.covariances_ξηζ.age[1], self.covariances_ξηζ.age_error[1]).replace('.', ','))
        ax.fill_between(-self.time,
            (self.covariances_ξηζ.values[:,1].T - self.covariances_ξηζ.values_error[:,1].T) / self.covariances_ξηζ.values[0,1],
            (self.covariances_ξηζ.values[:,1].T + self.covariances_ξηζ.values_error[:,1].T) / self.covariances_ξηζ.values[0,1], color='0.4', alpha=0.3, linewidth=0.)

        # ζ-ζ covariance
        ax.plot(-self.time, covariances_ξηζ[:,2], linestyle=':', color='0.6', linewidth=1.5,
            label='ζ-ζ : ({} ± {}) Ma'.format(
                self.covariances_ξηζ.age[2], self.covariances_ξηζ.age_error[2]).replace('.', ','))
        # ax.fill_between(self.time,
        #     (self.covariances_ξηζ.values[:,2].T - self.covariances_ξηζ.values_error[:,2].T) / self.covariances_ξηζ.values[0,2],
        #     (self.covariances_ξηζ.values[:,2].T + self.covariances_ξηζ.values_error[:,2].T) / self.covariances_ξηζ.values[0,2], color='0.6', alpha=0.3, linewidth=0.)

        # Robust covariance matrix determinant
        ax.plot(-self.time, self.covariances_ξηζ_matrix_robust_det.values,
        # ax.plot(self.time, self.covariances_ξηζ_matrix_robust_det.values / self.covariances_ξηζ_matrix_robust_det.values[0],
            linestyle='-', color='b', linewidth=1.5, label='Det (robust) : ({} ± {}) Ma'.format(
                self.covariances_ξηζ_matrix_robust_det.age,
                self.covariances_ξηζ_matrix_robust_det.age_error).replace('.', ','))

        # Robust covariance matrix trace
        ax.plot(-self.time, self.covariances_ξηζ_matrix_robust_trace.values,
        # ax.plot(self.time, self.covariances_ξηζ_matrix_robust_trace.values / self.covariances_ξηζ_matrix_robust_trace.values[0],
            linestyle='-', color='g', linewidth=1.5, label='Trace (robust) : ({} ± {}) Ma'.format(
                self.covariances_ξηζ_matrix_robust_trace.age,
                self.covariances_ξηζ_matrix_robust_trace.age_error).replace('.', ','))

        # Robust covariance normalization
        covariances_ξηζ_robust = self.covariances_ξηζ_robust.values
        # covariances_ξηζ_robust = self.covariances_ξηζ_robust.values / self.covariances_ξηζ_robust.values[0]

        # Robust ξ-ξ covariance
        ax.plot(-self.time, covariances_ξηζ_robust[:,0], linestyle='-.', color='#2635ff', linewidth=1.5,
            label='ξ-ξ (robust) : ({} ± {}) Ma'.format(
                self.covariances_ξηζ_robust.age[0],
                self.covariances_ξηζ_robust.age_error[0]).replace('.', ','))

        # Robust η-η covariance
        ax.plot(-self.time, covariances_ξηζ_robust[:,1], linestyle='--', color='#4d58ff', linewidth=1.5,
            label='η-η (robust) : ({} ± {}) Ma'.format(
                self.covariances_ξηζ_robust.age[1],
                self.covariances_ξηζ_robust.age_error[1]).replace('.', ','))

        # Robust ζ-ζ covariance
        ax.plot(-self.time, covariances_ξηζ_robust[:,2], linestyle=':', color='#737cff', linewidth=1.5,
            label='ζ-ζ (robust) : ({} ± {}) Ma'.format(
                self.covariances_ξηζ_robust.age[2],
                self.covariances_ξηζ_robust.age_error[2]).replace('.', ','))

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("r-r, θ-θ and z-z covariances of β Pictoris (without outliners) "
                    "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                    "errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("x-x, θ-θ and z-z covariances of {} moving group simulations with "
                    "kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=1, fontsize=6)
        ax.yaxis.set_major_formatter(format_ticks)
        ax.set_xlabel('Temps (Ma)')
        ax.set_ylabel('Covariances (pc)')
        ax.set_xlim(-self.final_time.value + 20., 0.)
        ax.set_ylim(0., 20.)
        # ax.set_ylim(0, 1.6)

        # Save figure
        save_figure(self.name, 'Covariances_ξηζ_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_cross_covariances_xyz_plot(self, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of x-u, y-v and z-w cross covariances. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # Cross covariance matrix determinant
        ax.plot(self.time, self.cross_covariances_xyz_matrix_det.values / self.cross_covariances_xyz_matrix_det.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='Det : ({} ± {}) Ma'.format(
                self.cross_covariances_xyz_matrix_det.age,
                self.cross_covariances_xyz_matrix_det.age_error).replace('.', ','))

        # Covariance normalization
        cross_covariances_xyz = self.cross_covariances_xyz.values / self.cross_covariances_xyz.values[0]

        # x-u covariance
        ax.plot(self.time, cross_covariances_xyz[:,0], linestyle='-.', color='0.2', linewidth=1.5,
            label='X-U : ({} ± {}) Ma'.format(
                self.cross_covariances_xyz.age[0],
                self.cross_covariances_xyz.age_error[0]).replace('.', ','))

        # y-v covariance
        ax.plot(self.time, cross_covariances_xyz[:,1], linestyle='--', color='0.4', linewidth=1.5,
            label='Y-V : ({} ± {}) Ma'.format(
                self.cross_covariances_xyz.age[1],
                self.cross_covariances_xyz.age_error[1]).replace('.', ','))

        # z-w covariance
        ax.plot(self.time, cross_covariances_xyz[:,2], linestyle=':', color='0.6', linewidth=1.5,
            label='Z-W : ({} ± {}) Ma'.format(
                self.cross_covariances_xyz.age[2],
                self.cross_covariances_xyz.age_error[2]).replace('.', ','))

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("X-U, Y-V and Z-W cross covariances of β Pictoris (without outliners) "
                    "over {} Myr\nwith {} km/s redshift correction and actual measurement "
                    "errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("X-U, Y-V and Z-W cross covariances of {} moving group simulations "
                    "with kinematics similar to β Pictoris \n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=1, fontsize=9)
        ax.yaxis.set_major_formatter(format_ticks)
        ax.set_xlabel('Temps (Ma)')
        ax.set_ylabel('Covariances croisée')
        ax.set_xlim(0, self.final_time.value)
        ax.set_ylim(0, 1.6)

        # Save figure
        save_figure(self.name, 'Cross_covariances_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_cross_covariances_ξηζ_plot(self, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of ξ-vξ, η-vη and ζ-vζ cross covariances. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(5, 4.2), facecolor='w')
        ax = fig.add_subplot(111)

        # Cross covariance matrix determinant
        ax.plot(self.time, self.cross_covariances_ξηζ_matrix_det.values / self.cross_covariances_ξηζ_matrix_det.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='Det : ({} ± {}) Ma'.format(
                self.cross_covariances_ξηζ_matrix_det.age,
                self.cross_covariances_ξηζ_matrix_det.age_error).replace('.', ','))
        # ax.fill_between(self.time,
        #     (self.cross_covariances_ξηζ_matrix_det.values - self.cross_covariances_ξηζ_matrix_det.values_error) / self.cross_covariances_ξηζ_matrix_det.values[0],
        #     (self.cross_covariances_ξηζ_matrix_det.values + self.cross_covariances_ξηζ_matrix_det.values_error) / self.cross_covariances_ξηζ_matrix_det.values[0], color='0.0', alpha=0.3, linewidth=0.)

        # Covariance normalization
        cross_covariances_ξηζ = self.cross_covariances_ξηζ.values / self.cross_covariances_ξηζ.values[0]

        # ξ-vξ covariance
        ax.plot(self.time, cross_covariances_ξηζ[:,0], linestyle='-.', color='0.2', linewidth=1.5,
            label='ξ-vξ: ({} ± {}) Ma'.format(
                self.cross_covariances_ξηζ.age[0],
                self.cross_covariances_ξηζ.age_error[0]).replace('.', ','))
        ax.fill_between(self.time,
            (self.cross_covariances_ξηζ.values[:,0].T - self.cross_covariances_ξηζ.values_error[:,0].T) / self.cross_covariances_ξηζ.values[0,0],
            (self.cross_covariances_ξηζ.values[:,0].T + self.cross_covariances_ξηζ.values_error[:,0].T) / self.cross_covariances_ξηζ.values[0,0], color='0.2', alpha=0.3, linewidth=0.)

        # η-vη covariance
        ax.plot(self.time, cross_covariances_ξηζ[:,1], linestyle='--', color='0.4', linewidth=1.5,
            label='η-vη: ({} ± {}) Ma'.format(
                self.cross_covariances_ξηζ.age[1],
                self.cross_covariances_ξηζ.age_error[1]).replace('.', ','))
        ax.fill_between(self.time,
            (self.cross_covariances_ξηζ.values[:,1].T - self.cross_covariances_ξηζ.values_error[:,1].T) / self.cross_covariances_ξηζ.values[0,1],
            (self.cross_covariances_ξηζ.values[:,1].T + self.cross_covariances_ξηζ.values_error[:,1].T) / self.cross_covariances_ξηζ.values[0,1], color='0.4', alpha=0.3, linewidth=0.)

        # ζ-vζ covariance
        ax.plot(self.time, cross_covariances_ξηζ[:,2], linestyle=':', color='0.6', linewidth=1.5,
            label='ζ-vζ: ({} ± {}) Ma'.format(
                self.cross_covariances_ξηζ.age[2],
                self.cross_covariances_ξηζ.age_error[2]).replace('.', ','))
        # ax.fill_between(self.time,
        #     (self.cross_covariances_ξηζ.values[:,2].T - self.cross_covariances_ξηζ.values_error[:,2].T) / self.cross_covariances_ξηζ.values[0,2],
        #     (self.cross_covariances_ξηζ.values[:,2].T + self.cross_covariances_ξηζ.values_error[:,2].T) / self.cross_covariances_ξηζ.values[0,2], color='0.6', alpha=0.3, linewidth=0.)

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                ax.set_title("R-vR, phi-vT and z-vz cross covariances of β Pictoris (without "
                    "outliners) over {} Myr\nwith {} km/s redshift correction and actual "
                    "measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                ax.set_title("R-vR, phi-vT and z-vz cross covariances of {} moving group simu"
                    "lations with kinematics similar to β Pictoris \n over {} Myr with {} km/s "
                    "redshift correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax.legend(loc=1, fontsize=9)
        ax.yaxis.set_major_formatter(format_ticks)
        ax.set_xlabel('Temps (Ma)')
        ax.set_ylabel('Covariances croisées')
        ax.set_xlim(0, self.final_time.value)
        ax.set_ylim(0, 1.6)

        # Save figure
        save_figure(self.name, 'Cross_covariances_ξηζ_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_scatter_mad_mst_cross_covariances_plots(
            self, other, secondary=False, title=True, forced=False, default=False, cancel=False):
        """ Creates a plot of scatter, median absolute deviation, and minimum spanning tree
            branches length mean and median absolute deviation over the entire duration of the
            data.
        """

        # Check if 'other' is valid
        self.stop(type(other) != type(self), 'TypeError',
            "'other' must be a Series object ({} given).", type(other))

        # Figure initialization
        self.check_traceback()
        fig, (ax0, ax1) = plt.subplots(
            ncols=2, constrained_layout=True, figsize=(10, 4.2), facecolor='w')

        # Scatter (other)
        ax0.plot(other.time, other.scatter_xyz_total.values / other.scatter_xyz_total.values[0],
            linestyle='-', color='0.0', linewidth=1.5, label='σ : ({} ± {}) Ma'.format(
                other.scatter_xyz_total.age, other.scatter_xyz_total.age_error).replace('.', ','))

        # Median absolute deviation (other)
        ax0.plot(other.time, other.mad_xyz_total.values / other.mad_xyz_total.values[0],
            linestyle='-.', color='0.2', linewidth=1.5, label='MAD : ({} ± {}) Ma'.format(
                other.mad_xyz_total.age, other.mad_xyz_total.age_error).replace('.', ','))

        # Minimum spanning tree mean branch length (other)
        ax0.plot(other.time, other.mst_mean.values / other.mst_mean.values[0],
            linestyle='--', color='0.4', linewidth=1.5, label='MST : ({} ± {}) Ma'.format(
                other.mst_mean.age, other.mst_mean.age_error).replace('.', ','))

        # Minimum spanning tree median absolute deviation branch length (other)
        ax0.plot(other.time, other.mst_mad.values / other.mst_mad.values[0],
            linestyle=':', color='0.6', linewidth=1.5, label='MAD MST : ({} ± {}) Ma'.format(
                other.mst_mad.age, other.mst_mad.age_error).replace('.', ','))

        # Scatter
        ax0.plot(self.time, self.scatter_xyz_total.values / self.scatter_xyz_total.values[0],
            linestyle='-', color='b', linewidth=1.5, label='σ : ({} ± {}) Ma'.format(
                self.scatter_xyz_total.age, self.scatter_xyz_total.age_error).replace('.', ','))

        # Median absolute deviation
        ax0.plot(self.time, self.mad_xyz_total.values / self.mad_xyz_total.values[0],
            linestyle='-.', color='#2635ff', linewidth=1.5, label='MAD : ({} ± {}) Ma'.format(
                self.mad_xyz_total.age, self.mad_xyz_total.age_error).replace('.', ','))

        # Minimum spanning tree mean branch length
        ax0.plot(self.time, self.mst_mean.values / self.mst_mean.values[0],
            linestyle='--', color='#4d58ff', linewidth=1.5, label='MST : ({} ± {}) Ma'.format(
                self.mst_mean.age, self.mst_mean.age_error).replace('.', ','))

        # Minimum spanning tree median absolute deviation branch length
        ax0.plot(self.time, self.mst_mad.values / self.mst_mad.values[0],
            linestyle=':', color='#737cff', linewidth=1.5, label='MAD MST : ({} ± {}) Ma'.format(
                self.mst_mad.age, self.mst_mad.age_error).replace('.', ','))

        # Legend and axes formatting
        ax0.legend(loc=1, fontsize=9.5)
        ax0.set_xlabel('Temps (Ma)')
        ax0.set_ylabel("Indicateurs d'âge")
        ax0.yaxis.set_major_formatter(format_ticks)
        ax0.set_xlim(0, self.final_time.value)
        ax0.set_ylim(0.0, 1.5)

        # Covariance normalization
        cross_covariances_xyz = self.cross_covariances_xyz.values / self.cross_covariances_xyz.values[0]
        other_cross_covariances_xyz = other.cross_covariances_xyz.values / other.cross_covariances_xyz.values[0]

        # X-U cross covariance (other)
        ax1.plot(other.time, other_cross_covariances_xyz[:,0],
            linestyle='-', color='k', linewidth=1.5, label='X-U : ({} ± {}) Ma'.format(
                other.cross_covariances_xyz.age[0],
                other.cross_covariances_xyz.age_error[0]).replace('.', ','))

        # Y-V cross covariance (other)
        ax1.plot(other.time, other_cross_covariances_xyz[:,1],
            linestyle='-.', color='0.2', linewidth=1.5, label='Y-V : ({} ± {}) Ma'.format(
                other.cross_covariances_xyz.age[1],
                other.cross_covariances_xyz.age_error[1]).replace('.', ','))

        # Z-W cross covariance (other)
        ax1.plot(other.time, other_cross_covariances_xyz[:,2],
            linestyle='--', color='0.4', linewidth=1.5, label='Z-W : ({} ± {}) Ma'.format(
                other.cross_covariances_xyz.age[2],
                other.cross_covariances_xyz.age_error[2]).replace('.', ','))

        # X-U cross covariance
        ax1.plot(self.time, cross_covariances_xyz[:,0],
            linestyle='-', color='b', linewidth=1.5, label='X-U : ({} ± {}) Ma'.format(
                self.cross_covariances_xyz.age[0],
                self.cross_covariances_xyz.age_error[0]).replace('.', ','))

        # Y-V cross covariance
        ax1.plot(self.time, cross_covariances_xyz[:,1],
            linestyle='-.', color='#2635ff', linewidth=1.5, label='Y-V : ({} ± {}) Ma'.format(
                self.cross_covariances_xyz.age[1],
                self.cross_covariances_xyz.age_error[1]).replace('.', ','))

        # Z-W cross covariance
        ax1.plot(self.time, cross_covariances_xyz[:,2],
            linestyle='--', color='#4d58ff', linewidth=1.5, label='Z-W : ({} ± {}) Ma'.format(
                self.cross_covariances_xyz.age[2],
                self.cross_covariances_xyz.age_error[2]).replace('.', ','))

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            if self.from_data:
                fig.suptitle("Size indicators of β Pictoris (without outliners) over {} "
                    "Myr\n with {} km/s redshift correction and actual measurement errors\n".format(
                        self.duration.value, round(self.rv_offset.to('km/s').value, 2)))
            elif self.from_model:
                fig.suptitle("Average size indicators of {} moving group simulations with "
                    "kinematics similar to β Pictoris\n over {} Myr with {} km/s redshift "
                    "correction and actual measurement errors of Gaia DR2\n".format(
                        self.number_of_groups, self.duration.value,
                        round(self.rv_offset.to('km/s').value, 2)))

        # Legend and axes formatting
        ax1.legend(loc=1, fontsize=9.2)
        ax1.set_xlabel('Temps (Ma)')
        ax1.set_ylabel('Covariances')
        ax1.yaxis.set_major_formatter(format_ticks)
        ax1.set_xlim(0, self.final_time.value)
        ax1.set_ylim(0, 1.5)

        # Save figure
        save_figure(self.name, 'Size_indicators_covariance_plots_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_histogram(self, title=True, forced=False, default=False, cancel=False):
        """ Creates an histogram of ages computed in a series. """

        # Figure initialization
        self.check_traceback()
        fig = plt.figure(figsize=(12, 8), facecolor='w')
        ax = fig.add_subplot(111)

        # Histogram plotting
        ages = [group.scatter_xyz_total.age for group in self]
        ax.hist(ages, bins='auto') # bins=np.arange(21.975, 26.025, 0.05)

        # Title formatting
        self.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title(
                'Distribution of {} moving groups age,\n'
                'without measurement errors. Average age: ({} ± {}) Myr\n'.format(
                    len(self), np.round(np.mean(ages), 3), np.round(np.std(ages), 3)))

        # Axes formatting
        ax.set_xlabel('Age (Myr)')
        ax.set_ylabel('Number of groups')

        # Save figure
        save_figure('Age Histogram', 'Age Historgram.pdf',
            forced=forced, default=default, cancel=cancel)
        # plt.show()

class Output_Group():
    """ Defines output methods from a Group object. """

    def trajectory(self, title=True, forced=False, default=False, cancel=False):
        """ Draw the XY trajectory of stars """

        # Figure initialization
        fig = plt.figure(figsize=(12, 5.5), facecolor='w')
        ax = fig.add_subplot(111)

        # Plot stars' trajectories
        for star in self.stars:
            ax.plot(star.position_xyz.T[1] / 1000, star.position_xyz.T[0] / 1000,
                c='k', linewidth=0.4)
            ax.plot(star.position_xyz_linear.T[1] / 1000, star.position_xyz_linear.T[0] / 1000,
                c='b', linewidth=0.4)

        # Plot outliers' trajectories
        for star in self.outliers:
            ax.plot(star.position_xyz.T[1] / 1000, star.position_xyz.T[0] / 1000,
                c='r', linewidth=0.4)
            ax.plot(star.position_xyz_linear.T[1] / 1000, star.position_xyz_linear.T[0] / 1000,
                c='r', linewidth=0.4)

        # Draw circles around the galactic center
        for r in range(1, 12):
            ax.add_artist(plt.Circle((0, 8.122), r, color='0.8', fill=False, linestyle=':'))

        # Draw vertical and horizontal lines going through the Sun's position at the current epoch
        ax.axhline(0., color='0.4', linestyle=':')
        ax.axvline(0., color='0.4', linestyle=':')

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("Trajectories of stars in βPMG")

        # Axes formatting
        ax.set_xlabel('Y (kpc)')
        ax.set_ylabel('X (kpc)')
        ax.set_aspect('equal')
        ax.set_adjustable('datalim')
        ax.invert_xaxis()

        # Save figure
        save_figure(self.name, 'Trajectories_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def trajectory_xyz_ξηζ(self, title=True, forced=False, default=False, cancel=False):
        """ Draws the average XYZ and ξηζ trajectory of stars over time. """

        # Figure initialization
        fig = plt.figure(figsize=(6, 5.5), facecolor='w')
        ax = fig.add_subplot(111)

        # Plot trajectories
        ax.plot(self.series.time, self.position_xyz[:,0].T / 1000, c='0.0', linewidth=1.5, linestyle='-', label='$X$')
        ax.plot(self.series.time, self.position_xyz[:,1].T / 1000, c='0.2', linewidth=1.5, linestyle='--', label='$Y$')
        ax.plot(self.series.time, self.position_xyz[:,2].T / 1000, c='0.4', linewidth=1.5, linestyle='-.', label='$Z$')
        ax.plot(self.series.time, (self.position_ξηζ[:,0].T) / 1000, c='b', linewidth=1.5, linestyle='-', label='ξ')
        ax.plot(self.series.time, (self.position_ξηζ[:,1].T) / 1000, c='#2635ff', linewidth=1.5, linestyle='--', label='η')
        ax.plot(self.series.time, (self.position_ξηζ[:,2].T) / 1000, c='#4d58ff', linewidth=1.5, linestyle='-.', label='ζ')

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("Trajectories of stars in βPMG")

        # Legend and axes formatting
        ax.legend(loc=3, fontsize=9.2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Position (kpc)')
        # ax.set_aspect('equal')
        # ax.set_adjustable('datalim')

        # Secondary x axis
        # slope, intercept, r_value, p_value, std_err = linregress(
        #     self.series.time, self.position_ξηζ[:,1].T * 180 / np.pi)
        # def time_to_θ(t):
        #     return slope * t + intercept
        # def θ_to_time(θ):
        #     return (θ - intercept) / slope
        # ax2 = ax.secondary_xaxis('top', functions=(time_to_θ, θ_to_time))
        # ax2.set_xlabel('θ (°)')

        # Save figure
        save_figure(self.name, 'Trajectories_XYZ_{}.pdf'.format(self.name),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def trajectory_ξηζ(self, title=True, forced=False, default=False, cancel=False):
        """ Draws the ξηζ trajectory of stars. """

        # Figure initialization
        fig = plt.figure(figsize=(10, 9.5), facecolor='w')
        ax1 = fig.add_axes([0.10, 0.34, 0.61, 0.61])
        ax2 = fig.add_axes([0.72, 0.34, 0.23, 0.61])
        ax3 = fig.add_axes([0.10, 0.10, 0.61, 0.23])

        # Plot trajectories
        t = int(18.5 / self.series.final_time.value * self.series.number_of_steps)
        for star in self.stars:
            ax1.plot(star.position_ξηζ.T[0], star.position_ξηζ.T[1], c='k', linewidth=0.4, zorder=1)
            ax2.plot(star.position_ξηζ.T[2], star.position_ξηζ.T[1], c='k', linewidth=0.4, zorder=1)
            ax3.plot(star.position_ξηζ.T[0], star.position_ξηζ.T[2], c='k', linewidth=0.4, zorder=1)

            # Plot current and birth positions
            ax1.scatter([star.position_ξηζ[0,0]], [star.position_ξηζ[0,1]], marker='s', color='k', zorder=2)
            ax1.scatter([star.position_ξηζ[t,0]], [star.position_ξηζ[t,1]], marker='o', color='b', zorder=2)
            ax2.scatter([star.position_ξηζ[0,2]], [star.position_ξηζ[0,1]], marker='s', color='k', zorder=2)
            ax2.scatter([star.position_ξηζ[t,2]], [star.position_ξηζ[t,1]], marker='o', color='b', zorder=2)
            ax3.scatter([star.position_ξηζ[0,0]], [star.position_ξηζ[0,2]], marker='s', color='k', zorder=2)
            ax3.scatter([star.position_ξηζ[t,0]], [star.position_ξηζ[t,2]], marker='o', color='b', zorder=2)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            fig.suptitle("Trajectories ξηζ of stars in βPMG")

        # Axes formatting
        ax1.set_xticklabels([])
        ax1.set_ylabel('$η\prime$ (pc)')
        ax2.set_xlabel('$ζ\prime$ (pc)')
        ax2.set_yticklabels([])
        ax3.set_xlabel('$ξ\prime$ (pc)')
        ax3.set_ylabel('$ζ\prime$ (pc)')

        # Axes limits
        ax1.set_xlim(-225, 25)
        ax1.set_ylim(-45, 145)
        ax2.set_xlim(-35, 45)
        ax2.set_ylim(-45, 145)
        ax3.set_xlim(-225, 25)
        ax3.set_ylim(-35, 45)

        # Axes ticks
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        ax3.tick_params(direction='in')

        # Save figure
        save_figure(self.name, 'Trajectories_ξηζ_{}.pdf'.format(self.name), forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_scatter(
            self, i, j, step=None, age=None, errors=False, labels=False, mst=False,
            title=True, forced=False, default=False, cancel=False):
        """ Creates a 2D scatter plot of star positions in i and j at a given 'step' or 'age' in
            Myr. If 'age' doesn't match a step, the closest step is used instead. 'age' overrules
            'steps' if both are given. 'labels' adds the stars' name and 'mst' adds the minimun
            spanning tree branches.
        """

        # Axis initialization
        axis = {'x': 0, 'y': 1, 'z': 2}
        keys = tuple(axis.keys())

        # X axis
        self.series.stop(type(i) != str, "X axis 'i' must be a string ({} given).", type(i))
        self.series.stop(i.lower() not in keys, 'ValueError',
            "X axis 'i' must be an axis key ('x', 'y' or 'z', {} given).", i)
        i = axis[i.lower()]

        # Y axis
        self.series.stop(type(j) != str, "Y axis 'j' must be a string ({} given).", type(j))
        self.series.stop(j.lower() not in keys, 'ValueError',
            "Y axis 'j' must be an axis key ('x', 'y' or 'z', {} given).", j)
        j = axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Figure initialization
        fig = plt.figure(figsize=(6, 5.5), facecolor='w')
        ax = fig.add_subplot(111)

        # Scatter
        ax.scatter(
            [star.position_xyz[step, i] for star in self.stars],
            [star.position_xyz[step, j] for star in self.stars], marker='o', color='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.stars:
                position = star.position_xyz[step]
                error = star.position_xyz_linear_error[step]

                # Horizontal error bars
                ax.plot(
                    (position[i] - error[i], position[i] + error[i]),
                    (position[j], position[j]), c='0.1', linewidth=0.7)

                # Vertical error bars
                ax.plot(
                    (position[i], position[i]),
                    (position[j] - error[j], position[j] + error[j]), c='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.stars:
                ax.text(star.position_xyz[step, i] + 1, star.position_xyz[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=7)

        # Branches creation
        self.series.stop(type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (branch.start.position[step, i], branch.end.position[step, i]),
                    (branch.start.position[step, j], branch.end.position[step, j]), c='b')

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("{} and {} positions of stars in β Pictoris at {} Myr "
                "wihtout outliers.\n".format(keys[i].upper(), keys[j].upper(), age))

        # Axes formatting
        ax.set_xlabel('${}$ (pc)'.format(keys[i].lower()))
        ax.set_ylabel('${}$ (pc)'.format(keys[j].lower()))

        # Save figure
        save_figure(self.name, '2D_Scatter_{}_{}{}_at_{}Myr.pdf'.format(
                self.name, keys[i].upper(), keys[j].upper(), int(round(age))),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_3D_scatter(
            self, step=None, age=None, errors=False, labels=False, mst=False,
            title=True, forced=False, default=False, cancel=False):
        """ Creates a 3D scatter plot of star positions in x, y and z at a given 'step' or 'age'
            in Myr. If 'age' doesn't match a step, the closest step is used instead. 'age'
            overrules 'step' if both are given. 'labels' adds the stars' name and 'mst' adds the
            minimum spanning tree branches.
        """

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(step * self.series.timestep.value, 2)

        # Figure initialization
        fig = plt.figure(figsize=(6, 5.5), facecolor='w')
        ax = fig.add_subplot(111, projection='3d')

        # Scatter
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.stars],
            [star.relative_position_xyz[step, 1] for star in self.stars],
            [star.relative_position_xyz[step, 2] for star in self.stars], marker='o', c='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.stars:
                position = star.relative_position_xyz[step]
                error = star.relative_position_xyz_error[step]

                # X axis error bars
                ax.plot(
                    (position[0] - error[0], position[0] + error[0]),
                    (position[1], position[1]), (position[2], position[2]), c='0.1', linewidth=0.7)

                # Y axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1] - error[1], position[1] + error[1]),
                    (position[2], position[2]), c='0.1', linewidth=0.7)

                # Z axis error bars
                ax.plot(
                    (position[0], position[0]), (position[1], position[1]),
                    (position[2] - error[2], position[2] + error[2]), c='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.stars:
                ax.text(star.position_xyz[step, 0] + 2, star.position_xyz[step, 1] + 2,
                    star.position_xyz[step, 2] + 2, star.name, horizontalalignment='left', fontsize=7)

        # Branches creation
        self.series.stop(type(mst) != bool, 'TypeError',
            "'mst' must be a boolean ({} given).", type(mst))
        if mst:
            for branch in self.mst[step]:
                ax.plot(
                    (branch.start.relative_position_xyz[step, 0],
                        branch.end.relative_position_xyz[step, 0]),
                    (branch.start.relative_position_xyz[step, 1],
                        branch.end.relative_position_xyz[step, 1]),
                    (branch.start.relative_position_xyz[step, 2],
                        branch.end.relative_position_xyz[step, 2]), c='b')

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("Minimum spanning tree of stars in β Pictoris at {} Myr.\n".format(age))

        # Axes formatting
        ax.set_xlabel('\n $x$ (pc)')
        ax.set_ylabel('\n $y$ (pc)')
        ax.set_zlabel('\n $z$ (pc)')

        # Save figure
        save_figure(self.name, '3D_Scatter_{}_at_{}Myr.pdf'.format(self.name, int(round(age))),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_and_3D_scatter(self, ages, title=True, forced=False, default=False, cancel=False):
        """ Creates a three 4-axis columns of xy, xz and yz 2D scatters, and a 3D scatter at three
            ages definied by a list or tuple.
        """

        # Check if ages is valid
        self.series.stop(type(ages) not in (tuple, list), 'TypeError',
            "'ages' must a tuple or a list ({} given).", type(ages))
        self.series.stop(len(ages) != 3, 'ValueError',
            "'ages' must be have a length of 3 ({} given).", ages)

        # Figure initialization
        self.fig = plt.figure(figsize=(9, 11.15), facecolor='w')

        # Axes creation
        row1 = 0.795
        row2 = 0.545
        row3 = 0.295
        row4 = 0.035
        col1 = 0.070
        col2 = 0.398
        col3 = 0.730
        self.create_2D_axis('x', 'y', age=ages[0], index=1, left=col1, bottom=row1)
        self.create_2D_axis('x', 'y', age=ages[1], index=2, left=col2, bottom=row1)
        self.create_2D_axis('x', 'y', age=ages[2], index=3, left=col3, bottom=row1)
        self.create_2D_axis('x', 'z', age=ages[0], index=4, left=col1, bottom=row2)
        self.create_2D_axis('x', 'z', age=ages[1], index=5, left=col2, bottom=row2)
        self.create_2D_axis('x', 'z', age=ages[2], index=6, left=col3, bottom=row2)
        self.create_2D_axis('y', 'z', age=ages[0], index=7, left=col1, bottom=row3)
        self.create_2D_axis('y', 'z', age=ages[1], index=8, left=col2, bottom=row3)
        self.create_2D_axis('y', 'z', age=ages[2], index=9, left=col3, bottom=row3)
        self.create_3D_axis(age=ages[0], index=10, left=0.0535, bottom=row4)
        self.create_3D_axis(age=ages[1], index=11, left=0.381, bottom=row4)
        self.create_3D_axis(age=ages[2], index=12, left=0.712, bottom=row4)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("XY, XZ, YZ and 3D scatters at {}, {} and {} Myr.\n".format(*ages))

        # Save figure
        save_figure(self.name, '2D_Scatter_{}_{}_{}_{}_Myr.pdf'.format(self.name, *ages),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_2D_axis(self, i, j, step=None, age=None, index=1, left=0, bottom=0):
        """ Creates a 2D axis. """

        # Axis initialization
        axis = {'x': 0, 'y': 1, 'z': 2}
        keys = tuple(axis.keys())

        # X axis
        self.series.stop(type(i) != str, "X axis 'i' must be a string ({} given).", type(i))
        self.series.stop(i.lower() not in keys, 'ValueError',
            "X axis 'i' must be an axis key ('x', 'y' or 'z', {} given).", i)
        i = axis[i.lower()]

        # Y axis
        self.series.stop(type(j) != str, "Y axis 'j' must be a string ({} given).", type(j))
        self.series.stop(j.lower() not in keys, 'ValueError',
            "Y axis 'j' must be an axis key ('x', 'y' or 'z', {} given).", j)
        j = axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Axis initialization
        ax = self.fig.add_subplot(4, 3, index, position=[left, bottom, 0.255, 0.20])

        # Star scatter
        ax.scatter(
            [star.relative_position_xyz[step, i] for star in self.stars],
            [star.relative_position_xyz[step, j] for star in self.stars],
            marker='o', s=8, color='k')

        # Outliers scatter
        ax.scatter(
            [star.relative_position_xyz[step, i] for star in self.outliers],
            [star.relative_position_xyz[step, j] for star in self.outliers],
            marker='o', s=8, color='#ff2634')

        # Error bars
        for star in self:
            position = star.relative_position_xyz[step]
            error = star.position_xyz_linear_error[step]
            color = 'k' if not star.outlier else '#ff2634'

            # Horizontal error bars
            ax.plot(
                (position[i] - error[i], position[i] + error[i]),
                (position[j], position[j]), c=color, linewidth=0.7)

            # Vertical error bars
            ax.plot(
                (position[i], position[i]),
                (position[j] - error[j], position[j] + error[j]), c=color, linewidth=0.7)

        # Axes formatting
        ax.set_xlabel('${}$ (pc)'.format(keys[i].upper()))
        ax.set_ylabel('${}$ (pc)'.format(keys[j].upper()), labelpad=-12.)

        # Limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

    def create_3D_axis(self, step=None, age=None, index=1, left=0, bottom=0):
        """ Creates a 3D axis. """

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            step = int(step)
            age = round(step * self.series.timestep.value, 2)

        # Figure initialization
        ax = self.fig.add_subplot(
            4, 3, index, projection='3d', position=[left, bottom, 0.29, 0.215])

        # Star scatter
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.stars],
            [star.relative_position_xyz[step, 1] for star in self.stars],
            [star.relative_position_xyz[step, 2] for star in self.stars], marker='o', c='0.0')

        # Outlier scatter
        ax.scatter(
            [star.relative_position_xyz[step, 0] for star in self.outliers],
            [star.relative_position_xyz[step, 1] for star in self.outliers],
            [star.relative_position_xyz[step, 2] for star in self.outliers], marker='o', c='#ff2634')

        # Branches creation
        for branch in self.mst[step]:
            ax.plot(
                (branch.start.relative_position_xyz[step, 0],
                    branch.end.relative_position_xyz[step, 0]),
                (branch.start.relative_position_xyz[step, 1],
                    branch.end.relative_position_xyz[step, 1]),
                (branch.start.relative_position_xyz[step, 2],
                    branch.end.relative_position_xyz[step, 2]), c='#2635ff')

        # Axes formatting
        ax.view_init(azim=45)
        ax.set_xlabel('$X$ (pc)')
        ax.set_ylabel('$Y$ (pc)')
        ax.set_zlabel('$Z$ (pc)')

        # Axes limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)

    def create_cross_covariances_scatter(
            self, i, j, step=None, age=None, errors=False, labels=False,
            title=True, forced=False, default=False, cancel=False):
        """ Creates a cross covariance scatter of star positions and velocities in i and j at a
            given 'step' or 'age' in Myr. If 'age' doesn't match a step, the closest step is used
            instead. 'age' overrules 'steps' if both are given. 'labels' adds the stars' name
            and 'mst' adds the minimum spanning tree branches.
        """

        # Axes initialization
        position_axis = {'x': 0, 'y': 1, 'z': 2}
        velocity_axis = {'u': 0, 'v': 1, 'w': 2}
        position_keys = tuple(position_axis.keys())
        velocity_keys = tuple(velocity_axis.keys())

        # Position axis
        self.series.stop(type(i) != str, "Position axis 'i' must be a string ({} given).", type(i))
        self.series.stop(i.lower() not in position_keys, 'ValueError',
            "Position axis 'i' must be an postion axis key ('x', 'y' or 'z', {} given).", i)
        i = position_axis[i.lower()]

        # Velocity axis
        self.series.stop(type(j) != str, "Velocity axis 'j' must be a string ({} given).", type(j))
        self.series.stop(j.lower() not in velocity_keys, 'ValueError',
            "Velocity axis 'j' must be an postion axis key ('u', 'v' or 'w', {} given).", j)
        j = velocity_axis[j.lower()]

        # Check if step and age are valid
        if step is not None:
            self.series.stop(type(step) not in (int, float), 'TypeError',
                "'step' must an integer or float ({} given).", type(step))
            self.series.stop(step % 1 != 0, 'ValueError',
                "'step' must be convertible to an integer ({} given).", step)
        if age is not None:
            self.series.stop(type(age) not in (int, float), 'TypeError',
                "'age' must be an integer or float ({} given).", type(age))
            self.series.stop(age < 0, 'ValueError',
                "'age' must be greater than or equal to 0.0 ({} given).", type(age))

        # Step or age calculation
        if age is not None:
            step = int(round(age / self.series.timestep.value))
            age = round(self.series.time[step], 2)
        else:
            age = round(step * self.series.timestep, 2)

        # Figure initialization
        fig = plt.figure(figsize=(6, 5.5), facecolor='w')
        ax = fig.add_subplot(111)

        # Scatter
        ax.scatter(
            [star.position_xyz[step, i] for star in self.stars],
            [star.velocity_xyz[step, j] for star in self.stars], marker='o', color='0.0')

        # Error bars
        self.series.stop(type(errors) != bool, 'TypeError',
            "'error' must be a boolean ({} given).", type(errors))
        if errors:
            for star in self.stars:
                position = star.position_xyz[step]
                position_error = star.position_xyz_linear_error[step]
                velocity = star.velocity_xyz[step]
                velocity_error = star.velocity_xyz_error[step]

                # Position (horizontal) error bars
                ax.plot(
                    (position[i] - position_error[i], position[i] + position_error[i]),
                    (velocity[j], velocity[j]),
                    c='0.1', linewidth=0.7)

                # Velocity (vertical) error bars
                ax.plot(
                    (position[i], position[i]),
                    (velocity[j] - velocity_error[j], velocity[j] + velocity_error[j]),
                    c='0.1', linewidth=0.7)

        # Star labels
        self.series.stop(type(labels) != bool, 'TypeError',
            "'labels' must be a boolean ({} given).", type(labels))
        if labels:
            for star in self.stars:
                ax.text(star.position_xyz[step, i] + 1, star.velocity_xyz[step, j] + 1, star.name,
                horizontalalignment='left', fontsize=7)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title("{} and {} covariance of stars in β Pictoris at {} Myr wihtout "
                "outliers.\n".format(position_keys[i].upper(), velocity_keys[j].upper(), age))

        # Axes formatting
        ax.set_xlabel('{} (pc)'.format(position_keys[i].upper()))
        ax.set_ylabel('{} (pc/Myr)'.format(velocity_keys[j].upper()))

        # Save figure
        save_figure(self.name, 'Covariances_Scatter_{}_{}{}.pdf'.format(
                self.name, position_keys[i].upper(), velocity_keys[j].upper()),
            forced=forced, default=default, cancel=cancel)
        # plt.show()

    def create_map(self, labels=False, title=True, forced=False, default=False, cancel=False):
        """ Creates a Mollweide projection of a traceback. """

        # Figure initialization
        fig = plt.figure(figsize=(8, 4), facecolor='w')
        ax = fig.add_subplot(111, projection="mollweide")

        # Coordinates computation
        from Traceback.coordinate import galactic_xyz_equatorial_rδα
        positions = np.array([[galactic_xyz_equatorial_rδα(*star.position_xyz[step])[0] \
            for step in range(self.series.number_of_steps)] for star in self])
        alphas = np.vectorize(lambda α: α - (2 * np.pi if α > np.pi else 0.0))(positions[:,:,2])
        deltas = positions[:,:,1]

        # Trajectories
        for star in range(len(self)):
            color = '#2635ff' if not self[star].outlier else '#ff2634'
            if star not in (15, 17):
                ax.plot(alphas[star], deltas[star], color=color, lw=1, zorder=2)
            else:
                limit = len(tuple(filter(lambda i: i <= np.pi and i > 0, alphas[star])))
                ax.plot(alphas[star, :limit], deltas[star, :limit], color=color, lw=1, zorder=2)
                ax.plot(alphas[star, limit:], deltas[star, limit:], color=color, lw=1, zorder=2)

            # Labels
            if labels:
                ax.text(alphas[star, 0] + 0.1, deltas[star, 0] + 0.1, star + 1,
                    horizontalalignment='left', fontsize=7, zorder=2)

        # Scatter
        ax.scatter(alphas[:,0], deltas[:,0], marker='.', color='k', zorder=3)

        # Proper motion arrows
        for star in self.series.data:
            ax.arrow(
                star.position.values[2] - (2 * np.pi if star.position.values[2] > np.pi else 0.0),
                star.position.values[1], -star.velocity.values[2]/4, -star.velocity.values[1]/4,
                head_width=0.03, head_length=0.03, color='k', zorder=4)

        # Title formatting
        self.series.stop(type(title) != bool, 'TypeError',
            "'title' must be a boolean ({} given).", type(title))
        if title:
            ax.set_title('Mollweide projection of tracebacks.')

        # Axes formatting
        ax.grid(zorder=1)

        # Save figure
        save_figure('Mollweide', forced=forced, default=default, cancel=cancel)
        # plt.show()

def create_histogram(ages, initial_scatter, number_of_stars, number_of_groups, age,
        title=True, forced=False, default=False, cancel=False):
    """ Creates an histogram of ages computed by multiple tracebacks. """

    # Check if ages are valid
    stop(type(ages) not in (tuple, list),
        "'ages' must either must be a tuple or list ({} given)", type(ages))
    for age in ages:
        stop(type(age) not in (int, float), 'TypeError',
            "All 'ages' must be an integer or float ({} given).", type(age))
        stop(age < 0, 'ValueError',
            "All 'ages' must be greater than or equal to 0.0 ({} given).", type(age))

    # Check if initial scatter is valid
    stop(type(initial_scatter) not in (int, float), 'TypeError',
        "'initial_scatter' must be an integer or float ({} given).", type(initial_scatter))
    stop(initial_scatter < 0, 'ValueError', "'initial_scatter' must be greater than "
        "or equal to 0.0 ({} given).", type(initial_scatter))

    # Check if number_of_stars is valid
    stop(type(number_of_stars) not in (int, float), 'TypeError',
        "'number_of_stars' must an integer or float ({} given).", type(number_of_stars))
    stop(number_of_stars % 1 != 0, 'ValueError',
        "'number_of_stars' must be convertible to an integer ({} given).", number_of_stars)
    number_of_stars = int(number_of_stars)

    # Check if number_of_groups is valid
    stop(type(number_of_groups) not in (int, float), 'TypeError',
        "'number_of_groups' must an integer or float ({} given).", type(number_of_groups))
    stop(number_of_groups % 1 != 0, 'ValueError',
        "'number_of_groups' must be convertible to an integer ({} given).", number_of_groups)
    number_of_groups = int(number_of_groups)

    # Check if age is valid
    stop(type(age) not in (int, float), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Figure initialization
    fig = plt.figure(figsize=(6, 5.5), facecolor='w')
    ax = fig.add_subplot(111)

    # Histogram plotting
    hist, bin_edges = np.histogram(ages, density=True)
    ax.hist(ages, bins='auto', density=True) # bins=np.arange(21.975, 26.025, 0.05)

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title(
            "Distribution of ages ({} groups, {} Myr, {} stars,\ninitial scatter = "
                "{} pc, {})".format(number_of_groups, age, number_of_stars, initial_scatter,
                    'calculated age = ({} ± {}) Myr'.format(
                        np.round(np.average(ages), 3), np.round(np.std(ages), 3))))

    # Axes formatting
    ax.set_xlabel('Age (Myr)')
    ax.set_ylabel('Number of groups')

    # Save figure
    save_figure('Histogram', 'Distribution of ages ({} groups, {} Myr, {} stars, initial '
            'scatter = {} pc).pdf'.format(number_of_groups, age, number_of_stars, initial_scatter),
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_color_mesh(initial_scatter, number_of_stars, errors, age, number_of_groups, method,
        title=True, forced=False, default=False, cancel=False):
    """ Creates a color mesh of errors over the initial scatter and number_of_stars.
        !!! Créer une fonction pour passer d'un array Numpy de shape (n, 3) à un !!!
        !!! color mesh + smoothing, genre create_color_mesh(x, y, z, smoothing). !!!
    """

    # Check if initial scatter is valid
    stop(type(initial_scatter) not in (tuple, list, np.ndarray),
        "'initial_scatter' must either must be a tuple or list ({} given)", type(initial_scatter))
    for scatter in np.array(initial_scatter).flatten():
        stop(type(scatter) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'initial_scatter' must be an integer or float ({} given).", type(scatter))
        stop(age < 0, 'ValueError',
            "All 'initial_scatter' must be greater than or equal to 0.0 ({} given).", type(scatter))

    # Check if number_of_stars is valid
    stop(type(number_of_stars) not in (tuple, list, np.ndarray),
        "'number_of_stars' must either must be a tuple or list ({} given)", type(errors))
    for star in np.array(number_of_stars).flatten():
        stop(type(star) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'initial_scatter' must be an integer or float ({} given).", type(star))
        stop(star < 0, 'ValueError',
            "All 'initial_scatter' must be greater than or equal to 0.0 ({} given).", type(star))
        stop(star % 1 != 0, 'ValueError',
            "All 'number_of_stars' must be convertible to an integer ({} given).", star)

    # Check if errors are valid
    stop(type(errors) not in (tuple, list, np.ndarray),
        "'errors' must either must be a tuple or list ({} given)", type(errors))
    for error in np.array(errors).flatten():
        stop(type(error) not in (int, float, np.int64, np.float64), 'TypeError',
            "All 'errors' must be an integer or float ({} given).", type(error))
        stop(error < 0, 'ValueError',
            "All 'errors' must be greater than or equal to 0.0 ({} given).", type(error))

    # Check if age is valid
    stop(type(age) not in (int, float, np.int64, np.float64), 'TypeError',
        "'age' must be an integer or float ({} given).", type(age))
    stop(age < 0, 'ValueError',
        "'age' must be greater than or equal to 0.0 ({} given).",type(age))

    # Check if number_of_groups is valid
    stop(type(number_of_groups) not in (int, float, np.int64, np.float64), 'TypeError',
        "'number_of_groups' must an integer or float ({} given).", type(number_of_groups))
    stop(number_of_groups % 1 != 0, 'ValueError',
        "'number_of_groups' must be convertible to an integer ({} given).", number_of_groups)
    number_of_groups = int(number_of_groups)

    # Check if method is valid
    stop(type(method) != str, 'TypeError', "'method' must a string ({} given).", type(method))

    # Figure initialization
    fig = plt.figure(figsize=(5, 4), facecolor='w')
    ax = fig.add_subplot(111)

    # Mesh plotting
    x, y = np.meshgrid(initial_scatter, number_of_stars)
    grid_x, grid_y = np.mgrid[0:20:81j, 20:100:81j]
    grid_z = griddata(np.array([(i, j) for i in initial_scatter for j in number_of_stars]),
        errors.T.flatten(), (grid_x, grid_y), method='linear')
    ax.pcolormesh(grid_x, grid_y, grid_z, cmap=plt.cm.PuBu_r, vmin=0, vmax=6)
    fig.colorbar(mappable=plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 6.0), cmap=plt.cm.PuBu_r),
        ax=ax, ticks=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], format='%0.1f')

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title('Scatter on age (Myr) over the initial scatter (pc)\n'
            'and the number of stars ({} groups, {} Myr)'.format(number_of_groups, age))

    # Axes formatting
    ax.set_xlabel('Dispersion initiale (pc)')
    ax.set_ylabel("Nombre de membres")
    ax.set_xticks([0.0, 5.0, 10.0, 15.0, 20.0])
    ax.set_yticks([20, 40, 60, 80, 100])

    # Save figure
    save_figure('Scatter on age ({} Myr, {})'.format(age, method),
        forced=forced, default=default, cancel=cancel)
    # plt.show()

def plot_age_error(title=True, forced=False, default=False, cancel=False):
    """ Creates a plot of ages obtained for diffrent measurement errors on radial velocity
        and offset due to gravitationnal redshift.
    """

    # Figure initialization
    fig = plt.figure(figsize=(7, 5.8), facecolor='w')
    ax = fig.add_subplot(111)

    # + 0.0 km/s points
    ax.errorbar(
        np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        [23.824, 23.506, 22.548, 21.238, 19.454, 17.639,
            16.008, 14.202, 12.67, 11.266, 9.732, 8.874, 8.044],
        yerr=[0.376, 0.517, 0.850, 1.062, 1.204, 1.383,
            1.534, 1.612, 1.544, 1.579, 1.576, 1.538, 1.504],
        fmt='o', color='0.0', ms=6.0, elinewidth=1.0, label='$\\Delta v_{r,grav}$ = 0,0 km/s')

    # + 0.5 km/s points
    ax.errorbar(
        np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        [19.858, 19.655, 19.116, 19.292, 17.246, 15.988,
            14.749, 13.577, 12.379, 11.222, 10.229, 9.210, 8.446],
        yerr=[0.376, 0.425, 0.641, 0.773, 0.992, 1.136,
            1.129, 1.251, 1.338, 1.331, 1.272, 1.345, 1.323],
        fmt='D', color='0.33', ms=6.0, elinewidth=1.0, label='$\\Delta v_{r,grav}$ = 0,5 km/s')

    # + 1.0 km/s points
    ax.errorbar(
        np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        [16.87, 16.743, 16.404, 15.884, 15.26, 14.522,
            13.529, 12.619, 11.751, 10.847, 9.982, 9.353, 8.461],
        yerr=[0.379, 0.453, 0.583, 0.685, 0.864, 0.93,
            0.951, 1.032, 1.147, 1.035, 1.142, 1.187, 1.149],
        fmt='^', color='0.67', ms=6.0, elinewidth=1.0, label='$\\Delta v_{r,grav}$ = 1,0 km/s')

    # β Pictoris typical error line
    ax.axvline(x=1.0105, ymin=0.0, ymax = 25, linewidth=1, color='k', ls='dashed')
    ax.text(1.15, 6.95, 'Erreur de mesure\nsur $v_r$ typique des\nmembres de $\\beta\\,$PMG',
        horizontalalignment='left', fontsize=14)

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title(
            "Measured age of a simulation of 1000 24 Myr-old groups\n"
            "over the measurement error on RV (other errors typical of Gaia DR2)\n")

    # Legend and axes formatting
    ax.legend(loc=1, fontsize=14)
    ax.xaxis.set_major_formatter(format_ticks)
    ax.yaxis.set_major_formatter(format_ticks)
    ax.set_xlabel('Erreur de mesure sur $v_r$ (km/s)', fontsize=14)
    ax.set_ylabel('Âge (Ma)', fontsize=14)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(6, 24.5)

    # Save figure
    save_figure('errors_rv_offset_plot', forced=forced, default=default, cancel=cancel)
    # plt.show()

def create_minimum_error_plots(title=True, forced=False, default=False, cancel=False):
    """ Creates a plot of the error on the age of minimal scatter as a function of the error
        on the uvw velocity.
    """

    # Figure initialization
    fig = plt.figure(figsize=(6, 5.5), facecolor='w')
    ax = fig.add_subplot(111)

    # Plotting
    errors = (0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
        4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    ages = (24.001, 23.966, 23.901, 23.74, 23.525, 22.224, 20.301, 18.113, 15.977, 11.293, 7.995,
        5.803, 4.358, 3.364, 2.665, 2.204, 1.756, 1.257, 0.933, 0.735, 0.580, 0.488, 0.346, 0.262,
        0.192, 0.160, 0.134)
    ax.plot(errors, ages, '.-', color='0.0', linewidth=1.0)

    # Title formatting
    stop(type(title) != bool, 'TypeError',
        "'title' must be a boolean ({} given).", type(title))
    if title:
        ax.set_title('Impact of UVW velocity on the age of minimal scatter.')

    # Axes formatting
    ax.set_xlabel('Error on UVW velcotiy (km/s)')
    ax.set_ylabel('Age of minimal scatter (Myr)')

    # Save figure
    save_figure('Minium Error Plot', 'minimum_error_plot.pdf',
        forced=forced, default=default, cancel=cancel)
    # plt.show()
