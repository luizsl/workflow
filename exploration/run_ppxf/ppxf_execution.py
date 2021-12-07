#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""

import numpy as np
import ppxf.ppxf_util as util

from scipy.constants import physical_constants
from ppxf.ppxf import ppxf
from time import perf_counter as clock

# from tempfile import mkdtemp
# import os.path as path


class ExecutePpxf:

    def __init__(self, data):
        self.data = data
        self.build_output_storage()
        self.run_all_data()

    def build_output_storage(self):
        np.memmap(filename = 'velocity.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'sigma.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'h3.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'h4.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'velocity_error.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'sigma_error.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'h3_error.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'h4_error.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'bestfit.dat', dtype = float, mode='w+',
                  shape = (self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'chi2.dat', dtype = float, mode='w+',
                  shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'apoly.dat', dtype = float, mode='w+',
                  shape = (self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'weights.dat', dtype = float, mode='w+',
                  shape = (self.data.o_n_model,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))

        np.memmap(filename = 'polyweights.dat', dtype = float, mode='w+',
                  shape = (13,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))

    def store_output(self, pp, index):

        velocity = np.memmap(filename = 'velocity.dat', dtype = float, mode='r+',
                             shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            velocity[index] = np.nan
        else:
            velocity[index] = pp.sol[0]
        del velocity

        sigma = np.memmap(filename = 'sigma.dat', dtype = float, mode='r+',
                          shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            sigma[index] = np.nan
        else:
            sigma[index] = pp.sol[1]
        del sigma

        h3 = np.memmap(filename = 'h3.dat', dtype = float, mode='r+',
                       shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            h3[index] = np.nan
        else:
            h3[index] = pp.sol[2]
        del h3

        h4 = np.memmap(filename = 'h4.dat', dtype = float, mode='r+',
                       shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            h4[index] = np.nan
        else:
            h4[index] = pp.sol[3]
        del h4

        velocity_error = np.memmap(filename = 'velocity_error.dat', dtype = float, mode='r+',
                                   shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            velocity_error[index] = np.nan
        else:
            velocity_error[index] = pp.error[0]
        del velocity_error

        sigma_error = np.memmap(filename = 'sigma_error.dat', dtype = float, mode='r+',
                                shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            sigma_error[index] = np.nan
        else:
            sigma_error[index] = pp.error[1]
        del sigma_error

        h3_error = np.memmap(filename = 'h3_error.dat', dtype = float, mode='r+',
                             shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            h3_error[index] = np.nan
        else:
            h3_error[index] = pp.error[2]
        del h3_error

        h4_error = np.memmap(filename = 'h4_error.dat', dtype = float, mode='r+',
                             shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            h4_error[index] = np.nan
        else:
            h4_error[index] = pp.error[3]
        del h4_error

        bestfit = np.memmap(filename = 'bestfit.dat', dtype = float, mode='r+',
                            shape = (self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            bestfit[:, index] = np.nan
        else:
            bestfit[:, index] = pp.bestfit
        del bestfit

        chi2 = np.memmap(filename = 'chi2.dat', dtype = float, mode='r+',
                         shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            chi2[index] = np.nan
        else:
            chi2[index] = pp.chi2
        del chi2

        apoly = np.memmap(filename = 'apoly.dat', dtype = float, mode='r+',
                          shape = (self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            apoly[:, index] = np.nan
        else:
            apoly[:, index] = pp.apoly
        del apoly

        weights = np.memmap(filename = 'weights.dat', dtype = float, mode='r+',
                            shape = (self.data.o_n_model,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            weights[:, index] = np.nan
        else:
            weights[:, index] = pp.weights
        del weights

        polyweights = np.memmap(filename = 'polyweights.dat', dtype = float, mode='r+',
                  shape = (13,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        if pp is None:
            polyweights[:, index] = np.nan
        else:
            polyweights[:, index] = pp.polyweights
        del polyweights

    def run_all_data(self):

        # result = np.memmap(filename = 'output.dat', dtype = object, mode='r+',
        #             shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))

        # shape_obs = (data.n_pixel_obs,) + (data.shape_obs[0] * data.shape_obs[1],)
        # shape_model = (data.n_pixel_model, data.o_n_model)

        shape_obs = (self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],)
        shape_model = (self.data.n_pixel_model, self.data.o_n_model)

        mmap_flux_obs = np.memmap('flux_obs.dat', dtype = 'float32',
                                  mode = 'r', shape = shape_obs)

        mmap_flux_obs_unc = np.memmap('flux_obs_unc.dat', dtype = 'float32',
                                      mode = 'r', shape = shape_obs)

        mmap_flux_model = np.memmap('flux_model.dat', dtype = 'float32',
                                    mode = 'r', shape = shape_model)

        for index, _ in np.ndenumerate(mmap_flux_obs[0, ...]):
            print('\n' + str(index[0]))
            flux_obs_slice = mmap_flux_obs[:, index[0]]
            flux_obs_unc_slice = mmap_flux_obs_unc[:, index[0]]
            flux_model = mmap_flux_model

            # result = np.memmap(filename = 'output.dat', dtype = object, mode='r+',
            #                    shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))


            if np.any(np.isnan(flux_obs_unc_slice) | np.isnan(flux_obs_slice)):
                print('nan')
                self.store_output(None, index[0])
                # result[index] = np.nan
            else:
                pp = self.execute_ppxf(flux_obs_slice, flux_obs_unc_slice,
                                       flux_model)
                # result[index] = pp
                self.store_output(pp, index[0])
            # result.flush()

    def execute_ppxf(self, flux_obs, flux_obs_unc, flux_model):
        C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

        frac = self.data.wave_obs[1]/self.data.wave_obs[0]    # Constant lambda fraction per pixel
        velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
        z = self.data.z # redshift estimate

        dv = C*np.log(self.data.wave_model[0]/self.data.wave_obs[0])    # eq.(8) of Cappellari (2017)
        goodpixels = util.determine_goodpixels(np.log(self.data.wave_obs),
                                               self.data.limit_model, z)

        vel = C*np.log(1 + z)   # eq.(8) of Cappellari (2017)
        start = [vel, 200.]  # (km/s), starting guess for [V, sigma]

        t = clock()
        pp = ppxf(flux_model, flux_obs, flux_obs_unc, velscale, start,
                  goodpixels=goodpixels, plot=False, moments=4,
                  degree=12, vsyst=dv, clean=False, lam=self.data.wave_obs,
                  quiet=True)

        # print("Formal errors:")
        # print("     dV    dsigma   dh3      dh4")
        # print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

        print('Elapsed time in PPXF: %.2f s' % (clock() - t))

        return pp
