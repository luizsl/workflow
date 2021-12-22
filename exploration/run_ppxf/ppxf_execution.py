#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""
# from numba import jit, cuda
import numpy as np
import ppxf.ppxf_util as util
import _pickle as pickle

from scipy.constants import physical_constants
from ppxf.ppxf import ppxf
from time import perf_counter as clock

# from tempfile import mkdtemp
# import os.path as path


class ExecutePpxf:

    def __init__(self, metadata_path):
        with open(metadata_path, 'rb') as inp:
            self.meta = pickle.load(inp)

        self.build_output_storage()
        self.run_all_data()

    def build_output_storage(self):
        np.memmap(filename = f'{self.meta.temp_output_dir}/velocity.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/sigma.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/h3.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/h4.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/velocity_error.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/sigma_error.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/h3_error.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/h4_error.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/bestfit.dat', dtype = float, mode='w+',
                  shape = (self.meta.n_pixel_obs,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/chi2.dat', dtype = float, mode='w+',
                  shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/apoly.dat', dtype = float, mode='w+',
                  shape = (self.meta.n_pixel_obs,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/weights.dat', dtype = float, mode='w+',
                  shape = (self.meta.o_n_model,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

        np.memmap(filename = f'{self.meta.temp_output_dir}/polyweights.dat', dtype = float, mode='w+',
                  shape = (13,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))

    def store_output(self, pp, index):

        velocity = np.memmap(filename = f'{self.meta.temp_output_dir}/velocity.dat', dtype = float, mode='r+',
                             shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            velocity[index] = np.nan
        else:
            velocity[index] = pp.sol[0]
        del velocity

        sigma = np.memmap(filename = f'{self.meta.temp_output_dir}/sigma.dat', dtype = float, mode='r+',
                          shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            sigma[index] = np.nan
        else:
            sigma[index] = pp.sol[1]
        del sigma

        h3 = np.memmap(filename = f'{self.meta.temp_output_dir}/h3.dat', dtype = float, mode='r+',
                       shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            h3[index] = np.nan
        else:
            h3[index] = pp.sol[2]
        del h3

        h4 = np.memmap(filename = f'{self.meta.temp_output_dir}/h4.dat', dtype = float, mode='r+',
                       shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            h4[index] = np.nan
        else:
            h4[index] = pp.sol[3]
        del h4

        velocity_error = np.memmap(filename = f'{self.meta.temp_output_dir}/velocity_error.dat', dtype = float, mode='r+',
                                   shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            velocity_error[index] = np.nan
        else:
            velocity_error[index] = pp.error[0]
        del velocity_error

        sigma_error = np.memmap(filename = f'{self.meta.temp_output_dir}/sigma_error.dat', dtype = float, mode='r+',
                                shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            sigma_error[index] = np.nan
        else:
            sigma_error[index] = pp.error[1]
        del sigma_error

        h3_error = np.memmap(filename = f'{self.meta.temp_output_dir}/h3_error.dat', dtype = float, mode='r+',
                             shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            h3_error[index] = np.nan
        else:
            h3_error[index] = pp.error[2]
        del h3_error

        h4_error = np.memmap(filename = f'{self.meta.temp_output_dir}/h4_error.dat', dtype = float, mode='r+',
                             shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            h4_error[index] = np.nan
        else:
            h4_error[index] = pp.error[3]
        del h4_error

        bestfit = np.memmap(filename = f'{self.meta.temp_output_dir}/bestfit.dat', dtype = float, mode='r+',
                            shape = (self.meta.n_pixel_obs,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            bestfit[:, index] = np.nan
        else:
            bestfit[:, index] = pp.bestfit
        del bestfit

        chi2 = np.memmap(filename = f'{self.meta.temp_output_dir}/chi2.dat', dtype = float, mode='r+',
                         shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            chi2[index] = np.nan
        else:
            chi2[index] = pp.chi2
        del chi2

        apoly = np.memmap(filename = f'{self.meta.temp_output_dir}/apoly.dat', dtype = float, mode='r+',
                          shape = (self.meta.n_pixel_obs,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            apoly[:, index] = np.nan
        else:
            apoly[:, index] = pp.apoly
        del apoly

        weights = np.memmap(filename = f'{self.meta.temp_output_dir}/weights.dat', dtype = float, mode='r+',
                            shape = (self.meta.o_n_model,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            weights[:, index] = np.nan
        else:
            weights[:, index] = pp.weights
        del weights

        polyweights = np.memmap(filename = f'{self.meta.temp_output_dir}/polyweights.dat', dtype = float, mode='r+',
                  shape = (13,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        if pp is None:
            polyweights[:, index] = np.nan
        else:
            polyweights[:, index] = pp.polyweights
        del polyweights
    
    def run_all_data(self):
        shape_obs = (self.meta.n_pixel_obs,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],)
        shape_model = (self.meta.n_pixel_model, self.meta.o_n_model)

        mmap_flux_obs = np.memmap(f'{self.meta.temp_input_dir}/flux_obs.dat',
                                  dtype = 'float32', mode = 'r',
                                  shape = shape_obs)

        mmap_flux_obs_unc = np.memmap(f'{self.meta.temp_input_dir}/flux_obs_unc.dat',
                                      dtype = 'float32', mode = 'r',
                                      shape = shape_obs)

        mmap_flux_model = np.memmap(f'{self.meta.temp_input_dir}/flux_model.dat',
                                    dtype = 'float32', mode = 'r',
                                    shape = shape_model)

        for index, _ in np.ndenumerate(mmap_flux_obs[0, ...]):
            print('\n' + str(index[0]))
            flux_obs_slice = mmap_flux_obs[:, index[0]]
            flux_obs_unc_slice = mmap_flux_obs_unc[:, index[0]]
            flux_model = mmap_flux_model

            if np.any(np.isnan(flux_obs_unc_slice) | np.isnan(flux_obs_slice)):
                print('nan')
                self.store_output(None, index[0])
            else:
                pp = self.execute_ppxf(flux_obs_slice, flux_obs_unc_slice,
                                       flux_model)
                self.store_output(pp, index[0])

    def execute_ppxf(self, flux_obs, flux_obs_unc, flux_model):
        C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

        frac = self.meta.wave_obs[1]/self.meta.wave_obs[0]    # Constant lambda fraction per pixel
        velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
        z = self.meta.z # redshift estimate

        dv = C*np.log(self.meta.wave_model[0]/self.meta.wave_obs[0])    # eq.(8) of Cappellari (2017)
        goodpixels = util.determine_goodpixels(np.log(self.meta.wave_obs),
                                               self.meta.limit_model, z)

        vel = C*np.log(1 + z)   # eq.(8) of Cappellari (2017)
        start = [vel, 200.]  # (km/s), starting guess for [V, sigma]

        t = clock()
        pp = ppxf(flux_model, flux_obs, flux_obs_unc, velscale, start,
                  goodpixels=goodpixels, plot=False, moments=4,
                  degree=12, vsyst=dv, clean=False, lam=self.meta.wave_obs,
                  quiet=True)

        # print("Formal errors:")
        # print("     dV    dsigma   dh3      dh4")
        # print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

        print('Elapsed time in PPXF: %.2f s' % (clock() - t))

        return pp
