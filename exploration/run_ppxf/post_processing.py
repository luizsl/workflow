#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 17:52:07 2021

@author: Luiz
"""

import _pickle as pickle
import numpy as np
from astropy.io import fits


class PostProcessing:

    def __init__(self, metadata_path):
        with open(metadata_path, 'rb') as inp:
            self.meta = pickle.load(inp)
        self.apoly()
        self.bestfit()
        self.chi2()
        self.h3()
        self.h3_error()
        self.h4()
        self.h4_error()
        self.polyweights()
        self.sigma()
        self.sigma_error()
        self.velocity()
        self.velocity_error()
        self.weights()

    def save_fits(self, data_param, name):
        hdu = fits.PrimaryHDU(data=data_param)
        hdul = fits.HDUList([hdu])
        hdul.writeto(f'{self.meta.output_run_ppxf}/{name}.fits', overwrite=True)

    def velocity(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/velocity.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'velocity')

    def sigma(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/sigma.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'sigma')

    def h3(self):

        out = np.memmap(filename = f'{self.meta.temp_output_dir}/h3.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'h3')

    def h4(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/h4.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'h4')

    def velocity_error(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/velocity_error.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'velocity_error')

    def sigma_error(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/sigma_error.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'sigma_error')

    def h3_error(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/h3_error.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'h3_error')

    def h4_error(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/h4_error.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'h4_error')

    def chi2(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/chi2.dat', dtype = float, mode='r',
                        shape = (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape(self.meta.shape_obs)
        self.save_fits(out, 'chi2')

    def polyweights(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/polyweights.dat', dtype = float, mode='r',
                        shape = (13,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape((-1,) + self.meta.shape_obs)
        self.save_fits(out, 'polyweights')

    def weights(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/apoly.dat', dtype = float, mode='r',
                        shape = (self.meta.o_n_model,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape((-1,) + self.meta.shape_obs)
        self.save_fits(out, 'weights')

    def apoly(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/apoly.dat', dtype = float, mode='r',
                        shape = (self.meta.n_pixel_obs,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape((-1,) + self.meta.shape_obs)
        self.save_fits(out, 'apoly')

    def bestfit(self):
        out = np.memmap(filename = f'{self.meta.temp_output_dir}/bestfit.dat', dtype = float, mode='r',
                        shape = (self.meta.n_pixel_obs,) + (self.meta.shape_obs[0] * self.meta.shape_obs[1],))
        out = out.reshape((-1,) + self.meta.shape_obs)
        self.save_fits(out, 'bestfit')
