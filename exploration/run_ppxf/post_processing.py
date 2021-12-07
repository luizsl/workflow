#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 17:52:07 2021

@author: Luiz
"""

import numpy as np
from astropy.io import fits

class PostProcessing:

    def __init__(self, ppxf_output, data):
        self.data = data
        self.ppxf_output = ppxf_output
        # self.result = np.memmap('output.dat', dtype = object, mode = 'r',
        #                         shape = (data.shape_obs[0] * data.shape_obs[1],))
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
        hdul.writeto(f'NGC613_1/{name}.fits', overwrite=True)

    def velocity(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].sol[0]
        # out = out.reshape(self.data.shape_obs)
        out = np.memmap(filename = 'velocity.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'velocity')

    def sigma(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].sol[1]
        # out = out.reshape(self.data.shape_obs)
        # self.save_fits(out, 'sigma')
        out = np.memmap(filename = 'sigma.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'sigma')

    def h3(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].sol[2]
        # out = out.reshape(self.data.shape_obs)
        # self.save_fits(out, 'h3')
        out = np.memmap(filename = 'h3.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'h3')

    def h4(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].sol[3]
        # out = out.reshape(self.data.shape_obs)
        # self.save_fits(out, 'h4')
        out = np.memmap(filename = 'h4.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'h4')

    def velocity_error(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].error[0]
        out = np.memmap(filename = 'velocity_error.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'velocity_error')

    def sigma_error(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].error[1]
        out = np.memmap(filename = 'sigma_error.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'sigma_error')

    def h3_error(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].error[2]
        out = np.memmap(filename = 'h3_error.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'h3_error')

    def h4_error(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].error[3]
        out = np.memmap(filename = 'h4_error.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'h4_error')

    def chi2(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out):
        #     out[index] = self.result[index].chi2
        out = np.memmap(filename = 'chi2.dat', dtype = float, mode='r',
                        shape = (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape(self.data.shape_obs)
        self.save_fits(out, 'chi2')

    def polyweights(self):
        # out = np.zeros((self.data.shape_obs[0] * self.data.shape_obs[1],) + (13,))
        # for index, _ in enumerate(out):
        #     out[index, :] = self.result[index].polyweights
        out = np.memmap(filename = 'polyweights.dat', dtype = float, mode='r',
                        shape = (13,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape((-1,) + self.data.shape_obs)
        self.save_fits(out, 'polyweights')

    def weights(self):
        # out = np.zeros((self.data.o_n_model,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out[0, ...]):
        #     out[:, index] = self.result[index].weights
        out = np.memmap(filename = 'apoly.dat', dtype = float, mode='r',
                        shape = (self.data.o_n_model,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape((-1,) + self.data.shape_obs)
        self.save_fits(out, 'weights')

    def apoly(self):
        # out = np.zeros((self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out[0, ...]):
        #     out[:, index] = self.result[index].apoly
        out = np.memmap(filename = 'apoly.dat', dtype = float, mode='r',
                        shape = (self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape((-1,) + self.data.shape_obs)
        self.save_fits(out, 'apoly')

    def bestfit(self):
        # out = np.zeros((self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        # for index, _ in enumerate(out[0, ...]):
        #     out[:, index] = self.result[index].bestfit

        out = np.memmap(filename = 'bestfit.dat', dtype = float, mode='r',
                        shape = (self.data.n_pixel_obs,) + (self.data.shape_obs[0] * self.data.shape_obs[1],))
        out = out.reshape((-1,) + self.data.shape_obs)
        self.save_fits(out, 'bestfit')
