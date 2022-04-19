#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:06:12 2022

@author: Luiz
"""

import os
from abc import ABC, abstractmethod

import numpy as np
import spectcube as sc
from astropy.io import fits


class Observation(ABC):
    meta = {}

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build_grid(self):
        pass

    def resample(self, old_wave=None, new_wave = None):
        self.meta['new_obs_sampling'] = 'ln'

        if old_wave is None:
            old_wave = self.meta['o_wave_obs']

        if new_wave is None:
            self.meta['wave_obs'] = sc.util.fit_wave_interval(
                old_wave,
                old_sampling = 'linear',
                new_sampling = self.meta['new_obs_sampling'])
        else:
            self.meta['wave_obs'] = new_wave

        self.flux_obs, _, self.flux_obs_unc = \
            sc.resampling(flux = self.flux_obs,
                          old_wave = old_wave,
                          old_sampling_type = self.meta['o_obs_sampling_type'],
                          new_wave = self.meta['wave_obs'],
                          new_sampling_type = self.meta['new_obs_sampling'],
                          flux_err = self.flux_obs_unc)

        self.meta['limit_obs'] = self.meta['wave_obs'][[0,-1]]
        self.meta['n_pixel_obs'] = len(self.meta['wave_obs'])

    def normalize(self):
        self.meta['obs_norm_factor'] = np.nanmedian(self.flux_obs)
        self.flux_obs = self.flux_obs/self.meta['obs_norm_factor']
        self.flux_obs_unc = self.flux_obs_unc/self.meta['obs_norm_factor']

    def reshape(self):
        self.meta['shape_obs'] = self.flux_obs.shape[1:]

        new_shape = (-1, np.array(self.meta['shape_obs']).prod())

        self.flux_obs = self.flux_obs.reshape(new_shape)
        self.flux_obs_unc = self.flux_obs_unc.reshape(new_shape)

    @staticmethod
    def correct_z(wave, z=None):
        assert z is not None, 'z cannot be None'
        new_wave = wave/(1. + z)
        return new_wave

    def convert_to_mmap(self, path='.', filename_data='flux_obs.dat',
                        filename_unc='flux_obs_unc.dat', clean=True):

        self.meta['mmap_filepath_data'] = os.path.join(path, filename_data)
        self.meta['mmap_filepath_unc'] = os.path.join(path, filename_unc)

        self.mmap_flux_obs = np.memmap(self.meta['mmap_filepath_data'],
                                       dtype='float32', mode='w+',
                                       shape= self.flux_obs.shape)
        self.mmap_flux_obs[:] = self.flux_obs[:]


        self.mmap_flux_obs_unc = np.memmap(self.meta['mmap_filepath_unc'],
                                           dtype='float32', mode='w+',
                                           shape= self.flux_obs_unc.shape)
        self.mmap_flux_obs_unc[:] = self.flux_obs_unc[:]

        if clean is True:
            del(self.flux_obs, self.flux_obs_unc)

    def process(self):
        self.build_grid()
        self.resample()
        self.normalize()
        self.reshape()


class Muse(Observation):

    def __init__(self, path_obs, z=None):
        assert os.path.isfile(path_obs), f'{path_obs} is NOT a file'
        self.meta['path_obs'] = path_obs

        with fits.open(self.meta['path_obs'], lazy_load_hdus=True) as hdu:
            self.meta['o_first_wave_obs'] = np.double(hdu['DATA'].header['CRVAL3'])
            self.meta['o_step_wave_obs'] = np.double(hdu['DATA'].header['CD3_3'])
            self.meta['o_n_pixel_obs'] = hdu['DATA'].header['NAXIS3']

        self.meta['o_obs_sampling_type'] = 'linear'

        wave = sc.util.build_wave_array(
            [self.meta['o_first_wave_obs'], self.meta['o_step_wave_obs']],
            sampling_type = self.meta['o_obs_sampling_type'],
            size = self.meta['o_n_pixel_obs'])

        if z is None:
            self.meta['o_wave_obs'] = wave
        else:
            self.meta['o_wave_obs_not_rest'] = wave
            self.meta['o_wave_obs'] = self.correct_z(wave=wave, z=z)

        self.meta['o_obs_limit'] = self.meta['o_wave_obs'][[0,-1]]

    def build_grid(self):
        with fits.open(self.meta['path_obs'], memmap = True,
                       lazy_load_hdus = True, cache = False) as hdu:
            self.flux_obs = np.array(hdu['DATA'].data)
            del hdu['DATA'].data
            flux_obs_unc = np.array(hdu['STAT'].data)
            self.flux_obs_unc = np.sqrt(flux_obs_unc)
            del hdu['STAT'].data


#%%
if __name__ == '__main__':
    obs = Muse('../../data/toy_3x3.fits', 0.004951)
    obs.build_grid()
    obs.resample()
    obs.normalize()
    obs.reshape()
    # obs.convert_to_mmap()

    import matplotlib.pyplot as plt

    plt.plot(obs.meta['o_wave_obs'], obs.flux_obs[: , 0,0])
    plt.plot(obs.meta['o_wave_obs'], obs.flux_obs_unc[: , 0,0])

    plt.plot(obs.meta['wave_obs'], obs.flux_obs[: , 0,0])
    plt.plot(obs.meta['wave_obs'], obs.flux_obs_unc[: , 0,0])
