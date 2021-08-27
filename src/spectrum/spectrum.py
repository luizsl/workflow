#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:57:24 2021

@author: chess-lin
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from ppxf.ppxf_util import gaussian_filter1d
import spectcube as sc


class Spectrum:
    def __init__(self, flux, wave, medium = None, sampling_type = None,
                 flux_unc = None, npix = None):
        if npix is None:
            self.flux = np.array(flux)
        else:
            self.flux = np.array(flux)[:npix]
            
        if flux_unc is None:
            self.flux_unc = np.full_like(self.flux, np.nan)
        else:
            self.flux_unc = np.array(flux_unc)
            
        self.medium = self._valid_key(medium,
                                      valid = ['air', 'vacuum'])    
        self.sampling_type = self._valid_key(sampling_type,
                                             valid = ['linear', 'log', 'ln'])
        self.wave = self._build_spec_wave(wave)

    @staticmethod
    def _valid_key(item, valid):
        if item is None:
            return item
        elif item.lower() in valid:
            return item.lower()
        else:
            raise ValueError(f"'{item}' is invalid. It should be in {valid}")

    def vac_to_air(self):
        #to do: Ciddor
        '''
        Convert wavelength from vacuun to air, using the recipe: Donald Morton
        2000, ApJ. Suppl., 130, 403
        https://ui.adsabs.harvard.edu/abs/2000ApJS..130..403M/abstract
        '''

        assert self.medium == 'vacuum', 'Not in vacuum'

        sig = (1.0e4/self.wave)**2.0
        factor = (1.0 + 8.34254e-5 + (2.406147e-2/(130.0-sig))
                  + (1.5998e-4/(38.9-sig)))

        self.wave /= factor
        self.medium = 'air'

    def _build_spec_wave(self, wave, npix = None):
        first_wave, step = wave

        if npix is None:
            npix = len(self.flux)

        if self.sampling_type == 'linear':
            wave = first_wave + step*np.arange(npix, dtype = np.double)
        elif self.sampling_type == 'log':
            wave = 10.0**(first_wave + step*np.arange(npix, dtype = np.double))
        elif self.sampling_type == 'ln':
            wave = np.e**(first_wave + step*np.arange(npix, dtype = np.double))
        return wave

    def convolve(self, sigma):
        self.flux = gaussian_filter1d(self.flux, sigma)

    def trim_w_mask(self, mask):
        self.wave = self.wave[mask]
        self.flux = self.flux[mask]
        self.flux_unc = self.flux_unc[mask]

    def resampling(self, new_wave, new_sampling_type):
        res_data, new_wave, res_err = \
            sc.resampling(flux = self.flux,
                          old_wave = self.wave,
                          old_sampling_type = self.sampling_type,
                          new_wave = new_wave,
                          new_sampling_type = new_sampling_type,
                          flux_err= self.flux_unc)
            
        self.sampling_type = new_sampling_type
        self.flux = res_data
        self.wave = new_wave
        
        if res_err is None:
            self.flux_unc  = np.full_like(res_data, None)
        else:
            self.flux_unc = res_err

    def normalize_median(self):
        self.flux = self.flux/np.median(self.flux)

    def plot(self):
        plt.step(self.wave, self.flux, where = 'mid')

    def __repr__(self):
        return repr(self.flux)

    def __str__(self):
        return repr(self.flux)
