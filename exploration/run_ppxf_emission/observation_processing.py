#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:47:28 2022

@author: Luiz
"""

import os
import tempfile
import warnings
import json
from abc import ABC, abstractmethod

import numpy as np
# import spectcube as sc
from astropy.io import fits
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# from normalize_median import normalize_band


class StellarContinuumFactory(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    
class Observation(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build_grid(self):
        pass

    def reshape(self):
        new_shape = (-1, np.array(self.meta['shape_obs']).prod())

        self.flux_grid = self.flux_grid.reshape(new_shape)
        self.flux_grid_unc = self.flux_grid_unc.reshape(new_shape)

    def convert_to_mmap(self):
        with tempfile.TemporaryFile() as f_flux, \
            tempfile.TemporaryFile() as f_flux_unc:

            mmap_flux_grid = np.memmap(
                f_flux,
                dtype='float32', mode='w+',
                shape= self.flux_grid.shape)
            mmap_flux_grid[:] = self.flux_grid[:]
            self.flux_grid = mmap_flux_grid
            self.flux_grid.flush()

            mmap_flux_grid_unc = np.memmap(
                f_flux_unc,
                dtype='float32', mode='w+',
                shape= self.flux_grid_unc.shape)
            mmap_flux_grid_unc[:] = self.flux_grid_unc[:]
            self.flux_grid_unc = mmap_flux_grid_unc
            self.flux_grid_unc.flush()

    def build_coordinate(self):
        "Adapted from Cappellari examples"
        # Create coordinates centred on the brightest spectrum

        # Hide warning of empty slices at edge of FoV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            flux = np.nanmean(self.flux_grid, 0)
            jm = np.nanargmax(flux)

        row, col = map(np.ravel, np.indices(self.meta['shape_obs']))

        pixsize = abs(self.header[1]['CD1_1'])*3600    # 0.2"
        x = (col - col[jm])*pixsize
        y = (row - row[jm])*pixsize

        self.meta['x_full'] = x
        self.meta['y_full'] = y
        self.meta['x_valid'] = self.meta['x_full'][self.meta['valid']]
        self.meta['y_valid'] = self.meta['y_full'][self.meta['valid']]

        self.meta['col'] = col + 1   # start counting from 1
        self.meta['row'] = row + 1

    def vorbin(self, target_sn=None):
        assert target_sn is not None
        signal = self.original_signal
        noise = self.original_noise

        pixelsize = abs(self.header[1]['CD1_1'])*3600

        out = voronoi_2d_binning(
           self.meta['x_valid'], self.meta['y_valid'],
            signal[self.meta['valid']], noise[self.meta['valid']],
           pixelsize=pixelsize, target_sn=target_sn, plot=0)
        bin_num, x_gen, y_gen, xbin, ybin, sn, nPixels, scale = out

        self.meta['bin_num'] = bin_num
        self.meta['x_gen'] = x_gen
        self.meta['y_gen'] =y_gen
        self.meta['xbin'] = xbin
        self.meta['ybin'] = ybin
        self.meta['sn'] = sn
        self.meta['nPixels'] = nPixels
        self.meta['scale'] = scale

        with tempfile.TemporaryFile() as temp_flux_file, \
            tempfile.TemporaryFile() as temp_flux_unc_file, \
            tempfile.TemporaryFile() as temp_flux_bin_file, \
            tempfile.TemporaryFile() as temp_flux_unc_bin_file:

            flux_valid = np.memmap(
                temp_flux_file,
                dtype='float32', mode='w+',
                shape=self.flux_grid[:,self.meta['valid']].shape)

            flux_unc_valid = np.memmap(
                temp_flux_unc_file,
                dtype='float32', mode='w+',
                shape=self.flux_grid_unc[:,self.meta['valid']].shape)

            flux_valid[:] =self.flux_grid[:,self.meta['valid']][:]
            flux_unc_valid[:] =self.flux_grid_unc[:,self.meta['valid']][:]

            n_pix = self.flux_grid.shape[0]

            flux_bin = np.memmap(
                temp_flux_bin_file,
                dtype='float32', mode='w+',
                shape= (n_pix, sn.size))

            flux_unc_bin = np.memmap(
                temp_flux_unc_bin_file,
                dtype='float32', mode='w+',
                shape= (n_pix, sn.size))

            for j in range(sn.size):
                w = bin_num == j
                flux_bin[:, j] = np.nansum(flux_valid[:, w], axis=1)
                flux_unc_bin[:, j] = np.sqrt(np.nansum(
                    flux_unc_valid[:, w]**2, axis=1))

            self.flux_grid = flux_bin
            self.flux_grid_unc = flux_unc_bin

    def compute_snr(self, snr_window:list=[-np.inf, np.inf]):
        # Hide warning of empty slices at edge of FoV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            w = ((self.meta['wave_obs'] > snr_window[0])
                 & (self.meta['wave_obs'] < snr_window[1]))
            signal = np.nanmean(self.flux_grid[w], axis=0)
            noise = np.nanmean(self.flux_grid_unc[w]**2, axis=0)
            noise= np.sqrt(noise)
            snr = signal / noise
            return snr, signal, noise

    def mask_spectral_axis(self, intervals=[-np.inf, np.inf], kind=None):
        assert kind in ['guess', 'fixed']
        kind = f'{kind}_goodpixels'
        wave = self.meta['wave_obs']
        mask = np.full_like(wave, False, dtype = bool)

        for lower, upper in intervals:
            temp = np.ma.masked_inside(wave, lower, upper)
            mask = np.logical_or(mask, temp.mask)

        # Invert mask to include in the fit
        mask = ~mask
        
        # Convert to goodpixels index
        goodpixels = np.arange(mask.size)[mask]

        self.meta[kind] = goodpixels

    def validate_spaxels(self, min_sn=None):
        sn_trigger = self.original_snr > min_sn
        finite = np.all(np.isfinite(self.flux_grid), axis = 0)
        valid = (sn_trigger & finite)
        self.meta['valid'] = np.asarray(valid, dtype=bool)


class Muse(Observation):
    def __init__(self, path_obs_flux, path_obs_flux_unc, wave):
        self.meta = {}
        assert os.path.isfile(path_obs_flux), f'{path_obs_flux} is NOT a file'
        assert os.path.isfile(path_obs_flux_unc), f'{path_obs_flux_unc} is NOT a file'
        self.meta['path_obs_flux'] = path_obs_flux
        self.meta['path_obs_flux_unc'] = path_obs_flux_unc

        self.meta['obs_sampling_type'] = 'ln'
        self.meta['wave_obs'] = wave

    def build_grid(self, min_valid_sn=0, snr_window=[-np.inf, np.inf]):
        with fits.open(self.meta['path_obs_flux'], memmap = True,
                       lazy_load_hdus = True) as hdul:
            self.flux_grid = np.array(hdul[0].data)

        with fits.open(self.meta['path_obs_flux_unc'], memmap = True,
                       lazy_load_hdus = True) as hdul:
            self.flux_grid_unc = np.array(hdul[0].data)
        assert self.flux_grid.shape == self.flux_grid_unc.shape     
        
        self.meta['shape_obs'] = self.flux_grid.shape[1:]
        
        if len(self.flux_grid.shape) == 3:
            self.reshape()
            
        self.original_snr, self.original_signal ,self.original_noise, = \
            self.compute_snr(snr_window=snr_window)

        self.validate_spaxels(min_sn=min_valid_sn)
        
        if len(self.flux_grid.shape) == 3:
            self.build_coordinate()


class StellarContinuum(StellarContinuumFactory):
    def __init__(self, path_stellar_continuum, wave):
        self.meta = {}
        assert os.path.isfile(path_stellar_continuum), f'{path_stellar_continuum} is NOT a file'
        self.meta['path_stellar_continuum'] = path_stellar_continuum
        
        self.meta['obs_sampling_type'] = 'ln'
        self.meta['wave'] = wave
        
    def build_grid(self):
        with fits.open(self.meta['path_stellar_continuum'], memmap = True,
                       lazy_load_hdus = True) as hdul:
            self.flux_grid = np.array(hdul[0].data)
        
        self.meta['shape_stellar'] = self.flux_grid.shape[1:]
        
        if len(self.flux_grid.shape) == 3:
            self.reshape('flux_grid')

        if len(self.flux_grid.shape) == 3:
            self.build_coordinate()

    def reshape(self, prop):
        new_shape = (-1, np.array(self.meta['shape_stellar']).prod())
        self.__setattr__(prop, self.__getattribute__(prop).reshape(new_shape))
        
    def convert_to_mmap(self):
        with tempfile.TemporaryFile() as f_stellar:

            mmap_flux_grid = np.memmap(
                f_stellar,
                dtype='float32', mode='w+',
                shape= self.flux_grid.shape)
            mmap_flux_grid[:] = self.flux_grid[:]
            self.flux_grid = mmap_flux_grid
            self.flux_grid.flush()

    def gather_kinematics(self, path_stellar_kinematics):
        self.meta['path_stellar_kinematics'] = path_stellar_kinematics
        with fits.open(self.meta['path_stellar_kinematics'], memmap = True,
                       lazy_load_hdus = True) as hdul:
            self.stellar_kinematics = np.array(hdul[0].data)
            
        if len(self.stellar_kinematics.shape) == 3:
            self.reshape('stellar_kinematics')
        
if __name__ == '__main__':
    path_obs_flux = '../../data_products/toy_100x100/MilesAgeMh/ppxf/galaxy.fits'
    path_obs_flux_unc = '../../data_products/toy_100x100/MilesAgeMh/ppxf/noise.fits'
    
    path_metadata = '../../data_products/toy_100x100/MilesAgeMh/ppxf/metadata.json'
    with open(path_metadata) as f:
        metadata = json.load(f)
    wave = np.array(metadata['obs']['wave_obs'])
    
    obs = Muse(path_obs_flux, path_obs_flux_unc, wave)
    
    obs.build_grid(min_valid_sn=3, snr_window=[5450, 5550])
    obs.convert_to_mmap()
    
    path_stellar_continuum = '../../data_products/toy_100x100/MilesAgeMh/ppxf/bestfit.fits'
    path_stellar_kinematics = '../../data_products/toy_100x100/MilesAgeMh/ppxf/sol.fits'
    stellar = StellarContinuum(path_stellar_continuum, wave)
    stellar.build_grid()
    stellar.convert_to_mmap()
    stellar.gather_kinematics(path_stellar_kinematics)