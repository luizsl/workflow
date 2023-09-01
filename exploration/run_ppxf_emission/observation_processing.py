#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:47:28 2022

@author: Luiz
"""

import json
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import spectcube as sc
from astropy.io import fits
from vorbin.voronoi_2d_binning import voronoi_2d_binning

from normalize_median import normalize_band


class StellarContinuumFactory(ABC):
    @abstractmethod
    def __init__(self):
        pass


class Observation(ABC):
    meta = {}
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build_grid(self):
        pass

    def resample(self, new_wave=None):
        old_wave = self.meta['wave_obs']

        if new_wave is None:
            self.meta['wave_obs'] = sc.util.fit_wave_interval(
                old_wave,
                old_sampling = self.meta['o_obs_sampling_type'],
                new_sampling = 'ln')
        else:
            self.meta['wave_obs'] = new_wave

        self.meta['new_obs_sampling'] = 'ln'

        self.flux_grid, _, self.flux_grid_unc = \
            sc.resampling(
                flux = self.flux_grid,
                old_wave = old_wave,
                old_sampling_type = self.meta['o_obs_sampling_type'],
                new_wave = self.meta['wave_obs'],
                new_sampling_type = self.meta['new_obs_sampling'],
                flux_err = self.flux_grid_unc)

        self.meta['limit_obs'] = self.meta['wave_obs'][[0,-1]]
        self.meta['n_pixel_obs'] = len(self.meta['wave_obs'])

    def normalize(self, **kwargs):
        if 'wave_obs' in self.meta:
            wave = self.meta['wave_obs']
        else:
            raise Exception
        assert len(wave) == self.flux_grid.shape[0]

        self.flux_grid, self.meta['obs_norm_factor'] = normalize_band(
            self.flux_grid, wave, **kwargs)

        self.flux_grid_unc = self.flux_grid_unc/self.meta['obs_norm_factor']

    def reshape(self):
        new_shape = (-1, np.array(self.meta['shape_obs']).prod())

        self.flux_grid = self.flux_grid.reshape(new_shape)
        self.flux_grid_unc = self.flux_grid_unc.reshape(new_shape)

    @staticmethod
    def correct_z(wave, z=None):
        assert z is not None, 'z cannot be None'
        new_wave = wave/(1. + z)
        return new_wave

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

    def vorbin(self, target_sn=None, sn_func=None):
        assert target_sn is not None

        signal = self.original_signal
        noise = self.original_noise

        pixelsize = abs(self.header[1]['CD1_1']) * 3600

        if sn_func is None:
            covar_a = 0
            covar_b = 1
            sn_func = partial(
                sn_function, covar_sn_a=covar_a, covar_sn_b=covar_b)

        try:
            bin_num, x_gen, y_gen, xbin, ybin, sn, nPixels, scale = \
                voronoi_2d_binning(
                    x=self.meta['x_valid'], y=self.meta['y_valid'],
                    signal=signal[self.meta['valid']],
                    noise=noise[self.meta['valid']],
                    sn_func=sn_func, pixelsize=pixelsize, target_sn=target_sn,
                    plot=0)

        except Exception as exp:
            if "All pixels have enough S/N" in str(exp):
                print(str(exp))
                n_spec = self.flux_grid[:, self.meta['valid']].shape[-1]

                bin_num = np.arange(n_spec)
                x_gen = np.zeros((n_spec), float)
                y_gen = np.zeros((n_spec), float)
                xbin = self.meta['x_valid']
                ybin = self.meta['y_valid']
                sn = self.meta['original_snr'][self.meta['valid']]
                nPixels = np.ones((n_spec,), float)
                scale = np.zeros((n_spec), float)
            else:
                raise exp

        self.meta['bin_num'] = bin_num
        self.meta['x_gen'] = x_gen
        self.meta['y_gen'] = y_gen
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
                # Average
                w = bin_num == j
                covar_a = sn_func.keywords['covar_sn_a']
                covar_b = sn_func.keywords['covar_sn_b']
                corr_factor = (1 + covar_a * np.log10(nPixels[j]) ** covar_b)

                flux_bin[:, j] = (1/w.sum())*np.nansum(flux_valid[:, w], 1)
                flux_unc_bin[:, j] = \
                    np.sqrt(np.nansum(flux_unc_valid[:, w]**2, 1))
                flux_unc_bin[:, j] = (1/w.sum())*flux_unc_bin[:, j]*corr_factor

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

    def trim_spectral_axis(self, lower=None, upper=None):
        mask = np.ma.masked_outside(self.meta['wave_obs'], lower, upper)
        if mask.mask.size == 1:
            pass
        else:
            self.meta['wave_obs'] = self.meta['wave_obs'][~mask.mask]
            self.flux_grid = self.flux_grid[~mask.mask, ...]
            self.flux_grid_unc = self.flux_grid_unc[~mask.mask, ...]
            self.meta['limit_obs'] = self.meta['wave_obs'][[0, -1]]

    def validate_spaxels(self, min_sn=None):
        sn_trigger = self.original_snr > min_sn
        finite = np.all(np.isfinite(self.flux_grid), axis = 0)
        valid = (sn_trigger & finite)
        self.meta['valid'] = np.asarray(valid, dtype=bool)


class Muse(Observation):
    def __init__(self, path_obs, z=None):
        assert os.path.isfile(path_obs), f'{path_obs} is NOT a file'
        self.meta['path_obs'] = path_obs

        with fits.open(self.meta['path_obs'], lazy_load_hdus=True) as hdul:
            self.meta['o_first_wave_obs'] = np.double(hdul['DATA'].header['CRVAL3'])
            self.meta['o_step_wave_obs'] = np.double(hdul['DATA'].header['CD3_3'])
            try:
                self.meta['o_n_pixel_obs'] = hdul['DATA'].header['NAXIS3']
            except:
                if hdul['DATA'].header['NAXIS'] == 1:
                    self.meta['o_n_pixel_obs'] = hdul['DATA'].header['NAXIS1']

        self.meta['o_obs_sampling_type'] = 'linear'

        wave = sc.util.build_wave_array(
            [self.meta['o_first_wave_obs'], self.meta['o_step_wave_obs']],
            sampling_type = self.meta['o_obs_sampling_type'],
            size = self.meta['o_n_pixel_obs'])

        if z is None:
            self.meta['wave_obs'] = wave
        else:
            self.meta['wave_obs'] = self.correct_z(wave=wave, z=z)
        self.meta['limit_obs'] = self.meta['wave_obs'][[0, -1]]

    def build_grid(self, min_valid_sn=0, snr_window=[-np.inf, np.inf]):
        with fits.open(self.meta['path_obs'], memmap = True,
                       lazy_load_hdus = True, cache = False) as hdul:
            # NOTE: A considerable number of the spaxel has a NaN at the last
            # pixel. To avoid further issues, when that occurs I'm assigning
            # to the last pixel the same value of the nearest one. <>
            where_nan = ~np.isfinite(hdul['DATA'].data[-1, ...])
            hdul['DATA'].data[-1, ...][where_nan] = \
                hdul['DATA'].data[-2, ...][where_nan]
            self.flux_grid = np.array(hdul['DATA'].data)
            del hdul['DATA'].data

            # NOTE: See previous note above
            # (A note that points to a note) <>
            hdul['STAT'].data[-1, ...][where_nan] = \
                hdul['STAT'].data[-2, ...][where_nan]
            flux_grid_unc = np.array(hdul['STAT'].data)
            self.flux_grid_unc = np.sqrt(flux_grid_unc)
            del hdul['STAT'].data

            header = []
            h = {card[0]: card[1] for card in hdul['PRIMARY'].header._cards}
            header.append(h)
            h = {card[0]: card[1] for card in hdul['DATA'].header._cards}
            header.append(h)
            h = {card[0]: card[1] for card in hdul['STAT'].header._cards}
            header.append(h)
            self.header = header

            self.meta['shape_obs'] = self.flux_grid.shape[1:]

            if hdul['DATA'].header['NAXIS'] == 3:
                self.reshape()

            self.original_snr, self.original_signal ,self.original_noise, = \
                self.compute_snr(snr_window=snr_window)

            self.validate_spaxels(min_sn=min_valid_sn)

            if hdul['DATA'].header['NAXIS'] == 3:
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

    def apply_binning(self, bin_num, npixels, valid):
        with tempfile.TemporaryFile() as temp_flux_file, \
            tempfile.TemporaryFile() as temp_flux_bin_file:

            flux_valid = np.memmap(
                temp_flux_file,
                dtype='float32', mode='w+',
                shape=self.flux_grid[:, valid].shape)

            flux_valid[:] = self.flux_grid[:, valid]

            n_pix = self.flux_grid.shape[0]

            flux_bin = np.memmap(
                temp_flux_bin_file,
                dtype='float32', mode='w+',
                shape=(n_pix, npixels.size))

            for j in range(npixels.size):
                w = bin_num == j
                flux_bin[:, j] = np.nansum(flux_valid[:, w], axis=1)

            self.flux_grid = flux_bin

    def normalize(self, **kwargs):
        if 'wave' in self.meta:
            wave = self.meta['wave']
        else:
            raise Exception
        assert len(wave) == self.flux_grid.shape[0]

        self.flux_grid, self.meta['obs_norm_factor'] = normalize_band(
            self.flux_grid, wave, **kwargs)

    def rescale(self, scale_template, wave=None, limits=[-np.inf, np.inf]):
        band = (limits[0] < wave) & (wave < limits[-1])
        scale = np.mean(scale_template[band], axis=0)
        mean_stellar =  np.mean(self.flux_grid[band], axis=0)
        scaled = self.flux_grid *  scale/mean_stellar

        return scaled


def sn_function(index, signal=None, noise=None, covar_sn_a=0, covar_sn_b=1):
    sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))
    sn = sn / (1 + covar_sn_a * (np.log10(index.size))**covar_sn_b)
    return sn


if __name__ == '__main__':
    path_obs = '../../data/fov_sample_1_3.fits'

    path_metadata = '../../data_products/fov_sample_1_3/XSLAgeMh/ppxf/metadata.json'
    with open(path_metadata) as f:
        metadata = json.load(f)
    wave = np.array(metadata['obs']['wave_obs'])

    obs = Muse(path_obs, z=0.0)

    obs.build_grid(min_valid_sn=3, snr_window=[5450, 5550])
    obs.vorbin(target_sn=200)
    obs.normalize(limits=[5450, 5550])
    obs.convert_to_mmap()

    path_stellar_continuum = '../../data_products/fov_sample_1_3/XSLAgeMh/ppxf/bestfit.fits'
    stellar = StellarContinuum(path_stellar_continuum, wave)
    stellar.build_grid()
    stellar.convert_to_mmap()

    valid = obs.meta['valid']
    npixels = obs.meta['nPixels']
    bin_num = obs.meta['bin_num']

    stellar.apply_binning(bin_num, npixels, valid)
    stellar.normalize(limits=[5450, 5550])

