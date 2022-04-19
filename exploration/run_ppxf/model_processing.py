#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:20:05 2022

@author: Luiz
"""

import glob
import os
import re
from abc import ABC, abstractmethod

import numpy as np
import spectcube as sc
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import compute_muse_lsf as lsf
from convolve import convolve


# class Miles(ABC):
#     name_pattern = {}

#     def __init__(self):
#         pass

#     def method_name(self, *args, **kwargs):
#         pass


#
class XSL(ABC):
    name_pattern = {}
    meta = {}
    model_resolving_power = 10_000

    @abstractmethod
    def __init__(self, path_model_dir):
        pass

    @abstractmethod
    def build_name_grid(self):
        pass

    @abstractmethod
    def build_flux_grid(self):
        pass

    @abstractmethod
    def build_head_grid(self):
        pass

    def convolve(self, bound = [4600, 9400]):
        fwhm_obs_ang = lsf.equation_lsf(self.meta['o_wave_model'],
                                        bound[0],
                                        bound[1])

        fwhm_model_ang = self.meta['o_wave_model'] / self.model_resolving_power

        fwhm_dif = np.sqrt((fwhm_obs_ang**2 - fwhm_model_ang**2).clip(0))

        edges = sc.util._build_edges(self.meta['o_wave_model'], sampling_type='log')
        sigma = fwhm_dif / (2.355 * np.diff(edges)) # Sigma difference in pixels

        self.flux_grid = convolve(flux = self.flux_grid, sigma = sigma)

    def resample(self, wave=None):
        self.meta['new_model_sampling'] = 'ln'

        if wave is None:
            self.meta['wave_model'] = sc.util.fit_wave_interval(
                self.meta['o_wave_model'],
                old_sampling = 'log',
                new_sampling = self.meta['new_model_sampling'])
        else:
            self.meta['wave_model'] = wave

        self.flux_grid, _, _ = sc.resampling(
            flux = self.flux_grid,
            old_wave = self.meta['o_wave_model'],
            old_sampling_type = 'log',
            new_wave = self.meta['wave_model'],
            new_sampling_type = self.meta['new_model_sampling'])

        self.meta['n_pixel_model'] = self.flux_grid.shape[0]
        self.meta['limit_model'] = self.meta['wave_model'][[0,-1]]

    def normalize(self):
        self.meta['model_norm_factor'] = np.nanmedian(self.flux_grid)
        self.flux_grid = self.flux_grid/self.meta['model_norm_factor']

    def convert_to_mmap(self, path='.', filename='flux_model.dat', clean=True):
        self.meta['mmap_filepath'] = os.path.join(path, filename)
        self.mmap_flux_model = np.memmap(self.meta['mmap_filepath'],
                                         dtype='float32', mode='w+',
                                         shape= self.flux_grid.shape)
        self.mmap_flux_model[:] = self.flux_grid[:]
        self.mmap_flux_model.flush()

        if clean is True:
            del self.flux_grid

    def build(self):
        self.build_name_grid()
        self.build_flux_grid()
        self.build_head_grid()

    def process(self):
        self.build()
        self.convolve()
        self.resample()
        self.normalize()


class XSLAgeMh(XSL):
    name_pattern = 'XSL_SSP_logT{0:.1f}_MH{1:.1f}_Kroupa_PC.fits'
    parameter_pattern = 'logT[0-9]{1,2}\.[0-9]{1,2}_MH[+/-]?[0-9]{1,2}\.[0-9]{1,2}'

    def __init__(self, path_model_dir):
        assert os.path.isdir(path_model_dir), f'{path_model_dir} is NOT a directory'
        self.path_model_dir = path_model_dir

        self.flux_grid = None
        self.head_grid = None
        self.name_grid = None
        self.mask = None

        model_files = glob.glob(os.path.join(self.path_model_dir, '*'))
        self.meta['model_files'] = model_files

        with fits.open(self.meta['model_files'][0]) as hdu:
            # Adding 1 to convert from nm to A
            self.meta['o_first_wave_model'] = np.double(1 + hdu['PRIMARY'].header['CRVAL1'])
            self.meta['o_step_wave_model'] = np.double(hdu['PRIMARY'].header['CDELT1'])
            self.meta['o_n_pixel_model'] = hdu['PRIMARY'].header['NAXIS1']

        self.meta['o_n_model'] = len(model_files)

        self.meta['o_wave_model'] = sc.util.build_wave_array(
            [self.meta['o_first_wave_model'], self.meta['o_step_wave_model']],
            sampling_type = 'log',
            size = self.meta['o_n_pixel_model'])

        self.meta['o_limit_model'] = self.meta['o_wave_model'][[0,-1]]

        self.get_parameter_range()

    def age_mh(self, filename):
        '''Adapted from ppxf routines (Cappellari+17)
        '''
        par = re.findall(self.parameter_pattern,
                         filename)[0]

        age, metal = par.split("_")
        age = float(age.replace('logT', ''))
        # Trick to remove -0 retrieved from file name
        age += 0
        metal = float(metal.replace('MH', ''))
        return age, metal

    def get_parameter_range(self):
        age_range = []
        mh_range = []
        for name in self.meta['model_files']:
            age, mh = self.age_mh(name)
            age_range.append(age)
            mh_range.append(mh)

        age_range = np.unique(age_range)
        mh_range = np.unique(mh_range)

        age_range = np.sort(age_range)
        mh_range = np.sort(mh_range)

        self.age_range = age_range
        self.mh_range = mh_range

    def build_name_grid(self):
        with np.nditer([self.age_range, self.mh_range, None],
                       flags = ['buffered'],
                       op_axes = [[0, -1], [-1, 0], None],
                       op_dtypes = [None, None, 'U256']) as it:
            for _age, _mh, z in it:
                z[...] = self.name_pattern.format(_age, _mh)
            self.name_grid = it.operands[-1]

    def build_flux_grid(self):
        out_shape = self.read_model(
            self.path_model_dir, self.name_grid[0,0]).shape + self.name_grid.shape
        out = np.zeros(out_shape)

        with np.nditer([self.name_grid], flags = ['multi_index']) as it:
            for x in it:
                out[(...,) + it.multi_index] = \
                    self.read_model(self.path_model_dir, x[()])

            self.flux_grid = out

    def build_head_grid(self):
        with np.nditer([self.name_grid, None]) as it:
            for x, y in it:
                y[...] = dict(self.read_model(self.path_model_dir, x[()], 'header'))

            self.head_grid = it.operands[-1]

    def isfile(self, directory, file):
        filepath = os.path.join(directory, file)

        if os.path.isfile(filepath):
            pass
        else:
            filepath = re.sub('MH0.0', 'MH-0.0', filepath)
            if os.path.isfile(filepath):
                pass
            else:
                raise Exception
        return filepath

    def read_model(self, directory, file, ext='data'):
        try:
            filepath = self.isfile(directory, file)
        except:
            print(f'{file} not found')
        else:
            with fits.open(filepath, memmap=True, lazy_load_hdus=True) as hdul:
                if ext == 'data':
                    data = hdul[0].data
                elif ext == 'header':
                    data = hdul[0].header
            return data

    # def set_mask(self, new_mask=None):
    #     if new_mask is None:
    #         new_mask = [
    #             24*[1] + 2*[0],     #-2.2
    #             23*[1] + 3*[0],     #-2
    #             23*[1] + 3*[0],     #-1.8
    #             13*[1] + 13*[0],    #-1.6
    #             13*[1] + 13*[0],    #-1.4
    #             10*[1] + 16*[0],    #-1.2
    #             9*[1] + 17*[0],     #-1.0
    #             9*[1] + 17*[0],     #-0.8
    #             8*[1] + 18*[0],     #-0.6
    #             8*[1] + 18*[0],     #-0.4
    #             5*[1] + 21*[0],     #-0.2
    #             3*[1] + 23*[0],     #0
    #             0*[1] + 26*[0],     #0.2
    #             ]
    #         new_mask = np.array(new_mask, dtype=bool)
    #         new_mask = new_mask.T

    #     self.mask = new_mask

    # def mask_grid(self):
    #     assert self.flux_grid is not None
    #     assert self.head_grid is not None
    #     assert self.name_grid is not None
    #     assert self.mask is not None

    #     self.name_grid = np.ma.masked_where(self.mask, self.name_grid)
    #     self.head_grid = np.ma.masked_where(self.mask, self.head_grid)
    #     self.flux_grid = np.ma.masked_where(
    #         np.broadcast_to(self.mask, self.flux_grid.shape),
    #         self.flux_grid)

    def plot(self, weigths=None):
        x_age, y_mh = np.meshgrid(self.age_range, self.mh_range)

        if weigths is None:
            weigths = np.zeros_like(x_age)

        fig, ax = plt.subplots()

        grid = ax.pcolormesh(x_age, y_mh, weigths.T)
        points = ax.scatter(x_age, y_mh,
                            marker ='.', color ='white', s = 1)
        cb = plt.colorbar(grid)

        ax.set_xlabel('$\log_{10}$ Age (yr)')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(axis='x', rotation=90)

        ax.set_ylabel('[Fe/H]')
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        plt.tight_layout()
        fig.show()

    def remove_param(self, param, values = []):
        attribute = getattr(self, param)
        _, _, index = np.intersect1d(values, attribute, return_indices=True)
        mask = np.zeros_like(attribute)
        mask[index] = 1
        masked_attribute = np.ma.masked_where(mask, attribute)
        masked_attribute = masked_attribute.compressed()
        setattr(self, param, masked_attribute)
        setattr(self, param + '_mask', mask)


if __name__ == '__main__':
    model = XSLAgeMh('../../data/models/XSL_SSP_PC_Kroupa/Kroupa')
    model.remove_param('age_range', values = [10.2])
    model.remove_param('mh_range', values = [-0.1, 0.1])
    model.build()

    # model.convert_to_mmap()

    model.plot()

