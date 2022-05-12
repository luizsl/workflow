#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:20:05 2022

@author: Luiz
"""

import glob
import os
import re
import tempfile
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import spectcube as sc
from astropy.io import fits

import compute_muse_lsf as lsf
from convolve import convolve


class AbstractModel(ABC):

    @abstractmethod    
    def __init__(self):
        pass
        
    @abstractmethod    
    def get_model(self):
        pass


class Model(AbstractModel):

    def __init__(self):
        pass
    
    def get_model(self, name):
        factories = {
            "XSLAgeMh" : XSLAgeMh(),
            "MilesAgeMh" : MilesAgeMh()
        }
        
        return factories[name]
    

class AbstractFactoryModel(ABC):
    name_pattern = {}
    meta = {}
    model_resolving_power = 10_000

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def load(self):
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

        if self.fwhm_model_ang is None:
            self.fwhm_model_ang = self.meta['o_wave_model'] / self.model_resolving_power

        fwhm_dif = np.sqrt((fwhm_obs_ang**2 - self.fwhm_model_ang**2).clip(0))

        edges = sc.util._build_edges(
            self.meta['o_wave_model'],
            sampling_type=self.meta['o_sampling_type'])
        sigma = fwhm_dif / (2.355 * np.diff(edges)) # Sigma difference in pixels

        self.flux_grid = convolve(flux = self.flux_grid, sigma = sigma)

    def resample(self, wave=None):
        self.meta['new_model_sampling'] = 'ln'

        if wave is None:
            self.meta['wave_model'] = sc.util.fit_wave_interval(
                self.meta['o_wave_model'],
                old_sampling = self.meta['o_sampling_type'],
                new_sampling = self.meta['new_model_sampling'])
        else:
            self.meta['wave_model'] = wave

        self.flux_grid, _, _ = sc.resampling(
            flux = self.flux_grid,
            old_wave = self.meta['o_wave_model'],
            old_sampling_type = self.meta['o_sampling_type'],
            new_wave = self.meta['wave_model'],
            new_sampling_type = self.meta['new_model_sampling'])

        self.meta['n_pixel_model'] = self.flux_grid.shape[0]
        self.meta['limit_model'] = self.meta['wave_model'][[0,-1]]

    def normalize(self, limits=None):
        if 'wave_model' in self.meta:
            wave = self.meta['wave_model']
        elif 'o_wave_model' in self.meta:
            wave = self.meta['o_wave_model']
        else:
            raise Exception

        if limits is not None:
            band = (limits[0] < wave) & (wave < limits[-1])
        else:
            band = self.flux_grid > 0

        self.meta['model_norm_factor'] = np.nanmedian(self.flux_grid[band])
        self.flux_grid = self.flux_grid/self.meta['model_norm_factor']

    def reshape(self):
        assert self.flux_grid.ndim > 2

        reg_dim = self.flux_grid.shape[1:]

        npix = self.flux_grid.shape[0]
        self.flux_grid = self.flux_grid.reshape(npix, -1)

        self.meta['reg_dim'] = reg_dim

    def convert_to_mmap(self):
        with tempfile.TemporaryFile() as f:
            flux_grid = np.memmap(
                f,
                dtype='float32', mode='w+',
                shape= self.flux_grid.shape)
            flux_grid[:] = self.flux_grid[:]
            self.flux_grid = flux_grid
            self.flux_grid.flush()

    def build(self):
        self.build_name_grid()
        self.build_flux_grid()
        self.build_head_grid()

    def process(self):
        self.convolve()
        self.resample()
        self.normalize()


class XSLAgeMh(AbstractFactoryModel):
    name_pattern = 'XSL_SSP_logT{0:.1f}_MH{1:.1f}_Kroupa_PC.fits'
    parameter_pattern = 'logT[0-9]{1,2}\.[0-9]{1,2}_MH[+/-]?[0-9]{1,2}\.[0-9]{1,2}'
    fwhm_model_ang = None

    def __init__(self):
        pass
    
    def load(self, path_model_dir):
        assert os.path.isdir(path_model_dir), f'{path_model_dir} is NOT a directory'
        self.path_model_dir = path_model_dir
        self.meta['o_sampling_type'] = 'log'

        self.flux_grid = None
        self.head_grid = None
        self.name_grid = None
        self.mask = None

        model_files = glob.glob(os.path.join(self.path_model_dir, '*.fits'))
        self.meta['model_files'] = model_files

        with fits.open(self.meta['model_files'][0]) as hdu:
            # Adding 1 to convert from nm to A
            self.meta['o_first_wave_model'] = np.double(1 + hdu['PRIMARY'].header['CRVAL1'])
            self.meta['o_step_wave_model'] = np.double(hdu['PRIMARY'].header['CDELT1'])
            self.meta['o_n_pixel_model'] = hdu['PRIMARY'].header['NAXIS1']

        self.meta['o_n_model'] = len(model_files)

        self.meta['o_wave_model'] = sc.util.build_wave_array(
            [self.meta['o_first_wave_model'], self.meta['o_step_wave_model']],
            sampling_type = self.meta['o_sampling_type'],
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

    def plot(self, weigths=None):
        x_age, y_mh = np.meshgrid(self.age_range, self.mh_range)

        if weigths is None:
            weigths = np.zeros_like(x_age.T)

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


class MilesAgeMh(AbstractFactoryModel):
    name_pattern = 'Eun1.30Z{:+05.2f}T{:07.4f}_iPp0.00_baseFe_linear_FWHM_variable.fits'
    parameter_pattern = '[m/p][0-9]\.[0-9]{2}T[0-9]{2}\.[0-9]{4}'
    fwhm_model_ang = 2.51

    def __init__(self):
        pass
    
    def load(self, path_model_dir):
        assert os.path.isdir(path_model_dir), f'{path_model_dir} is NOT a directory'
        self.path_model_dir = path_model_dir
        self.meta['o_sampling_type'] = 'linear'

        self.flux_grid = None
        self.head_grid = None
        self.name_grid = None
        self.mask = None

        model_files = glob.glob(os.path.join(self.path_model_dir, '*.fits'))
        self.meta['model_files'] = model_files

        with fits.open(self.meta['model_files'][0]) as hdu:
            self.meta['o_first_wave_model'] = np.double(hdu['PRIMARY'].header['CRVAL1'])
            self.meta['o_step_wave_model'] = np.double(hdu['PRIMARY'].header['CDELT1'])
            self.meta['o_n_pixel_model'] = hdu['PRIMARY'].header['NAXIS1']

        self.meta['o_n_model'] = len(model_files)

        self.meta['o_wave_model'] = sc.util.build_wave_array(
            [self.meta['o_first_wave_model'], self.meta['o_step_wave_model']],
            sampling_type = self.meta['o_sampling_type'],
            size = self.meta['o_n_pixel_model'])

        self.meta['o_limit_model'] = self.meta['o_wave_model'][[0,-1]]

        self.get_parameter_range()

    def age_mh(self, filename):
        '''Adapted from ppxf routines (Cappellari+17)
        '''
        par = re.findall(self.parameter_pattern,
                         filename)[0]
        metal, age = par.split("T")
        age = float(age)
        metal = metal.replace('m', '-')
        metal = float(metal.replace('p', '+'))
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
                a = self.parameter_pattern.replace(
                    'T[0-9]{2}\\.[0-9]{4}', f'T{_age:07.4f}')
                a = a.replace('[m/p][0-9]\\.[0-9]{2}', f'{_mh:+0.2f}')
                a = a.replace('+', 'p')
                a = a.replace('-', 'm')
                
                b = glob.glob(os.path.join(self.path_model_dir, '*') + a + '*')
                assert len(b) == 1
                name = os.path.split(b[0])[-1]
                z[...] = name
            self.name_grid = it.operands[-1]
        self.meta['age_range'] = self.age_range 
        self.meta['mh_range'] = self.mh_range
        
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

    def plot(self, weigths=None):
        x_age, y_mh = np.meshgrid(np.log10(self.age_range*1e9).round(2),
                                  self.mh_range)

        if weigths is None:
            weigths = np.zeros_like(x_age.T)

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

#%%
if __name__ == '__main__':
    
    model = MilesAgeMh()
    # model.load('../../data/models/tmpWzZ2t1')
    model.load('../../data/models/miles_Padova00_UN_baseFe_v10.0')
    # model.build()
    model.build_name_grid()
    model.build_flux_grid()
    model.build_head_grid()
    model.reshape()
    model.convolve()
    model.resample()
    model.normalize()
    

    # log_step = np.log(prep.obs.meta['wave_obs'][1]/prep.obs.meta['wave_obs'][0])
    # wave = np.exp(np.arange(
    #     np.log(t.meta['o_wave_model'][0]),
    #     np.log(t.meta['o_wave_model'][-1]),
    #     log_step))

    # model = XSLAgeMh('../../data/models/XSL_SSP_PC_Kroupa/Kroupa')
    # # model.remove_param('age_range', values = [10.2])
    # # model.remove_param('mh_range', values = [-0.1, 0.1])
    # model.build()
    # model.reshape()
    # model.convolve()
    # model.resample()
    # model.normalize()

    # model.convert_to_mmap()
    # # model.plot()

    # t = MilesAgeMh('/home/chess-lin/miniforge3/lib/python3.9/site-packages/ppxf/miles_models')
    # # t.remove_param('age_range', values = [15.8489])
    # t.build()
    # t.convolve()



    # t.resample(wave)
    # t.normalize()

    # import ppxf.miles_util as lib
    # t1 = lib.miles('/home/chess-lin/miniforge3/lib/python3.9/site-packages/ppxf/miles_models/*.fits',
    #                 velscale = 55.16655145380999,
    #                 FWHM_gal=2.7
    #                 )
