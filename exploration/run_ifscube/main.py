#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:02:39 2022

@author: Luiz
"""

import json
import os
import sys
import shutil
from configparser import ConfigParser, ExtendedInterpolation

import numpy as np
import spectcube as sc
from astropy.io import fits


class IFSCubeInput:
    def __init__(self, initialization_file):
        self.initialization_file = initialization_file
        self.read_conf()
        self.meta = self.read_meta()
        self.create_output_dir()
        self.get_headers()
        self.hdul_out = fits.HDUList()
        self.normalization_factor = self.recover_norm_factor()
        self.velocity_map = self.recover_kinematics()
        self.rest_wave_grid = self.build_rest_wave_grid(self.meta['obs']['wave_obs'],
                                                        self.velocity_map)
        self.commom_wave = self.find_commom_range(self.rest_wave_grid,
                                                  self.velocity_map)
        self.add_muse_data()
        self.add_bestfit_ssp()
        self.save_cube()

    def save_cube(self):
        if self.count is None:
            self.meta['cube_path'] = os.path.join(
                self.meta['conf']['output_run_ifscube'], 'input_cube.fits')
        else:
            self.meta['cube_path'] = os.path.join(
                self.meta['conf']['output_run_ifscube'],
                f'input_cube_{self.count}.fits')

        self.hdul_out.writeto(self.meta['cube_path'], overwrite = False)

    def create_output_dir(self):
        # # later add a function to write a more complex _sec_dir name
        # sec_dir_ = f'{dir_}miles'
        dir_ = os.path.join(self.meta['conf']['output_run'], 'ifscube')

        # create unique name
        if os.path.isdir(dir_) is False:
            self.meta['conf']['output_run_ifscube'] = dir_
            self.count = None
        else:
            count = 1
            while os.path.isdir(f'{dir_}_{str(count)}') is True:
                count += 1
            self.count = count
            self.meta['conf']['output_run_ifscube'] = f'{dir_}_{str(count)}'

        os.makedirs(self.meta['conf']['output_run_ifscube'], exist_ok=True)

    def read_conf(self):
        configur = ConfigParser(interpolation=ExtendedInterpolation())
        configur.read(self.initialization_file)

        self.ppxf_data_products = configur.get('resources', 'ppxf_results')
        self.ifscube_conf = configur.get('ifscube', 'conf_path')

    def read_meta(self):
        metadata_path = os.path.join(self.ppxf_data_products, "metadata.json")
        with open(metadata_path) as f:
            meta = json.load(f)
        return meta

    def get_headers(self):
        with fits.open(self.meta['obs']['path_obs']) as hdul:
            self.header_primary = hdul[0].header
            self.header_sci = hdul[1].header
            self.header_err = hdul[2].header

    def recover_parameter(self, parameter):
        parameter_path = os.path.join(
            self.meta['conf']['output_run_ppxf'],f'{parameter}.fits')

        with fits.open(parameter_path) as hdul:
            parameter = hdul['PRIMARY'].data
        return parameter
    
    def recover_kinematics(self, ext=0):
        parameter_path = os.path.join(
            self.meta['conf']['output_run_ppxf'], 'sol.fits')

        with fits.open(parameter_path) as hdul:
            parameter = hdul['PRIMARY'].data[ext]
        return parameter

    def recover_norm_factor(self):
        # normalization_factor_path = os.path.join(
        #     self.meta['meta']['output_run_ppxf'],'normalization_factor.fits')

        # with fits.open(normalization_factor_path) as hdul:
        #     normalization_factor = hdul['PRIMARY'].data
        normalization_factor = self.meta['model']['model_norm_factor']
        return normalization_factor

    def add_muse_data(self):
        global flux, old_wave, wave
        primary_hdu = fits.PrimaryHDU(header = self.header_primary)
        self.hdul_out.append(primary_hdu)

        with fits.open(self.meta['obs']['path_obs']) as hdul:
            where_nan = ~np.isfinite(hdul['DATA'].data[-1, ...])
            
            hdul['DATA'].data[-1, ...][where_nan] = \
                hdul['DATA'].data[-2, ...][where_nan]
            hdul['STAT'].data[-1, ...][where_nan] = \
                hdul['STAT'].data[-2, ...][where_nan]
            
            flux = np.single(hdul['DATA'].data)
            uncertainty = np.single(hdul['STAT'].data)
            
            z = self.meta['conf']['observation']['redshift']
            first = hdul['DATA'].header['CRVAL3']
            step = hdul['DATA'].header['CD3_3']
            size =  hdul['DATA'].header['NAXIS3']
            
            old_wave = sc.util.build_wave_array(
                wave = [first, step],
                sampling_type='linear',
                size = size)
            old_wave = old_wave/(1. + z)
        
            flux, wave, uncertainty = sc.resampling(
                flux = flux,
                old_wave = old_wave,
                old_sampling_type = 'log',
                new_wave = np.array(self.meta['obs']['wave_obs']),
                new_sampling_type = 'ln',
                flux_err = uncertainty)
        
        flux_new = np.full(self.commom_wave.shape + self.rest_wave_grid.shape[1:],
                            fill_value=np.nan)
        uncertainty_new = np.full(self.commom_wave.shape + self.rest_wave_grid.shape[1:],
                            fill_value=np.nan)
        
        for i, j in np.ndindex(self.rest_wave_grid.shape[1:]):
            flux_new[:, i, j], _, uncertainty_new[:, i, j] = sc.resampling(
                flux = flux[:, i, j],
                old_wave = self.wave_to_rest(wave, v=self.velocity_map[i, j]),
                old_sampling_type = 'ln',
                new_wave = np.array(self.commom_wave),
                new_sampling_type = 'linear',
                flux_err = uncertainty[:, i, j])
        
        sci_hdu = fits.ImageHDU(
            np.single(flux_new), name = 'SCI', header = self.header_sci)
        sci_hdu.header['CTYPE3'] = 'AWAV'
        sci_hdu.header['CRVAL3'] = self.commom_wave[0]
        sci_hdu.header['CD3_3'] = self.commom_wave[1] - self.commom_wave[0]
        self.hdul_out.append(sci_hdu)

        err_hdu = fits.ImageHDU(
            np.single(uncertainty_new), name = 'ERR', header = self.header_err)
        err_hdu.header['CTYPE3'] = 'AWAV'
        err_hdu.header['CRVAL3'] = self.commom_wave[0]
        err_hdu.header['CD3_3'] = self.commom_wave[1] - self.commom_wave[0]
        self.hdul_out.append(err_hdu)

        mask = np.any(np.isnan(flux_new[:, ...]), axis = 0)
        mask = np.array(mask, dtype = int)
        mask_hdu = fits.ImageHDU(
            mask, name = 'MASK', header = self.header_sci)
        mask_hdu.header['CTYPE3'] = 'AWAV'
        mask_hdu.header['CRVAL3'] = self.commom_wave[0]
        mask_hdu.header['CD3_3'] = self.commom_wave[1] - self.commom_wave[0]
        self.hdul_out.append(mask_hdu)

    def add_bestfit_ssp(self):
        bestfit_path = os.path.join(
            self.meta['conf']['output_run_ppxf'],'bestfit.fits')

        with fits.open(bestfit_path) as hdul:
            stellar = hdul['PRIMARY'].data
            stellar, _, _ = sc.resampling(
                flux = stellar,
                old_wave = self.rest_wave_grid,
                old_sampling_type = 'ln',
                new_wave = np.array(self.commom_wave),
                new_sampling_type = 'linear')

        scale = np.nanmedian(self.hdul_out['SCI'].data, axis = 0) / np.nanmedian(stellar, axis = 0)
        stellar = stellar * scale

        self.stellar = stellar
        stellar_hdu = fits.ImageHDU(
            np.single(stellar), name = 'STELLAR', header = self.header_sci)
        stellar_hdu.header['CTYPE3'] = 'AWAV'
        stellar_hdu.header['CRVAL3'] = self.commom_wave[0]
        stellar_hdu.header['CD3_3'] = self.commom_wave[1] - self.commom_wave[0]
        self.hdul_out.append(stellar_hdu)

    def build_rest_wave_grid(self, wave, velocity_map):
        wave = np.asarray(wave)
        velocity_map = np.asarray(velocity_map)

        rest_wave_grid = np.zeros(wave.shape + velocity_map.shape)

        for i, j in np.ndindex(velocity_map.shape):
            rest_wave_grid[:, i, j] = self.wave_to_rest(
                wave = wave, v = velocity_map[i, j])
            # print(rest_wave_grid[:, i, j][0])
        return rest_wave_grid

    def wave_to_rest(self, wave, v=0, c=299792.458):
        wave = np.asarray(wave)
        rest_wave = wave / (1. + (v/c))
        return rest_wave

    def find_commom_range(self, wave_rest_grid, velocity_map,
                          percentile: float = 100):
        velocity_map = np.asarray(velocity_map)

        lower_lim = np.nanpercentile(velocity_map, 100-percentile,
                                     interpolation = 'nearest')
        upper_lim = np.nanpercentile(velocity_map, percentile,
                                     interpolation = 'nearest')

        lower_index = np.argwhere(velocity_map == lower_lim)[0]
        upper_index = np.argwhere(velocity_map == upper_lim)[0]

        lower_wave = wave_rest_grid[:, lower_index[0], lower_index[1]]
        upper_wave = wave_rest_grid[:, upper_index[0], upper_index[1]]

        commom_wave = lower_wave[(lower_wave >= upper_wave[0])
                                 & (lower_wave <= upper_wave[-1])]
        commom_wave = sc.util.fit_wave_interval(commom_wave,
                                                old_sampling = 'log',
                                                new_sampling = 'linear')
        return commom_wave


class RunIFScube:
    def __init__(self, initialization_file):
        self.initialization_file = initialization_file
        self.read_conf()
        
        meta = IFSCubeInput(self.initialization_file).meta
        self.meta = meta
        self.copy_metadata()
    
        os.system(f"cubefit -lc {self.ifscube_conf} {self.meta['cube_path']}")

        self.remove_temporary()
        self.move_data_products()

    def read_conf(self):
        configur = ConfigParser(interpolation=ExtendedInterpolation())
        configur.read(self.initialization_file)

        self.ppxf_data_products = configur.get('resources', 'ppxf_results')
        self.ifscube_conf = configur.get('ifscube', 'conf_path')

    def move_data_products(self):
        shutil.move(
            self.meta['cube_path'].split('/')[-1].split('.')[0] + '_linefit.fits',
            self.meta['conf']['output_run_ifscube'])

    def copy_metadata(self):
        shutil.copy(
            self.ifscube_conf,
            self.meta['conf']['output_run_ifscube'])

        shutil.copy(
            self.initialization_file,
            self.meta['conf']['output_run_ifscube'])

    def remove_temporary(self):
        os.remove(self.meta['cube_path'])
        try:
            os.remove(self.meta['cube_path'].split('/')[-1].split('.')[0] \
                      + '_linefit.fits.lock')
        except FileNotFoundError:
            pass

#%%
if __name__ == '__main__':
    initialization_file = sys.argv[1]
    RunIFScube(initialization_file)

    # t = IFSCubeInput(initialization_file = 'test_ifscube.ini')
    # a = RunIFScube(initialization_file = 'test_ifscube.ini')

#%%
# import matplotlib.pyplot as plt

# plt.plot(old_wave, flux[:, 20, 0])

# plt.plot(old_wave, flux[:, 50, 0])
# plt.plot(old_wave2, flux_n[:, 50, 0])
# plt.plot(old_wave2/ (1. + (t.velocity_map[50,0]/299792)), flux_n[:, 50, 0])
# plt.plot(flux[:, 50, 0])

# plt.plot(t.commom_wave, t.hdul_out['sci'].data[:, ::2, 0])
#%%
# cube = RunIFScube(initialization_file = 'test_ifscube.ini')

# input_cube = IFSCubeInput(
#     obs_path = cube.meta['obs_path'],
#     metadata_path = os.path.join(cube.meta['output_run_ppxf'], 'metadata.json'),
#     output_dir = cube.meta['output_run_ifscube'],
#     count = cube.count)

# # plt.plot(input_cube.commom_wave, input_cube.hdul_out['sci'].data[:, 0,0])
# # plt.plot(input_cube.commom_wave, input_cube.hdul_out['stellar'].data[:, 0,0])


# # input_cube.hdul_out.info()
# # flux = input_cube.hdul_out['sci'].data

# # plt.plot(flux[:, 0, 0])

# from astropy.io import fits
# cube = fits.open('../../data_products/toy_50x50/miles/ifscube_7/input_cube_7.fits')
# rest = input_cube.rest_wave_grid
# a = cube[1].

# with fits.open(input_cube.meta['obs_path']) as hdul:
#     flux = np.single(hdul['data'].data)
#     uncertainty = np.single(hdul['stat'].data)
#     flux, wave, uncertainty = sc.resampling(
#         flux = flux,
#         old_wave = np.array(input_cube.meta['o_wave_obs']),
#         old_sampling_type = 'linear',
#         new_wave = np.array(input_cube.meta['wave_obs']),
#         new_sampling_type = 'log',
#         flux_err = uncertainty)


# import matplotlib.pyplot as plt

# # plt.plot(input_cube.commom_wave, input_cube.hdul_out['sci'].data[:, 30, 40])
# # plt.plot(input_cube.commom_wave, input_cube.hdul_out['sci'].data[:, 2, 0])

# # plt.plot(input_cube.meta['wave_obs'], flux[:, 30, 40])
# plt.plot(np.asarray(input_cube.meta['wave_obs']), flux[:, 2, 0])
# plt.plot(np.asarray(input_cube.meta['wave_obs']) / (1. + 1369.8646841041664/299792.458) , flux[:, 2, 0])
# plt.plot(input_cube.commom_wave, input_cube.hdul_out['sci'].data[:, 2, 0])

# plt.imshow(input_cube.velocity_map)
