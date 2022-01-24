#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:02:39 2022

@author: Luiz
"""

import json
import os
import shutil
import sys
from configparser import ConfigParser, ExtendedInterpolation

import numpy as np
import spectcube as sc
from astropy.io import fits


class IFSCubeInput:
    def __init__(self, obs_path, metadata_path):
        self.obs_path = obs_path
        self.metadata_path = metadata_path
        self.meta = self.read_meta()
        self.get_headers()
        self.hdul_out = fits.HDUList()
        self.normalization_factor = self.recover_norm_factor()
        self.add_muse_data()
        self.add_bestfit_ssp()
        self.hdul_out.writeto('input_cube.fits', overwrite = True)

    def read_meta(self):
        with open(self.metadata_path) as f:
            meta = json.load(f)
        return meta

    def get_headers(self):
        with fits.open(self.obs_path) as hdul:
            self.header_primary = hdul[0].header
            self.header_sci = hdul[1].header
            self.header_err = hdul[2].header

    def recover_norm_factor(self):
        normalization_factor_path = os.path.join(
            self.meta['output_run_ppxf'],'normalization_factor.fits')
        
        with fits.open(normalization_factor_path) as hdul:
            normalization_factor = hdul['PRIMARY'].data
        return normalization_factor

    def add_muse_data(self):
        primary_hdu = fits.PrimaryHDU(header = self.header_primary)
        self.hdul_out.append(primary_hdu)

        with fits.open(self.obs_path) as hdul:
            flux = hdul['data'].data
            uncertainty = hdul['stat'].data
            flux, wave, uncertainty = sc.resampling(
                flux = flux,
                old_wave = np.array(self.meta['o_wave_obs']),
                old_sampling_type = 'linear',
                new_wave = np.array(self.meta['wave_obs']),
                new_sampling_type = 'log',
                flux_err = uncertainty)

        new_wave = sc.util.fit_wave_interval(
            wave = np.array(self.meta['wave_obs']),
            old_sampling = 'log', new_sampling = 'linear')

        flux, wave, uncertainty = sc.resampling(
            flux = flux,
            old_wave = np.array(self.meta['wave_obs']),
            old_sampling_type = 'log',
            new_wave = np.array(new_wave),
            new_sampling_type = 'linear',
            flux_err = uncertainty)

        sci_hdu = fits.ImageHDU(flux, name = 'SCI', header = self.header_sci)
        # sci_hdu.header['CTYPE3'] = 'AWAV-LOG'
        sci_hdu.header['CTYPE3'] = 'AWAV'
        sci_hdu.header['CRVAL3'] = new_wave[0]
        sci_hdu.header['CD3_3'] = wave[1] - new_wave[0]
        self.hdul_out.append(sci_hdu)

        err_hdu = fits.ImageHDU(
            uncertainty, name = 'ERR', header = self.header_err)
        # err_hdu.header['CTYPE3'] = 'AWAV-LOG'
        err_hdu.header['CTYPE3'] = 'AWAV'
        err_hdu.header['CRVAL3'] = new_wave[0]
        err_hdu.header['CD3_3'] = new_wave[1] - new_wave[0]
        self.hdul_out.append(err_hdu)

        mask = np.any(np.isnan(flux[:, ...]), axis = 0)
        mask = np.array(mask, dtype = int)
        mask_hdu = fits.ImageHDU(
            mask, name = 'MASK', header = self.header_sci)
        # mask_hdu.header['CTYPE3'] = 'AWAV-LOG'
        mask_hdu.header['CTYPE3'] = 'AWAV'
        mask_hdu.header['CRVAL3'] = new_wave[0]
        mask_hdu.header['CD3_3'] = new_wave[1] - new_wave[0]
        self.hdul_out.append(mask_hdu)

    def add_bestfit_ssp(self):
        bestfit_path = os.path.join(
            self.meta['output_run_ppxf'],'bestfit.fits')
                
        new_wave = sc.util.fit_wave_interval(
            wave = np.array(self.meta['wave_obs']),
            old_sampling = 'log', new_sampling = 'linear')
        
        with fits.open(bestfit_path) as hdul:
            stellar = hdul['PRIMARY'].data
            stellar, wave, _ = sc.resampling(
                flux = stellar,
                old_wave = np.array(self.meta['wave_obs']),
                old_sampling_type = 'log',
                new_wave = np.array(new_wave),
                new_sampling_type = 'linear')
            
        stellar = stellar * self.normalization_factor
        self.stellar = stellar
        stellar_hdu = fits.ImageHDU(stellar, name = 'STELLAR', header = self.header_sci)
        # stellar_hdu.header['CTYPE3'] = 'AWAV-LOG'
        stellar_hdu.header['CTYPE3'] = 'AWAV'
        stellar_hdu.header['CRVAL3'] = wave[0]
        stellar_hdu.header['CD3_3'] = wave[1] - wave[0]
        self.hdul_out.append(stellar_hdu)

class RunIFScube:
    def __init__(self, initialization_file):
        self.initialization_file = initialization_file
        self.read_conf()
        self.meta = self.read_meta()
        self.create_output_dir()

        IFSCubeInput(
            obs_path = self.meta['obs_path'],
            metadata_path = os.path.join(self.meta['output_run_ppxf'], 'metadata.json'))

        os.system(f'cubefit -oc {self.ifscube_conf} input_cube.fits')

        self.move_data_products()
        self.remove_temporary()
        
    def read_conf(self):
        configur = ConfigParser(interpolation=ExtendedInterpolation())
        configur.read(self.initialization_file)

        self.ppxf_data_products = configur.get('resources', 'ppxf_results')
        self.ifscube_conf = configur.get('ifscube', 'conf_path')

    def read_meta(self):
        dir_ = os.path.join(self.ppxf_data_products, 'metadata.json')
        with open(dir_) as f:
            meta = json.load(f)
        return meta

    def create_output_dir(self):
        # # later add a function to write a more complex _sec_dir name
        # sec_dir_ = f'{dir_}miles'
        dir_ = os.path.join(self.meta['output_run'], 'ifscube')

        # create unique name
        if os.path.isdir(dir_) is False:
            self.meta['output_run_ifscube'] = dir_
        else:
            count = 1
            while os.path.isdir(f'{dir_}_{str(count)}') is True:
                count += 1
            self.meta['output_run_ifscube'] = f'{dir_}_{str(count)}'

        os.makedirs(self.meta['output_run_ifscube'], exist_ok=True)

    def move_data_products(self):
        shutil.move(
            'input_cube_linefit.fits',
            self.meta['output_run_ifscube'])

        shutil.copy(
            self.ifscube_conf,
            self.meta['output_run_ifscube'])

        shutil.copy(
            self.initialization_file,
            self.meta['output_run_ifscube'])
        
    def remove_temporary(self):
        os.remove('input_cube.fits')

if __name__ == '__main__':
    # RunIFScube(initialization_file = 'test_ifscube.ini')
    initialization_file = sys.argv[1]
    RunIFScube(initialization_file)


#%%
import matplotlib.pyplot as plt

# metadata_path = '../../data_products/toy_20x20/miles/ppxf/metadata.json'
# bestfit_path = '../../data_products/toy_20x20/miles/ppxf/bestfit.fits'
# obs_path = '../../data/toy_20x20.fits'
# normalization_factor_path = '../../data_products/toy_20x20/miles/ppxf/normalization_factor.fits'


# with open(metadata_path) as f:
#     meta = json.load(f)

# with fits.open(obs_path) as hdul:
#     flux = hdul['data'].data
#     uncertainty = hdul['stat'].data
#     flux, wave, uncertainty = sc.resampling(
#         flux = flux,
#         old_wave = np.array(meta['o_wave_obs']),
#         old_sampling_type = 'linear',
#         new_wave = np.array(meta['wave_obs']),
#         new_sampling_type = 'log',
#         flux_err = uncertainty)

# with fits.open(bestfit_path) as hdul:
#     stellar = hdul['PRIMARY'].data
    

# with fits.open(normalization_factor_path) as hdul:
#     normalization_factor = hdul['PRIMARY'].data
    
# plt.plot(wave, flux[:,0,0])
# plt.plot(wave, stellar[:,0,0] * normalization_factor[0,0])
# plt.plot(wave, stellar[:,0,0])
# stellar_corr =  (stellar[:,0,0] * normalization_factor[0,0])
# plt.plot(wave, flux[:,0,0] - stellar_corr)

# #%%

res = '../../data_products/toy_20x20/miles/ifscube/input_cube_linefit.fits'
# # res = 'input_cube.fits'
hdul = fits.open(res)

# plt.plot(hdul['restwave'].data, hdul['stellar'].data[:,0,0])
# plt.plot(hdul['restwave'].data, hdul['fitspec'].data[:,0,0] - hdul['stellar'].data[:,0,0])
# plt.plot(hdul['restwave'].data, hdul['fitcont'].data[:,0,0])
# plt.plot(hdul['restwave'].data, hdul['fitspec'].data[:,0,0] - hdul['model'].data[:,0,0])
# a = hdul['stellar'].data
plt.imshow(hdul['red_chi'].data)
