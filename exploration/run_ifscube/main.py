#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:02:39 2022

@author: chess-lin
"""

import json
import os

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
        self.add_muse_data()
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
        new_wave = sc.util.fit_wave_interval(
            wave = np.array(self.meta['wave_obs']), 
            old_sampling = 'log', new_sampling = 'linear')
        
        with fits.open(self.obs_path) as hdul:
            stellar = hdul['PRIMARY'].data
            stellar, wave, _ = sc.resampling(
                flux = stellar, 
                old_wave = np.array(self.meta['wave_obs']),
                old_sampling_type = 'log',
                new_wave = np.array(new_wave), 
                new_sampling_type = 'linear')
            
        stellar_hdu = fits.ImageHDU(
            stellar, name = 'STELLAR', header = self.header_sci)
        # stellar_hdu.header['CTYPE3'] = 'AWAV-LOG'
        stellar_hdu.header['CTYPE3'] = 'AWAV'
        stellar_hdu.header['CRVAL3'] = wave[0]
        stellar_hdu.header['CD3_3'] = wave[1] - wave[0]
        self.hdul_out.append(stellar_hdu)


class RunIFScube:
    def __init__(self, conf_file, input_cube):
        os.system(f'cubefit -oc {conf_file} {input_cube}')


# class ControlIFSCube:
#     pass
#     IFSCubeInput(obs_path, metadata_path)
#     RunIFScube(conf_file, input_cube)

if __name__ == '__main__':
    
    bestfit_path = '../../data_products/toy_20x20/miles/ppxf/bestfit.fits'
    metadata_path = '../../data_products/toy_20x20/miles/ppxf/metadata.json'
    obs_path = '../../data/toy_20x20.fits'
     
    conf_file = 'halpha_cube_muse.cfg'
    input_cube = 'input_cube.fits'
    
    IFSCubeInput(obs_path, metadata_path)
    RunIFScube(conf_file, input_cube)

