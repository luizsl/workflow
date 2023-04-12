#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:04:13 2022

@author: Luiz
"""

import json
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


class PopMeanProperties:
    def __init__(self, datapath=None, metadatapath=None, age_log10=None):
        assert metadatapath is not None
        self.metadatapath = metadatapath
        self.read_metadata()
        
        self.start_logging()
        
        assert isinstance(age_log10, bool)
        self.age_log10 = age_log10
        
        assert datapath is not None
        self.datapath = datapath
        self.read_data()
        
        self.mh_range = self.meta['model']['mh_range']
        self.logger.info('\nMH range:')
        self.logger.info(self.mh_range)
        
        self.age_range = np.asarray(self.meta['model']['age_range'])
        if self.age_log10 is False:
            self.age_range = np.asarray(self.age_range) * 1e9
            self.logger.info('\nAge range [yr]:') 
            self.logger.info([''.join("{:0.2e}".format(i)) for i in self.age_range])
            self.logger.info('\nAge is not on a logarithmic scale, it will be converted to:',)
            self.age_range = np.log10(self.age_range)
        self.logger.info('\nlog10 age range [yr]:')
        self.logger.info(np.round(self.age_range, 3))
          
    def start_logging(self):
        name_log_file = os.path.join(
            self.meta['conf']['output_run_ppxf'],
            'log_post_processing.log')
        
        formatter = logging.Formatter('%(message)s')
        loglevel = logging.DEBUG
        
        file_handler = logging.FileHandler(name_log_file )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(loglevel)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(loglevel)
        
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(loglevel)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        
        self.logger = logger
        
    def read_data(self):
        with fits.open(self.datapath) as hdul:
            self.data = hdul[0].data
            
        if self.data.ndim==1:
            self.data = np.expand_dims(self.data, axis=(1,2))
        
    def read_metadata(self):
        with open(self.metadatapath) as f:
            self.meta = json.load(f)
            
    @property
    def mh_light(self):
        reg_dim = self.meta['model']['reg_dim']
        shape_out = self.data.shape[1:]
        out = np.full(shape_out, fill_value=np.nan)
        mh, age = np.meshgrid(self.mh_range,
                              self.age_range)
        for i, j in np.ndindex(out.shape):
            weight = self.data[:, i, j].reshape(reg_dim)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[i,j] = np.nansum(weight*mh)/np.nansum(weight)
        return out
    
    @property
    def log10_age_light(self):
        reg_dim = self.meta['model']['reg_dim']
        shape_out = self.data.shape[1:]
        out = np.full(shape_out, fill_value=np.nan)
        mh, age = np.meshgrid(self.mh_range,
                              self.age_range)
        for i, j in np.ndindex(out.shape):
            weight = self.data[:, i, j].reshape(reg_dim)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[i,j] = np.nansum(weight*age)/np.nansum(weight)
        return out
    
    def save(self, data, param:str, output_path:str):
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
        name = os.path.join(output_path, f'{param}.fits')
        hdul.writeto(name, overwrite=True)


if __name__ == '__main__':
    
    # datapath = '../../data_products/NGC0613_full_stacked_spectrum/MilesAgeMh_31/ppxf/weights.fits'
    # metadatapath = '../../data_products/NGC0613_full_stacked_spectrum/MilesAgeMh_31/ppxf/metadata.json'
    
    # datapath = '../../data_products/fov_sample_1_5/MilesAgeMh/ppxf/weights.fits'
    # metadatapath = '../../data_products/fov_sample_1_5/MilesAgeMh/ppxf/metadata.json'

    datapath = '../../data_products/fov_sample_1_5/XSLAgeMh_1/ppxf/weights.fits'
    metadatapath = '../../data_products/fov_sample_1_5/XSLAgeMh_1/ppxf/metadata.json'
        
    t = PopMeanProperties(datapath, metadatapath, age_log10=True)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(10**(t.log10_age_light - 9), origin='lower', cmap='jet', vmin=3, vmax=7)
    ax[1].imshow(t.mh_light, origin='lower', cmap='jet', vmin=0, vmax=0.18)
    # t.save(t.mh, 'mean_mh',  '../../data_products/ngc613/regul_50_sn_100/')
    # t.save(t.age, 'mean_log_age',  '../../data_products/ngc613/regul_50_sn_100/')
    
    # plt.imshow(t.data[:, 169, 155].reshape((24, 6)))
