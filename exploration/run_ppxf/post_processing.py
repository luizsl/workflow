#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:04:13 2022

@author: Luiz
"""

import os
import json

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


class PopMeanPropertie:
    def __init__(self, datapath, metadatapath):
        self.datapath = datapath
        self.metadatapath = metadatapath
        self.read_data()
        self.read_metadata()
        
    def read_data(self):
        with fits.open(self.datapath) as hdul:
            self.data = hdul[0].data
        
    def read_metadata(self):
        with open(metadatapath) as f:
            self.meta = json.load(f)
            
    @property
    def mh(self):
        reg_dim = self.meta['model']['reg_dim']
        out = np.full(self.data.shape[1:], fill_value=np.nan)
        mh, age = np.meshgrid(self.meta['model']['mh_range'],
                              np.log10(self.meta['model']['age_range']) + 9
                              )
        for i, j in np.ndindex(out.shape):
            weight = self.data[:, i, j].reshape(reg_dim)
            out[i,j] = np.nansum(weight*mh)/np.nansum(weight)
        return out
    
    @property
    def age(self):
        global mh, age
        reg_dim = self.meta['model']['reg_dim']
        out = np.full(self.data.shape[1:], fill_value=np.nan)
        mh, age = np.meshgrid(self.meta['model']['mh_range'],
                              np.log10(self.meta['model']['age_range']) + 9,
                              )
        for i, j in np.ndindex(out.shape):
            weight = self.data[:, i, j].reshape(reg_dim)
            out[i,j] = np.nansum(weight*age)/np.nansum(weight)
        return out
    
    def save(self, data, param:str, output_path:str):
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
        name = os.path.join(output_path, f'{param}.fits')
        hdul.writeto(name, overwrite=True)


if __name__ == '__main__':
    # datapath = '../../data_products/regularization_parameter/MilesAgeMh_reg_0d5/ppxf/weights.fits'
    # metadatapath = '../../data_products/regularization_parameter/MilesAgeMh_reg_0d4/ppxf/metadata.json'
    
    datapath = '../../data_products/ngc613/regul_50_sn_100/weights.fits'
    metadatapath = '../../data_products/ngc613/regul_50_sn_100/metadata.json'
    
    # datapath = '../../data_products/toy_100x100/MilesAgeMh_3/ppxf/weights.fits'
    # metadatapath = '../../data_products/toy_100x100/MilesAgeMh_3/ppxf/metadata.json'
    
    t = PopMeanPropertie(datapath, metadatapath)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(10**(t.age - 9), origin='lower', cmap='jet', vmin=3, vmax=7)
    ax[1].imshow(t.mh, origin='lower', cmap='jet', vmin=0, vmax=0.18)
    t.save(t.mh, 'mean_mh',  '../../data_products/ngc613/regul_50_sn_100/')
    t.save(t.age, 'mean_log_age',  '../../data_products/ngc613/regul_50_sn_100/')
    
    plt.imshow(t.data[:, 169, 155].reshape((24, 6)))
