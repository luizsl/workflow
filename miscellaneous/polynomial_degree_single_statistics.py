#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:36:30 2022

@author: Luiz

Residuals analysis of Legendre polynomial test 
"""
import os
import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits


def read_data(directory, file, ext=0):
    path = os.path.join(directory, file)
    with fits.open(path) as hdul:
        data = hdul[ext].data
        return data
    
def build_spectra_array(prop=None, order_range=None, directory=None, lib=None):
    assert prop is not None
    assert order_range is not None
    assert directory is not None
    assert lib is not None
    
    # # Build variables
    # for i in range(1,length+1):
    #     locals()[f'm{i}'] = []
    
    out = []
    # Read and unpack data
    for i in range(len(order_range)):
        _directory = os.path.join(directory, lib) + f'_{i}/ppxf'
        file = f'{prop}.fits'

        # NOTE: Exception due to limitations on directory name assignment
        if i == 0:
            _directory = _directory.replace('_0','')
            
        d = read_data(_directory, file)
        out.append(d)
        # for i in range(1,length+1):
        #     if length > 1:
        #         locals()[f'm{i}'].append(d[i-1])
        #     else:
        #         locals()[f'm{i}'].append(d)

    # Convert to numpy arrays for sake of simplicity
    out = np.asarray(out).T
    return out

if __name__ == '__main__':
    degree_range = np.arange(-1, 21, 1)
    mdegree_range = np.arange(0, 21, 1)
    
    # Build spectral axis from some of the metadata file
    metadata_path = '../data_products/polynomial_single/emiles/additive_polynomial/NGC0613_full_stacked_spectrum/MilesAgeMh/ppxf/metadata.json'
    with open(metadata_path) as f:
        metadata=json.load(f)
        
    # Read bestfit
    # Additive
    bestfit_add_emiles = build_spectra_array(
        prop='bestfit',
        order_range=degree_range, 
        directory='../data_products/polynomial_single/emiles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    bestfit_add_miles = build_spectra_array(
        prop='bestfit',
        order_range=degree_range, 
        directory='../data_products/polynomial_single/miles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    bestfit_add_xsl = build_spectra_array(
        prop='bestfit',
        order_range=degree_range, 
        directory='../data_products/polynomial_single/xsl/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')
    
    # Multiplicative
    bestfit_mlt_emiles = build_spectra_array(
        prop='bestfit',
        order_range=mdegree_range, 
        directory='../data_products/polynomial_single/emiles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    bestfit_mlt_miles = build_spectra_array(
        prop='bestfit',
        order_range=mdegree_range, 
        directory='../data_products/polynomial_single/miles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    bestfit_mlt_xsl = build_spectra_array(
        prop='bestfit',
        order_range=mdegree_range, 
        directory='../data_products/polynomial_single/xsl/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')
    
    # Read observation (They must be all the same)
    # Additive
    galaxy_add_emiles = build_spectra_array(
        prop='galaxy',
        order_range=degree_range, 
        directory='../data_products/polynomial_single/emiles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    galaxy_add_miles = build_spectra_array(
        prop='galaxy',
        order_range=degree_range, 
        directory='../data_products/polynomial_single/miles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    galaxy_add_xsl = build_spectra_array(
        prop='galaxy',
        order_range=degree_range, 
        directory='../data_products/polynomial_single/xsl/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')
    
    # Multiplicative
    galaxy_mlt_emiles = build_spectra_array(
        prop='galaxy',
        order_range=mdegree_range, 
        directory='../data_products/polynomial_single/emiles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    galaxy_mlt_miles = build_spectra_array(
        prop='galaxy',
        order_range=mdegree_range, 
        directory='../data_products/polynomial_single/miles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')
    
    galaxy_mlt_xsl = build_spectra_array(
        prop='galaxy',
        order_range=mdegree_range, 
        directory='../data_products/polynomial_single/xsl/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')
    
    # Plot bestfit
    # E-MILES
    
    
    # Read provisional mask
    goodpixels_file = '../data_products/fov_sample_1_5/MilesAgeMh_1/ppxf/goodpixels.fits'
    with fits.open(goodpixels_file)as hdul:
        goodpixels = hdul[0].data[:, 30, 30]
        goodpixels = goodpixels[np.isfinite(goodpixels)]
        goodpixels = np.asarray(goodpixels, dtype=int)
        
    data_bestfit = bestfit_mlt_emiles
    data_galaxy =  galaxy_mlt_emiles
    
    fig, ax = plt.subplots(figsize=(12,4))
    wave = np.asarray(metadata['obs']['wave_obs'])
    c_seq = plt.get_cmap('rainbow', 21)
    # c_seq = sns.color_palette(sns.cubehelix_palette(start=2, rot=1))
    # c_seq = plt.matplotlib.colors.ListedColormap(c_seq)
    n_spec = data_bestfit.shape[1]
    for i in range(0,1):
        residual = data_galaxy[:, i] - data_bestfit[:, i]
        plt.step(wave, residual, alpha=1, where='mid')
        plt.axhline(np.nanmedian(residual), alpha=0.5, color='k', lw=0.5)
        plt.step(wave, data_galaxy[:, i], alpha=1, where='mid')
        
    ax.set_xlabel(r'Wavelength [$\AA$]')
    ax.set_ylabel(r'Flux density [a.u.]')
    ax.set_title('Additive Legendre polynomial')
    
    import spectcube as sc
    bound = sc.util._build_edges(wave, 'ln')
    
    for i in range(wave.size):
        if i not in goodpixels:
            lw = bound[i]
            up = bound[i+1]
            ax.axvspan(lw, up, color='lightgray')
            
    plt.tight_layout()