#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 10:50:54 2022

@author: Luiz

Legendre polynomial degree statistics
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import fits


def read_data(directory, file, ext=0):
    path = os.path.join(directory, file)
    with fits.open(path) as hdul:
        data = hdul[ext].data
        return data

def extract_maps(prop=None, length=None, order_range=None, directory=None, 
                 lib=None):
    assert prop is not None
    assert length is not None
    assert order_range is not None
    assert directory is not None
    assert lib is not None
    
    # Build variables
    for i in range(1,length+1):
        locals()[f'm{i}'] = []

    # Read and unpack data
    for i in range(len(order_range)):
        _directory = os.path.join(directory, lib) + f'_{i}/ppxf'
        file = f'{prop}.fits'

        # NOTE: Exception due to limitations on directory name assignment
        if i == 0:
            _directory = _directory.replace('_0','')
            
        d = read_data(_directory, file)
        for i in range(1,length+1):
            if length > 1:
                locals()[f'm{i}'].append(d[i-1])
            else:
                locals()[f'm{i}'].append(d)

    # Convert to numpy arrays for sake of simplicity
    for i in range(1,length+1):
        locals()[f'm{i}'] = np.asarray(locals()[f'm{i}'])

    # Pack output
    if length > 1:
        out = []
        for i in range(1,length+1):
            out.append(locals()[f'm{i}'])
    else:
        out = locals()[f'm{i}']

    return out


if __name__ == '__main__':
    
    #%% Velocity dispersion
    
    # Additive polynomial
    degree_range = np.arange(-1, 21, 5)

    h1_add_emiles, h2_add_emiles, h3_add_emiles, h4_add_emiles = extract_maps(
        prop='sol', length=4, order_range=degree_range,
        directory = '../data_products/emiles/additive_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')
    
    h1_add_miles, h2_add_miles, h3_add_miles, h4_add_miles = extract_maps(
        prop='sol', length=4, order_range=degree_range,
        directory = '../data_products/miles/additive_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')
    
    h1_add_xsl, h2_add_xsl, h3_add_xsl, h4_add_xsl = extract_maps(
        prop='sol', length=4, order_range=degree_range,
        directory = '../data_products/xsl/additive_polynomial/fov_sample_1_5/',
        lib='XSLAgeMh')

    h2_mean_add_emiles = np.nanmean(h2_add_emiles, axis=(1,2))
    h2_std_add_emiles = np.nanstd(h2_add_emiles, axis=(1,2))
    h2_median_add_emiles = np.nanmedian(h2_add_emiles, axis=(1,2))

    h2_mean_add_miles = np.nanmean(h2_add_miles, axis=(1,2))
    h2_std_add_miles = np.nanstd(h2_add_miles, axis=(1,2))
    h2_median_add_miles = np.nanmedian(h2_add_miles, axis=(1,2))
    
    h2_mean_add_xsl = np.nanmean(h2_add_xsl, axis=(1,2))
    h2_std_add_xsl = np.nanstd(h2_add_xsl, axis=(1,2))
    h2_median_add_xsl = np.nanmedian(h2_add_xsl, axis=(1,2))
    
    # Plot

    # E-MILES
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(degree_range, h2_mean_add_emiles / np.median(h2_mean_add_emiles),
            color=plt.cm.tab20(1), label=r'\texttt{E-MILES}')
    
    # MILES
    ax.plot(degree_range, h2_mean_add_miles / np.median(h2_mean_add_miles),
            color=plt.cm.tab20(3), label=r'\texttt{MILES}')
    
    # XSL
    ax.plot(degree_range, h2_mean_add_xsl / np.median(h2_mean_add_xsl),
            color=plt.cm.tab20(5), label=r'\texttt{XSL}')

    
    # Multiplicative polynomial
    
    mdegree_range = np.arange(0, 21, 5)

    h1_mlt_emiles, h2_mlt_emiles, h3_mlt_emiles, h4_mlt_emiles = extract_maps(
        prop='sol', length=4, order_range=mdegree_range,
        directory = '../data_products/emiles/multiplicative_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')

    h1_mlt_miles, h2_mlt_miles, h3_mlt_miles, h4_mlt_miles = extract_maps(
        prop='sol', length=4, order_range=mdegree_range,
        directory = '../data_products/miles/multiplicative_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')
    
    h1_mlt_xsl, h2_mlt_xsl, h3_mlt_xsl, h4_mlt_xsl = extract_maps(
        prop='sol', length=4, order_range=degree_range,
        directory = '../data_products/xsl/multiplicative_polynomial/fov_sample_1_5/',
        lib='XSLAgeMh')
    
    h2_mean_mlt_emiles = np.nanmean(h2_mlt_emiles, axis=(1,2))
    h2_std_mlt_emiles = np.nanstd(h2_mlt_emiles, axis=(1,2))
    h2_median_mlt_emiles = np.nanmedian(h2_mlt_emiles, axis=(1,2))
    
    h2_mean_mlt_miles = np.nanmean(h2_mlt_miles, axis=(1,2))
    h2_std_mlt_miles = np.nanstd(h2_mlt_miles, axis=(1,2))
    h2_median_mlt_miles = np.nanmedian(h2_mlt_miles, axis=(1,2))
    
    h2_mean_mlt_xsl = np.nanmean(h2_mlt_xsl, axis=(1,2))
    h2_std_mlt_xsl = np.nanstd(h2_mlt_xsl, axis=(1,2))
    h2_median_mlt_xsl = np.nanmedian(h2_mlt_xsl, axis=(1,2))
    
    # Plot
    # E-MILES
    ax.plot(mdegree_range, h2_mean_mlt_emiles / np.median(h2_mean_mlt_emiles),
            color=plt.cm.tab20(0), label=r'\texttt{E-MILES}', 
            ls='dashdot')
    
    # MILES
    ax.plot(mdegree_range, h2_mean_mlt_miles / np.median(h2_mean_mlt_miles),
            color=plt.cm.tab20(2), label=r'\texttt{MILES}', ls='dashdot')
    
    # XSL
    ax.plot(mdegree_range, h2_mean_mlt_xsl / np.median(h2_mean_mlt_xsl),
            color=plt.cm.tab20(4), label=r'\texttt{XSL}', ls='dashdot',)
    
    ax.axhline(1, color='gray', ls='dotted')

    ax.set_xlabel('Polynomial degree (n)')
    ax.set_ylabel('Normalized velocity dispersion')
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    
    plt.legend()
    plt.tight_layout()
    
    #%% Reduced Chi-squared
    
    # Additive polynomial
    degree_range = np.arange(-1, 21, 5)
    
    chi2_add_miles = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/miles/additive_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')
    
    chi2_add_emiles = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/emiles/additive_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')
    
    chi2_add_xsl = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/xsl/additive_polynomial/fov_sample_1_5/',
        lib='XSLAgeMh')
    
    # Multiplicative polynomial
    mdegree_range = np.arange(-1, 21, 5)
    
    chi2_mlt_miles = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/miles/multiplicative_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')
    
    chi2_mlt_emiles = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/emiles/multiplicative_polynomial/fov_sample_1_5/',
        lib='MilesAgeMh')
    
    chi2_mlt_xsl = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/xsl/multiplicative_polynomial/fov_sample_1_5/',
        lib='XSLAgeMh')
    
    chi2_mean_add_emiles = np.nanmean(chi2_add_emiles, axis=(1,2))
    chi2_median_add_emiles = np.nanmedian(chi2_add_emiles, axis=(1,2))
    
    chi2_mean_mlt_emiles = np.nanmean(chi2_mlt_emiles, axis=(1,2))
    chi2_median_mlt_emiles = np.nanmedian(chi2_mlt_emiles, axis=(1,2))
    
    chi2_mean_add_miles = np.nanmean(chi2_add_miles, axis=(1,2))
    chi2_median_add_miles = np.nanmedian(chi2_add_miles, axis=(1,2))
    
    chi2_mean_mlt_miles = np.nanmean(chi2_mlt_miles, axis=(1,2))
    chi2_median_mlt_miles = np.nanmedian(chi2_mlt_miles, axis=(1,2))
    
    chi2_mean_add_xsl = np.nanmean(chi2_add_xsl, axis=(1,2))
    chi2_median_add_xsl = np.nanmedian(chi2_add_xsl, axis=(1,2))
    
    chi2_mean_mlt_xsl = np.nanmean(chi2_mlt_xsl, axis=(1,2))
    chi2_median_mlt_xsl = np.nanmedian(chi2_mlt_xsl, axis=(1,2))
    
    # Plot
    # Additive polynomial
    
    # E-MILES
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(degree_range, chi2_mean_add_emiles,
            color=plt.cm.tab20(1), label=r'\texttt{E-MILES}')
    
    # MILES
    ax.plot(degree_range, chi2_mean_add_miles,
            color=plt.cm.tab20(3), label=r'\texttt{MILES}')
    
    # XSL
    ax.plot(degree_range, chi2_mean_add_xsl,
            color=plt.cm.tab20(5), label=r'\texttt{XSL}')
    
    # Multiplicatica polynomial
    
    # E-MILES
    ax.plot(mdegree_range, chi2_mean_mlt_emiles,
            color=plt.cm.tab20(0), label=r'\texttt{E-MILES}', ls='dashdot')
    
    # MILES
    ax.plot(mdegree_range, chi2_mean_mlt_miles,
            color=plt.cm.tab20(2), label=r'\texttt{MILES}', ls='dashdot')
    
    # XSL
    ax.plot(mdegree_range, chi2_mean_mlt_xsl,
            color=plt.cm.tab20(4), label=r'\texttt{XSL}', ls='dashdot')
    
    ax.axhline(1, color='gray', ls='dotted')

    ax.set_xlabel('Polynomial degree (n)')
    ax.set_ylabel(r'Reduced $\chi^2_{\nu}$')
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    
    plt.legend()
    plt.tight_layout()
    