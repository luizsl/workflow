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
import spectcube as sc
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats


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

    # Convert to numpy arrays for sake of simplicity
    out = np.asarray(out).T
    return out

def check_inside(wave, fixed_mask:list):
    for bound in fixed_mask:
        if (wave > bound[0]) & (wave < bound[1]):
            return True
    return False

def residual_and_degree(bestfit, galaxy, goodpixels, degree, degree_plot,
                        metadata, spectral_limits=[4736.1, 9285.3], title=None,
                        save_title=None):

    assert galaxy.shape[1] == degree.shape[0]

    wave = np.asarray(metadata['obs']['wave_obs'])
    bound = sc.util._build_edges(wave, 'ln')

    fixed_mask = metadata['conf']['observation']['fixed_spectral_mask']
    goodpixels = goodpixels[np.isfinite(goodpixels)]
    goodpixels = np.asarray(goodpixels, dtype=int)

    mask = np.full_like(wave, fill_value=True, dtype=bool)
    mask[goodpixels] = False

    # Build figure frame
    plt.style.use('../src/fig_conf.mplstyle')
    fig = plt.figure(figsize=(11,6),)
    gs = fig.add_gridspec(nrows=degree_plot.size, ncols=2,
                          left=0.08, right=0.99,
                          top=0.95, bottom=0.1,
                          hspace=0.1, wspace=0,
                          width_ratios = [30,1],
                          )
    ax = np.full((degree_plot.size, 2), fill_value=None, dtype=object)
    for i in range(degree_plot.size) :
        ax[i, 0] = fig.add_subplot(gs[i, 0], sharex=ax[0,0], sharey=ax[0,0])
        ax[i, 1] = fig.add_subplot(gs[i, 1], sharey=ax[0,0])

    # Plot
    i = 0
    for k, p in enumerate(degree):
        if p in degree_plot:
            residual = galaxy[:, k] / bestfit[:, k]
            # print(p, np.nanmax(np.ma.masked_where(mask, residual)).round(3) , (np.nanmin(np.ma.masked_where(mask, residual))).round(3))

            # scatter
            ax[i, 0].scatter(wave, np.ma.masked_where(mask, residual),
                             alpha=1, color='navy', s=1, marker='.')

            # shaded line
            ax[i, 0].axhline(1, alpha=0.5, color='k', lw=0.5)
            ax[i, 0].axhspan(1 - 0.01, 1 + 0.01, color='lightcoral', alpha=0.5,
                             lw=0, zorder=0)
            # ax[i, 0].axhspan(1 - 0.03, 1 + 0.03, color='lightcoral', alpha=0.1,
            #                   lw=0, zorder=0)

            ax[i, 0].set_ylabel(f'$p={p}$')

            # shade regions
            for j in range(wave.size):
                if check_inside(wave[j], fixed_mask):
                    lw = bound[j]
                    up = bound[j+1]
                    ax[i, 0].axvspan(lw, up, color='silver', alpha=1, lw=0)
                elif j not in goodpixels:
                    lw = bound[j]
                    up = bound[j+1]
                    ax[i, 0].axvspan(lw, up, color='darkseagreen', alpha=1, lw=0)

            # shade bounds
            ax[i, 0].axvspan(4680, spectral_limits[0], color='silver',
                             alpha=1, lw=0)
            ax[i, 0].axvspan(spectral_limits[1], 9350, color='silver',
                             alpha=1, lw=0)

            # ax[i, 0].set_ylim(-0.055, 0.1)
            ax[i, 0].set_ylim(1 - 0.06, 1 + 0.13)
            ax[i, 0].set_xlim(4680, 9350)

            #  Distribution
            sns.kdeplot(ax=ax[i, 1], y=residual, bw_method='silverman',
                        bw_adjust=.1,
                        # clip=(-0.1,0.1),
                        clip=(1 + 0.1 , 1 - 0.1),
                        fill=True, common_norm=True, alpha=.5, linewidth=0.5,
                        color='navy',
                    )
            # ax[i, 1].set_ylim((0.93,1.15))

            i = i + 1

        ax[-1,0].set_xlabel(r'Wavelength [\AA]', fontsize='large')
        ax[-1,1].set_xlabel(r'Density', fontsize='large')

        fig.supylabel(r'Residual [$\%$]', x=0.01)

    for i in range(ax.shape[0]-1):
        ax[i,0].get_xaxis().set_ticklabels([])
        ax[i,1].get_xaxis().set_ticklabels([])
        ax[i,1].set_xlabel('')
    for i in range(ax.shape[0]):
        ax[i, 1].label_outer()

    ax[0,0].set_title(title, loc='left')

    plt.savefig(f'../plots/polynomial_degree/residual_{save_title}.pdf')

if __name__ == '__main__':
    degree_range = np.arange(-1, 21, 1)
    mdegree_range = np.arange(0, 21, 1)

    # Read bestfit

    ## Additive
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

    ## Multiplicative
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

    ## Additive
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

    ## Multiplicative
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

    # Read goodpixels

    ## Additive
    goodpixels_add_emiles = build_spectra_array(
        prop='goodpixels',
        order_range=degree_range,
        directory='../data_products/polynomial_single/emiles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    goodpixels_add_miles = build_spectra_array(
        prop='goodpixels',
        order_range=degree_range,
        directory='../data_products/polynomial_single/miles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    goodpixels_add_xsl = build_spectra_array(
        prop='goodpixels',
        order_range=degree_range,
        directory='../data_products/polynomial_single/xsl/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')

    ## Multiplicative
    goodpixels_mlt_emiles = build_spectra_array(
        prop='goodpixels',
        order_range=mdegree_range,
        directory='../data_products/polynomial_single/emiles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    goodpixels_mlt_miles = build_spectra_array(
        prop='goodpixels',
        order_range=mdegree_range,
        directory='../data_products/polynomial_single/miles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    goodpixels_mlt_xsl = build_spectra_array(
        prop='goodpixels',
        order_range=mdegree_range,
        directory='../data_products/polynomial_single/xsl/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')

    # Read polynomial
    ## Additive
    poly_add_miles = build_spectra_array(
        prop='apoly',
        order_range=degree_range,
        directory='../data_products/polynomial_single/miles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    poly_add_emiles = build_spectra_array(
        prop='apoly',
        order_range=degree_range,
        directory='../data_products/polynomial_single/emiles/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    poly_add_xsl = build_spectra_array(
        prop='apoly',
        order_range=degree_range,
        directory='../data_products/polynomial_single/xsl/additive_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')

    ## Multiplicative
    poly_mlt_miles = build_spectra_array(
        prop='mpoly',
        order_range=mdegree_range,
        directory='../data_products/polynomial_single/miles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    poly_mlt_emiles = build_spectra_array(
        prop='mpoly',
        order_range=mdegree_range,
        directory='../data_products/polynomial_single/emiles/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='MilesAgeMh')

    poly_mlt_xsl = build_spectra_array(
        prop='mpoly',
        order_range=mdegree_range,
        directory='../data_products/polynomial_single/xsl/multiplicative_polynomial/NGC0613_full_stacked_spectrum/',
        lib='XSLAgeMh')

    #%% Residual against polynomial degree

    # Common polynomial degree list
    degree_plot = np.array([-1,4,6,8,10,12,16])
    mdegree_plot = np.array([0,4,6,8,10,12,16])

    # MILES

    # Commom metadata
    metadata_path = ('../data_products/polynomial_single/miles/'
                      'multiplicative_polynomial/NGC0613_full_stacked_spectrum/'
                      'MilesAgeMh/ppxf/metadata.json')
    with open(metadata_path) as f:
        metadata_miles=json.load(f)

    ## Additive
    residual_and_degree(bestfit=bestfit_add_miles,
                        galaxy=galaxy_add_miles,
                        goodpixels=goodpixels_add_miles,
                        degree = degree_range,
                        degree_plot=degree_plot,
                        metadata=metadata_miles,
                        spectral_limits=[4736.1, 7300],
                        title=r'\texttt{MILES} / Additive polynomial',
                        save_title='miles_dditive_polynomial')

    ## Multiplicative
    residual_and_degree(bestfit=bestfit_mlt_miles,
                        galaxy=galaxy_mlt_miles,
                        goodpixels=goodpixels_mlt_miles,
                        degree = mdegree_range,
                        degree_plot=mdegree_plot,
                        metadata=metadata_miles,
                        spectral_limits=[4736.1, 7300],
                        title=r'\texttt{MILES} / Multiplicative polynomial',
                        save_title='miles_ultiplicative_polynomial')

    # E-MILES

    # Commom metadata
    metadata_path = ('../data_products/polynomial_single/emiles/'
                      'multiplicative_polynomial/NGC0613_full_stacked_spectrum/'
                      'MilesAgeMh/ppxf/metadata.json')
    with open(metadata_path) as f:
        metadata_miles=json.load(f)

    ## Additive
    residual_and_degree(bestfit=bestfit_add_emiles,
                        galaxy=galaxy_add_emiles,
                        goodpixels=goodpixels_add_emiles,
                        degree = degree_range,
                        degree_plot=degree_plot,
                        metadata=metadata_miles,
                        title=r'\texttt{E-MILES} / Additive polynomial',
                        save_title='emiles_additive_polynomial')

    ## Multiplicative
    residual_and_degree(bestfit=bestfit_mlt_emiles,
                        galaxy=galaxy_mlt_emiles,
                        goodpixels=goodpixels_mlt_emiles,
                        degree = mdegree_range,
                        degree_plot=mdegree_plot,
                        metadata=metadata_miles,
                        title=r'\texttt{E-MILES} / Multiplicative polynomial',
                        save_title='emiles_multiplicative_polynomial')

    # XSL

    # Commom metadata
    metadata_path = ('../data_products/polynomial_single/xsl/'
                      'multiplicative_polynomial/NGC0613_full_stacked_spectrum/'
                      'XSLAgeMh/ppxf/metadata.json')
    with open(metadata_path) as f:
        metadata_miles=json.load(f)

    ## Additive
    residual_and_degree(bestfit=bestfit_add_xsl,
                        galaxy=galaxy_add_xsl,
                        goodpixels=goodpixels_add_xsl,
                        degree = degree_range,
                        degree_plot=degree_plot,
                        metadata=metadata_miles,
                        title=r'\texttt{XSL} / Additive polynomial',
                        save_title='xsl_additive_polynomial')

    ## Multiplicative
    residual_and_degree(bestfit=bestfit_mlt_xsl,
                        galaxy=galaxy_mlt_xsl,
                        goodpixels=goodpixels_mlt_xsl,
                        degree = mdegree_range,
                        degree_plot=mdegree_plot,
                        metadata=metadata_miles,
                        title=r'\texttt{XSL} / Multiplicative polynomial',
                        save_title='xsl_multiplicative_polynomial')
