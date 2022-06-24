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

def boxplot(ax, data=None, norm=None, color=None, order_range=None, n_offset=0):
    widths=0.08
    offset=0.2
    sym='.'
    
    if norm is None:
        norm = np.nanmedian(data)
    if norm is False:
        norm = 1
        
    data = data / norm
    data = data.reshape(len(order_range), -1).T 
    data = [data[:, i][~np.isnan(data[:, i])] for i in range(len(order_range))]
    
    bp = ax.boxplot(data, sym=sym, widths=widths,
                    positions=order_range+n_offset*offset,
                    patch_artist=True,
                    boxprops={'facecolor':'white',
                              'edgecolor':color},
                    capprops={'color':color},
                    whiskerprops={'color':color},
                    flierprops={'markeredgecolor':color,
                                'markersize':1},
                    medianprops={'color':'k'},
                    meanprops={'color':'k'})
    return bp

def mean_plot(ax, data=None, norm=None, color=None, order_range=None, **kwargs):
    if norm is None:
        norm = np.nanmedian(data)
    if norm is False:
        norm = 1
        
    data = np.nanmean(data, axis=(1,2)) / norm
    lp = ax.plot(order_range, data, color=color, **kwargs)
    return lp
#%%
if __name__ == '__main__':
    
    degree_range = np.arange(-1, 20, 1)
    mdegree_range = np.arange(0, 20, 1)
    
    # NOTE: Create output directory <>
    results_dir = '../plots/polynomial_degree/'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    # Read kinematics data
    
    # Additive polynomial
    h1_add_emiles, h2_add_emiles, h3_add_emiles, h4_add_emiles = extract_maps(
        prop='sol', length=4, order_range=degree_range,
        directory = '../data_products/polynomial_ifu/emiles/additive_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')
    
    h1_add_miles, h2_add_miles, h3_add_miles, h4_add_miles = extract_maps(
        prop='sol', length=4, order_range=degree_range,
        directory = '../data_products/polynomial_ifu/miles/additive_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')
    
    h1_add_xsl, h2_add_xsl, h3_add_xsl, h4_add_xsl = extract_maps(
        prop='sol', length=4, order_range=degree_range,
        directory = '../data_products/polynomial_ifu/xsl/additive_polynomial/fov_sample_1_3/',
        lib='XSLAgeMh')

    # Multiplicative polynomial
    h1_mlt_emiles, h2_mlt_emiles, h3_mlt_emiles, h4_mlt_emiles = extract_maps(
        prop='sol', length=4, order_range=mdegree_range,
        directory = '../data_products/polynomial_ifu/emiles/multiplicative_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')

    h1_mlt_miles, h2_mlt_miles, h3_mlt_miles, h4_mlt_miles = extract_maps(
        prop='sol', length=4, order_range=mdegree_range,
        directory = '../data_products/polynomial_ifu/miles/multiplicative_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')
    
    h1_mlt_xsl, h2_mlt_xsl, h3_mlt_xsl, h4_mlt_xsl = extract_maps(
        prop='sol', length=4, order_range=mdegree_range,
        directory = '../data_products/polynomial_ifu/xsl/multiplicative_polynomial/fov_sample_1_3/',
        lib='XSLAgeMh')
    
    # Reduced Chi-squared
    
    # Additive polynomial
    chi2_add_miles = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/polynomial_ifu/miles/additive_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')
    
    chi2_add_emiles = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/polynomial_ifu/emiles/additive_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')
    
    chi2_add_xsl = extract_maps(
        prop='chi2', length=1, order_range=degree_range,
        directory = '../data_products/polynomial_ifu/xsl/additive_polynomial/fov_sample_1_3/',
        lib='XSLAgeMh')
    
    # Multiplicative polynomial
    chi2_mlt_miles = extract_maps(
        prop='chi2', length=1, order_range=mdegree_range,
        directory = '../data_products/polynomial_ifu/miles/multiplicative_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')
    
    chi2_mlt_emiles = extract_maps(
        prop='chi2', length=1, order_range=mdegree_range,
        directory = '../data_products/polynomial_ifu/emiles/multiplicative_polynomial/fov_sample_1_3/',
        lib='MilesAgeMh')
    
    chi2_mlt_xsl = extract_maps(
        prop='chi2', length=1, order_range=mdegree_range,
        directory = '../data_products/polynomial_ifu/xsl/multiplicative_polynomial/fov_sample_1_3/',
        lib='XSLAgeMh')

    # Sigma
    # Additive
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(2, 1, figsize=(12,4), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    
    # Box
    ax[0].axhline(1, color='gray', ls='dotted')
    
    norm_add = np.nanmedian(
        np.array([h2_add_xsl, h2_add_emiles, h2_add_miles]))
    # norm_add = None
    
    bp_miles = boxplot(ax=ax[0], data=h2_add_miles, norm=norm_add,
                       color=plt.cm.tab20(0), order_range=degree_range,
                       n_offset=-1)
    bp_emiles = boxplot(ax=ax[0],data=h2_add_emiles, norm=norm_add,
                        color=plt.cm.tab20(2), order_range=degree_range)
    bp_xsl = boxplot(ax=ax[0],data=h2_add_xsl, norm=norm_add,
                     color=plt.cm.tab20(4), order_range=degree_range,
                     n_offset=1)
    
    # Lines
    ax[1].axhline(1, color='gray', ls='dotted')
    mean_miles = mean_plot(ax=ax[1], data=h2_add_miles, color=plt.cm.tab20(0),
                       order_range=degree_range)
    mean_emiles = mean_plot(ax=ax[1],data=h2_add_emiles, color=plt.cm.tab20(2),
                        order_range=degree_range)
    mean_xsl = mean_plot(ax=ax[1],data=h2_add_xsl, color=plt.cm.tab20(4),
                     order_range=degree_range)
    
    # Customize the major grid
    ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray',
               axis='y')
    # Customize the minor grid
    ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
               axis='y')
    
    ax[0].set_xticks(degree_range)
    ax[0].set_xticklabels(degree_range)
    
    ax[1].set_xlabel('Polynomial degree (n)')
    ax[0].set_ylabel(r'$\sigma_{\star, \rm{med}}$')
    ax[1].set_ylabel(r'Average $\sigma_{\star, \rm{med}}$')
    ax[0].set_title('Additive Legendre polynomial')
    ax[1].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    ax[0].legend(
        [bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
        [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    plt.xlim(-1.5,)
    plt.savefig('../plots/polynomial_degree/boxplot_sigma_additive.pdf')
    
    # Multiplicative
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(2, 1, figsize=(12,4), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    
    # Box
    ax[0].axhline(1, color='gray', ls='dotted')
    norm_mlt = np.nanmedian(np.array([h2_mlt_xsl, h2_mlt_emiles, h2_mlt_miles]))
    # norm_mlt = None
    
    bp_miles = boxplot(ax=ax[0], data=h2_mlt_miles, norm=norm_mlt,
                       color=plt.cm.tab20(0), order_range=mdegree_range,
                       n_offset=-1)
    bp_emiles = boxplot(ax=ax[0], data=h2_mlt_emiles, norm=norm_mlt,
                        color=plt.cm.tab20(2), order_range=mdegree_range)
    bp_xsl = boxplot(ax=ax[0], data=h2_mlt_xsl, norm=norm_mlt,
                     color=plt.cm.tab20(4), order_range=mdegree_range,
                     n_offset=1)
    
    # Lines
    ax[1].axhline(1, color='gray', ls='dotted')
    mean_miles = mean_plot(ax=ax[1], data=h2_mlt_miles, color=plt.cm.tab20(0),
                       order_range=mdegree_range)
    mean_emiles = mean_plot(ax=ax[1],data=h2_mlt_emiles, color=plt.cm.tab20(2),
                        order_range=mdegree_range)
    mean_xsl = mean_plot(ax=ax[1],data=h2_mlt_xsl, color=plt.cm.tab20(4),
                     order_range=mdegree_range)
    
    # Customize the major grid
    ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray',
               axis='y')
    # Customize the minor grid
    ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
               axis='y')
    
    ax[0].set_xticks(mdegree_range)
    ax[0].set_xticklabels(mdegree_range)
    
    ax[1].set_xlabel('Polynomial degree (n)')
    ax[0].set_ylabel(r'$\sigma_{\star, \rm{med}}$')
    ax[1].set_ylabel(r'Average $\sigma_{\star, \rm{med}}$')
    ax[0].set_title('Multiplicative Legendre polynomial')
    ax[0].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    ax[0].legend(
        [bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
        [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
    plt.tight_layout()
    plt.xlim(-1.5,)
    plt.savefig('../plots/polynomial_degree/boxplot_sigma_multiplicative.pdf')
    
    # H3 e H4
    
    # Additive
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(2, 1, figsize=(12,3), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1]})
    
    h3_norm_add = np.nanmedian(
        np.array([h3_add_xsl, h3_add_emiles, h3_add_miles]))
    h4_norm_add = np.nanmedian(
        np.array([h4_add_xsl, h4_add_emiles, h4_add_miles]))
    # h3_norm_add = None
    # h4_norm_add = None
    
    # Lines
    # h3
    mean_miles = mean_plot(ax=ax[0], data=h3_add_miles, norm=h3_norm_add,
                           color=plt.cm.tab20(0), order_range=degree_range)
    mean_emiles = mean_plot(ax=ax[0],data=h3_add_emiles, norm=h3_norm_add,
                            color=plt.cm.tab20(2), order_range=degree_range)
    mean_xsl = mean_plot(ax=ax[0],data=h3_add_xsl, norm=h3_norm_add,
                         color=plt.cm.tab20(4), order_range=degree_range)
    
    # Customize the major grid
    ax[0].grid(which='major', linestyle='-', linewidth='0.4', color='gray',
               axis='y')
    # Customize the minor grid
    ax[0].grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
               axis='y')
    
    ax[0].set_xticks(degree_range)
    ax[0].set_xticklabels(degree_range)
    
    # h4
    # ax[1].axhline(1, color='gray', ls='dotted')
    mean_miles = mean_plot(ax=ax[1], data=h4_add_miles, norm=h4_norm_add,
                           color=plt.cm.tab20(0), order_range=degree_range)
    mean_emiles = mean_plot(ax=ax[1],data=h4_add_emiles, norm=h4_norm_add,
                            color=plt.cm.tab20(2), order_range=degree_range)
    mean_xsl = mean_plot(ax=ax[1],data=h4_add_xsl, norm=h4_norm_add,
                         color=plt.cm.tab20(4), order_range=degree_range)
    
    # Customize the major grid
    ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray',
               axis='y')
    # Customize the minor grid
    ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
               axis='y')
    
    ax[1].set_xticks(degree_range)
    ax[1].set_xticklabels(degree_range)
    
    ax[1].set_xlabel('Polynomial degree (n)')
    ax[0].set_ylabel(r'h$_3$')
    ax[1].set_ylabel(r'h$_4$')
    ax[0].set_title('Additive Legendre polynomial')
    ax[1].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    ax[1].legend(
        [bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
        [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    plt.xlim(-1.5,)
    plt.savefig('../plots/polynomial_degree/mean_h3_h4_additive.pdf')
    
    # Multiplicative
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(2, 1, figsize=(12,3), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1]})
    
    h3_norm_mlt = np.nanmedian(
        np.array([h3_mlt_xsl, h3_mlt_emiles, h3_mlt_miles]))
    h4_norm_mlt = np.nanmedian(
        np.array([h4_mlt_xsl, h4_mlt_emiles, h4_mlt_miles]))
    # h4_norm_mlt = None
    # h3_norm_mlt = None
    
    # Lines
    # h3
    mean_miles = mean_plot(ax=ax[0], data=h3_mlt_miles, norm=h3_norm_mlt,
                           color=plt.cm.tab20(0), order_range=mdegree_range )
    mean_emiles = mean_plot(ax=ax[0],data=h3_mlt_emiles, norm=h3_norm_mlt,
                            color=plt.cm.tab20(2), order_range=mdegree_range )
    mean_xsl = mean_plot(ax=ax[0],data=h3_mlt_xsl, norm=h3_norm_mlt,
                         color=plt.cm.tab20(4), order_range=mdegree_range )
    
    # Customize the major grid
    ax[0].grid(which='major', linestyle='-', linewidth='0.4', color='gray',
               axis='y')
    # Customize the minor grid
    ax[0].grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
               axis='y')
    
    ax[0].set_xticks(mdegree_range )
    ax[0].set_xticklabels(mdegree_range )
    
    # h4
    # ax[1].axhline(1, color='gray', ls='dotted')
    mean_miles = mean_plot(ax=ax[1], data=h4_mlt_miles, norm=h4_norm_mlt,
                           color=plt.cm.tab20(0), order_range=mdegree_range)
    mean_emiles = mean_plot(ax=ax[1],data=h4_mlt_emiles, norm=h4_norm_mlt,
                            color=plt.cm.tab20(2), order_range=mdegree_range)
    mean_xsl = mean_plot(ax=ax[1],data=h4_mlt_xsl, norm=h4_norm_mlt,
                         color=plt.cm.tab20(4), order_range=mdegree_range)
    
    # Customize the major grid
    ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray',
               axis='y')
    # Customize the minor grid
    ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
               axis='y')
    
    ax[1].set_xticks(mdegree_range )
    ax[1].set_xticklabels(mdegree_range )
    
    ax[1].set_xlabel('Polynomial degree (n)')
    ax[0].set_ylabel(r'Average h$_3$')
    ax[1].set_ylabel(r'Average h$_4$')
    ax[0].set_title('Multiplicative Legendre polynomial')
    ax[1].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    ax[1].legend(
        [bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
        [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    plt.xlim(-1.5,)
    plt.savefig('../plots/polynomial_degree/mean_h3_h4_multiplicative.pdf')
    
    # Reduced Chi-squared
    
    # Additive
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(figsize=(12,2))
    
    chi2_norm_add = np.nanmedian(
        np.array([chi2_add_xsl, chi2_add_emiles, chi2_add_miles]))
    # chi2_norm_add = False
    
    # Lines
    # chi2
    mean_miles = mean_plot(ax=ax, data=chi2_add_miles, norm=chi2_norm_add,
                           color=plt.cm.tab20(0), order_range=degree_range)
    mean_emiles = mean_plot(ax=ax,data=chi2_add_emiles, norm=chi2_norm_add,
                            color=plt.cm.tab20(2), order_range=degree_range)
    mean_xsl = mean_plot(ax=ax,data=chi2_add_xsl, norm=chi2_norm_add,
                         color=plt.cm.tab20(4), order_range=degree_range)
    
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.4', color='gray',
            axis='y')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
            axis='y')
    
    ax.set_xticks(degree_range)
    ax.set_xticklabels(degree_range)
    
    ax.set_xlabel('Polynomial degree (n)')
    ax.set_ylabel(r'Average reduced $\chi^{2}$')
    ax.set_title('Additive Legendre polynomial')
    ax.legend([mean_miles[0], mean_emiles[0], mean_xsl[0]],
              [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    plt.tight_layout()
    plt.xlim(-1.5, )
    plt.savefig('../plots/polynomial_degree/reduced_chi2_additive.pdf')
    
    # Multiplicative    
    plt.style.use('../src/fig_conf.mplstyle')
    fig, ax = plt.subplots(figsize=(12,2))
    
    chi2_norm_mlt = np.nanmedian(
        np.array([chi2_mlt_xsl, chi2_mlt_emiles, chi2_mlt_miles]))
    # chi2_norm_mlt = False
    
    # Lines
    # chi2
    mean_miles = mean_plot(ax=ax, data=chi2_mlt_miles, norm=chi2_norm_mlt,
                           color=plt.cm.tab20(0), order_range=mdegree_range)
    mean_emiles = mean_plot(ax=ax,data=chi2_mlt_emiles, norm=chi2_norm_mlt,
                            color=plt.cm.tab20(2), order_range=mdegree_range)
    mean_xsl = mean_plot(ax=ax,data=chi2_mlt_xsl, norm=chi2_norm_mlt,
                         color=plt.cm.tab20(4), order_range=mdegree_range)
    
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.4', color='gray',
            axis='y')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray',
            axis='y')
    
    ax.set_xticks(mdegree_range)
    ax.set_xticklabels(mdegree_range)
    
    ax.set_xlabel('Polynomial degree (n)')
    ax.set_ylabel(r'Average reduced $\chi^{2}$')
    ax.set_title('Multiplicative Legendre polynomial')
    ax.legend([mean_miles[0], mean_emiles[0], mean_xsl[0]],
              [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    plt.tight_layout()
    plt.xlim(-1.5,)
    plt.savefig('../plots/polynomial_degree/reduced_chi2_multiplicative.pdf')