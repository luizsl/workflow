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
    
    # Velocity dispersion
    
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
    ax.set_ylabel(r'Reduced $\chi^2$')
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
    
    plt.legend()
    plt.tight_layout()
    
#%%  Boxplot

'''
- Plot the gauss-hermite moments as a function of legendre polynomial degree
- the data is normalized with the median of the sequence
- boxplot is the candidate to the geometry
- panels of additive and multiplicative polynomials
'''

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
    
#%% Sigma

# Additive

degree_range = np.arange(-1, 21, 5)

plt.style.use('../src/fig_conf.mplstyle')
fig, ax = plt.subplots(2, 1, figsize=(12,4), sharex=True,
                       gridspec_kw={'height_ratios': [2, 1]})

# Box
ax[0].axhline(1, color='gray', ls='dotted')

# norm_add = np.nanmedian(np.array([h2_add_xsl, h2_add_emiles, h2_add_miles]))
norm_add = None

bp_miles = boxplot(ax=ax[0], data=h2_add_miles, norm=norm_add, color=plt.cm.tab20(0),
                   order_range=degree_range, n_offset=-1)
bp_emiles = boxplot(ax=ax[0],data=h2_add_emiles, norm=norm_add, color=plt.cm.tab20(2),
                    order_range=degree_range)
bp_xsl = boxplot(ax=ax[0],data=h2_add_xsl, norm=norm_add, color=plt.cm.tab20(4),
                 order_range=degree_range, n_offset=1)

# Lines
ax[1].axhline(1, color='gray', ls='dotted')
mean_miles = mean_plot(ax=ax[1], data=h2_add_miles, color=plt.cm.tab20(0),
                   order_range=degree_range)
mean_emiles = mean_plot(ax=ax[1],data=h2_add_emiles, color=plt.cm.tab20(2),
                    order_range=degree_range)
mean_xsl = mean_plot(ax=ax[1],data=h2_add_xsl, color=plt.cm.tab20(4),
                 order_range=degree_range)

# Customize the major grid
ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax[0].set_xticks(degree_range)
ax[0].set_xticklabels(degree_range)

ax[1].set_xlabel('Polynomial degree (n)')
ax[0].set_ylabel(r'$\sigma_{\star, \rm{med}}$')
ax[1].set_ylabel(r'Average $\sigma_{\star, \rm{med}}$')
ax[0].set_title('Additive Legendre polynomial')
ax[1].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
ax[0].legend([bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
          [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
fig.align_ylabels(ax[:])
plt.tight_layout()

# Multiplicative

mdegree_range = np.arange(0, 21, 5)

plt.style.use('../src/fig_conf.mplstyle')
fig, ax = plt.subplots(2, 1, figsize=(12,4), sharex=True,
                       gridspec_kw={'height_ratios': [2, 1]})

# Box
ax[0].axhline(1, color='gray', ls='dotted')
# norm_mlt = np.nanmedian(np.array([h2_mlt_xsl, h2_mlt_emiles, h2_mlt_miles]))
norm_mlt = None

bp_miles = boxplot(ax=ax[0], data=h2_mlt_miles, norm=norm_mlt, color=plt.cm.tab20(0),
                   order_range=mdegree_range, n_offset=-1)
bp_emiles = boxplot(ax=ax[0], data=h2_mlt_emiles, norm=norm_mlt, color=plt.cm.tab20(2),
                    order_range=mdegree_range)
bp_xsl = boxplot(ax=ax[0], data=h2_mlt_xsl, norm=norm_mlt, color=plt.cm.tab20(4),
                 order_range=mdegree_range, n_offset=1)

# Lines
ax[1].axhline(1, color='gray', ls='dotted')
mean_miles = mean_plot(ax=ax[1], data=h2_mlt_miles, color=plt.cm.tab20(0),
                   order_range=mdegree_range)
mean_emiles = mean_plot(ax=ax[1],data=h2_mlt_emiles, color=plt.cm.tab20(2),
                    order_range=mdegree_range)
mean_xsl = mean_plot(ax=ax[1],data=h2_mlt_xsl, color=plt.cm.tab20(4),
                 order_range=mdegree_range)

# Customize the major grid
ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax[0].set_xticks(mdegree_range)
ax[0].set_xticklabels(mdegree_range)

ax[1].set_xlabel('Polynomial degree (n)')
ax[0].set_ylabel(r'$\sigma_{\star, \rm{med}}$')
ax[1].set_ylabel(r'Average $\sigma_{\star, \rm{med}}$')
ax[0].set_title('Multiplicative Legendre polynomial')
ax[0].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
ax[0].legend([bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
          [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
plt.tight_layout()

#%% H3 e H4

# Additive

degree_range = np.arange(-1, 21, 5)

plt.style.use('../src/fig_conf.mplstyle')
fig, ax = plt.subplots(2, 1, figsize=(12,3), sharex=True,
                       gridspec_kw={'height_ratios': [1, 1]})

# norm_add = np.nanmedian(np.array([h3_add_xsl, h3_add_emiles, h3_add_miles]))
norm_add = None

# Lines

# h3
mean_miles = mean_plot(ax=ax[0], data=h3_add_miles, norm=norm_add,
                       color=plt.cm.tab20(0), order_range=degree_range)
mean_emiles = mean_plot(ax=ax[0],data=h3_add_emiles, norm=norm_add,
                        color=plt.cm.tab20(2), order_range=degree_range)
mean_xsl = mean_plot(ax=ax[0],data=h3_add_xsl, norm=norm_add,
                     color=plt.cm.tab20(4), order_range=degree_range)

# Customize the major grid
ax[0].grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax[0].grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax[0].set_xticks(degree_range)
ax[0].set_xticklabels(degree_range)

# h4
# ax[1].axhline(1, color='gray', ls='dotted')
mean_miles = mean_plot(ax=ax[1], data=h4_add_miles, norm=norm_add,
                       color=plt.cm.tab20(0), order_range=degree_range)
mean_emiles = mean_plot(ax=ax[1],data=h4_add_emiles, norm=norm_add,
                        color=plt.cm.tab20(2), order_range=degree_range)
mean_xsl = mean_plot(ax=ax[1],data=h4_add_xsl, norm=norm_add,
                     color=plt.cm.tab20(4), order_range=degree_range)

# Customize the major grid
ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax[1].set_xticks(degree_range)
ax[1].set_xticklabels(degree_range)

ax[1].set_xlabel('Polynomial degree (n)')
ax[0].set_ylabel(r'h$_3$')
ax[1].set_ylabel(r'h$_4$')
ax[0].set_title('Additive Legendre polynomial')
ax[1].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
ax[0].legend([bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
          [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
fig.align_ylabels(ax[:])
plt.tight_layout()

# Multiplicative

mdegree_range  = np.arange(0, 21, 5)

plt.style.use('../src/fig_conf.mplstyle')
fig, ax = plt.subplots(2, 1, figsize=(12,3), sharex=True,
                       gridspec_kw={'height_ratios': [1, 1]})

# norm_mlt = np.nanmedian(np.array([h3_mlt_xsl, h3_mlt_emiles, h3_mlt_miles]))
norm_mlt = None

# Lines

# h3
mean_miles = mean_plot(ax=ax[0], data=h3_mlt_miles, norm=norm_mlt,
                       color=plt.cm.tab20(0), order_range=mdegree_range )
mean_emiles = mean_plot(ax=ax[0],data=h3_mlt_emiles, norm=norm_mlt,
                        color=plt.cm.tab20(2), order_range=mdegree_range )
mean_xsl = mean_plot(ax=ax[0],data=h3_mlt_xsl, norm=norm_mlt,
                     color=plt.cm.tab20(4), order_range=mdegree_range )

# Customize the major grid
ax[0].grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax[0].grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax[0].set_xticks(mdegree_range )
ax[0].set_xticklabels(mdegree_range )

# h4
# ax[1].axhline(1, color='gray', ls='dotted')
mean_miles = mean_plot(ax=ax[1], data=h4_mlt_miles, norm=norm_mlt,
                       color=plt.cm.tab20(0), order_range=mdegree_range )
mean_emiles = mean_plot(ax=ax[1],data=h4_mlt_emiles, norm=norm_mlt,
                        color=plt.cm.tab20(2), order_range=mdegree_range )
mean_xsl = mean_plot(ax=ax[1],data=h4_mlt_xsl, norm=norm_mlt,
                     color=plt.cm.tab20(4), order_range=mdegree_range )

# Customize the major grid
ax[1].grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax[1].grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax[1].set_xticks(mdegree_range )
ax[1].set_xticklabels(mdegree_range )

ax[1].set_xlabel('Polynomial degree (n)')
ax[0].set_ylabel(r'Average h$_3$')
ax[1].set_ylabel(r'Average h$_4$')
ax[0].set_title('Multiplicative Legendre polynomial')
ax[1].xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
ax[0].legend([bp_miles["whiskers"][0], bp_emiles["whiskers"][0], bp_xsl["whiskers"][0]],
          [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
fig.align_ylabels(ax[:])
plt.tight_layout()

#%% Reduced Chi-squared

# Additive

degree_range = np.arange(-1, 21, 5)

plt.style.use('../src/fig_conf.mplstyle')
fig, ax = plt.subplots(figsize=(12,2))

# norm_add = np.nanmedian(np.array([chi2_add_xsl, chi2_add_emiles, chi2_add_miles]))
norm_add = False

# Lines

# chi2
mean_miles = mean_plot(ax=ax, data=chi2_add_miles, norm=norm_add,
                       color=plt.cm.tab20(0), order_range=degree_range)
mean_emiles = mean_plot(ax=ax,data=chi2_add_emiles, norm=norm_add,
                        color=plt.cm.tab20(2), order_range=degree_range)
mean_xsl = mean_plot(ax=ax,data=chi2_add_xsl, norm=norm_add,
                     color=plt.cm.tab20(4), order_range=degree_range)

# Customize the major grid
ax.grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax.set_xticks(degree_range)
ax.set_xticklabels(degree_range)

ax.set_xlabel('Polynomial degree (n)')
ax.set_ylabel(r'Average Reduced $\chi^{2}$')
ax.set_title('Additive Legendre polynomial')
ax.legend([mean_miles[0], mean_emiles[0], mean_xsl[0]],
          [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
plt.tight_layout()

# Multiplicative
mdegree_range = np.arange(0, 21, 5)

plt.style.use('../src/fig_conf.mplstyle')
fig, ax = plt.subplots(figsize=(12,2))

# norm_mlt = np.nanmedian(np.array([chi2_mlt_xsl, chi2_mlt_emiles, chi2_mlt_miles]))
norm_mlt = False

# Lines

# chi2
mean_miles = mean_plot(ax=ax, data=chi2_mlt_miles, norm=norm_mlt,
                       color=plt.cm.tab20(0), order_range=mdegree_range)
mean_emiles = mean_plot(ax=ax,data=chi2_mlt_emiles, norm=norm_mlt,
                        color=plt.cm.tab20(2), order_range=mdegree_range)
mean_xsl = mean_plot(ax=ax,data=chi2_mlt_xsl, norm=norm_mlt,
                     color=plt.cm.tab20(4), order_range=mdegree_range)

# Customize the major grid
ax.grid(which='major', linestyle='-', linewidth='0.4', color='gray', axis='y')
# Customize the minor grid
ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray', axis='y')

ax.set_xticks(mdegree_range)
ax.set_xticklabels(mdegree_range)

ax.set_xlabel('Polynomial degree (n)')
ax.set_ylabel(r'Average Reduced $\chi^{2}$')
ax.set_title('Multiplicative Legendre polynomial')
ax.legend([mean_miles[0], mean_emiles[0], mean_xsl[0]],
          [r'\texttt{MILES}', r'\texttt{E-MILES}', r'\texttt{XSL}'])
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%i'))
plt.tight_layout()
