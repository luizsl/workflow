#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:26:35 2023

@author: Luiz
"""

import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits



with plt.style.context(['science', 'nature']):

    flux_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
                 'XSLAgeMh_3/ppxf_emission_line_1components/'
                 'corrected_flux.fits'
                 )
   
    with fits.open(flux_path) as hdul:
        flux = hdul[0].data
    
    fig, ax = plt.subplots()

    # ax.imshow(np.arcsinh(eqw), origin='lower')

    ax.scatter(np.log10((3/4)*flux[4] / flux[9]),
               flux[9]*1e-20/cont,
               alpha=0.02, marker='o', s=3, linewidths=0.01)
    ax.set_yscale('log')

    ax.set_xlim(-1.1, 0.6)
    ax.set_ylim(0.1, 100)
    
    ax.hlines(0.5, -5, 5)
    ax.hlines(6, -0.4, 5, color='k')
    ax.vlines(-0.4, 0, 200, color='k')
    # ax.set_xlabel(r'$\log_{10} \left(\rm{[NII] \lambda6583} / \rm{H}\alpha\right)$')
    # ax.set_ylabel(r'$\log_{10} \left(\rm{[OIII]\lambda5007} / \rm{H}\beta\right)$')
    # fig.xlim(-1.5, 1)
    # fig.ylim(-1.3, 1.5)
    
    fig.show()
