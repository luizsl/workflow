#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:19:41 2021

@author: Luiz
"""

import glob
import numpy as np
from astropy.io import fits
from ppxf.ppxf import ppxf
from scipy.constants import physical_constants
from os import path
import ppxf as ppxf_package
import matplotlib.pyplot as plt
import spectcube as sc

import compute_muse_lsf as lsf
from convolve import convolve

ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
vazdekis = glob.glob('../data/models/tmpWzZ2t1/Mku1.30Z*.fits')

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

# create a single model to be used as template
with fits.open(vazdekis[0]) as hdu:
    first_wave = hdu['PRIMARY'].header['CRVAL1']
    step_wave = hdu['PRIMARY'].header['CDELT1']
    n_pixel = hdu['PRIMARY'].header['NAXIS1']

template_w = sc.util.build_wave_array([first_wave, step_wave],
                                         sampling_type = 'linear',
                                         size = n_pixel)

new_wave = sc.util.fit_wave_interval([template_w[0], template_w[-1]],
                                     sampling_type = 'ln', size = n_pixel)

templates = np.zeros((n_pixel, len(vazdekis)))
for j, file in enumerate(vazdekis):
    with fits.open(vazdekis[j]) as hdu:
        flux = hdu['PRIMARY'].data
        templates[:, j] = flux
        
############ Model properties ##############

fwhm_model = 2.51 #Vazdekis+10
fwhm_obs = lsf.equation_lsf(template_w, 4750, 9350)
fwhm_dif = np.sqrt((fwhm_obs**2 - fwhm_model**2).clip(0))
sigma = fwhm_dif / (2.355 * step_wave) # Sigma difference in pixels


y = convolve(flux = templates, sigma = sigma)

y, w, _ = sc.resampling(flux = templates,
                                old_wave = template_w, 
                                old_sampling_type = 'linear',
                                new_wave = new_wave,
                                new_sampling_type ='ln')

plt.plot(template_w, templates[:,0])
plt.plot(w, y[:,0])