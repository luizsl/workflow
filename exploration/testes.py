#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:12:48 2021

@author: chess-lin
"""

import glob
from time import perf_counter as clock
from os import path
import matplotlib.pyplot as plt

from astropy.io import fits
import numpy as np

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from spectrum import Spectrum
from scipy import interpolate


c = 299792.458  

ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

file = ppxf_dir + '/spectra/NGC4550_SAURON.fits'
hdu = fits.open(file)
h2 = hdu[0].header
data = hdu[0].data
wave = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
plt.plot(wave, data)

spec = Spectrum(data, [h2['CRVAL1'], h2['CDELT1']], sampling_type = 'linear')

# resampling and rebinning
wave_ln = np.e**np.array(np.log(spec.wave[0]) + np.arange(395) * np.log(wave[1]/wave[0]))
spec.resampling(new_wave = wave_ln, new_sampling_type = 'ln')
spec.plot()

spec.rebinning(wave_ln, new_sampling_type = 'ln')
spec.plot()

data_r, wave_r, vel_r = util.log_rebin([wave[0], wave[-1]], data, flux = False)
plt.plot(np.e**wave_r, data_r)

# convolution
sigma = np.full_like(spec.flux, 2)
spec.convolve(sigma)
spec.plot()
