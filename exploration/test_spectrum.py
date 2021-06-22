#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:15:31 2021

@author: chess-lin
"""

import unittest
from spectrum import Spectrum
from astropy.io import fits
import numpy as np

file = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

with fits.open(file) as hdu:
    y = hdu[1].data[:,
                    np.random.choice(hdu[1].shape[1],1),
                    np.random.choice(hdu[1].shape[2],1)]
    x = Spectrum(y, wave = [4750., 1.25], medium = 'air', sampling_type = 'linear')
    
class TesteSpectrum(unittest.TestCase):
    
    def teste___init__(self):
        self.assertEqual(x.flux.size, x.wave.size)

    def test__resampling_linear(self):
        wave_log = 10**(3.6766936096248664 + 0.00011427298629357851*np.arange(3600))
        x.resampling(wave_log, new_sampling_type = 'log')
        self.assertEqual(x.flux.size, x.wave.size)
        
    def test__resampling_log(self):
        wave_lin = 4700 + 1.5*np.arange(3000, dtype = np.double)
        x.resampling(wave_lin, new_sampling_type = 'linear')
        self.assertEqual(x.flux.size, x.wave.size)
        
    def test__resampling_ln(self):
        wave_ln = np.e**(8.465899897028686 + 0.0002631232747715068*np.arange(4000))
        x.resampling(wave_ln, new_sampling_type = 'ln')
        self.assertEqual(x.flux.size, x.wave.size)
        
if __name__ == '__main__':
    unittest.main()



