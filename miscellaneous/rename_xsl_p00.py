#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:48:42 2023

@author: Luiz
"""
import os
import glob
import shutil

import numpy as np
from astropy.io import fits


model_directory = '/mnt/m2/Git/workflow/data/models/XSL_SSP_P00_Kroupa/Kroupa'

new_directory = list(os.path.split(model_directory))
new_directory[0] = f'{new_directory[0]}_renamed'
new_directory = os.path.join(*new_directory)

try:
    os.makedirs(new_directory)
except Exception as e:
    raise e

files = glob.glob(os.path.join(model_directory, '*'))
new_names = set()
age_range = set()
z_range = set()
mh_range = set()
for file in files:
    age = float(fits.getval(file, 'LOGAGE'))
    z = float(fits.getval(file, 'MH'))

    z_sol = 0.019
    x_sol = 1 - (0.23 + 2.25*z_sol) - z_sol
    x = 1 - (0.23 + 2.25*z) - z

    mh = np.log10(z/x) - np.log10(z_sol/x_sol)

    age_range.add(age)
    z_range.add(z)
    mh_range.add(mh)
    if mh == 0:
        new_name = f'XSL_SSP_logT{np.log10(age):.2f}_MH-{mh:.1f}_Kroupa_P00.fits'
    else:
        new_name = f'XSL_SSP_logT{np.log10(age):.2f}_MH{mh:.1f}_Kroupa_P00.fits'
    print(new_name)
    new_names.add(new_name)

    src = os.path.join(model_directory, file)
    dst = os.path.join(new_directory, new_name)
    shutil.copy(src, dst)
