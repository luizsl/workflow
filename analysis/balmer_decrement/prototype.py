#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:30:36 2022

@author: Luiz

Compute extinction based on Balmer decrement"""

import json
# import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

file_metadata = ('../../data_products/toy_trick/MilesAgeMh/'
                 'ppxf_emission_line_binned300_2components/'
                 'metadata.json'
                 )

file_output = ('../../data_products/toy_trick/MilesAgeMh/'
               'ppxf_emission_line_binned300_2components/'
               'ppxf_output.json'
               )

with open(file_output) as fp:
    out_ppxf = json.load(fp)

with open(file_metadata) as fp:
    out_metadata = json.load(fp)

gas_names = np.asarray(out_ppxf['results']['gas_names'])
corrected_flux = np.asarray(out_ppxf['results']['corrected_flux'])

xbin = out_metadata['obs']['xbin']
ybin = out_metadata['obs']['ybin']

ha_1 = corrected_flux[8]
hb_1 = corrected_flux[9]
bal_dec_1 = ha_1/hb_1

ha_2 = corrected_flux[18]
hb_2 = corrected_flux[19]
bal_dec_2 = ha_2/hb_2

# Scatter plot
fig, ax = plt.subplots(1, 2)
ax[0].scatter(xbin, ybin, c=bal_dec_1)
ax[1].scatter(xbin, ybin, c=bal_dec_2)

#  Plot of interpolation

inter = interpolate.LinearNDInterpolator
# inter = interpolate.RBFInterpolator

x_full = np.asarray(out_metadata['obs']['x_full'])
y_full = np.asarray(out_metadata['obs']['y_full'])
grid = np.column_stack((x_full, y_full))

_interp_rbf_bal_dec_1 = inter(
    np.column_stack([ybin, xbin]), bal_dec_1)

rbf_bal_dec_1 = _interp_rbf_bal_dec_1(grid)

_interp_rbf_bal_dec_2 = inter(
    np.column_stack([ybin, xbin]), bal_dec_2)

rbf_bal_dec_2 = _interp_rbf_bal_dec_2(grid)

shape = (len(set(y_full.round(5))),
         len(set(x_full.round(5))))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(rbf_bal_dec_1.reshape(shape), origin='lower')
ax[1].imshow(rbf_bal_dec_2.reshape(shape), origin='lower')

#%%
# Interpolation with selection
fig, ax = plt.subplots(1, 2)
ax[0].scatter(xbin, ybin, c=bal_dec_1)
ax[1].scatter(xbin, ybin, c=bal_dec_2)