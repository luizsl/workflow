#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 23:13:13 2022

@author: Luiz

Plot MUSE LSF
"""

import sys
sys.path.insert(1, '/home/chess-lin/Documents/Git/workflow/exploration/run_ppxf')
from compute_muse_lsf import equation_lsf

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

C = speed_of_light/1e3
Z = 0.004951

wave = 4750 + 1.25*np.arange(3680)
wave_z = wave / (1 + Z)
lsf_ang = equation_lsf(wave)
lsf_ang_z = equation_lsf(wave, z=Z)
lsf_kms = equation_lsf(wave, unit='kms')
lsf_kms_z = equation_lsf(wave, unit='kms', z=Z)

# Plot
plt.style.use('../src/fig_conf.mplstyle')

fig, ax1 = plt.subplots(figsize=(7,3), tight_layout=True)

ax1.plot(wave, lsf_ang, color='navy', label='rest [\AA]')
ax1.plot(wave_z, lsf_ang_z, color='royalblue', label=r'z $\approx$ 0.00495 [\AA]', ls='dashed')

ax2 = ax1.twinx()
ax2.plot(wave, lsf_kms, color='darkred', label ='rest [km\,s$^{-1}$]')
ax2.plot(wave_z, lsf_kms_z, color='indianred', label=r'z $\approx$ 0.00495 [km\,s$^{-1}$]', ls='dashed')

# fig.legend(loc = "upper right", 
#            bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes)
ax1.set_xlabel(r'Wavelength [\AA]')
ax1.set_ylabel(r'FWHM [\AA]')
ax2.set_ylabel(r'Instrumental dispersion [km/s]')

plt.savefig('../plots/muse_lsf_bacon_redshift.pdf')
