#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 17:52:01 2021

@author: chess-lin
"""

import pandas as pd
import matplotlib.pyplot as plt

lsf_file = '../../data/misc_data/muse_manual_resolution.csv'

lsf = pd.read_csv(lsf_file)

# Plot
plt.style.use('fig_conf.mplstyle')

fig, ax = plt.subplots()
ax.plot(lsf['lambda'], lsf['fwhm_A'], label = r'FWHM in $\AA$',
        color = 'C0')

ax2 = ax.twinx()
ax2.plot(lsf['lambda'], lsf['sigma_kms'],
         label = r'$\sigma_{\text{ins}}$ in $\kmps$', color = 'C1')

fig.legend(loc = "upper right", 
           bbox_to_anchor = (1, 1), bbox_transform = ax.transAxes)
ax.set_xlabel(r'$\textbf{Wavelength} \mathbf{(\AA)}$')
ax.set_ylabel(r'\textbf{FWHM of LSF (\AA)}')
ax2.set_ylabel(r'\textbf{$\sigma_{\text{ins}}$ ($\kmps$})')
ax.set_title(r'MUSE LSF')

plt.savefig('../../plots/muse_lsf.pdf')
