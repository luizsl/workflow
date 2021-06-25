#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:20:09 2021

@author: Luiz
"""
from matplotlib.pyplot import style
from plot_util import sn_muse_image as snr_map

style.use('fig_conf.mplstyle')
cube_file = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'
snr_map.build_map(cube_file, '../plots/snr_muse_ngc613.pdf')
