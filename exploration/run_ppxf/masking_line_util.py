#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:27:47 2022

@author: Luiz
"""
import numpy as np
from scipy.constants import physical_constants

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

#%% Masking emission-line

linewidth = 750 # km/s

emission_list = [     # air
    4861.333, 	# Hβ
    4958.911, 	# [O III]
    5006.843, 	# [O III]
    5197.577, 	# Fe II
    5200.257, 	# [N I]
    5875.624,   # He I
    6300.304, 	# [O I]
    6363.776, 	# [O I]
    6548.050, 	# [N II]
    6562.819, 	# Hα
    6583.460, 	# [N II]
    6716.440, 	# [S II]
    6730.810, 	# [S II]
    9068.600, 	# [S III]
]

name_list = [
    'Hb',
    '[O III]',
    '[O III]',
    'Fe II',
    '[N I]',
    'He I',
    '[O I]',
    '[O I]',
    '[N II]',
    'Ha',
    '[N II]',
    '[S II]',
    '[S II]',
    '[S III]',
]

emission_list = np.asarray(emission_list)

lower_bound = emission_list / np.e**((linewidth*0.5) / C)
upper_bound = emission_list * np.e**((linewidth*0.5) / C)

np.round(lower_bound, 1, out=lower_bound)
np.round(upper_bound, 1, out=upper_bound)


for i in range(emission_list.size):
   print(f'[{lower_bound[i]}, {upper_bound[i]}],  # {emission_list[i]:.2f} {name_list[i]:10} ({linewidth} km/s)')
print('')   

#%% Masking edges

sigma_star = 200 # km/s
z = 0.004951

spectral_edge = [
    4750,
    9350,
]

name_list = [
    'Nominal start',
    'Nominal end',
]

spectral_edge = np.asarray(spectral_edge)

spectral_edge_rest = spectral_edge / (1+z)

lower_bound = spectral_edge / np.e**((3*sigma_star) / C)
upper_bound = spectral_edge * np.e**((3*sigma_star) / C)

np.round(lower_bound, 1, out=lower_bound)
np.round(upper_bound, 1, out=upper_bound)


for i in range(spectral_edge.size):
   print(f'[{lower_bound[i]}, {upper_bound[i]}],  # {spectral_edge[i]:.2f} {name_list[i]:14} (3*sigma_star ~ {3*sigma_star} km/s)')
print('')   

#%% Ansatz

ansatz_region = [
    [7553.0, 7575.0],
    [7587.0, 7620.0],
]

name_list = [
    'Noisy region',
    'Noisy region',
]

ansatz_region = np.asarray(ansatz_region)

for i in range(len(ansatz_region)):
   print(f'[{ansatz_region[i, 0]}, {ansatz_region[i, 1]}],  # {name_list[i]:10}')
print('')   
