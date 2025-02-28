#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:27:47 2022

@author: Luiz
"""
import numpy as np
from scipy.constants import physical_constants

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

# Emission-lines mask

linewidth = 1200 # km/s

wave_list = [     # air
    4861.333, 	# Hb
    4958.911, 	# [O III]
    5006.843, 	# [O III]
    5197.577, 	# Fe II
    5200.257, 	# [N I]
    5875.624,   # He I
    6300.304, 	# [O I]
    6363.776, 	# [O I]
    6548.050, 	# [N II]
    6562.819, 	# Ha
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

wave_list = np.asarray(wave_list)

lower_bound = wave_list / np.e**((linewidth*0.5) / C)
upper_bound = wave_list * np.e**((linewidth*0.5) / C)

# line_bound = np.column_stack([lower_bound, upper_bound])
np.round(lower_bound, 1, out=lower_bound)
np.round(upper_bound, 1, out=upper_bound)


for i in range(wave_list.size):
   print(f'[{lower_bound[i]}, {upper_bound[i]}],  # {wave_list[i]:.2f} {name_list[i]:10} ({linewidth} km/s)')

#%% Spectrum edges mask

edges = [4750, 7350]
edge_name = ["Nominal start", "Nominal end  "]
sigma = 200
edge_width = 2*3*sigma

edges = np.asarray(edges)

lower_bound_edge = edges / np.e**((edge_width*0.5) / C)
upper_bound_edge = edges * np.e**((edge_width*0.5) / C)

np.round(lower_bound_edge, 1, out=lower_bound_edge)
np.round(upper_bound_edge, 1, out=upper_bound_edge)

for i in range(edges.size):
   print(f'[{lower_bound_edge[i]}, {upper_bound_edge[i]}], # {edges[i]:.2f} {edge_name[i]:10} (+-3*sigma_star ~{edge_width} km/s)')
