#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:30:36 2022

@author: Luiz

Compute extinction based on Balmer decrement"""

import json
# import re

import numpy as np

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


corrected_flux = np.asarray(out_ppxf['results']['corrected_flux'])
ha_1 = corrected_flux[8]
hb_1 = corrected_flux[9]

balmer_dec_1 = ha_1/hb_1

ha_2 = corrected_flux[18]
hb_2 = corrected_flux[19]

balmer_dec_2 = ha_2/hb_2


plt.hist(balmer_dec_1, alpha=0.3, bins=5)
plt.hist(balmer_dec_2, alpha=0.3, bins=5)
