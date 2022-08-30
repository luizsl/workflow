#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:47:28 2022

@author: Luiz
"""

import json
from abc import ABC, abstractmethod

# import matplotlib.pyplot as plt
import numpy as np
import ppxf.ppxf_util as util


def lsf_muse(lamb):
    lsf = 5.866e-8*lamb**2 - 9.187e-4*lamb + 6.040
    return lsf


class AbstractModel(ABC):
    @abstractmethod    
    def __init__(self):
        pass
        
    
class Model(AbstractModel):
    meta = {}
    def __init__(self):
        pass
    
    def build_model(self, wave, z=0):
        self.meta['wave'] = np.array(wave)
        self.meta['z'] = z
        
        lam_range_gal = [np.min(self.meta['wave']), np.max(self.meta['wave'])]
        
        self.gas_templates, self.gas_names, self.line_wave = \
            util.emission_lines(
                ln_lam_temp=np.log(self.meta['wave']),
                lam_range_gal=lam_range_gal, FWHM_gal=lsf_muse,
                pixel=True, tie_balmer=False, 
                limit_doublets=False, vacuum=False)


if __name__ == '__main__':
    
    path_metadata = '../../data_products/toy_100x100/MilesAgeMh/ppxf/metadata.json'
    with open(path_metadata) as f:
        metadata = json.load(f)
        
    wave = np.array(metadata['obs']['wave_obs'])
    em_model = Model()
    em_model.build_model(wave, z=0.004)
