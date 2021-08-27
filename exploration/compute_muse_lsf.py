#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:20:00 2021

@author: Luiz

Produce the LSF for MUSE based on equation 8 in Bacon et al. 2017

"""

import numpy as np

def equation_lsf(x, lower = None, upper = None):
    lsf = 5.866e-8*x**2 - 9.187e-4*x + 6.040
    
    if lower is not None:
        lsf = [10 for i in x if i < lower]
        
    return lsf

if __name__ == '__main__':
    x = np.arange(5000, 6000, 1)
    lsf = equation_lsf(x)
