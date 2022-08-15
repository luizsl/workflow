#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:20:00 2021

@author: Luiz

Produce the LSF for MUSE based on equation 8 in Bacon et al. 2017

"""

import numpy as np
from scipy.constants import physical_constants

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s


def find_nearest(array, number):
    diff = np.abs(array - number)    
    nearest_index = diff.argmin()
    return nearest_index
    
def truncate_domain(lamb, lsf, lower_lamb=None, upper_lamb=None):
    if lower_lamb is not None:
        lower_index = find_nearest(lamb, lower_lamb)
        lower_value = lsf[lower_index]
        lsf = [lower_value if l<lower_lamb else lsf[i] 
               for i, l in enumerate(lamb)]
    
    if upper_lamb is not None:
        upper_index = find_nearest(lamb, upper_lamb)
        upper_value = lsf[upper_index]
        lsf = [upper_value if l > upper_lamb else lsf[i] 
               for i, l in enumerate(lamb)]
        
    return np.array(lsf)

def equation_lsf(lamb, lower_lamb=None, upper_lamb=None, unit=None, z=None):
    # udf10, equation (8), Bacon+17
    lsf = 5.866e-8*lamb**2 - 9.187e-4*lamb + 6.040
    
    if lower_lamb or upper_lamb is not None:
        lsf = truncate_domain(lamb, lsf, lower_lamb, upper_lamb)
        
    if unit is None:
        pass
    elif unit == 'a':
        pass
    elif unit == 'kms':
        lsf = (lsf*C)/(2.355*lamb)
    else:
        raise Exception

    if z is not None:
        lsf = lsf/(1+z)
        
    return lsf


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.arange(4500, 9400, 1)
    lsf = equation_lsf(x, 4600, 9300, 'a')
    plt.plot(x, lsf)
    
    x = np.arange(4500, 9400, 1)
    lsf = equation_lsf(x, 4600, 9300, 'a', z=0.005)
    plt.plot(x, lsf)
