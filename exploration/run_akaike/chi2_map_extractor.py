#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:24:20 2022

@author: Luiz
"""
import os
import glob

from astropy.io import fits

def main(pathlist):
    for i, name in enumerate(pathlist):
        name = os.path.join(name, '*.fits')
        filepath = glob.glob(name)[0]
        
        with fits.open(filepath) as hdul:
            # IDEA: Removing the harcode on the following line can make the 
            # script reuseble for extracting other parameters
            chi_map = hdul['RED_CHI'].data
            hdu = fits.PrimaryHDU(chi_map)
            hdul = fits.HDUList([hdu])
            hdul.writeto(f'comp_{i}.fits', overwrite=True)
            
if __name__ == '__main__':
    # IDEA: Employ command line input parameter to a more flexible script 
    path_1comp = '../../data_products/NGC613/miles/ifscube'
    path_2comp = '../../data_products/NGC613/miles/ifscube_5'
    path_3comp = '../../data_products/NGC613/miles/ifscube_6'

    pathlist = [path_1comp, path_2comp, path_3comp]
    main()
    