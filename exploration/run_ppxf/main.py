#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:55:58 2021

@author: Luiz
"""
import os
import tempfile
import shutil
from build_metadata import Meta
from data_preprocessing import DataPreprocessing
from ppxf_execution import ExecutePpxf
from post_processing import PostProcessing
import _pickle as pickle

if __name__ == '__main__':
    # read configurations
    conf_file = 'config.ini'
    metadata = Meta(conf_file = 'config.ini')
    metadata.temp_input_dir = tempfile.mkdtemp(dir =  '.')
    metadata.temp_output_dir = tempfile.mkdtemp(dir =  '.')
    
    # create file to record metadata
    with open('metadata.pkl', 'wb') as out:
        pickle.dump(metadata, out)
        
    # create output folder
    if os.path.isdir(metadata.output_dir) is True:
        raise Exception
    else:
        os.mkdir(metadata.output_dir)
    
    DataPreprocessing('metadata.pkl')
    ExecutePpxf('metadata.pkl')
    PostProcessing('metadata.pkl')

    shutil.move('metadata.pkl', f'{metadata.output_dir}/metadata.pkl')
    
    # Removing temporary files
    shutil.rmtree(metadata.temp_input_dir)
    shutil.rmtree(metadata.temp_output_dir)
    
# import numpy as np
# from astropy.io import fits
# import matplotlib.pyplot as plt

# hdu = fits.open('NGC613_1/velocity.fits')
# plt.imshow(hdu[0].data)
# plt.show()

# hdu = fits.open('NGC613_1/bestfit.fits')
# plt.plot(hdu[0].data[:, 10, 10])
