#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:55:58 2021

@author: Luiz
"""

import json
import logging
import os
import shutil
import sys

import yaml
from astropy.utils.misc import JsonCustomEncoder

from data_preprocessing import DataPreprocessing
from post_processing import PopMeanProperties
from ppxf_execution import ExecutePpxf


class Main:
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.start_logging()

    def start_logging(self):
        formatter = logging.Formatter('%(message)s')
        loglevel = logging.DEBUG

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(loglevel)
        
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(loglevel)
        logger.addHandler(stream_handler)
        
        self.logger = logger
        
    def run_all(self):
        self.read_config()
        self.create_output_folder()
        self.keep_conf_copy()

        self.data = DataPreprocessing(self.meta)
        self.ppxf_out = ExecutePpxf(self.data, self.meta)

        self.meta_to_json()

        if 'secondary' in self.meta['output']:
            self.logger.info(f'\nComputing secondary properties\n{30*"-"}')
            self.compute_secondary()

    def read_config(self):
        with open(self.conf_file) as f:
            self.meta = yaml.load(f, Loader=yaml.Loader)

    def create_output_folder(self):
        dir_ = os.path.join(
            self.meta['output']['output_root'],
            self.meta['observation']['obs_name'])
        if os.path.isdir(dir_) is False:
            os.makedirs(dir_, exist_ok=True)

        sec_dir_ = os.path.join(dir_, self.meta['model']['class_'])

        # create unique name
        if os.path.isdir(sec_dir_) is False:
            self.meta['output_run'] = sec_dir_
        else:
            count = 1
            name = sec_dir_
            while os.path.isdir(name) is True:
                name = f"{sec_dir_}_{count}"
                count += 1
            self.meta['output_run'] = name

        self.meta['output_run_ppxf'] = os.path.join(
            self.meta['output_run'], 'ppxf')
        os.makedirs(self.meta['output_run_ppxf'], exist_ok=True)

    def keep_conf_copy(self):
        shutil.copy(self.conf_file, self.meta['output_run_ppxf'])

    def meta_to_json(self):
        meta={}
        meta.update({'conf' : self.meta})
        meta.update({'obs' : self.data.obs.meta})
        meta.update({'model' : self.data.model.meta})

        path = os.path.join(self.meta['output_run_ppxf'], 'metadata.json')

        with open(path, 'w') as out:
            json.dump(meta, fp=out, indent=4, cls=JsonCustomEncoder)

    def compute_secondary(self):
        base = self.meta['output_run_ppxf']
        datapath = os.path.join(base, 'weights.fits')
        metadatapath = os.path.join(base, 'metadata.json')
        stellar = PopMeanProperties(
            datapath=datapath,
            metadatapath=metadatapath,
            age_log10=self.data.model.meta['age_log10'])

        if 'mean_log_age_light' in self.meta['output']['secondary']:
            stellar.save(stellar.mh_light, 'mean_log10_age_light', 
                         self.meta['output_run_ppxf'])

        if 'mean_mh_light' in self.meta['output']['secondary']:
            stellar.save(stellar.mh_light, 'mean_mh_light', 
                         self.meta['output_run_ppxf'])

            
#%%
if __name__ == '__main__':
    conf = sys.argv[1]
    ppxf_control = Main(conf)
    ppxf_control.run_all()


# %% Debug

   	# conf = 'test.yaml'
  
   	# ppxf_prep = Main(conf)
   	# ppxf_prep.read_config()
   	# ppxf_prep.run_all()

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(10**(age-9), origin='lower')
    # ax[1].imshow(mh, origin='lower')


    # w = ppxf_prep.ppxf_out.ppxf.weights
    # ax[2].imshow(w[:, 0].reshape(24,6))
