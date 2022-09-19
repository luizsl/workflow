#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:00:00 2022

@author: Luiz
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from astropy.utils.misc import JsonCustomEncoder

from data_preprocessing import DataPreprocessing
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
        ExecutePpxf(self.data, self.meta)
        self.meta_to_json()
    
    def read_config(self):
        with open(self.conf_file) as f:
            self.meta = yaml.load(f, Loader=yaml.Loader)

    def create_output_folder(self):
        dir_ = str(Path(self.meta['resources']['ppxf_stellar_dir']).parents[0])
        if os.path.isdir(dir_) is False:
            raise Exception
            
        sec_dir_ = os.path.join(dir_, 'ppxf_emission_line')

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

        os.makedirs(self.meta['output_run'], exist_ok=True)

    def keep_conf_copy(self):
        shutil.copy(self.conf_file, self.meta['output_run'])

    def meta_to_json(self):
        meta={}
        meta.update({'conf' : self.meta})
        meta.update({'obs' : self.data.obs.meta})
        meta.update({'stellar' : self.data.stellar.meta})
        meta.update({'stellar_kinematics' : self.data.stellar_kinematics.meta})

        path = os.path.join(self.meta['output_run'], 'metadata.json')

        with open(path, 'w') as out:
            json.dump(meta, fp=out, indent=4, cls=JsonCustomEncoder)

#
if __name__ == '__main__':
    # conf = sys.argv[1]
    # ppxf_control = Main(conf)
    # ppxf_control.run_all()

#  Debug

   	conf = 'test.yaml'
  
   	ppxf_prep = Main(conf)
   	ppxf_prep.run_all()
