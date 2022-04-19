#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:55:58 2021

@author: Luiz
"""
import json
import os
import shutil
import sys
import tempfile

import _pickle as pickle
from astropy.utils.misc import JsonCustomEncoder

from build_metadata import Meta
from data_preprocessing import DataPreprocessing
from post_processing import PostProcessing
from ppxf_execution import ExecutePpxf


class Main:
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def run_all(self):
        self.read_config()
        self.create_output_folder()
        self.create_temporary()
        self.keep_conf_copy()
        self.create_metadata_file()
        self.execute()
        self.remove_temporary()
        self.meta_to_json()

    def read_config(self):
        self.meta = Meta(conf_file = self.conf_file)

    def create_output_folder(self):
        dir_ = os.path.join(self.meta.output_root, self.meta.output_dir)
        if os.path.isdir(dir_) is False:
            os.makedirs(dir_, exist_ok=True)

        # later add a function to write a more complex _sec_dir name
        sec_dir_ = f'{dir_}miles'
        
        # create unique name
        if os.path.isdir(sec_dir_) is False:
            self.meta.output_run = sec_dir_
        else:
            count = 1
            while os.path.isdir(f'{sec_dir_}_{str(count)}') is True:
                count += 1
            self.meta.output_run = f'{sec_dir_}_{str(count)}'
            
        self.meta.output_run_ppxf = \
            os.path.join(self.meta.output_run, 'ppxf')
        os.makedirs(self.meta.output_run_ppxf, exist_ok=True)
        
    def keep_conf_copy(self):
        shutil.copy(self.conf_file, self.meta.output_run_ppxf)

    def create_temporary(self):
        self.temp_input_dir = tempfile.mkdtemp(dir = self.meta.output_root)
        self.temp_output_dir = tempfile.mkdtemp(dir = self.meta.output_root)
        self.meta.temp_input_dir = self.temp_input_dir
        self.meta.temp_output_dir = self.temp_output_dir

    def create_metadata_file(self):
        self.metadata_path = f'{self.meta.output_run_ppxf}/metadata.pkl'
        with open(self.metadata_path, 'wb') as out:
            pickle.dump(self.meta, out)

    def execute(self):
        DataPreprocessing(self.metadata_path)
        ExecutePpxf(self.metadata_path)
        PostProcessing(self.metadata_path)

    def remove_temporary(self):
        shutil.rmtree(self.meta.temp_input_dir)
        shutil.rmtree(self.meta.temp_output_dir)

    def meta_to_json(self):
        with open(f'{self.meta.output_run_ppxf}/metadata.pkl', 'rb') as inp:
            meta = pickle.load(inp)
            meta = meta.__dict__
        with open(f'{self.meta.output_run_ppxf}/metadata.json', 'w') as out:
            json.dump(meta, fp = out, indent=4, cls = JsonCustomEncoder)

#%%
if __name__ == '__main__':
    # conf = sys.argv[1]
    # ppxf_control = Main(conf)
    # ppxf_control.run_all()
    
    conf = 'test.ini'
    t = Main(conf)
    t.read_config()
    
    t.create_output_folder()
    
    t.create_temporary()
    t.keep_conf_copy()
    t.create_metadata_file()
    # t.execute()
    

