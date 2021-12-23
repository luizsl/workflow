#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:55:58 2021

@author: Luiz
"""
import os
import tempfile
import shutil
import _pickle as pickle
from build_metadata import Meta
from data_preprocessing import DataPreprocessing
from ppxf_execution import ExecutePpxf
from post_processing import PostProcessing


class Main:
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.read_config()
        self.create_temporary()
        self.create_output_folder()
        self.create_metadata_file()
        self.exectute()
        self.remove_temporary()
        
    def read_config(self): 
        self.meta = Meta(conf_file = self.conf_file)
        
    def create_temporary(self):
        self.temp_input_dir = tempfile.mkdtemp(dir =  '.')
        self.temp_output_dir = tempfile.mkdtemp(dir =  '.')
        self.meta.temp_input_dir = self.temp_input_dir
        self.meta.temp_output_dir = self.temp_output_dir
        
    def create_output_folder(self):
        if os.path.isdir(self.meta.output_dir) is False:
            pass
        else:
            name = self.meta.output_dir
            count = 1
            while os.path.isdir(f'{name}_{str(count)}') is True:
                count += 1
            self.meta.output_dir = f'{name}_{str(count)}'
            
        os.mkdir(self.meta.output_dir)
            
    def create_metadata_file(self):
        self.metadata_path = f'{self.meta.output_dir}/metadata.pkl'
        with open(self.metadata_path, 'wb') as out:
            pickle.dump(self.meta, out)
        
    def exectute(self):
        DataPreprocessing(self.metadata_path)
        ExecutePpxf(self.metadata_path)
        PostProcessing(self.metadata_path)
        
    def remove_temporary(self):
        shutil.rmtree(self.meta.temp_input_dir)
        shutil.rmtree(self.meta.temp_output_dir)

if __name__ == '__main__':
    Main('config.ini')
