#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:55:58 2021

@author: Luiz
"""

import os
import shutil
import sys
import yaml
import json

from astropy.utils.misc import JsonCustomEncoder

from data_preprocessing import DataPreprocessing
from ppxf_execution import ExecutePpxf


class Main:
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def run_all(self):
        self.read_config()
        self.create_output_folder()
        self.keep_conf_copy()

        self.data = DataPreprocessing(self.meta)
        self.ppxf_out = ExecutePpxf(self.data, self.meta)
        self.ppxf_out.reconstruct_map(
            self.data, par=self.meta['output']['to_save'])

        self.meta_to_json()

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
                name = os.path.join(dir_, f"{sec_dir_}_{count}")
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
            json.dump(meta, fp = out, indent=4, cls = JsonCustomEncoder)


if __name__ == '__main__':
    conf = sys.argv[1]
    # conf = 'test.yaml'
    ppxf_control = Main(conf)
    ppxf_control.run_all()
