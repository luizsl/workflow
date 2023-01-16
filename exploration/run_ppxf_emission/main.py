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

from bounds_processing import (bounds_fixed_constructor,
                               bounds_interval_constructor)
from data_preprocessing import DataPreprocessing
from ppxf_execution import ExecutePpxf
from reconstruct_map import reconstruct_map


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
        self.execution = ExecutePpxf(self.data, self.meta)
        self.meta_to_json()
        self.results_to_json()
        self.results_to_map()

    def read_config(self):
        with open(self.conf_file) as f:
            loader = yaml.Loader
            loader.add_constructor(tag='!Interval',
                                   constructor=bounds_interval_constructor)
            loader.add_constructor(tag='!Fixed',
                                   constructor=bounds_fixed_constructor)

            self.meta = yaml.load(f, Loader=loader)

    def create_output_folder(self):
        dir_ = str(Path(self.meta['resources']['ppxf_stellar_dir']).parents[0])
        if os.path.isdir(dir_) is False:
            raise Exception

        comp = self.meta['gas_template']['components']

        if self.meta['vorbin']['apply']:
            sn = self.meta['vorbin']['target_sn']
            binned = f'_binned{sn}'
        else:
            binned = None

        sec_dir_ = os.path.join(
            dir_,
            f"ppxf_emission_line{binned or ''}_{comp}components")

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
        meta = {}
        meta.update({'conf': self.meta})
        # NOBUG: Implement method to the serialization of 'bounds' is required.
        # It won't be done for now, just removing it instead.
        meta['conf']['gas_template'].pop('bounds')

        meta.update({'obs': self.data.obs.meta})
        meta.update({'stellar': self.data.stellar.meta})

        path = os.path.join(self.meta['output_run'], 'metadata.json')

        with open(path, 'w') as out:
            json.dump(meta, fp=out, indent=4, cls=JsonCustomEncoder)

    def results_to_json(self):
        parameters = self.meta['output']['to_save']

        for parameter in parameters:
            data = self.execution.out_ppxf[parameter].values
            path = os.path.join(self.meta['output_run'], f'{parameter}.json')

            with open(path, 'w') as out:
                json.dump(data, fp=out, indent=4, cls=JsonCustomEncoder)

    def results_to_map(self):
        parameters = self.meta['output']['to_map']
        file_metadata = os.path.join(self.meta['output_run'], 'metadata.json')

        with open(file_metadata) as fp:
            out_metadata = json.load(fp)

        try:
            binned = self.meta['vorbin']['apply']
        except Exception:
            binned = False

        for parameter in parameters:
            data = self.execution.out_ppxf[parameter].values
            reconstruct_map(data=data, out_metadata=out_metadata,
                            parameter=parameter, binned=binned)


if __name__ == '__main__':
    conf = sys.argv[1]
    ppxf_control = Main(conf)
    ppxf_control.run_all()

#%%  Debug

   	# conf = 'test.yaml'

   	# ppxf_prep = Main(conf)
   	# ppxf_prep.run_all()
