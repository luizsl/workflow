#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:00:39 2022

@author: Luiz
"""

import concurrent.futures
import subprocess

sn = 100
regul=[0,20,30,40,50,60,100]
script_path = '../exploration/run_ppxf/main.py'

def run_test(sn, regul, script_path, index):
    conf_file = f'assess_regularization/sn{sn}_regul{regul}_fov1x5.yaml'
    p0 = subprocess.run(f"taskset -c 0,1,2,3 nohup python {script_path} {conf_file} > nohup{index}.out",
                   shell=True, check=True,
                   )
    p0.wait()

limit = 2
with concurrent.futures.ThreadPoolExecutor(limit) as executor:
    for index, r in enumerate(regul):
        executor.submit(run_test, sn, r, script_path, index)
