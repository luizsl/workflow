#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:00:39 2022

@author: Luiz
"""
import multiprocessing as mp
import subprocess

def run_test(sn, regul, script_path, index):
    conf_file = f'assess_regularization/sn{sn}_regul{regul}_fov1x5.yaml'
    p0 = subprocess.run(f"nohup python {script_path} {conf_file} > nohup{index}.out",
                   shell=True, check=True,
                   )

limit = 1

with mp.Pool(processes=limit) as pool:
    sn = 40
    regul = [0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    for index, r in enumerate(regul):
        r = round(r, 2)
        r = f'{r:0.2f}'
        r = r.replace('.', 'd')
        script_path = '../exploration/run_ppxf/main.py'
        pool.apply(run_test, args=[sn, r, script_path, index])
