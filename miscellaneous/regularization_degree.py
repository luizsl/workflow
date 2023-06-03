#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:00:39 2022

@author: Luiz
"""
import multiprocessing as mp
import subprocess

def run_test(sn, regul, script_path, index):
    conf_file = f'eval_regularization/sn{sn}_regul{regul}_fov1x3_norm_obs.yaml'
    p0 = subprocess.run(
        f"nohup python {script_path} {conf_file} > nohup{index}.out",
        shell=True, check=True)

if __name__ == '__main__':
    limit = 5

    with mp.Pool(processes=limit) as pool:
        sn = 100
        regul = [0, 0.05, 0.04, 0.03, 0.02, 0.01]
        for index, r in enumerate(regul):
            r = round(r, 2)
            r = f'{r:0.2f}'
            r = r.replace('.', 'd')
            print(r)
            script_path = '../exploration/run_ppxf/main.py'
            pool.apply(run_test, args=[sn, r, script_path, index])
