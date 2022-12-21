#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:50:42 2022

@author: Luiz
"""

import json
from time import perf_counter as clock

import matplotlib.pyplot as plt
import numpy as np
import ppxf.ppxf_util as util
import spectcube as sc
from astropy.io import fits
from ppxf.ppxf import ppxf, robust_sigma

from bounds_processing import build_bounds

C = 299792.458  # speed of light in km/s

data = ppxf_prep.data

t = clock()

# for i, index in enumerate(np.arange(10,3000, 200), 1):
#     print(str(i).center(80, '-'))
index = 20

galaxy = data.obs.flux_grid[:, index]
noise = data.obs.flux_grid_unc[:, index]
stellar = data.stellar.flux_grid[:, index]
goodpixels = data.obs.meta['fixed_goodpixels']

stellar_kinematics = data.stellar_kinematics.kinematics_grid[:, index]
gas_kinematics = data.gas_kinematics.kinematics_grid[:, index]

gas_templates = data.em_model.template
gas_names = data.em_model.label
lam = data.obs.meta['wave_obs']
velscale = C*np.diff(np.log(lam[-2:]))

# 0 Component
ngas_comp = 0
component = np.array([0])
moments = [-4]

start_stellar_kinematics = [0, 1, 0, 0]

pp = ppxf(stellar, galaxy, noise, velscale, start=start_stellar_kinematics,
          moments=moments, degree=-1, mdegree=2, component=component,
          goodpixels=goodpixels,
          plot=0,
          lam=lam, vsyst=0,
          )

# fig, ax = plt.subplots()
# pp.plot()

try:
    A_ineq_kin = data.main_meta['gas_template']['A_ineq_kin']
    b_ineq_kin = data.main_meta['gas_template']['b_ineq_kin']
    A_ineq_kin = np.asarray(A_ineq_kin, dtype=float)
    b_ineq_kin = np.asarray(b_ineq_kin, dtype=float)
    b_ineq_kin = b_ineq_kin/velscale
    constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}
except Exception:
    constr_kinem = None


ngas_comp = data.main_meta['gas_template']['components']
em_shape = data.em_model.size
comp = [1] + em_shape * ngas_comp
component = np.concatenate(
    [[i]*value for i, value in enumerate(comp)]
    )

gas_moments = data.main_meta['gas_template']['moments']
stellar_moments = data.stellar_kinematics.kinematics_grid.shape[0]
moments = [-stellar_moments] + [gas_moments] * ngas_comp * len(em_shape)

gas_templates = np.tile(gas_templates, ngas_comp)
gas_names = np.asarray(
    [a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
label_wave = np.tile(data.em_model.label_wave, ngas_comp)
gas_component = np.array(component) > 0
stars_gas_templates = np.column_stack([pp.bestfit, gas_templates])

start_stellar_kinematics = np.array([0, 1, 0, 0])
start_gas_kinematics = gas_kinematics.reshape(-1, gas_moments)
start = [start_stellar_kinematics.tolist()] + start_gas_kinematics.tolist()

bounds_gas = np.array(build_bounds(gas_kinematics, data.bounds_rule))
bounds_gas = bounds_gas.reshape(ngas_comp * len(em_shape), -1, gas_moments)
bounds_gas = bounds_gas.tolist()
bounds_stellar = [[-1, 1], [1, 2], [-1, 1], [-1, 1]]
bounds = [bounds_stellar] + bounds_gas

pp = ppxf(stars_gas_templates, galaxy, noise, velscale, start,
          moments=moments, degree=-1, mdegree=-1, component=component,
          gas_component=gas_component, gas_names=gas_names,
          # goodpixels=goodpixels,
          bounds=bounds,
          # constr_kinem=constr_kinem,
          # constr_templ=constr_templ,
          # linear_method='cvxopt',
          # method='capfit',
          # linear_method='lsq_lin',
          plot=False,
          lam=lam, vsyst=0,
          global_search={'tol': 0.1, 'disp': 1, 'popsize': 5, 'recombination': 0.8},
          # global_search=False,
          )

corrected_flux = np.full_like(gas_names, np.nan, dtype=float)
amplitude_rms = np.full_like(gas_names, np.nan, dtype=float)
rms = robust_sigma(pp.galaxy - pp.bestfit, zero=1)
# names = ['Halpha', 'Hbeta' , '[NII]6583_d', '[OIII]5007_d']
# names = gas_names.tolist()
for p, name in enumerate(gas_names):
    kk = gas_names == name             # Extract first gas kinematic component
    dlam = label_wave[kk]*velscale/C   # Angstrom per pixel at line wavelength (dlam/lam = dv/c)
    corrected_flux[p] = (pp.gas_flux[kk]*dlam)[0]   # Convert to ergs/(cm^2 s)
    amplitude_rms[p] = np.max(pp.gas_bestfit_templates[:, kk]) / rms
    print(f"{name:20s}",
          f"-Amp/Res: {amplitude_rms[p]:6.2f};",
          f"flux: {corrected_flux[p]:6.0f} ergs/(cm^2 s)")

# fig, ax = plt.subplots()
# pp.plot()

print(f'\n{round(clock()-t,2)} s')
