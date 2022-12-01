#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:50:42 2022

@author: Luiz
"""

import matplotlib.pyplot as plt
import numpy as np
import ppxf.ppxf_util as util
import spectcube as sc
from astropy.io import fits
import json
from ppxf.ppxf import ppxf, robust_sigma
from time import perf_counter as clock

C = 299792.458  # speed of light in km/s

data = ppxf_prep.data

# ppxf

t = clock()

# for index in range(10):
index = 5000
vlim = lambda x: gas_kinematics[0] + x*np.array([-100, 100])

galaxy = data.obs.flux_grid[:, index]
noise = data.obs.flux_grid_unc[:, index]
stellar = data.stellar.flux_grid[:, index]
stellar_kinematics = data.stellar_kinematics.kinematics_grid[:, index]

# if data.gas_kinematics is None:
#     gas_kinematics = stellar_kinematics[:2]
#     start_gas_kinematics = np.tile(gas_kinematics[:2], (ngas_comp,1)).tolist()
# else:
#     gas_kinematics = data.gas_kinematics.kinematics_grid[:, index]
#     start_gas_kinematics = gas_kinematics.reshape(ngas_comp,-1).tolist()

gas_kinematics = data.gas_kinematics.kinematics_grid[:, index]


goodpixels = data.obs.meta['fixed_goodpixels']
gas_templates = data.em_model.template
gas_names = data.em_model.label
# line_wave = data.em_model.line_wave
lam = data.obs.meta['wave_obs']
velscale = C*np.diff(np.log(lam[-2:]))

### 0 Component
ngas_comp = 0
component = np.array([0])
moments = [-4]

start_stellar_kinematics = [0, 1, 0, 0]
bounds = np.asarray([vlim(1), [20, 300], [-1,1], [-1,1]])       # I force the component=2 to lie +/-600 km/s from the stellar velocity

pp = ppxf(stellar, galaxy, noise, velscale, start=start_stellar_kinematics,
          moments=moments, degree=-1, mdegree=8, component=component,
           goodpixels=goodpixels,
          plot=0, 
          lam=lam, vsyst=0, 
          )

# fig, ax = plt.subplots()
# pp.plot()

# ### 1 Component
# ngas_comp = 1
# component = np.array([0] + [1]*7)
# moments = [-4, 2]

# bounds = [[[-1, 1], [1, 2], [-1, 1], [-1, 1]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
#           [vlim(6), [20, 500]],]       # I force the component=2 to lie +/-600 km/s from the stellar velocity

# A_ineq_kin = np.array([[0, 0, 0, 0, 0, 0]])
# b_ineq_kin = np.array([0])/velscale


### 2 Component
ngas_comp = 4
# component = np.array([0] + [1]*10 + [2]*10)
component = np.array([0] + [1]*2 + [2]*8 + [3]*2 + [4]*8)
moments = [-4, 2, 2, 2, 2]

bounds = [[[-1,1], [1, 2], [-1, 1], [-1, 1]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
          [vlim(2), [20, 150]],       # I force the component=1 to lie +/-200 km/s from the stellar velocity
          [vlim(3), [20, 400]],
          [vlim(2), [20, 150]],       # I force the component=1 to lie +/-200 km/s from the stellar velocity
          [vlim(3), [20, 400]],]       # I force the component=2 to lie +/-600 km/s from the stellar velocity
  
A_ineq_kin = np.array([[0, 0, 0, 0,   0, 1,   0, 0,    0,-1,   0, 0],
                       [0, 0, 0, 0,   0, 0,   0, 1,    0, 0,   0,-1],
                       ])
b_ineq_kin = np.array([0, 0])/velscale


# ## 3 Component
# ngas_comp = 3
# component = np.array([0] + [1]*7 + [2]*7 + [3]*7)
# moments = [-4, 2, 2, 2]

# bounds = [[[-1,1], [1, 2], [-1, 1], [-1, 1]],      # Bounds are ignored for the stellar component=0 which has fixed kinematic
#           [vlim(2), [20, 150]],       # I force the component=1 to lie +/-200 km/s from the stellar velocity
#           [vlim(6), [20, 300]],  
#           [vlim(6), [20, 400]]]       # I force the component=2 to lie +/-600 km/s from the stellar velocity
  
# A_ineq_kin = np.array([[0, 0, 0, 0, 0, 1, 0, -1, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 1, 0, -1]])
# b_ineq_kin = np.array([0, 0])/velscale

###

gas_kinematics = np.concatenate(
    [np.tile(gas_kinematics[:2], 2), np.tile(gas_kinematics[2:], 2)])
start_gas_kinematics = gas_kinematics.reshape(-1,2).tolist()
start = [start_stellar_kinematics] +  start_gas_kinematics

gas_templates = np.tile(gas_templates, 2)
gas_names = np.asarray([a + f"_({p+1})" for p in range(2) for a in gas_names])
# line_wave = np.tile(line_wave, ngas_comp)

constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}    

gas_component = np.array(component) > 0
# stars_gas_templates = np.column_stack([stellar, gas_templates])

stars_gas_templates = np.column_stack([pp.bestfit, gas_templates])

# start[2][1] = start[1][1]
print(start)

pp = ppxf(stars_gas_templates, galaxy, noise, velscale, start,
          moments=moments, degree=-1, mdegree=-1, component=component, 
          gas_component=gas_component, gas_names=gas_names,
            # goodpixels=goodpixels,
          bounds=bounds,
           constr_kinem=constr_kinem,
          # constr_templ=constr_templ,
          # linear_method='cvxopt',
           # method='capfit',
          # linear_method='lsq_lin',
          plot=False, 
          lam=lam, vsyst=0, 
           global_search={
               'tol': 0.1, 'disp': 1, 'popsize': 5, 'recombination': 0.9
               },
           # global_search=True,
          )

# rms = robust_sigma(pp.galaxy - pp.bestfit, zero=1)
# # names = ['Halpha', 'Hbeta' , '[NII]6583_d', '[OIII]5007_d']
# names = gas_names.tolist()
# for p, name in enumerate(names):
#     kk = gas_names == name            # Extract first gas kinematic component
#     dlam = line_wave[kk]*velscale/C   # Angstrom per pixel at line wavelength (dlam/lam = dv/c)
#     flux = (pp.gas_flux[kk]*dlam)[0]  # Convert to ergs/(cm^2 s)
#     an = np.max(pp.gas_bestfit_templates[:, kk])/rms
#     print(f"{name:20s} - Amplitude/Noise: {an:6.4g}; gas flux: {flux:6.0f} ergs/(cm^2 s)")
    
print(f'\n{round(clock()-t,2)} s')

fig, ax = plt.subplots()
pp.plot()