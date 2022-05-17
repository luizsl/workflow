# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:35:32 2022

@author: Luiz
"""
import json
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


class Data(ABC):
    @abstractmethod
    def __init__(self):
        pass    
    
    @abstractmethod
    def read_data(self):
        pass
    
    
class FactoryValid(ABC):
    @property
    @abstractmethod
    def valid(self):
        pass
    
    
class Map(Data):
    def __init__(self, path, unit=None):
        self.path = path 
        self.unit = unit
        self.data, self.x_coor, self.y_coor = self.read_data()
        
    def read_data(self):
        hdul_data = fits.open(self.path)
        if self.unit is None:
            data = np.array(hdul_data[0].data)
        else:
            data = np.array(hdul_data[0].data[self.unit])
    
        y = np.arange(data.shape[0]) + 0.11
        x = np.arange(data.shape[1]) + 0.11
        x_coor, y_coor = np.meshgrid(x, y)

        return data, x_coor, y_coor
       

class Valid(FactoryValid):
    def __init__(self, data, nan=True, null=True, interval:list=None):
        self.data = data
        self.nan = nan
        self.null = null
        self.interval = interval
        
    @property    
    def valid(self):
        condition = np.full_like(self.data, fill_value=True)
        if self.nan:
            condition = np.logical_and(condition, self.not_nan())
        if self.null:
            condition = np.logical_and(condition, self.not_null())
        if self.interval:
            condition = np.logical_and(condition, self.between(*self.interval))
        return condition
    
    def not_nan(self):
        condition = ~np.isnan(self.data)
        return condition

    def not_null(self):
        condition = self.data!=0
        return condition
    
    def between(self, min_=None, max_=None):
        if min_ is None:
            condition_min = np.full_like(self.data, fill_value = True)
        else:
            condition_min = self.data > min_
            
        if max_ is None:
            condition_max = np.full_like(self.data, fill_value = True)
        else:
            condition_max = self.data < max_
            
        condition = np.logical_and(condition_min, condition_max)    
        return condition
    

class InlaInput:
    def __init__(self, input_data:Map, input_chi2:Map,
                 validator:FactoryValid=Valid, 
                 filters:dict={}):
        self.input_data = input_data
        self.input_chi2 = input_chi2
        self.abstract_validator = validator
        self.filters = filters
        self.raw_input = self.build()
        self.input = self.masking()
        self.shape = input_data.data.shape
        
    def build(self):
        data_coor = np.stack((self.input_data.y_coor.flatten(),
                              self.input_data.x_coor.flatten(),
                              self.input_data.data.flatten(),
                              self.input_chi2.data.flatten()))
        data_coor = data_coor.T
        return data_coor
    
    def validate(self):
        validator = self.abstract_validator(self.input_data.data, **self.filters)
        condition = validator.valid
        return condition
    
    def masking(self):
        mask = self.validate().flatten()
        data = self.raw_input[mask]
        return data
        
    def to_file(self, path):
        np.savetxt(path, self.input, fmt='%6.2f %6.2f %10.2f %10.2f', delimiter='\t')


class InlaExecution:
    def __init__(self, meta_dict:dict):
        self.meta = meta_dict
        
        with open('meta.json', '+w') as f:
            json.dump(meta_dict, f)
            
    def build(self):
        data = Map(self.meta['data_path'], self.meta['data_extension'])
        chi2 = Map(self.meta["chi2_path"] )

        self.inla_input = InlaInput(data, chi2, filters = self.meta['filters'])
        self.inla_input.input
        self.meta['data_dim'] = self.inla_input.shape
        self.inla_input.to_file(self.meta['input_inla'])
        
        with open('meta.json', '+w') as f:
            json.dump(self.meta, f)
            
    def run(self):
        try:
            os.system('Rscript interface.R meta.json')
        except:
            raise Exception
        
    def retrieve(self):
        file = self.meta['output_inla'] + '.json'
        with open(file) as f:
            output = json.load(f)
        return output

#%%
############## velocity ################
dt = {'data_path': '../Git/workflow/data_products/fov_sample_1_3/MilesAgeMh/ppxf/sol.fits',
      'chi2_path': '../Git/workflow/data_products/fov_sample_1_3/MilesAgeMh/ppxf/chi2.fits',
      'filters': {'nan':True, 'null':True, 'interval':[-0.5,0.5]},
      'data_extension': 3,
      'input_inla': 'input_inla.text',
      'output_inla': 'output_inla'}

main = InlaExecution(dt)    
main.build()
main.run()
a = main.retrieve()

im = np.array(a['out']).T

old_im = np.array(a['image']).T
old_im[np.where(old_im == 'NA')] = 'nan'
old_im = np.array(old_im, dtype=float)

fig, ax = plt.subplots(1,2)
ax[0].imshow(im, cmap='jet', origin='lower')
ax[1].imshow(old_im, cmap='jet', origin='lower')

# ############## sigma ################
# dt = {'data_path': '../Git/workflow/data_products/fov_sample_1_5/MilesAgeMh/ppxf/sol.fits',
#       'chi2_path': '../Git/workflow/data_products/fov_sample_1_5/MilesAgeMh/ppxf/chi2.fits',
#       'filters': {'nan':True, 'null':True, 'interval':[30,150]},
#       'data_extension': 1,
#       'input_inla': 'inla_input.text',
#       'output_inla': 'inla_output'}

# main = InlaExecution(dt)    
# main.build()
# main.run()
# b = main.retrieve()

# ############## h3 ################
# dt = {'data_path': '../Git/workflow/data_products/fov_sample_1_5/MilesAgeMh/ppxf/sol.fits',
#       'chi2_path': '../Git/workflow/data_products/fov_sample_1_5/MilesAgeMh/ppxf/chi2.fits',
#       'filters': {'nan':True, 'null':True, 'interval':[-0.2,0.2]},
#       'data_extension': 2,
#       'input_inla': 'inla_input.text',
#       'output_inla': 'inla_output'}

# main = InlaExecution(dt)    
# main.build()
# main.run()
# c = main.retrieve()


# ############## h4 ################
# dt = {'data_path': '../Git/workflow/data_products/fov_sample_1_5/MilesAgeMh/ppxf/sol.fits',
#       'chi2_path': '../Git/workflow/data_products/fov_sample_1_5/MilesAgeMh/ppxf/chi2.fits',
#       'filters': {'nan':True, 'null':True, 'interval':[-0.3,0.3]},
#       'data_extension': 3,
#       'input_inla': 'inla_input.text',
#       'output_inla': 'inla_output'}

# main = InlaExecution(dt)    
# main.build()
# main.run()
# d = main.retrieve()
#%%



dt = {'data_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/sol.fits',
      'chi2_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/chi2.fits',
      'filters': {'nan':True, 'null':True,},
      'data_extension': 0,
      'input_inla': 'input_inla.txt',
      'output_inla': 'output_inla'}

main = InlaExecution(dt)    
main.build()
main.run()
a = main.retrieve()

# im = np.array(a['out']).T

# # old_im = np.array(a['image']).T
# im[np.where(im == 'NA')] = 'nan'
# im = np.array(im, dtype=float)

# # fig, ax = plt.subplots(1,2)
# ax[0,0].imshow(im, cmap='jet', origin='lower')
# # ax[1].imshow(old_im, cmap='jet', origin='lower')

dt = {'data_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/sol.fits',
      'chi2_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/chi2.fits',
      'filters': {'nan':True, 'null':True,},
      'data_extension': 1,
      'input_inla': 'input_inla.txt',
      'output_inla': 'output_inla'}

main = InlaExecution(dt)    
main.build()
main.run()
b = main.retrieve()


dt = {'data_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/sol.fits',
      'chi2_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/chi2.fits',
      'filters': {'nan':True, 'null':True,},
      'data_extension': 2,
      'input_inla': 'input_inla.txt',
      'output_inla': 'output_inla'}

main = InlaExecution(dt)    
main.build()
main.run()
c = main.retrieve()


dt = {'data_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/sol.fits',
      'chi2_path': '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/chi2.fits',
      'filters': {'nan':True, 'null':True,},
      'data_extension': 3,
      'input_inla': 'input_inla.txt',
      'output_inla': 'output_inla'}

main = InlaExecution(dt)    
main.build()
main.run()
d = main.retrieve()

#%%
plt.style.use('science')
fig, ax = plt.subplots(2,2, figsize=(6,5))

im = np.array(a['out']).T
# im[np.where(im == 'NA')] = 'nan'
im = np.array(im, dtype=float)
i = ax[0,0].imshow(im, cmap='jet', origin='lower', vmin = -120, vmax=120)
plt.colorbar(i, ax=ax[0,0])
ax[0,0].set_title(r'$V_{*}$')

im = np.array(b['out']).T
# im[np.where(im == 'NA')] = 'nan'
im = np.array(im, dtype=float)
i = ax[0,1].imshow(im, cmap='jet', origin='lower', vmin = 60, vmax=120)
plt.colorbar(i, ax=ax[0,1])
ax[0,1].set_title(r'$\sigma_{*}$')

im = np.array(c['out']).T
# im[np.where(im == 'NA')] = 'nan'
im = np.array(im, dtype=float)
i = ax[1,0].imshow(im, cmap='jet', origin='lower', vmin = -0.13, vmax=0.13)
plt.colorbar(i, ax=ax[1,0])
ax[1,0].set_title(r'$h_{3}$')

im = np.array(d['out']).T
# im[np.where(im == 'NA')] = 'nan'
im = np.array(im, dtype=float)
i = ax[1,1].imshow(im, cmap='jet', origin='lower', vmin = -0.05, vmax=0.15)
plt.colorbar(i, ax=ax[1,1])
ax[1,1].set_title(r'$h_{4}$')

# fig.tight_layout()
plt.savefig('kinematics_with_INLA.pdf', bbox_inches='tight', format='pdf')
plt.close()
#%%
path = '../../data_products/fov_sample_1_3/MilesAgeMh/ppxf/sol.fits'
hdul = fits.open(path)

plt.style.use('science')
fig, ax = plt.subplots(2,2, figsize=(6,5))

im = hdul[0].data[0, ...]
im = np.array(im, dtype=float)
i = ax[0,0].imshow(im, cmap='jet', origin='lower', vmin = -120, vmax=120)
plt.colorbar(i, ax=ax[0,0])
ax[0,0].set_title(r'$V_{*}$')

im = hdul[0].data[1, ...]
im = np.array(im, dtype=float)
i = ax[0,1].imshow(im, cmap='jet', origin='lower',  vmin = 60, vmax=120)
plt.colorbar(i, ax=ax[0,1])
ax[0,1].set_title(r'$\sigma_{*}$')

im = hdul[0].data[2, ...]
im = np.array(im, dtype=float)
i = ax[1,0].imshow(im, cmap='jet', origin='lower', vmin = -0.13, vmax=0.13)
plt.colorbar(i, ax=ax[1,0])
ax[1,0].set_title(r'$h_{3}$')

im = hdul[0].data[3, ...]
im = np.array(im, dtype=float)
i = ax[1,1].imshow(im, cmap='jet', origin='lower', vmin = -0.05, vmax=0.15)
plt.colorbar(i, ax=ax[1,1])
ax[1,1].set_title(r'$h_{4}$')

plt.savefig('kinematics_without_INLA.pdf', bbox_inches='tight', format='pdf')
plt.close()
#%%

class PostInla:
    
    def retrieve(self):
        file = self.meta['output_inla'] + '.json'
        with open(file) as f:
            output = json.load(f)
        return output
    
if __name__ == "__main__":
    pass

