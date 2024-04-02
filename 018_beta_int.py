# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:41:37 2023

@author: lwh41
"""

import glob
import numpy as np
import ds_format as ds
import os.path
import xarray as xr
from scipy import interpolate
import scipy.signal as signal
from netCDF4 import Dataset
from datetime import datetime
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


# Used to get list of lidar dates
old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

## Update 14.3.2023
ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'

output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'



vdr_flat = []
temps_flat = []
targ_flat = []
beta_flat = []
zfull_flat = []
beta_int_flat = []
beta_int_cm_pix_flat = []

def process(date):
    input_path = ref_mask_dir.format(date[:-2])+str(date)+'.nc'
    
    d = ds.read(input_path)
    
    beta = ds.var(d, 'backscatter').data
    cloud_mask = ds.var(d, 'cloud_mask').data
    zfull = ds.var(d, 'zfull').data
    temps  = ds.var(d, 'temp').data
    vdr = ds.var(d, 'vdr').data
    vdr_cm = ds.var(d, 'vdr_cm').data
    amps_flag = ds.var(d, 'amps_flag').data
    
    LCC_ij = ds.var(d, 'LCC_ij').data
    LCC_j = ds.var(d, 'LCC_j').data
    SLCC_ij = ds.var(d, 'SLCC_ij').data
    SLCC_j = ds.var(d, 'SLCC_j').data
    WLCC_ij = ds.var(d, 'WLCC_ij').data
    WLCC_j = ds.var(d, 'WLCC_j').data
    ICC_ij = ds.var(d, 'ICC_ij').data
    ICC_j = ds.var(d, 'ICC_j').data

    beta_cm = np.where(cloud_mask,beta,np.nan)
    temps_cm = np.where(cloud_mask,temps,np.nan)
    zfull_cm = np.where(cloud_mask,zfull,np.nan)
    
    beta_int = np.nansum(beta, axis=1)
    beta_int_cm = np.nansum(beta, axis=1)

    beta_int_pix = np.zeros(np.shape(beta))
    beta_int_cm_pix = np.zeros(np.shape(beta))
    
    for i in range(0,288):
        beta_int_pix[i,:] = beta_int[i]
        beta_int_cm_pix[i,:] = beta_int_cm[i]
    
    targ_df = pd.DataFrame({'ICC_ij':d['ICC_ij'].flatten(),'SLCC_ij':d['SLCC_ij'].flatten(), 'WLCC_ij':d['WLCC_ij'].flatten()})
    targ = targ_df[['ICC_ij','SLCC_ij','WLCC_ij']].idxmax(axis=1)

    return vdr.flatten(), temps_cm.flatten(), targ, beta.flatten(), zfull_cm.flatten(), beta_int_pix.flatten(), beta_int_cm_pix.flatten()


dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]

for date in dates[:5]:
    print('Processing', date)

    vdr, temps, targ, beta, zfull, beta_int_pix, beta_int_cm_pix = process(date)
    vdr_flat= np.concatenate((vdr_flat, vdr))
    temps_flat = np.concatenate((temps_flat, temps))
    targ_flat = np.concatenate((targ_flat, targ))
    beta_flat = np.concatenate((beta_flat, beta))
    zfull_flat = np.concatenate((zfull_flat, zfull))
    beta_int_flat = np.concatenate((beta_int_flat, beta_int_pix))
    beta_int_cm_pix_flat = np.concatenate((beta_int_cm_pix_flat, beta_int_cm_pix))
    

hist_vars = pd.DataFrame({'x':beta_flat, 'y':zfull_flat, 'hue':targ_flat})
ax = sns.histplot(hist_vars, x='x', y='y', binrange=((3e-6,5e-6),(0,12000)))

# ax = sns.histplot(hist_vars, x='x', y='y',binrange=((0,1e-3),(0,12000)))
