# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:46:52 2023

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

mpl2nc_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//mpl2nc{}/'



vdr_flat = []
temps_flat = []
targ_flat = []
beta_flat = []
zfull_flat = []
beta_int_flat = []
beta_int_cm_pix_flat = []

def process(date):
    filepath = (mpl2nc_dir+'{}*').format(date[:-2],date)
    d = xr.open_mfdataset(filepath,combine='nested',concat_dim='profile')
    print(date,' - Opened mpl2nc data')
        
    nrb_cross_pol = d['nrb_crosspol'].values
    nrb_co_pol = d['nrb_copol'].values
    time = d['time'].values
    
    c = d['c'][0].values
    bin_time= d['bin_time'][0].values
    range_ = d['range'].values
    
    
    height = 0.5*bin_time*c*(range_+0.5)
    vdr = nrb_cross_pol/nrb_co_pol
    


    return nrb_cross_pol, nrb_co_pol, height


dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]

for date in dates[50:51]:
    print('Processing', date)
    nrb_cross_pol, nrb_co_pol, height= process(date)
    
    full = nrb_co_pol + 2*nrb_cross_pol

    hist_dict = {'nrb_cross_vals':[],'nrb_co_vals':[],'full_vals':[], 'height_vals':[]}

    for j in range(0,2490):
        for i in range(0,2000):
            height_val = height[i]
            nrb_cross_val = nrb_cross_pol[j, i]
            nrb_co_val = nrb_co_pol[j, i]
            full_val = full[j, i]
            hist_dict['nrb_cross_vals'] += [nrb_cross_val]
            hist_dict['nrb_co_vals'] += [nrb_co_val]
            hist_dict['full_vals'] += [full_val]
            hist_dict['height_vals'] += [height_val]
            



    
ax = sns.histplot(hist_dict, x='nrb_cross_vals', y='height_vals')


# ax = sns.histplot(hist_vars, x='x', y='y',binrange=((0,1e-3),(0,12000)))
