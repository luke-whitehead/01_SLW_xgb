# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:46:36 2023

@author: lwh41
"""

import glob
import numpy as np
import os.path
import xarray as xr
from scipy import interpolate
import scipy.signal as signal
from netCDF4 import Dataset
from datetime import datetime
import time
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold,GroupKFold,LeaveOneGroupOut,StratifiedGroupKFold
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict,permutation_test_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score,classification_report,confusion_matrix
import pickle
import shap
import seaborn as sns
import matplotlib.pyplot as plt
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

# Used to get list of lidar dates
old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

## Update 14.3.2023
ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'

# output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask_nohp/mask{}/'
output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//multiclass_model_mask_nohp/mask{}/'


META = {
	'Z_ij': {
		'.dims': ['time', 'level'],
		'long_name': 'model cloud mask',
		'units': '1',
	},
    'Z_j': {
		'.dims': ['time'],
		'long_name': 'model cloud mask',
		'units': '1',
	},
    'Z_j_ICC': {
		'.dims': ['time'],
		'long_name': 'model cloud mask, 1 if ICC in profile',
		'units': '1',
	},
    'Z_j_SLCC': {
		'.dims': ['time'],
		'long_name': 'model cloud mask, 1 if SLCC in profile',
		'units': '1',
	},
    'Z_j_WLCC': {
		'.dims': ['time'],
		'long_name': 'model cloud mask, 1 if WLCC in profile',
		'units': '1',
	},
}

# Load model

# TARGETS = [
#     'LCC_ij',
#     'LCC_j',
#     'SLCC_ij',
#     'SLCC_j',
#     ]

# TARGET = TARGETS[3]

# model_save_name = 'models/X_'+TARGET+'_nohp.json'
model_save_name = 'models/multi_ij_no_hp.json'

model = xgb.XGBClassifier();
model.load_model(model_save_name)

# Load ref mask (to use as base to create model mask)
VARS = [
    'backscatter',
    'cloud_mask',
    'temp',
    'LCC_ij',
    'LCC_j',
    'ICC_ij',
    'ICC_j',
    'SLCC_ij',
    'SLCC_j',
    'WLCC_ij',
    'WLCC_j',
    'amps_flag',
    'time',
    'time_bnds',
    'zfull'
    ]

def process(date,beta_threshold=2e-5,width_threshold=1):
    
    input_path = ref_mask_dir.format(date[:-2])+str(date)+'.nc'
    
    d = ds.read(input_path, VARS)
    beta = ds.var(d, 'backscatter').data
    zfull = ds.var(d, 'zfull').data
    temp  = ds.var(d, 'temp').data
    
    Z_ij = np.full(np.shape(beta),np.nan)
    
    # For multiclass classification
    Z_j_ICC = np.full(np.shape(beta)[0],np.nan)
    Z_j_SLCC = np.full(np.shape(beta)[0],np.nan)
    Z_j_WLCC = np.full(np.shape(beta)[0],np.nan)
    
    # For binary classification
    # Z_j = np.full(np.shape(beta)[0],np.nan)
    
    for time_idx, profile in enumerate(beta):
        peaks, properties = signal.find_peaks(profile,height=beta_threshold, width=width_threshold, rel_height=0.5, prominence=(None,None))
        for peak_idx, peak_alt_idx in enumerate(peaks):            
                peak_dict = {                
                    # Backscatter properties
                    'peak_backscatter' : properties['peak_heights'][peak_idx],
                    'peak_width' : properties['widths'][peak_idx],
                    'peak_width_height' : properties['width_heights'][peak_idx],
                    'peak_prominence' : properties['prominences'][peak_idx],
                    'peak_alt' : zfull[peak_alt_idx],
                    'num_peaks' : len(peaks),
                    'peak_order' : peak_idx,
                    
                    # AMPS temperature
                    'peak_temp' : temp[time_idx, peak_alt_idx],                
                    }
                peak_df = pd.DataFrame.from_dict([peak_dict])
                Z = model.predict(peak_df) #0=ICC, 1=SLCC, 2=WLCC
                
                min_alt = int(peak_alt_idx-properties['widths'][peak_idx])
                max_alt = int(peak_alt_idx+properties['widths'][peak_idx])
                
                min_idx = max(0,min_alt)
                max_idx = min(299,max_alt)
            
                Z_ij[time_idx,min_idx:max_idx] = Z
                
                # For multiclass classification
                if Z == 0:
                    Z_j_ICC[time_idx]=1
                elif Z == 1:
                    Z_j_SLCC[time_idx]=1
                elif Z == 2:
                    Z_j_WLCC[time_idx]=1
                    
    # For binary classification         
    # Z_j = np.nanmax(Z_ij,axis=1)
    
    
    ds.var(d,'Z_ij',Z_ij)
    ds.meta(d, 'Z_ij', META.get('Z_ij'))
    
    # For multiclass classification
    ds.var(d,'Z_j_ICC',Z_j_ICC)
    ds.var(d,'Z_j_SLCC',Z_j_SLCC)
    ds.var(d,'Z_j_WLCC',Z_j_WLCC)
    
    ds.meta(d, 'Z_j_ICC', META.get('Z_j_ICC'))
    ds.meta(d, 'Z_j_SLCC', META.get('Z_j_SLCC'))
    ds.meta(d, 'Z_j_WLCC', META.get('Z_j_WLCC'))
    
    # For binary classification
    # ds.var(d,'Z_ij',Z_ij)
    # ds.meta(d, 'Z_j', META.get('Z_j'))
    
    
    
    # output_path = output_dir.format(TARGET,date[:-2])+str(date)+'.nc'
    output_path = output_dir.format(date[:-2])+str(date)+'.nc'

    
    print('Writing to',output_path)
    ds.write(output_path, d)
            
def main():
    dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]
    unprocessed_dates = []
    for date in dates:
        if not os.path.exists(output_dir.format(date[:-2])+str(date)+'.nc'):
            unprocessed_dates+=[date]

    # dates = ['20210617']
    
    for date in dates:
        print('Processing', date)
        start_time = time.time()
        process(date)
        print("--- %s seconds ---" % (time.time() - start_time))
        

if __name__ == "__main__":
    main()