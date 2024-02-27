# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:18:07 2023

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

# mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask/mask{}/'
# mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask_nohp/mask{}/'
# mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//multiclass_model_mask_nohp/mask{}/'

mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_davis_mask/'


def run_all(xLCC_j, Z_j_SLCC_zeros, xLCC_j_unique_ts, Z_j_unique_ts):
    
    print('-------------------------------------------------')
    print('All dates')
    
    ## Selecting only time steps with peaks
    num_ts_with_pk = int(len(xLCC_j_unique_ts))
    frac_with_pk = 100*num_ts_with_pk/len(xLCC_j)
    print('{} time steps with peaks detected, out of {} time steps ({:.1f}%)'.format(num_ts_with_pk,len(xLCC_j),frac_with_pk))
    
    num_xLCC_vdr_ts_with_pk = np.nansum(xLCC_j_unique_ts)
    frac_xLCC_vdr_from_ts_with_pk = 100*num_xLCC_vdr_ts_with_pk/num_ts_with_pk
    print('VDR: {:.0f} time steps have xLCC, out of {} time steps with peaks detected ({:.1f}%)'.format(num_xLCC_vdr_ts_with_pk,num_ts_with_pk,frac_xLCC_vdr_from_ts_with_pk))
    
    num_xLCC_vdr = np.nansum(xLCC_j)
    frac_xLCC_vdr_from_all_ts = 100*num_xLCC_vdr/len(xLCC_j)
    print('VDR: {:.0f} time steps have xLCC, out of {} total time steps ({:.1f}%)'.format(num_xLCC_vdr,len(xLCC_j),frac_xLCC_vdr_from_all_ts))
    
    num_xLCC = np.nansum(Z_j_SLCC_zeros)
    frac_xLCC_from_ts_with_pk = 100*num_xLCC/num_ts_with_pk
    print('Model: {:.0f} time steps have xLCC, out of {} time steps with peaks detected ({:.1f}%)'.format(num_xLCC,num_ts_with_pk,frac_xLCC_from_ts_with_pk))
    
    frac_xLCC_from_all_ts = 100*num_xLCC/len(xLCC_j)
    print('Model: {:.0f} time steps have xLCC, out of {} total time steps ({:.1f}%)'.format(num_xLCC,len(xLCC_j),frac_xLCC_from_all_ts))
    
    # Z_j_zeros = np.where(np.isnan(Z_j),0,Z_j)
    chch_bal_acc = balanced_accuracy_score(xLCC_j,Z_j_SLCC_zeros)
    print('Balanced accuracy of the xLCC classification, examining all time steps: {:.3f}'.format(chch_bal_acc))
    chch_bal_acc_pkts = balanced_accuracy_score(xLCC_j_unique_ts,Z_j_unique_ts)
    print('Balanced accuracy of the xLCC classification, examining only time steps where peaks detected: {:.3f}'.format(chch_bal_acc_pkts))
    
    print('-------------------------------------------------')

def run_date(date):

    # Open ref mask
    ref_mask_path = ref_mask_dir.format(date[:-2])+str(date)+'.nc'
    d_ref = ds.read(ref_mask_path, ['vdr','vdr_cm','amps_flag','SLCC_j'])
    
    amps_flag = ds.var(d_ref, 'amps_flag')
    
    
    if not amps_flag:
        # print(f'{date} - AMPS not available')
        return None, None, None, None
    
    SLCC_j = ds.var(d_ref, 'SLCC_j').data
    vdr = ds.var(d_ref,'vdr')
    vdr_cm = ds.var(d_ref,'vdr_cm')
    
    # print('Opening peak dataset')
    peaks_fpath = peaks_dir+'peaks_df_{}.csv'.format('2e-5')
    peaks_df = pd.read_csv(peaks_fpath, index_col=[0]).query('date=={}'.format(date))  
    
    model_mask_load_ag = np.load(mod_mask_dir+date+'.npz')
    g22davis_liq = model_mask_load_ag['grid']   
    g22davis_liq_prof = model_mask_load_ag['profile']

    Z_j_SLCC = g22davis_liq_prof
    
    # mod_mask_path = mod_mask_dir.format(date[:-2])+str(date)+'.nc'
    # d_mod = ds.read(mod_mask_path)
    
    # alcf_cloud_mask = ds.var(d_mod,'cloud_mask')
    # alcf_backscatter = ds.var(d_mod,'backscatter')
    # temp = ds.var(d_mod,'temp')
    # LCC_ij = ds.var(d_mod,'LCC_ij')
    # LCC_j = ds.var(d_mod,'LCC_j')
    # SLCC_ij = ds.var(d_mod,'SLCC_ij')
    # SLCC_j = ds.var(d_mod,'SLCC_j')
    # # WLCC_ij = ds.var(d_mod,'WLCC_ij')
    # WLCC_j = ds.var(d_mod,'WLCC_j')
    # # ICC_ij = ds.var(d_mod,'ICC_ij')
    # ICC_j = ds.var(d_mod,'ICC_j')
    
    # Z_ij = ds.var(d_mod,'Z_ij')
    # # Z_j = ds.var(d_mod,'Z_j')
    
    # Z_j_ICC = ds.var(d_mod,'Z_j_ICC')
    # Z_j_SLCC = ds.var(d_mod,'Z_j_SLCC')
    # Z_j_WLCC = ds.var(d_mod,'Z_j_WLCC')
    
    # Z_any = np.any(((Z_j_ICC==1),(Z_j_SLCC==1),(Z_j_WLCC==1)),axis=0)
    
    
    
   
    
    print(date)
    # print(TARGET)
    

    ## Selecting only time steps with peaks
    unique_ts = np.unique(peaks_df['peak_time_idx'])
    num_ts_with_pk = len(unique_ts)
    frac_with_pk = 100*len(unique_ts)/len(SLCC_j)
    print('{} time steps with peaks detected, out of {} time steps ({:.1f}%)'.format(num_ts_with_pk,len(SLCC_j),frac_with_pk))
    
    num_SLCC_vdr_ts_with_pk = np.nansum(SLCC_j[unique_ts])
    frac_SLCC_vdr_from_ts_with_pk = 100*num_SLCC_vdr_ts_with_pk/num_ts_with_pk
    print('VDR: {:.0f} time steps have SLCC, out of {} time steps with peaks detected ({:.1f}%)'.format(num_SLCC_vdr_ts_with_pk,num_ts_with_pk,frac_SLCC_vdr_from_ts_with_pk))
    
    num_SLCC_vdr = np.nansum(SLCC_j)
    frac_SLCC_vdr_from_all_ts = 100*num_SLCC_vdr/len(SLCC_j)
    print('VDR: {:.0f} time steps have SLCC, out of {} total time steps ({:.1f}%)'.format(num_SLCC_vdr,len(SLCC_j),frac_SLCC_vdr_from_all_ts))
    
    num_SLCC = np.nansum(Z_j_SLCC)
    frac_SLCC_from_ts_with_pk = 100*num_SLCC/num_ts_with_pk
    print('Model: {:.0f} time steps have SLCC, out of {} time steps with peaks detected ({:.1f}%)'.format(num_SLCC,num_ts_with_pk,frac_SLCC_from_ts_with_pk))
    
    frac_SLCC_from_all_ts = 100*num_SLCC/len(SLCC_j)
    print('Model: {:.0f} time steps have SLCC, out of {} total time steps ({:.1f}%)'.format(num_SLCC,len(SLCC_j),frac_SLCC_from_all_ts))
    
    Z_j_SLCC_zeros = np.where(np.isnan(Z_j_SLCC),0,Z_j_SLCC)
    chch_bal_acc = balanced_accuracy_score(SLCC_j,Z_j_SLCC_zeros)
    print('Balanced accuracy of the SLCC classification, examining all time steps: {:.3f}'.format(chch_bal_acc))
    chch_bal_acc_pkts = balanced_accuracy_score(SLCC_j[unique_ts],Z_j_SLCC_zeros[unique_ts])
    print('Balanced accuracy of the SLCC classification, examining only time steps where peaks detected: {:.3f}'.format(chch_bal_acc_pkts))
  
    print('-------------------------------------------------')

    return SLCC_j, Z_j_SLCC_zeros, SLCC_j[unique_ts], Z_j_SLCC_zeros[unique_ts]
 
    
def main():
    dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]
    # dates = ['20210518', '20210605']    
    
    # TARGET = 'SLCC_j'
    
    xLCC_j_all = [] 
    Z_j_all = [] 
    xLCC_j_unique_ts_all = [] 
    Z_j_unique_ts_all = []

    
    for date in dates:
        xLCC_j, Z_j, xLCC_j_unique_ts, Z_j_unique_ts = run_date(date)
        
        if xLCC_j is not None:
            xLCC_j_all = np.hstack((xLCC_j_all,xLCC_j))
            Z_j_all = np.hstack((Z_j_all,Z_j))
            xLCC_j_unique_ts_all = np.hstack((xLCC_j_unique_ts_all,xLCC_j_unique_ts))
            Z_j_unique_ts_all = np.hstack((Z_j_unique_ts_all,Z_j_unique_ts))

        
        
    run_all(xLCC_j_all, Z_j_all, xLCC_j_unique_ts_all, Z_j_unique_ts_all)
    
    return xLCC_j_all, Z_j_all, xLCC_j_unique_ts_all, Z_j_unique_ts_all
    
    
if __name__ == "__main__":
    xLCC_j_all, Z_j_all, xLCC_j_unique_ts_all, Z_j_unique_ts_all = main()