# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 00:50:11 2022

@author: lwh41


Load Chch MPL reference mask NC (includes ALCF output, AMPS temperature)
and analyses peaks of ALCF backscatter to create dataset of peak properties.

Saves peak dataset to CSV
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


# Used to get list of lidar dates
old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

## Update 14.3.2023
ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'

output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'


def date_peaks(date,beta_threshold,width_threshold=1):
    
    input_path = ref_mask_dir.format(date[:-2])+str(date)+'.nc'
    
    d = ds.read(input_path)
    
    beta = ds.var(d, 'backscatter').data
    zfull = ds.var(d, 'zfull').data
    temp  = ds.var(d, 'temp').data
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


    ## Iterate through each profile, finding peaks
    
    date_peak_dict = []
    
    for time_idx, profile in enumerate(beta):
       
         ### For each vertical profile, gets peak properties
        peaks, properties = signal.find_peaks(profile,height=beta_threshold, width=width_threshold, rel_height=0.5, prominence=(None,None))
        
        for peak_idx, peak_alt_idx in enumerate(peaks):

            
            peak_dict = {
                # Meta data
                'date' : date,
                'peak_time_idx' : time_idx,
                'peak_alt_idx' : peak_alt_idx,
                
                # Backscatter properties
                'peak_backscatter' : properties['peak_heights'][peak_idx],
                'peak_width' : properties['widths'][peak_idx],
                'peak_width_height' : properties['width_heights'][peak_idx],
                'peak_prominence' : properties['prominences'][peak_idx],
                'peak_alt' : zfull[peak_alt_idx],
                'num_peaks' : len(peaks),
                'peak_order' : peak_idx,
                
                # AMPS temperature
                'amps_flag' : amps_flag,
                'peak_temp' : temp[time_idx, peak_alt_idx],
                
                # VDR
                'peak_vdr' : vdr_cm[time_idx, peak_alt_idx],
                
                # Reference mask
                'LCC_j' : LCC_j[time_idx],
                'SLCC_j' : SLCC_j[time_idx],
                'WLCC_j' : WLCC_j[time_idx],
                'ICC_j' : ICC_j[time_idx],
                
                'LCC_ij' : LCC_ij[time_idx,peak_alt_idx],
                'SLCC_ij' : SLCC_ij[time_idx,peak_alt_idx],
                'WLCC_ij' : WLCC_ij[time_idx,peak_alt_idx],
                'ICC_ij' : ICC_ij[time_idx,peak_alt_idx],
                            
                }
            
            date_peak_dict.append(peak_dict)
    
    return date_peak_dict


def run(beta_threshold):
    dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]
    
    peak_dict = []
    
    for date in dates:
        print('Processing', date)

        date_peak_dict = date_peaks(date,beta_threshold)
        peak_dict += date_peak_dict
        
    peaks_df = pd.DataFrame.from_dict(peak_dict)
            
    return peaks_df

def main():
    
    start_time = time.time()
    
    peaks_df = run(2e-5)
    
    print('Saving to CSV...')
    output_path = output_dir+'peaks_df_2e-5.csv'
    peaks_df.to_csv(output_path)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return peaks_df

if __name__ == "__main__":
    peaks_df = main()