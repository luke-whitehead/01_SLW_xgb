# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:57:38 2023

@author: lwh41

Create the NetCDF framework for assimilating data from ALCF, VDR reference
mask and AMPS. Future may add G22 model masks and MRR.
"""

import numpy as np
import ds_format as ds
import os
import glob
import aquarius_time as aq
import xarray as xr
from scipy import interpolate
from netCDF4 import Dataset
from datetime import datetime
import pandas as pd
import time

# Used to get list of lidar dates
old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

## Update 2023-03-13 for recalibrated MPL (cloud mask threshold 4e-6)
mpl2nc_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//mpl2nc{}/'
alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf{}//lidar/'
amps_dir = 'C://Local//lwh41//DATA//AMPS_NC/'

output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'

VARIABLES = [
            'vdr',
            'vdr_cm',
            'temp',
            'amps_flag',
            'vdr_threshold',
            'LCC_ij',
            'LCC_j',
            'ICC_ij',
            'ICC_j',
            'SLCC_ij',
            'SLCC_j',
            'WLCC_ij',
            'WLCC_j',
            ]

META = {
	'vdr': {
		'.dims': ['time', 'level'],
		'long_name': 'linear volume depolarization ratio',
		'units': '1',
	},
    'vdr_cm': {
		'.dims': ['time', 'level'],
		'long_name': 'linear volume depolarization ratio restricted to cloud mask',
		'units': '1',
	},
    'temp': {
		'.dims': ['time', 'level'],
		'long_name': 'air temperature from interpolated AMPS temperature',
        'standard_name': 'air temperature',
		'units': 'degrees_C',
	},
	'amps_flag': {
# 		'.dims': [],
		'long_name': 'true if AMPS temperature data present, false if not',

	},
	'vdr_threshold': {
# 		'.dims': [],
		'long_name': 'vdr threshold below which reference mask determines cloud is liquid',
		'units': '1',
	},
	'LCC_ij': {
		'.dims': ['time', 'level'],
		'long_name': 'reference mask classification of cloud as liquid containing cloud',
	},
    'LCC_j': {
		'.dims': ['time'],
		'long_name': 'reference mask classification of cloud as liquid containing cloud',
	},
	'ICC_ij': {
		'.dims': ['time', 'level'],
		'long_name': 'reference mask classification of cloud as ice containing cloud',
	},
    'ICC_j': {
		'.dims': ['time'],
		'long_name': 'reference mask classification of cloud as ice containing cloud',
	},
	'SLCC_ij': {
		'.dims': ['time', 'level'],
		'long_name': 'reference mask classification of cloud as supercooled liquid containing cloud',
	},
    'SLCC_j': {
		'.dims': ['time'],
		'long_name': 'reference mask classification of cloud as supercooled liquid containing cloud',
	},
	'WLCC_ij': {
		'.dims': ['time', 'level'],
		'long_name': 'reference mask classification of cloud as warm liquid containing cloud',
	},
    'WLCC_j': {
		'.dims': ['time'],
		'long_name': 'reference mask classification of cloud as warm liquid containing cloud',
	},
}




def get_ok_dates():
    mpl_file_list = glob.glob(mpl2nc_dir+'*')
    mpl_date_list_all = [file[-15:-7] for file in mpl_file_list]
    mpl_date_list = np.unique(np.array(mpl_date_list_all))
    
    checked_date_list = []
    reject_date_list = []
    
    for date in mpl_date_list:
        alcf_format = '{}-{}-{}T000000.nc'.format(date[:4],date[4:6],date[6:8])
    
        ### Raises error if ALCF data not present
        if not os.path.exists(alcf_dir+alcf_format):
            raise FileNotFoundError('ALCF Output not found', date)
            
        ### Only includes data if AMPS data present
        amps_format = '{}00'.format(date)
        if os.path.exists(amps_dir+amps_format):
            checked_date_list.append(date)

        else:
            reject_date_list.append(date)
  
    # print('Rejected the following dates because AMPS data not available:')
    print(reject_date_list)
    
    return checked_date_list

            

def calc_vdr(date):
    """
    Opens mpl2nc output files for a given date and concatenates to one NC file.
    Calculates VDR from cross- and co-polarised backscatter.
    Calculates heights from range.
    Returns VDR, heights and time

    Parameters
    ----------
    date : str
        example: '20210518'
    """
    ## Loads MPL depolarization data from mpl2nc output and calcualtes VDR
    
    # Opens all files in mpl2nc directory from date, merges into one dataset
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

    d.close()
    
    return vdr, height, time

def pres2alt(pres):
    """
    pressure: units hPa
    alt: units m
    """
    
    scale_height = 7000# use tsurface
    sfc_press = 1013
    
    alt = -scale_height*np.log(pres/sfc_press)
    return alt#.replace([np.inf, -np.inf], np.nan).dropna()

def open_amps(date):
    ### Opens AMPS
        
    amps_filepath= amps_dir+"{0}00/{0}00_WRF_NZ_f*.nc".format(date)
    amps_ds = xr.open_mfdataset(amps_filepath,combine='nested',concat_dim='time',drop_variables=['A_PCP_GDS5_SFC_acc3h','A_PCP_GDS5_SFC_acc'])
    
    temps=amps_ds.variables['TMP_GDS5_ISBL'].values #temp as function of pressure level (altitude)
    iso_lvls=amps_ds.variables['lv_ISBL2'].values #pressure levels (ie altitudes) 
    
    site_temp = temps[:8,:,228,257]-273 # For ER site
    
    ## Interpolating AMPS temperature grid over MPL grid
    x = np.linspace(0,287, 8)
    y = np.linspace(0,999,21)
    X,Y = np.meshgrid(x,y)
    
    Z = site_temp
    
    x2 = np.linspace(0,287,288)
    y2 = np.linspace(0,999,1000)
    X2, Y2 = np.meshgrid(x2, y2)
        
    f = interpolate.interp2d(y, x, Z, kind='cubic')
    Z2 = f(y2, x2)
    
    site_temp_interpol = np.flip(Z2,axis=1)
    
    ## Interpolation of AMPS iso Levels
    xx = np.linspace(0,999,21)
    yy = iso_lvls
    
    xx_new = np.linspace(0,999,1000)
    
    f = interpolate.interp1d(xx, yy)
    pres_levels_interpolate = np.flip(f(xx_new))
    
    alt_levels = pres2alt(pres_levels_interpolate)
    
    return site_temp_interpol, alt_levels

def convert_times(jtime):
    utime = (jtime - 2440587.5)*86400

    utc_time = np.zeros(np.size(utime),datetime)
        
    for index, x in np.ndenumerate(utime):
        utc_time[index] = datetime.utcfromtimestamp(x)

    return utc_time

def open_alcf(date):
    alcf_format = '{}-{}-{}T000000.nc'.format(date[:4],date[4:6],date[6:8])
    alcf_ds = Dataset(alcf_dir.format(date[:-2])+alcf_format)
    alcf_cloud_mask = alcf_ds.variables['cloud_mask'][:]

    jtime = alcf_ds.variables['time'][:] #size 288

    alcf_utc_time = convert_times(jtime)
    zfull = alcf_ds.variables['zfull'][:] #size 300
    
    alcf_ds.close()
    
    d = ds.read(alcf_dir.format(date[:-2])+alcf_format)
    
    return d, alcf_cloud_mask, zfull, alcf_utc_time 

def find_nearest(array, value):
    ### Inputs an array 'a' and value 'x'; returns the closest element of a to x
    array = np.asarray(array)
    diffs = (np.abs(array - value))
    min_diff = diffs.min()
    idx= diffs.argmin()
    return idx,min_diff

def convert_time_idx(alcf_idx, alcf_times,mpl_times_dt):
    ### Inputs the index of the ALCF times; outputs the index of the nearest MPL time
    alcf_times_np = np.array([alcf_times]).astype('datetime64[ms]')
    alcf_time = alcf_times_np[0,alcf_idx]

    mpl_idx,diff = find_nearest(mpl_times_dt,alcf_time)

    return mpl_idx, diff

def subsample(alcf_cloud_mask, alcf_heights, alcf_time, amps_temps, amps_heights, mpl_vdr, mpl_heights, mpl_time):
    ### Regrid MPL data to same size as ALCF data
        
    vdr_grid = np.full(np.shape(alcf_cloud_mask),-999.)

    temps_grid = np.full(np.shape(alcf_cloud_mask),np.nan)
    
    for time_idx, profile in enumerate(alcf_cloud_mask):

        #Get time index of mpl2nc grid
        mpl_time_idx, mpl_time_diff = convert_time_idx(time_idx, alcf_time,mpl_time)
            
        for alt_idx, mask in enumerate(profile):
                
            alt = alcf_heights[alt_idx]
            
            # #Value of VDR for that pixel:
            if mpl_time_diff <= np.timedelta64(1,'m'):
                
                # altitude scale conversion
                mpl_alt_idx,_ = find_nearest(mpl_heights, alt)
                pixel_VDR = mpl_vdr[mpl_time_idx, mpl_alt_idx]
                
                if pixel_VDR==np.inf or pixel_VDR==-np.inf:
                        pixel_VDR = -999
                        
                vdr_grid[time_idx,alt_idx] = pixel_VDR
                            
            if amps_temps is not None:
                amps_alt_idx,_ = find_nearest(amps_heights,alt)
                temp = amps_temps[time_idx, amps_alt_idx]
                temps_grid[time_idx,alt_idx] = temp
            
    return vdr_grid, temps_grid
    
def group_non_nan(vdr_arr,cloud_mask_arr):
        """
        Looks at individual cloud mask profile. 
        Identifies groups (ie different cloud layers).
        Gets the nanmean VDR of each group.
        Fills group with nanmean VDR and returns array same size as cloud mask.

        Parameters
        ----------
        vdr_arr : TYPE
            VDR array, ALCF grid size.
        cloud_mask_arr : TYPE
            Cloud mask array, ALCF grid size, 1 where cloud, 0 else.

        Returns
        -------
        vdr_mean_arr : TYPE
            DESCRIPTION.

        """
        cm_df = pd.Series(cloud_mask_arr)
        vdr_df = pd.Series(vdr_arr)
        vdr_mean_arr = np.full(np.shape(cloud_mask_arr),np.nan)
        vdr_median_arr = np.full(np.shape(cloud_mask_arr),np.nan)
        for _,v in cm_df[cm_df==1].groupby((cm_df==0).cumsum()):
            v_mean=np.nanmean(vdr_df[v.index])
            v_median=np.nanmedian(vdr_df[v.index])
            vdr_mean_arr[v.index]=v_mean
            vdr_median_arr[v.index]=v_median
        return vdr_mean_arr,vdr_median_arr


def get_vdr_cm(vdr_subsamp, alcf_cloud_mask):
    # Layer-averaging of VDR
    
    vdr_subsamp = np.where(vdr_subsamp<0,np.nan,vdr_subsamp)
    vdr_subsamp = np.where(vdr_subsamp>1,np.nan,vdr_subsamp)

    vdr_grouped_mean = np.full(np.shape(vdr_subsamp),np.nan)

    for time_idx, profile in enumerate(alcf_cloud_mask):
        vdr_grouped_mean[time_idx],_= group_non_nan(vdr_subsamp[time_idx],profile)
      
    return vdr_grouped_mean


def process(date):
        
    print(date, ' - Calculating VDR')
    mpl_vdr, mpl_heights, mpl_time = calc_vdr(date)
    
    print(date, ' - Opening ALCF')
    d, alcf_cloud_mask, alcf_heights, alcf_time = open_alcf(date)

    if os.path.exists(amps_dir+'{}00'.format(date)):
        print(date, ' - Opening AMPS')
        amps_temps, amps_heights = open_amps(date)
        amps_flag = True
    
    else:
        print(date, ' - AMPS not available')
        amps_temps, amps_heights = None, None
        amps_flag = False
        
       
    print(date, ' - Subsampling VDR and temps')
    vdr_subsamp, temps_subsamp = subsample(alcf_cloud_mask, alcf_heights, alcf_time, amps_temps, amps_heights, mpl_vdr, mpl_heights, mpl_time)

    print(date, ' - Layer averaging of VDR')
    vdr_cloud_mask = get_vdr_cm(vdr_subsamp, alcf_cloud_mask)
    
    print(date, ' - Creating reference mask')

    vdr_threshold = 0.1 # below this, classify as liquid
                        # above this, classify as ice

    LCC_ij = np.where(vdr_cloud_mask<=vdr_threshold,1,0)
    LCC_j = np.max(LCC_ij,axis=1)

    ICC_ij = np.where(vdr_cloud_mask>vdr_threshold,1,0)
    ICC_j = np.max(ICC_ij,axis=1)
    
    if amps_flag:
        # If AMPS temperature data available, classifies SLCC according to temps
        slcc_cond = np.logical_and(vdr_cloud_mask<=vdr_threshold,temps_subsamp<=0)
        SLCC_ij = np.where(slcc_cond,1,0)
        SLCC_j = np.max(SLCC_ij,axis=1)
        
        wlcc_cond = np.logical_and(vdr_cloud_mask<=vdr_threshold,temps_subsamp>0)
        WLCC_ij = np.where(wlcc_cond,1,0)
        WLCC_j = np.max(WLCC_ij,axis=1)
    else:
        SLCC_ij = np.full(np.shape(LCC_ij),np.nan)
        SLCC_j = np.full(np.shape(LCC_j),np.nan)
        WLCC_ij = np.full(np.shape(LCC_ij),np.nan)
        WLCC_j = np.full(np.shape(LCC_j),np.nan)
    
    
    ds.var(d,'vdr',vdr_subsamp)
    ds.var(d,'vdr_cm',vdr_cloud_mask)
    ds.var(d,'temp',temps_subsamp)
    ds.var(d,'amps_flag',int(amps_flag))
    ds.var(d,'vdr_threshold',vdr_threshold)
    ds.var(d,'LCC_ij',LCC_ij)
    ds.var(d,'LCC_j',LCC_j)
    ds.var(d,'ICC_ij',ICC_ij)
    ds.var(d,'ICC_j',ICC_j)
    ds.var(d,'SLCC_ij',SLCC_ij)
    ds.var(d,'SLCC_j',SLCC_j)
    ds.var(d,'WLCC_ij',WLCC_ij)
    ds.var(d,'WLCC_j',WLCC_j)
    
    for var in VARIABLES:
        ds.meta(d, var, META.get(var))
        
        
    ds.meta(d,'.',{'created':datetime.utcnow().strftime('%Y-%m-%dT:%H:%M:%SZ')})
        
    output_path = output_dir.format(date[:-2])+str(date)+'.nc'
    
    print('Writing to',output_path)
    ds.write(output_path, d)
    
    

def main():
    dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]
    unprocessed_dates = []
    for date in dates:
        if not os.path.exists(output_dir.format(date[:-2])+str(date)+'.nc'):
            unprocessed_dates+=[date]

    
    for date in unprocessed_dates:
        print('Processing', date)
        start_time = time.time()
        process(date)
        print("--- %s seconds ---" % (time.time() - start_time))
        

if __name__ == "__main__":
    main()