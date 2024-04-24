## IMPORTS

import numpy as np
import os
import glob
import xarray as xr
from scipy import interpolate
from netCDF4 import Dataset
from datetime import datetime
import pandas as pd
import time


## VARIABLES

data_dir = '/home/lukeew/Local_work/data/01_SLW_xgb'

amps_dir = data_dir + '/AMPS_NC/'
mpl2nc_dir = data_dir + '/CHCH_preprocessed/mpl2nc{}/'

alcf_dir = data_dir + '/CHCH_preprocessed/alcf{}/lidar/'

output_dir = data_dir + '/CHCH_v6_reference/preprocessed{}/'


## FUNCTIONS

# Opening MPL

def getDropVars():
    ### Function to identify unnecessary variables from the input dataset that should be dropped
    date = '20210518'
    path = (mpl2nc_dir+'{}1200.nc').format(date[:-2],date)
    
    d = xr.open_dataset(path)
    drop_vars = [i for i in d.data_vars if i not in ['nrb_crosspol', 'nrb_copol', 'c', 'bin_time', 'range', 'time']]
    d.close()

    return drop_vars

def getMPL(date, drop_vars):

    filepath = (mpl2nc_dir+'{}*').format(date[:-2],date)
    ds = xr.open_mfdataset(filepath,combine='nested',concat_dim='profile', drop_variables = drop_vars)
        
    nrb_cross_pol = ds['nrb_crosspol']
    nrb_co_pol = ds['nrb_copol']
    time = ds['time']

    c = ds['c'][0]
    bin_time= ds['bin_time'][0]
    range_ = ds['range']

    height = 0.5*bin_time*c*(range_+0.5)

    ds.close()

    new_ds = xr.Dataset({'nrb_cross_pol':nrb_cross_pol,'nrb_co_pol':nrb_co_pol}, coords = {'time':time, 'height':height})

    return new_ds


# Opening AMPS

def pres2alt(pres):
    """
    pressure: units hPa
    alt: units m
    """
    
    scale_height = 7000# use tsurface
    sfc_press = 1013
    
    alt = -scale_height*np.log(pres/sfc_press)
    return alt#.replace([np.inf, -np.inf], np.nan).dropna()

def openAMPS(date):
    ### Opens AMPS

    amps_filepath= amps_dir+"{0}00/{0}00_WRF_NZ_f*.nc".format(date)
    amps_ds = xr.open_mfdataset(amps_filepath,combine='nested',concat_dim='time',drop_variables=['A_PCP_GDS5_SFC_acc3h','A_PCP_GDS5_SFC_acc'])
    
    temps=amps_ds.variables['TMP_GDS5_ISBL'] #temp as function of pressure level (altitude)
    iso_lvls=amps_ds.variables['lv_ISBL2'] #pressure levels (ie altitudes) 
    
    site_temp = temps[:8,:,228,257]-273 # For ER site
    
    ## Interpolating AMPS temperature grid to 5 minute resolution and 1000 levels
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

    amps_ds.close()
    
    return site_temp_interpol, alt_levels

# Subsampling

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

def subsample_mpl(alcf_ds, mpl_ds):

    alcf_cloud_mask = alcf_ds.cloud_mask.values
    alcf_heights = alcf_ds.zfull.values
    alcf_time = alcf_ds.time.values
    mpl_nrb_copol = mpl_ds.nrb_co_pol.values
    mpl_nrb_crosspol = mpl_ds.nrb_cross_pol.values
    mpl_heights = mpl_ds.height.values
    mpl_time = mpl_ds.time.values

    ### Regrid MPL data to same size as ALCF data
        
    copol_grid = np.full(np.shape(alcf_cloud_mask),np.nan)
    crosspol_grid = np.full(np.shape(alcf_cloud_mask),np.nan)
    
    for time_idx, profile in enumerate(alcf_cloud_mask):

        #Get time index of mpl2nc grid
        mpl_time_idx, mpl_time_diff = convert_time_idx(time_idx, alcf_time,mpl_time)
            
        for alt_idx, mask in enumerate(profile):
                
            alt = alcf_heights[alt_idx]
            
            # #Value of NRB co/cross for that pixel:
            if mpl_time_diff <= np.timedelta64(1,'m'):
                
                # altitude scale conversion
                mpl_alt_idx,_ = find_nearest(mpl_heights, alt)
                pixel_copol = mpl_nrb_copol[mpl_time_idx, mpl_alt_idx]
                pixel_crosspol = mpl_nrb_crosspol[mpl_time_idx, mpl_alt_idx]
                
                if pixel_copol==np.inf or pixel_copol==-np.inf:
                    pixel_copol = np.nan
                if pixel_crosspol==np.inf or pixel_crosspol==-np.inf:
                    pixel_crosspol = np.nan
                        
                copol_grid[time_idx,alt_idx] = pixel_copol
                crosspol_grid[time_idx,alt_idx] = pixel_crosspol
                            
    copol_da = xr.DataArray(copol_grid, coords=[('time',alcf_ds.time.data), ('level',alcf_ds.zfull.data)])
    crosspol_da = xr.DataArray(crosspol_grid, coords=[('time',alcf_ds.time.data), ('level',alcf_ds.zfull.data)])
    new_mpl_ds = xr.Dataset({'copol':copol_da, 'crosspol':crosspol_da})
    
    return new_mpl_ds


def subsample_amps(alcf_ds, amps_temps, amps_heights):
    alcf_cloud_mask = alcf_ds.cloud_mask.values
    alcf_heights = alcf_ds.zfull.values
    
    temps_grid = np.full(np.shape(alcf_cloud_mask),np.nan)

    for time_idx, profile in enumerate(alcf_cloud_mask):
        for alt_idx, _ in enumerate(profile):
            alt = alcf_heights[alt_idx]

            if amps_temps is not None:
                amps_alt_idx,_ = find_nearest(amps_heights,alt)
                temp = amps_temps[time_idx, amps_alt_idx]
                temps_grid[time_idx,alt_idx] = temp

    temps_da = xr.DataArray(temps_grid, coords=[('time',alcf_ds.time.data), ('level',alcf_ds.zfull.data)])

    return temps_da



def process(date):

    print(date, ' - Opening MPL')
    drop_vars = getDropVars()
    mpl_ds = getMPL(date, drop_vars)

    print(date,' - Opening AMPS')
    amps_temps, amps_range = openAMPS(date)

    print(date,' - Opening ALCF')
    alcf_infile = alcf_dir.format(date[:-2])+'{}-{}-{}T000000.nc'.format(date[:4],date[4:6],date[6:8])
    alcf_ds = xr.open_dataset(alcf_infile)

    print(date,' - Subsampling MPL')
    ds = subsample_mpl(alcf_ds, mpl_ds)

    print(date,' - Subsampling AMPS')
    temps_da = subsample_amps(alcf_ds,amps_temps, amps_range)

    ds['cloud_mask'] = alcf_ds.cloud_mask
    ds['backscatter'] = alcf_ds.backscatter
    ds['temp'] = temps_da

    print(date,' - Calculating VDR')
    ds = ds.assign(vdr = ds.crosspol/ds.copol)

    print('Finished {}.'.format(date))

    outfile = output_dir.format(date[:-2])+str(date)+'.nc'
    ds.to_netcdf(outfile, format='NETCDF4')
    print('  ---> ' + outfile)

    mpl_ds.close()
    alcf_ds.close()
    ds.close()


def getUnproDates():

    dates = [file[-20:-10].replace('-','') for file in sorted(glob.glob(alcf_dir.format('*')+'/*.nc'))]
    unprocessed_dates = []

    for date in dates:
        if not os.path.exists(output_dir.format(date[:-2])+str(date)+'.nc'):
            unprocessed_dates+=[date]

    return unprocessed_dates


def main():

    date_list = getUnproDates()

    for date in date_list:
        try:
            process(date)
        except OSError:
            print('OSError due to missing files')
            pass



if __name__ == "__main__":
    main()