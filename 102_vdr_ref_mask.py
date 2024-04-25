## IMPORTS

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd


## VARIABLES

data_dir = '/home/lukeew/Local_work/data/01_SLW_xgb'
vdr_mask_data_dir = data_dir + '/CHCH_v6_reference/preprocessed{}'
output_dir = data_dir + '/CHCH_v6_reference/processed{}/'


liq_threshold = 0.1
mix_threshold = 0.4

# date_list = ['20210518']

## FUNCTIONS

def classify_bin(ds, liq_threshold = liq_threshold, mix_threshold = mix_threshold):

    ds['undet_nan']   = xr.where((ds.vdr.isnull()),1,0).astype('int8') # Gets Nans due to missing data

    # Applies (0,1) threshold for VDR
    ds['vdr']  = xr.where(ds.vdr<0,np.nan,ds.vdr)
    ds['vdr']  = xr.where(ds.vdr>1,np.nan,ds.vdr)

    ds['bin_wlwc'] = xr.where((ds.cloud_mask == 1) & (ds.vdr <= liq_threshold) & (ds.temp >= 0),1,0).astype('int8') # Warm liquid water clouds
    ds['bin_slwc'] = xr.where((ds.cloud_mask == 1) & (ds.vdr <= liq_threshold) & (ds.temp <  0),1,0).astype('int8') # Supercooled liquid water clouds
    ds['bin_mpc'] = xr.where((ds.cloud_mask == 1) & (ds.vdr <= mix_threshold) & (ds.vdr > liq_threshold) & (ds.temp <  0),1,0).astype('int8') # Mixed phase clouds
    ds['bin_slcc'] = xr.where((ds.cloud_mask == 1) & (ds.vdr <= mix_threshold) & (ds.temp <  0),1,0).astype('int8') # Supercooled liquid water containing clouds 
    ds['bin_ic']   = xr.where((ds.cloud_mask == 1) & (ds.vdr >  mix_threshold) & (ds.temp <  0),1,0).astype('int8') # Ice clouds

    ds['bin_liq'] = xr.where((ds.bin_wlwc) | (ds.bin_slwc),1,0).astype('int8')

    ds['bin_undet']   = xr.where((ds.cloud_mask == 1) & (ds.vdr > liq_threshold) & (ds.temp > 0),1,0).astype('int8') # Undetermined from VDR and temperatures alone

    return ds


def get_layer_modes(cloud_mask, liq, mix, ice, undet):

    cm_df = pd.Series(cloud_mask)
    liq_df = pd.Series(liq)
    mix_df = pd.Series(mix)
    ice_df = pd.Series(ice)
    undet_df = pd.Series(undet)

    modes_arr = np.full(np.shape(cloud_mask),np.nan)  # array of length (n altitude bins)

    for _,v in cm_df[cm_df==1].groupby((cm_df==0).cumsum()):
        n_liq = liq_df[v.index].sum()
        n_mix = mix_df[v.index].sum()
        n_ice = ice_df[v.index].sum()
        n_undet = undet_df[v.index].sum()

        counts = np.array([n_undet,n_mix,n_liq,n_ice])  #order important because if two n are the same, the first will be chosen (so undetermined the best)
        mode = np.argmax(counts)
        modes_arr[v.index] = mode

    return modes_arr


def classify_layer(ds):
    modes_all = np.full(np.shape(ds.cloud_mask),np.nan)

    for time_idx, cm_profile in enumerate(ds.cloud_mask):   
        ds_time =  ds.isel(time=time_idx)  
        modes = get_layer_modes(ds_time.cloud_mask.values, ds_time.bin_liq.values, ds_time.bin_mpc.values, ds_time.bin_ic.values, ds_time.bin_undet.values)
        modes_all[time_idx] = modes

    modes_da = xr.DataArray(modes_all, coords=[('time',ds.time.data),('level',ds.level.data)])

    ds['layer_ice'] = xr.where((modes_da == 3)|((~np.isnan(modes_da))&(ds.temp<=-40)),1,0).astype('int8')
    ds['layer_liq'] = xr.where((modes_da == 2)&(ds.temp>-40),1,0).astype('int8')
    ds['layer_mix'] = xr.where((modes_da == 1)&(ds.temp>-40),1,0).astype('int8')
    ds['layer_undet'] = xr.where((modes_da == 0),1,0).astype('int8')

    ds['layer_wlwc'] = xr.where((ds.layer_liq == 1) & (ds.temp >= 0),1,0).astype('int8')
    ds['layer_slwc'] = xr.where((ds.layer_liq == 1) & (ds.temp < 0),1,0).astype('int8')
    ds['layer_slcc'] = xr.where((ds.layer_slwc == 1) | (ds.layer_mix == 1),1,0).astype('int8')

    return ds


def process(date):

    print('{} starting...'.format(date))

    infile = vdr_mask_data_dir.format(date[:6]) + '/{}.nc'.format(date)
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile)

    ds = classify_bin(ds)
    ds = classify_layer(ds)

    print('Finished {}.'.format(date))

    outfile = output_dir.format(date[:-2])+str(date)+'.nc'
    ds.to_netcdf(outfile, format='NETCDF4')
    print('  ---> ' + outfile)

    ds.close()


def getUnproDates():

    dates = [file[-11:-3].replace('-','') for file in sorted(glob.glob(vdr_mask_data_dir.format('*')+'/*.nc'))]
    unprocessed_dates = []

    for date in dates:
        if not os.path.exists(output_dir.format(date[:-2])+str(date)+'.nc'):
            unprocessed_dates+=[date]

    return unprocessed_dates


def main():

    date_list = getUnproDates()

    for date in date_list:
        process(date)


if __name__ == "__main__":
    main()

