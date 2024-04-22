## IMPORTS

import os
import glob
import xarray as xr


## VARIABLES

data_dir = '/home/lukeew/Local_work/data/01_SLW_xgb'
vdr_mask_data_dir = data_dir + '/CHCH_v4_reference/preprocessed{}'
output_dir = data_dir + '/CHCH_v4_reference/processed{}/'


liq_threshold = 0.1
mix_threshold = 0.4

# date_list = ['20210518']

## FUNCTIONS

def classify(ds, liq_threshold = liq_threshold, mix_threshold = mix_threshold):

    ds['wlwc'] = xr.where((ds.vdr_layer <= liq_threshold) & (ds.temp >= 0),1,0).astype('int8') # Warm liquid water clouds
    ds['slwc'] = xr.where((ds.vdr_layer <= liq_threshold) & (ds.temp <  0),1,0).astype('int8') # Supercooled liquid water clouds
    ds['mpc'] = xr.where((ds.vdr_layer <= mix_threshold) & (ds.vdr_layer > liq_threshold) & (ds.temp <  0),1,0).astype('int8') # Mixed phase clouds
    ds['slcc'] = xr.where((ds.vdr_layer <= mix_threshold) & (ds.temp <  0),1,0).astype('int8') # Supercooled liquid water containing clouds 
    ds['ic']   = xr.where((ds.vdr_layer >  mix_threshold) & (ds.temp <  0),1,0).astype('int8') # Ice clouds

    ds['undet']   = xr.where((ds.vdr_layer > liq_threshold) & (ds.temp > 0),1,0).astype('int8') # Undetermined from VDR and temperatures alone
    ds['u_nan']   = xr.where((ds.vdr_layer.isnull()) & (ds.cloud_mask == 1),1,0).astype('int8') # Data missing
    ds['unknown']   = xr.where((ds.undet) | (ds.u_nan),1,0).astype('int8') # Unknown, either undetermined or missing data

    return ds


def process(date):

    print('{} starting...'.format(date))

    infile = vdr_mask_data_dir.format(date[:6]) + '/{}.nc'.format(date)
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile)

    ds = classify(ds)

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

