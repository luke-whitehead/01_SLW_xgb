## IMPORTS

import xarray as xr

## VARIABLES

data_dir = '/home/lukeew/Local_work/data/01_SLW_xgb'
vdr_mask_data_dir = data_dir + '/CHCH_v1_reference/processed{}'
output_dir = data_dir + '/CHCH_v2_reference/processed{}/'

EXCL_VARS = ['ICC_ij', 'ICC_j', 'LCC_ij', 'LCC_j', 'SLCC_ij', 'SLCC_j', 'WLCC_ij', 'WLCC_j', 
             'vdr_threshold', 'lr','backscatter_sd','amps_flag','lat','lon', 'time_bnds','altitude','cbh',
             ]

liq_threshold = 0.1
mix_threshold = 0.4

## FUNCTIONS

def classify(ds, liq_threshold = liq_threshold, mix_threshold = mix_threshold):

    ds['wlwc'] = xr.where((ds.vdr_cm <= liq_threshold) & (ds.temp >= 0),1,0).astype('int8') # Warm liquid water clouds
    ds['slwc'] = xr.where((ds.vdr_cm <= liq_threshold) & (ds.temp <  0),1,0).astype('int8') # Supercooled liquid water clouds
    ds['slcc'] = xr.where((ds.vdr_cm <= mix_threshold) & (ds.temp <  0),1,0).astype('int8') # Supercooled liquid water containing clouds 
    ds['ic']   = xr.where((ds.vdr_cm >  mix_threshold) & (ds.temp <  0),1,0).astype('int8') # Ice clouds

    ds['undet']   = xr.where((ds.vdr_cm > liq_threshold) & (ds.temp > 0),1,0).astype('int8') # Undetermined from VDR and temperatures alone
    ds['u_nan']   = xr.where((ds.vdr_cm.isnull()) & (ds.cloud_mask == 1),1,0).astype('int8') # Data missing
    ds['unknown']   = xr.where((ds.undet) | (ds.u_nan),1,0).astype('int8') # Unknown, either undetermined or missing data

    return ds


def process(date):

    print('{} starting...'.format(date))

    infile = vdr_mask_data_dir.format(date[:6]) + '/{}.nc'.format(date)
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile, drop_variables=EXCL_VARS)
    ds = ds.set_coords('zfull')

    ds = classify(ds)

    print('Finished {}.'.format(date))

    outfile = output_dir.format(date[:-2])+str(date)+'.nc'
    ds.to_netcdf(outfile, format='NETCDF4')
    print('  ---> ' + outfile)

    ds.close()


process('20210520')

