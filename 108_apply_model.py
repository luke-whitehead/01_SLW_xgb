## IMPORTS

import xarray as xr
import scipy.signal as signal
import xgboost as xgb
import glob
import os
import numpy as np
import pandas as pd


## VARIABLES

data_dir = '/home/lukeew/Local_work/data/01_SLW_xgb'
reference_dir = data_dir + '/CHCH_v3_reference/'
reference_mask_dir = reference_dir + 'processed{}/'
model_save_dir = reference_dir + 'models/'

output_dir = data_dir + '/CHCH_v3_model_processed/{}/model_processed_{}/'

EXCL_VARS = ['vdr', 'vdr_cm', 'wlwc', 'slwc', 'mpc', 'slcc', 'ic', 'undet', 'u_nan', 'unknown']

magnitude_threshold = 2e-5
width_threshold = 1

date_list = ['20210521', '20210522']


## FUNCTIONS

def loadModel(model_name):

    print('Loading {}...'.format(model_name))
    model_infile = model_save_dir + model_name +'.json'
    print('  <--- ' + model_infile)

    model = xgb.XGBClassifier();
    model.load_model(model_infile)

    return model


def process(date, model):

    print('{} starting...'.format(date))

    ### Loading input dataset

    infile = reference_mask_dir.format(date[:-2])+str(date)+'.nc'
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile, drop_variables=EXCL_VARS)

    time = ds.time
    backscatter = ds.backscatter.values
    temp = ds.temp.values
    cloud_mask = ds.cloud_mask
    altitude = ds.zfull.values

    ds_new = ds.copy()
    ds.close()

    ### Initializing output dataset

    ds_new['classified_peaks'] = xr.full_like(ds_new.backscatter, np.nan)

    ds_new['wlwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['slwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['mpc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['ic'] = xr.zeros_like(ds_new.cloud_mask)

    ds_new['wlwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
    ds_new['slwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
    ds_new['mpc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
    ds_new['ic_t'] = xr.zeros_like(ds_new.time, dtype='int8')

    ### Classifying profiles sequantially

    for time_idx, profile in enumerate(backscatter):
        
        ### For each vertical profile, gets peak properties
        peaks, properties = signal.find_peaks(profile,height=magnitude_threshold, width=width_threshold, rel_height=0.5, prominence=(None,None))

        for peak_idx, peak_alt_idx in enumerate(peaks):  
                        
                peak_dict = {                
                    # Backscatter properties
                    'peak_magnitude' : properties['peak_heights'][peak_idx],
                    'peak_width' : properties['widths'][peak_idx],
                    'peak_width_height' : properties['width_heights'][peak_idx],
                    'peak_prominence' : properties['prominences'][peak_idx],
                    'peak_alt' : altitude[peak_alt_idx]/1000,
                    'num_peaks' : len(peaks),
                    'peak_order' : peak_idx,
                    
                    # AMPS temperature
                    'peak_temp' : temp[time_idx, peak_alt_idx],                                
                    }
                
                peak_df = pd.DataFrame.from_dict([peak_dict])

                ### Properties passed to model for prediction
                Z = model.predict(peak_df)[0] #0=WLWC, 1=SLWC, 2=MPC 3=IC

                min_alt = int(peak_alt_idx-properties['widths'][peak_idx])
                max_alt = int(peak_alt_idx+properties['widths'][peak_idx])
                
                min_idx = max(0,min_alt)
                max_idx = min(299,max_alt)

                ### Model result saved to output dataset

                ds_new['classified_peaks'][dict(time=time_idx,level=peak_alt_idx)] = Z

                if Z == 0:
                    ds_new['wlwc_t'][dict(time=time_idx)] = 1
                    ds_new['wlwc'][dict(time=time_idx)][min_idx:max_idx] = 1 
                elif Z == 1:
                    ds_new['slwc_t'][dict(time=time_idx)] = 1
                    ds_new['slwc'][dict(time=time_idx)][min_idx:max_idx] = 1
                elif Z == 2:
                    ds_new['mpc_t'][dict(time=time_idx)] = 1
                    ds_new['mpc'][dict(time=time_idx)][min_idx:max_idx] = 1
                elif Z == 3:
                    ds_new['ic_t'][dict(time=time_idx)] = 1
                    ds_new['ic'][dict(time=time_idx)][min_idx:max_idx] = 1


    print('Finished {}.'.format(date))

    return ds_new



def getUnproDates(model_name):

    dates = [file[-11:-3].replace('-','') for file in sorted(glob.glob(reference_mask_dir.format('*')+'/*.nc'))]
    unprocessed_dates = []

    for date in dates:
        if not os.path.exists(output_dir.format(model_name,date[:-2])+str(date)+'.nc'):
            unprocessed_dates+=[date]

    return unprocessed_dates


def main():
    
    model_name = 'model_004'
    model = loadModel(model_name)

    date_list = getUnproDates(model_name)

    for date in date_list:
        ds = process(date, model)

        outfile = output_dir.format(model_name,date[:-2])+str(date)+'.nc'
        ds.to_netcdf(outfile, format='NETCDF4')
        print('  ---> ' + outfile)


if __name__ == "__main__":
    main()