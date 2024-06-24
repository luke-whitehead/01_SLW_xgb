## IMPORTS

import xarray as xr
import scipy.signal as signal
import xgboost as xgb
import glob
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

import joblib


## VARIABLES

data_dir = '/home/lukeew/Local_work/data/01_SLW_xgb'
reference_dir = data_dir + '/CHCH_v6_reference/'
reference_mask_dir = reference_dir + 'processed{}/'
model_save_dir = reference_dir + 'models/'

output_dir = data_dir + '/CHCH_v6_model/{}/model_processed_{}/'

EXCL_VARS = ['copol', 'crosspol','vdr', 'undet_nan', 'bin_wlwc', 'bin_slwc', 'bin_mpc', 'bin_slcc', 'bin_ic', 'bin_liq', 'bin_undet', 'layer_ice', 'layer_liq', 'layer_mix', 'layer_undet','layer_wlwc', 'layer_slwc', 'layer_slcc']
new_cols = ['peak_order','num_peaks','peak_prominence','peak_magnitude','peak_width','peak_width_height','peak_temp','peak_alt']

magnitude_threshold = 2e-5
width_threshold = 1

## FUNCTIONS


def loadModel(model_name):

    print('Loading {}...'.format(model_name))

    model_infile = model_save_dir + model_name +'.json'
    print('  <--- ' + model_infile)

    if model_name == 'AG_XGBoost':
        model_ag_load = xgb.XGBClassifier();
        model_ag_load.load_model(model_infile)
        std = joblib.load(model_save_dir+'AG_scaler.save') 
        model = Pipeline([('standardscaler',std),('xgbclassifier',model_ag_load)])

    else:
        model = xgb.XGBClassifier();
        model.load_model(model_infile)

    return model


def process_AG(date, model):

    print('{} starting...'.format(date))

    ### Loading input dataset

    infile = reference_mask_dir.format(date[:-2])+str(date)+'.nc'
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile, drop_variables=EXCL_VARS)

    time = ds.time
    backscatter = ds.backscatter.values
    temp = ds.temp.values
    cloud_mask = ds.cloud_mask
    altitude = ds.level.values

    ds_new = ds.copy()
    ds.close()

    ### Initializing output dataset

    ds_new['classified_peaks'] = xr.full_like(cloud_mask, -1)
    ds_new['classified_peaks_mask'] = xr.zeros_like(cloud_mask)

    ds_new['slwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['slwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')

    ds_new['other'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['other_t'] = xr.zeros_like(ds_new.time, dtype='int8')

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

                peaks_df_ag = peak_df[new_cols]
                Z = model.predict(peaks_df_ag.values)[0]

                min_alt = int(peak_alt_idx-(properties['widths'][peak_idx]))
                max_alt = int(peak_alt_idx+(properties['widths'][peak_idx]))
                
                min_idx = max(0,min_alt)
                max_idx = min(299,max_alt)

                ### Model result saved to output dataset

                ds_new['classified_peaks'][dict(time=time_idx,level=peak_alt_idx)] = Z
                ds_new['classified_peaks_mask'][dict(time=time_idx)][min_idx:max_idx] = 1

                if Z == 1:
                    ds_new['slwc_t'][dict(time=time_idx)] = 1
                    ds_new['slwc'][dict(time=time_idx)][min_idx:max_idx] = 1
                elif Z == 0:
                    ds_new['other_t'][dict(time=time_idx)] = 1
                    ds_new['other'][dict(time=time_idx)][min_idx:max_idx] = 1

    print('Finished {}.'.format(date))

    return ds_new



def process_001(date, model):

    print('{} starting...'.format(date))

    ### Loading input dataset

    infile = reference_mask_dir.format(date[:-2])+str(date)+'.nc'
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile, drop_variables=EXCL_VARS)

    time = ds.time
    backscatter = ds.backscatter.values
    temp = ds.temp.values
    cloud_mask = ds.cloud_mask
    altitude = ds.level.values

    ds_new = ds.copy()
    ds.close()


    ### Initializing output dataset

    ds_new['classified_peaks'] = xr.full_like(cloud_mask, -1)
    ds_new['classified_peaks_mask'] = xr.zeros_like(cloud_mask)

    ds_new['slwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['slwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')

    ds_new['wlwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['mpc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['ic'] = xr.zeros_like(ds_new.cloud_mask)

    ds_new['wlwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
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

                min_alt = int(peak_alt_idx-(properties['widths'][peak_idx]))
                max_alt = int(peak_alt_idx+(properties['widths'][peak_idx]))
                
                min_idx = max(0,min_alt)
                max_idx = min(299,max_alt)

                ### Model result saved to output dataset

                ds_new['classified_peaks'][dict(time=time_idx,level=peak_alt_idx)] = Z
                ds_new['classified_peaks_mask'][dict(time=time_idx)][min_idx:max_idx] = 1

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


def process_002(date, model):

    print('{} starting...'.format(date))

    ### Loading input dataset

    infile = reference_mask_dir.format(date[:-2])+str(date)+'.nc'
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile, drop_variables=EXCL_VARS)

    time = ds.time
    backscatter = ds.backscatter.values
    temp = ds.temp.values
    cloud_mask = ds.cloud_mask
    altitude = ds.level.values

    ds_new = ds.copy()
    ds.close()


    ### Initializing output dataset

    ds_new['classified_peaks'] = xr.full_like(cloud_mask, -1)
    ds_new['classified_peaks_mask'] = xr.zeros_like(cloud_mask)

    ds_new['wlwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['slcc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['ic'] = xr.zeros_like(ds_new.cloud_mask)

    ds_new['wlwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
    ds_new['slcc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
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
                Z = model.predict(peak_df)[0] #0=WLWC, 1=SLCC, 2=IC

                min_alt = int(peak_alt_idx-(properties['widths'][peak_idx]))
                max_alt = int(peak_alt_idx+(properties['widths'][peak_idx]))
                
                min_idx = max(0,min_alt)
                max_idx = min(299,max_alt)

                ### Model result saved to output dataset

                ds_new['classified_peaks'][dict(time=time_idx,level=peak_alt_idx)] = Z
                ds_new['classified_peaks_mask'][dict(time=time_idx)][min_idx:max_idx] = 1

                if Z == 0:
                    ds_new['wlwc_t'][dict(time=time_idx)] = 1
                    ds_new['wlwc'][dict(time=time_idx)][min_idx:max_idx] = 1 
                elif Z == 1:
                    ds_new['slcc_t'][dict(time=time_idx)] = 1
                    ds_new['slcc'][dict(time=time_idx)][min_idx:max_idx] = 1
                elif Z == 2:
                    ds_new['ic_t'][dict(time=time_idx)] = 1
                    ds_new['ic'][dict(time=time_idx)][min_idx:max_idx] = 1

    print('Finished {}.'.format(date))

    return ds_new


def process_003(date, model):

    print('{} starting...'.format(date))

    ### Loading input dataset

    infile = reference_mask_dir.format(date[:-2])+str(date)+'.nc'
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile, drop_variables=EXCL_VARS)

    time = ds.time
    backscatter = ds.backscatter.values
    temp = ds.temp.values
    cloud_mask = ds.cloud_mask
    altitude = ds.level.values

    ds_new = ds.copy()
    ds.close()


    ### Initializing output dataset

    ds_new['classified_peaks'] = xr.full_like(cloud_mask, -1)
    ds_new['classified_peaks_mask'] = xr.zeros_like(cloud_mask)

    ds_new['wlwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['slwc'] = xr.zeros_like(ds_new.cloud_mask)
    ds_new['neither'] = xr.zeros_like(ds_new.cloud_mask)

    ds_new['wlwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
    ds_new['slwc_t'] = xr.zeros_like(ds_new.time, dtype='int8')
    ds_new['neither_t'] = xr.zeros_like(ds_new.time, dtype='int8')

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
                Z = model.predict(peak_df)[0] #0=WLWC, 1=SLWC, 2=neither (MPC/IC)

                min_alt = int(peak_alt_idx-(properties['widths'][peak_idx]))
                max_alt = int(peak_alt_idx+(properties['widths'][peak_idx]))
                
                min_idx = max(0,min_alt)
                max_idx = min(299,max_alt)

                ### Model result saved to output dataset

                ds_new['classified_peaks'][dict(time=time_idx,level=peak_alt_idx)] = Z
                ds_new['classified_peaks_mask'][dict(time=time_idx)][min_idx:max_idx] = 1

                if Z == 0:
                    ds_new['wlwc_t'][dict(time=time_idx)] = 1
                    ds_new['wlwc'][dict(time=time_idx)][min_idx:max_idx] = 1 
                elif Z == 1:
                    ds_new['slwc_t'][dict(time=time_idx)] = 1
                    ds_new['slwc'][dict(time=time_idx)][min_idx:max_idx] = 1
                elif Z == 2:
                    ds_new['neither_t'][dict(time=time_idx)] = 1
                    ds_new['neither'][dict(time=time_idx)][min_idx:max_idx] = 1


    print('Finished {}.'.format(date))

    return ds_new


def getUnproDates(model_name):

    print('Getting list of valid dates...')

    dates = [file[-11:-3].replace('-','') for file in sorted(glob.glob(reference_mask_dir.format('*')+'/*.nc'))]
    # valid_dates = getValidDates(dates)

    unprocessed_dates = []

    for date in dates:
        if not os.path.exists(output_dir.format(model_name,date[:-2])+str(date)+'.nc'):
            unprocessed_dates+=[date]

    return unprocessed_dates


def main():
    
    model_name = 'model_003'
    model = loadModel(model_name)

    date_list = getUnproDates(model_name)

    print('Got dates, start processing...')

    for date in date_list:
        ds = process_AG(date, model)

        outfile = output_dir.format(model_name,date[:-2])+str(date)+'.nc'
        ds.to_netcdf(outfile, format='NETCDF4')
        print('  ---> ' + outfile)


if __name__ == "__main__":
    main()