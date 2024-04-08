## IMPORTS

import glob
import xarray as xr
import scipy.signal as signal
import pandas as pd

## VARIABLES

data_dir = '/home/lukeew/Local_work/data/01_SLW_xgb'
reference_mask_dir = data_dir + '/CHCH_v2_reference/processed{}/'
output_dir = data_dir + '/CHCH_v2_reference/'

## FUNCTIONS

def getPeaks(date):

    infile = reference_mask_dir.format(date[:6]) + '{}.nc'.format(date)
    print('  <--- ' + infile)

    ds = xr.open_dataset(infile)

    backscatter = ds.backscatter.values 
    altitude = ds.zfull.values
    temperature = ds.temp.values

    wlwc_target = ds.wlwc.values
    slwc_target = ds.slwc.values
    slcc_target = ds.slcc.values
    ic_target = ds.ic.values
    undet_target = ds.undet.values # Undetermined by reference mask
    unan_target = ds.u_nan.values # Unknown due to missing data

    magnitude_threshold = 2e-5
    width_threshold = 1

    date_peak_dict = []

    for time_idx, profile in enumerate(backscatter):
        
        ### For each vertical profile, gets peak properties
        peaks, properties = signal.find_peaks(profile,height=magnitude_threshold, width=width_threshold, rel_height=0.5, prominence=(None,None))
        
        for peak_idx, peak_alt_idx in enumerate(peaks):
            
            peak_dict = {
                # Meta data
                'date' : date,
                'peak_time_idx' : time_idx,
                'peak_alt_idx' : peak_alt_idx,
                
                # Backscatter properties
                'peak_magnitude' : properties['peak_heights'][peak_idx],
                'peak_width' : properties['widths'][peak_idx],
                'peak_width_height' : properties['width_heights'][peak_idx],
                'peak_prominence' : properties['prominences'][peak_idx],
                'peak_alt' : altitude[peak_alt_idx],
                'num_peaks' : len(peaks),
                'peak_order' : peak_idx,
                
                # AMPS temperature
                'peak_temp' : temperature[time_idx, peak_alt_idx],
                            
                # Reference mask target            
                'wlwc_target' : wlwc_target[time_idx,peak_alt_idx],
                'slwc_target' : slwc_target[time_idx,peak_alt_idx],
                'slcc_target' : slcc_target[time_idx,peak_alt_idx],
                'ic_target' : ic_target[time_idx,peak_alt_idx],
                'undet_target' : undet_target[time_idx,peak_alt_idx],
                'unan_target' : unan_target[time_idx,peak_alt_idx],
                            
                }
            
            date_peak_dict.append(peak_dict)

    ds.close()
    
    return date_peak_dict


def main():

    dates = [file[-11:-3].replace('-','') for file in sorted(glob.glob(reference_mask_dir.format('*')+'/*.nc'))]

    peak_dict = []
    
    for date in dates:
        date_peak_dict = getPeaks(date)
        peak_dict += date_peak_dict

    print('All dates processed...')
        
    peaks_df = pd.DataFrame.from_dict(peak_dict)

    print('Saving to CSV...')

    outfile = output_dir+'peaks_training_df.csv'
    peaks_df.to_csv(outfile)
    print('  ---> ' + outfile)


if __name__ == "__main__":
    peaks_df = main()
        