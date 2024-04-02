# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:41:47 2022

@author: lwh41

Create individual backscatter profiles for a day.

"""
import glob
import numpy as np
import os.path
import xarray as xr
from scipy import interpolate
from netCDF4 import Dataset
from datetime import datetime
import seaborn as sns

import pandas as pd
import pickle
import itertools



import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import accuracy_score, classification_report,plot_roc_curve,plot_precision_recall_curve,confusion_matrix


raw_data_dir = 'C://Local//lwh41//DATA//'
# data_dir = 'C://Local//lwh41//DATA//vers_2//'
# mpl2nc_dir = raw_data_dir+'mpl2nc/'
# alcf_dir = raw_data_dir+'alcf_output/'
amps_dir = raw_data_dir+'AMPS_NC/'

# vdr_dir = raw_data_dir+'vdr/'

# mpl_pre_processed_nan_dir = data_dir+'mpl_pre_processed_nan/'
# reference_mask_dir = data_dir+'vdr_reference_mask/'

# model_classified_dir = data_dir+'model_classified_correct_width/'

# model = pickle.load(open(data_dir+'models/model.pickle.dat', 'rb'))


## Update 2023-01-08 for recalibrated MPL
mpl2nc_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_mpl2nc/'
alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'
vdr_dir =  'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_vdr/'
mpl_pre_processed_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_mpl_pre_processed/'
mpl_pre_processed_nan_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_mpl_pre_processed_nan/'
reference_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_vdr_reference_mask/'
### EDIT!
g22_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_slw_mask/'

### EDIT!
figs_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_peak_scatter_figs//'

pks_fpath_model_result = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_peaks//MPL_peak_df_bs_5e-6_new_ref_class_new_cal_with_result_slw_model.csv'
model_save_path = 'g22_chch.json'

# vars_to_drop = ['date','peak_time_idx', 'peak_alt_idx',
#                             'ref_slw', 'ref_wliq','ref_liq','ref_ice',
#                             'peak_vdr','peak_vdr_mask','peak_vdr_mean',
#                             'lr','lr_grad_abs',
#                             # 'peak_temp',
#                             # 'peak_alt'
#                             ]

# peaks_df = pd.read_csv(data_dir+'MPL_peak_df.csv', index_col=[0]) 

# X = peaks_df.drop(vars_to_drop, axis=1)

# y_pred = model.predict(X)
# peaks_df = peaks_df.assign(model_result=y_pred)


## Update 14.3.2023
# ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'

peaks_fpath = peaks_dir+'peaks_df_{}.csv'.format('2e-5')

peaks_df = pd.read_csv(peaks_fpath, index_col=[0])
# peaks_df = peaks_df.query('amps_flag == 1')

# peaks_df = pd.read_csv(pks_fpath_model_result, index_col=[0])
    

""" Psuedo Code:

For each date:
    Open ALCF output, extract backscatter + cloud mask
    Open reference mask, extract grid + profile view
    Open model classified mask, extract grid + profile view

"""

def make_figure(date):
    print('Plotting: ',date)
    
    data_mask_load = np.load(mpl_pre_processed_nan_dir+date+'.npz')
    loaded_vdr = data_mask_load['vdr']
    loaded_vdr_mask = data_mask_load['vdr_mask']
    loaded_vdr_mask_mean = data_mask_load['vdr_mask_mean']
    loaded_vdr_mask_median = data_mask_load['vdr_mask_med']
    loaded_temps = data_mask_load['temps_grid']
    
    ## Opens reference mask
    ref_mask_load = np.load(reference_mask_dir+date+'_mean.npz')
    ref_slw = ref_mask_load['ref_slw']
    ref_wliq = ref_mask_load['ref_wliq']
    ref_liq = ref_mask_load['ref_liq']
    ref_ice = ref_mask_load['ref_ice']
    
    ref_slw_prof = ref_mask_load['ref_slw_prof']
    ref_wliq_prof = ref_mask_load['ref_wliq_prof']
    ref_liq_prof = ref_mask_load['ref_liq_prof']
    ref_ice_prof = ref_mask_load['ref_ice_prof']

    
    alcf_format = '{}-{}-{}T000000.nc'.format(date[:4],date[4:6],date[6:8])
    alcf_ds = Dataset(alcf_dir+alcf_format)
    alcf_cloud_mask = alcf_ds.variables['cloud_mask']
    alcf_backscatter = alcf_ds['backscatter']
    
    # Open model mask
    model_mask_load = np.load(g22_mask_dir+date+'.npz')
    model_mask_grid = model_mask_load['grid']
    model_mask_profiles = model_mask_load['profile']
    

    # peaks_df_date = peaks_df.query('date == {}  and model_result ==1'.format(date))
    peaks_df_date = peaks_df.query('date == {}'.format(date))
    
    ice_bs = ([],[])
    liq_bs = ([],[])
    
    ice_width = ([],[])
    liq_width = ([],[])
    # result_width = np.full((288),np.nan)
    
    for peak_time_idx in np.unique(peaks_df_date['peak_time_idx']):
        peaks = peaks_df_date.query('peak_time_idx == {}'.format(peak_time_idx))
        

        for idx,peak in peaks.iterrows():
            alt_idx = peak.peak_alt_idx
            width = peak.peak_width
            bs = peak.peak_backscatter
            width_height = peak.peak_width_height
            prom = peak.peak_prominence
            vdr = peak.peak_vdr
            temp = peak.peak_temp
            peak_order = peak.peak_order
            
            if peak.LCC_ij == 1:
                liq_bs[0].append(peak_time_idx)
                liq_bs[1].append(bs)
                
                liq_width[0].append(peak_time_idx)
                liq_width[1].append(width)
                width
            if peak.ICC_ij == 1:
                ice_bs[0].append(peak_time_idx)
                ice_bs[1].append(bs)
                
                ice_width[0].append(peak_time_idx)
                ice_width[1].append(width)
            # else:
                # print('error')
                
    peak_scatter_dict = {
       'peak_backscatter':{
           'ice':ice_bs,
           'liq':liq_bs
           },
       'peak_width':{
           'ice':ice_width,
           'liq':liq_width
           }
        }
    

    
    ### Plot figure
    
    fig,axes  = plt.subplots(nrows=2,ncols=2,figsize=(18,12), gridspec_kw={'width_ratios':[3.5,1]})

    # (ax1,ax2), (ax3, ax4) = axes
    var_list = ['peak_backscatter', 'peak_width']
    
    for ax_idx, (ax1, ax2) in enumerate(axes):
        
        var = var_list[ax_idx]
        
        liq_tuple = peak_scatter_dict[var]['liq']
        ice_tuple = peak_scatter_dict[var]['ice']
    
        ## Fig 1 - ALCF backscatter
        
        Im1=ax1.imshow(alcf_backscatter[:].T,
           	extent=(0, 288, 0, 15),
           	aspect='auto',
           	origin='lower',
            interpolation='None',
            cmap='Blues',
           	norm=colors.LogNorm(vmin=1e-8,vmax=1e-4),
            
        )
        
        ax1.set_ylabel('Altitude (km)', fontsize=15)
        ax1.set_xticklabels([])
        ax1.set_xticks(np.arange(36,288,36))
        ax1.xaxis.set_minor_locator(MultipleLocator(12))   
        
        ax1.set_xticklabels([ '03:00',
                               '06:00',
                               '09:00',
                               '12:00',
                               '15:00',
                               '18:00',
                               '21:00'], fontsize=15)
        
        # ax1.vlines(profile_idcs,0,15,colors='r')
        
        
        # for idx, i in enumerate(profile_idcs):
            # ax1.text((i/288),1.02,letters[idx].format(i),fontsize=12,transform=ax1.transAxes,ha='center',color='r')
    
        
        
        ax1.set_title(var,fontsize=20)
        # ax1.text(0.5,1.3,date,fontsize=30,transform=ax1.transAxes,ha='center')
        
        ax1.tick_params(axis='y', labelsize=13)
        # ax1.text(-0.095,0.85,'(a)',fontsize=20,transform=ax1.transAxes)
        
        
        ax1a = ax1.twinx()
        
        # ax1a.plot(result_width, 'r.')
        c4 = (0.8,0.8,0)
        ax1a.scatter(ice_tuple[0], ice_tuple[1],c=[c4],marker='o')
        ax1a.scatter(liq_tuple[0], liq_tuple[1],c='r',marker='o')
        
        
        
        # ax1a.set_ylim(0,300)
        
        ax1a.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        
        

    
        sns.distplot(peaks_df_date.query("ICC_ij == 1")[var],label='Ice peaks',ax=ax2, color=c4,bins=30,vertical=True)
        sns.distplot(peaks_df_date.query("LCC_ij == 1")[var],label='Liquid peaks',ax=ax2, color='r',bins=30,vertical=True)
        
    
        # sns.kdeplot(peaks_df_date.query("ref_liq == 0")[var],label='Ice peaks',ax=ax2, color=c4,vertical=True)
        # sns.kdeplot(peaks_df_date.query("ref_liq == 1")[var],label='Liquid peaks',ax=ax2, color='r',vertical=True)
            
        
        ## Set y limits, either:
        # try:
        #     ymin_ice=min(ice_tuple[1])+np.subtract(*np.percentile(ice_tuple[1],[0,75]))
        #     ymax_ice=max(ice_tuple[1])+np.subtract(*np.percentile(ice_tuple[1],[100,25]))
            
        #     ymin_liq=min(liq_tuple[1])+np.subtract(*np.percentile(liq_tuple[1],[0,75]))
        #     ymax_liq=max(liq_tuple[1])+np.subtract(*np.percentile(liq_tuple[1],[100,25]))
                
        #     ymin=min(ymin_ice,ymin_liq)
        #     ymax=max(ymax_ice,ymax_liq)
            
        #     ax1a.set_ylim(ymin,ymax)
        #     ax2.set_ylim(ymin,ymax)
            
        #     print(ymin,ymax)
            
        # except ValueError as e:
        #     print(e)
        
        
        
        ## or:
        ax2.get_shared_y_axes().join(ax1a,ax2)
        
        ax1a.spines['right'].set_color('red')
        ax2.yaxis.label.set_color('red')
        ax1a.tick_params(axis='y',colors='red')

        ax2.ticklabel_format(axis='both',style='sci',scilimits=(-2,2))
        # ax2.set_ylabel('')
        ax2.set_yticks([])
        
        # ax2.set_xlabel('')
        # ax2.set_xticks([])
        
        colour_labels = ['Liquid Peaks (n={})'.format(len(liq_tuple[1])),'Ice Peaks (n={})'.format(len(ice_tuple[1]))]
        patch_colors = [ 'r',c4]
        patches_colours = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(2) ]
        ax2.legend(handles=patches_colours, loc='upper right', fontsize=13)#,borderaxespad=0. )

    
    axes[0,0].text(0.01,1.04,date,fontsize=25,transform=axes[0,0].transAxes,ha='left')

    axes[1,0].set_xlabel('Time (UTC)', fontsize=20)
    
    
    # Axis limits
    axes[0,1].set_ylim(0,8e-4)
    axes[1,1].set_ylim(0,25)
    
    
    plt.tight_layout()
      
    
    
    print('Saving: ',date)
    # plt.savefig(figs_dir+date,dpi=300)
    plt.show()
        
    

# Defining list of dates

file_list = glob.glob(g22_mask_dir+'*')
dates = [file[-12:-4] for file in file_list]
    
dates = ['20210518']

for date in dates:
    print(date)
    make_figure(date)

