# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:31:11 2022

@author: lwh41

Mask intercomparison (ref, G22-Davis, G22-Chch) as func of altitude

"""


import glob
import numpy as np
import os.path
import xarray as xr
from scipy import interpolate
import scipy.signal as signal
from netCDF4 import Dataset
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle
import ds_format as ds

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score,balanced_accuracy_score, classification_report,confusion_matrix

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

# raw_data_dir = 'C://Local//lwh41//DATA//'
# data_dir = 'C://Local//lwh41//DATA//vers_2//'
# # mpl2nc_dir = raw_data_dir+'mpl2nc/'
# # alcf_dir = raw_data_dir+'alcf_output/'
# amps_dir = raw_data_dir+'AMPS_NC/'

# # vdr_dir = raw_data_dir+'vdr/'

# # mpl_pre_processed_nan_dir = data_dir+'mpl_pre_processed_nan/'
# # reference_mask_dir = data_dir+'vdr_reference_mask/'

# # model_classified_dir = data_dir+'model_classified_correct_width/'
# # model_classified_dir_ag = data_dir+'ag_model_classified_correct_width/'

# ## Update 2023-01-08 for recalibrated MPL
# mpl2nc_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_mpl2nc/'
# alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'
# vdr_dir =  'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_vdr/'
# mpl_pre_processed_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_mpl_pre_processed/'
# mpl_pre_processed_nan_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_mpl_pre_processed_nan/'
# reference_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_vdr_reference_mask/'

# g22_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_slw_mask/'
# g22_davis_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_davis_mask/'

# ### EDIT!
# figs_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//figures//'



# Used to get list of lidar dates
old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

g22_davis_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_davis_mask/'


## Update 14.3.2023
ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'
mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask/mask{}/'

figs_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//cloud_occurrence/'


c2 = (0.8,0,0,1)     #SLW
c3 = (0,0,1)    #Warm Liquid
c4 = (0.8,0.8,0)    #Ice/Non-SLW



def process_date(date, TARGET):
    
    
    mod_mask_path = mod_mask_dir.format(TARGET,date[:-2])+str(date)+'.nc'
    
    print(mod_mask_path)
    d_mod = ds.read(mod_mask_path)
    
    alcf_cloud_mask = ds.var(d_mod,'cloud_mask')
    alcf_backscatter = ds.var(d_mod,'backscatter')
    temp = ds.var(d_mod,'temp')
    LCC_ij = ds.var(d_mod,'LCC_ij')
    LCC_j = ds.var(d_mod,'LCC_j')
    SLCC_ij = ds.var(d_mod,'SLCC_ij')
    SLCC_j = ds.var(d_mod,'SLCC_j')
    WLCC_ij = ds.var(d_mod,'WLCC_ij')
    WLCC_j = ds.var(d_mod,'WLCC_j')
    ICC_ij = ds.var(d_mod,'ICC_ij')
    ICC_j = ds.var(d_mod,'ICC_j')
    
    Z_ij = ds.var(d_mod,'Z_ij')
    Z_ij_zeros = np.where(np.isnan(Z_ij),0,Z_ij)
    # Z_j = ds.var(d_mod,'Z_j')
    
    zfull = ds.var(d_mod,'zfull')

    # Open G22-Davis mask
    model_mask_load_ag = np.load(g22_davis_mask_dir+date+'.npz')
    g22davis_liq = model_mask_load_ag['grid']
    g22davis_liq_prof = model_mask_load_ag['profile']

    # g22davis_ice = np.where((alcf_cloud_mask >= g22davis_liq),alcf_cloud_mask-g22davis_liq,alcf_cloud_mask)
    # g22chch_ice  = np.where((alcf_cloud_mask >= Z_ij), alcf_cloud_mask-Z_ij, alcf_cloud_mask)

    g22chch_ice = Z_ij==0
    g22davis_ice = np.where(np.logical_and(~np.isnan(Z_ij), g22davis_liq==0),1,0)
    
    return LCC_ij,ICC_ij,SLCC_ij,WLCC_ij,g22davis_liq,g22davis_ice,Z_ij_zeros,g22chch_ice,zfull
    


def process_scores(ref_mask_all_all,cloud_mask_all):
    # Make cloud altitude accuracy intercomp
    
    scores_f1 = np.full(300,np.nan)
    scores_recall = np.full(300,np.nan)
    scores_prec = np.full(300,np.nan)
    scores_acc = np.full(300,np.nan)
    

    for alt_idx, ref_alt_arr in enumerate(ref_mask_all_all.T):
        cloud_mask_all_alt_arr = cloud_mask_all[:,alt_idx]
        
        if ref_alt_arr.max() != 0:

            scores_f1[alt_idx] = f1_score(ref_alt_arr,cloud_mask_all_alt_arr)
            scores_recall[alt_idx] = recall_score(ref_alt_arr,cloud_mask_all_alt_arr)
            scores_prec[alt_idx] = precision_score(ref_alt_arr,cloud_mask_all_alt_arr)
            scores_acc[alt_idx] = accuracy_score(ref_alt_arr,cloud_mask_all_alt_arr)

    return scores_f1, scores_recall, scores_prec, scores_acc


def plot_fig(ref_clouds, LCC_clouds, SLCC_clouds, davis_clouds, all_scores,zfull, acc_label):
    
    LCC_ij_all, ICC_ij_all, SLCC_ij_all, WLCC_ij_all = ref_clouds
    
    g22chch_liq_all_LCC, g22chch_ice_all_LCC = LCC_clouds
    g22chch_liq_all_SLCC, g22chch_ice_all_SLCC = SLCC_clouds
    
    g22davis_liq_all, g22davis_ice_all = davis_clouds
    
    scores_davis, scores_chch_LCC, scores_chch_SLCC = all_scores
    
    
    # Plot cloud height distributions
    ref_liq_alt = LCC_ij_all.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    ref_ice_alt = ICC_ij_all.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    ref_slw_alt = SLCC_ij_all.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    ref_wliq_alt = WLCC_ij_all.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    
    g22davis_liq_alt = g22davis_liq_all.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    g22davis_ice_alt = g22davis_ice_all.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    
    g22chch_LCC_liq_alt = g22chch_liq_all_LCC.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    g22chch_LCC_ice_alt = g22chch_ice_all_LCC.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    
    g22chch_SLCC_liq_alt = g22chch_liq_all_SLCC.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    g22chch_SLCC_ice_alt = g22chch_ice_all_SLCC.sum(axis=0)*100/np.shape(LCC_ij_all)[0]
    
    
    fig = plt.figure(figsize=(16,6))
    gs_left = gridspec.GridSpec(1, 1)
    gs_mid = gridspec.GridSpec(1, 2,width_ratios=[4,1])
    gs_right = gridspec.GridSpec(1, 2,width_ratios=[4,1])
    gs_far_right = gridspec.GridSpec(1, 2,width_ratios=[4,1])
    
    ax1 = fig.add_subplot(gs_left[0,0])
    ax2 = fig.add_subplot(gs_mid[0,0])
    ax2a = fig.add_subplot(gs_mid[0,1])
    ax3 = fig.add_subplot(gs_right[0,0])
    ax3a = fig.add_subplot(gs_right[0,1])
    ax4 = fig.add_subplot(gs_far_right[0,0])
    ax4a = fig.add_subplot(gs_far_right[0,1])
    
    gs_left.update(left=0.0,right=0.195)
    gs_mid.update(left=0.219,right=0.463)
    gs_right.update(left=0.487,right=0.731)
    gs_far_right.update(left=0.756,right=1.0)
    
    gs_mid.update(wspace=0.15)
    gs_right.update(wspace=0.15)
    gs_far_right.update(wspace=0.15)
    
    # fig, (ax1,ax2,ax2a,ax3,ax3a) = plt.subplots(nrows=1, ncols=5,figsize=(12,6),sharey=True,gridspec_kw={'width_ratios':[4,4,1,4,1]})
    # ax2.yaxis.set_tick_params(labelleft=True)
    # ax3.yaxis.set_tick_params(labelleft=True)
    
    ax1.fill_betweenx(zfull/1000, ref_wliq_alt, label='WLCC',alpha=0.5,color=c3)
    ax1.fill_betweenx(zfull/1000, ref_ice_alt, label='ICC',alpha=0.5,color=c4)
    ax1.fill_betweenx(zfull/1000, ref_slw_alt, label='SLCC',alpha=0.5,color=c2)
    
    # ax1.plot(ref_liq_alt,alcf_range/1000, label='Liquid')
    # ax1.plot(ref_ice_alt,alcf_range/1000, label='Ice')
    # ax1.plot(ref_slw_alt,alcf_range/1000, label='SLW')
    # ax1.plot(ref_wliq_alt,alcf_range/1000, label='Warm Liquid')
    ax1.text(0.5,1.02,'Reference',fontsize=14,transform=ax1.transAxes,ha='center',va='bottom')
    ax1.set_ylabel('Altitude (km)', fontsize=13)
    ax1.set_xlabel('Frequency (%)', fontsize=13)
    ax1.tick_params(axis='both', labelsize=13)
    
    
    
    ax2.fill_betweenx(zfull/1000, g22davis_ice_alt, label='Non-SLCC',alpha=0.5,color=c4)
    ax2.fill_betweenx(zfull/1000, g22davis_liq_alt, label='SLCC',alpha=0.5,color=c2)

    # ax2a = ax2.twiny()
    ax2a.plot(scores_davis,zfull/1000, 'k', label='F1 Score')
    ax2a.set_xlabel(acc_label, fontsize=13)
    ax2a.tick_params(axis='x',labelsize=13)
    ax2a.set_xlim(0,1)
    # ax2.plot(g22davis_liq_alt,alcf_range/1000, label='Liquid')
    # ax2.plot(g22davis_ice_alt,alcf_range/1000, label='Ice')
    ax2.text(0.5,1.02,'G22-Davis',fontsize=14,transform=ax2.transAxes,ha='center',va='bottom')
    ax2.set_xlabel('Frequency (%)', fontsize=13)
    ax2.tick_params(axis='both', labelsize=13)
    
    ax3.fill_betweenx(zfull/1000, g22chch_LCC_ice_alt, label='Non-LCC',alpha=0.5,color=c4)
    ax3.fill_betweenx(zfull/1000, g22chch_LCC_liq_alt, label='LCC',alpha=0.5,color=c2)
    
    # ax3a = ax3.twiny()
    ax3a.plot(scores_chch_LCC,zfull/1000, 'k',label='F1 Score',)
    ax3a.set_xlabel(acc_label, fontsize=13)
    ax3a.tick_params(axis='x',labelsize=13)
    ax3a.set_xlim(0,1)
    # ax3.plot(g22chch_liq_alt,alcf_range/1000, label='Liquid')
    # ax3.plot(g22chch_ice_alt,alcf_range/1000, label='Ice')
    ax3.text(0.5,1.02,'X_LCC_ij',fontsize=14,transform=ax3.transAxes,ha='center',va='bottom')
    ax3.set_xlabel('Frequency (%)', fontsize=13)
    ax3.tick_params(axis='both', labelsize=13)
    
    
    ax4.fill_betweenx(zfull/1000, g22chch_SLCC_ice_alt, label='Non-SLCC',alpha=0.5,color=c4)
    ax4.fill_betweenx(zfull/1000, g22chch_SLCC_liq_alt, label='SLCC',alpha=0.5,color=c2)
    
    # ax3a = ax3.twiny()
    ax4a.plot(scores_chch_SLCC,zfull/1000, 'k',label='F1 Score',)
    ax4a.set_xlabel(acc_label, fontsize=13)
    ax4a.tick_params(axis='x',labelsize=13)
    ax4a.set_xlim(0,1)
    # ax3.plot(g22chch_liq_alt,alcf_range/1000, label='Liquid')
    # ax3.plot(g22chch_ice_alt,alcf_range/1000, label='Ice')
    ax4.text(0.5,1.02,'X_SLCC_ij',fontsize=14,transform=ax4.transAxes,ha='center',va='bottom')
    ax4.set_xlabel('Frequency (%)', fontsize=13)
    ax4.tick_params(axis='both', labelsize=13)
    
    ax1.get_shared_x_axes().join(ax1,ax2,ax3,ax4)
    ax1.get_shared_y_axes().join(ax1,ax2,ax2a,ax3,ax3a,ax4,ax4a)
    
    ax1.set_xlim(0,10)
    ax2.set_xlim(0,10)
    ax3.set_xlim(0,10)
    ax4.set_xlim(0,10)
    
    ax2a.set_yticklabels([])
    ax3a.set_yticklabels([])
    ax4a.set_yticklabels([])
    
    ax2a.tick_params(axis='y',left=False)
    ax3a.tick_params(axis='y',left=False)
    ax4a.tick_params(axis='y',left=False)
    
    ax1.legend()
    handles, labels = ax1.get_legend_handles_labels()
    order = [2,0,1]
    ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=13)
    
    ax2.legend()
    handles, labels = ax2.get_legend_handles_labels()
    order = [1,0]
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=13)
    
    ax3.legend()
    handles, labels = ax3.get_legend_handles_labels()
    order = [1,0]
    ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=13)
    
    ax4.legend()
    handles, labels = ax4.get_legend_handles_labels()
    order = [1,0]
    ax4.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=13)
    # ax2a.legend()
    # ax3a.legend()
    
    ax1.margins(x=0,y=0)
    ax2.margins(x=0,y=0)
    ax3.margins(x=0,y=0)
    ax4.margins(x=0,y=0)
    # ax2a.margins(x=1,y=0)
    # ax3a.margins(x=1,y=0)
    
    ax1.text(0.07,1.02,'a',fontsize=22,transform=ax1.transAxes,ha='left',va='bottom')
    ax2.text(0.07,1.02,'b',fontsize=22,transform=ax2.transAxes,ha='left',va='bottom')
    ax2a.text(0.07,1.02,'c',fontsize=22,transform=ax2a.transAxes,ha='left',va='bottom')
    ax3.text(0.07,1.02,'d',fontsize=22,transform=ax3.transAxes,ha='left',va='bottom')
    ax3a.text(0.07,1.02,'e',fontsize=22,transform=ax3a.transAxes,ha='left',va='bottom')
    ax4.text(0.07,1.02,'f',fontsize=22,transform=ax4.transAxes,ha='left',va='bottom')
    ax4a.text(0.07,1.02,'g',fontsize=22,transform=ax4a.transAxes,ha='left',va='bottom')


    
    # plt.tight_layout(pad=1.3)
    
    # plt.savefig(figs_dir+'alt_intercomp_ij_'+acc_label,dpi=300,bbox_inches='tight')
    
    
def get_chch_clouds(okdates, TARGET):
    
    LCC_ij_all = np.empty((0,300))
    ICC_ij_all = np.empty((0,300))
    SLCC_ij_all = np.empty((0,300))
    WLCC_ij_all = np.empty((0,300))
    g22davis_liq_all = np.empty((0,300))
    g22davis_ice_all = np.empty((0,300))
    g22chch_liq_all = np.empty((0,300))
    g22chch_ice_all = np.empty((0,300))
    
    for date in okdates:
        LCC_ij,ICC_ij,SLCC_ij,WLCC_ij,g22davis_liq,g22davis_ice,Z_ij_zeros,g22chch_ice,zfull = process_date(date, TARGET)
        
        LCC_ij_all = np.concatenate((LCC_ij_all, LCC_ij))
        ICC_ij_all = np.concatenate((ICC_ij_all, ICC_ij))
        SLCC_ij_all = np.concatenate((SLCC_ij_all, SLCC_ij))
        WLCC_ij_all = np.concatenate((WLCC_ij_all, WLCC_ij))
        
        g22davis_liq_all = np.concatenate((g22davis_liq_all, g22davis_liq))
        g22davis_ice_all = np.concatenate((g22davis_ice_all, g22davis_ice))
        
        g22chch_liq_all = np.concatenate((g22chch_liq_all, Z_ij_zeros))
        g22chch_ice_all = np.concatenate((g22chch_ice_all, g22chch_ice))  
        
    chch_clouds = LCC_ij_all, ICC_ij_all, SLCC_ij_all, WLCC_ij_all, g22chch_liq_all, g22chch_ice_all
    davis_clouds = g22davis_liq_all, g22davis_ice_all
    
    return chch_clouds, davis_clouds, zfull
    


dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]
okdates = []

for date in dates:
    if os.path.exists(g22_davis_mask_dir+date+'.npz'):
        okdates+=[date]

# okdates = ['20210518']

chch_clouds_LCC, davis_clouds, zfull = get_chch_clouds(okdates, 'LCC_j')

LCC_ij_all, ICC_ij_all, SLCC_ij_all, WLCC_ij_all, g22chch_liq_all_LCC, g22chch_ice_all_LCC = chch_clouds_LCC
g22davis_liq_all, g22davis_ice_all = davis_clouds

mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask_nohp/mask{}/'
chch_clouds_SLCC, _, _ = get_chch_clouds(okdates, 'SLCC_j')
_, _, _, _, g22chch_liq_all_SLCC, g22chch_ice_all_SLCC = chch_clouds_SLCC

scores_davis = process_scores(SLCC_ij_all,g22davis_liq_all)
scores_chch_LCC = process_scores(LCC_ij_all,g22chch_liq_all_LCC)
scores_chch_SLCC = process_scores(SLCC_ij_all,g22chch_liq_all_SLCC)

all_scores_f1 = scores_davis[0], scores_chch_LCC[0], scores_chch_SLCC[0]
all_scores_recall = scores_davis[1], scores_chch_LCC[1], scores_chch_SLCC[1]
all_scores_prec = scores_davis[2], scores_chch_LCC[2], scores_chch_SLCC[2]
all_scores_acc = scores_davis[3], scores_chch_LCC[3], scores_chch_SLCC[3]

acc_label = 'F1'
all_scores = all_scores_f1

ref_clouds = LCC_ij_all, ICC_ij_all, SLCC_ij_all, WLCC_ij_all
LCC_clouds =  g22chch_liq_all_LCC, g22chch_ice_all_LCC
SLCC_clouds =  g22chch_liq_all_SLCC, g22chch_ice_all_SLCC

plot_fig(ref_clouds, LCC_clouds, SLCC_clouds, davis_clouds, all_scores,zfull, acc_label)