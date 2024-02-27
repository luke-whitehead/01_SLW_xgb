# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:00:18 2023

@author: lwh41
"""


import glob
import numpy as np
import os.path
import xarray as xr
from scipy import interpolate
from netCDF4 import Dataset
from datetime import datetime

import pandas as pd
import pickle
import time

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import ds_format as ds

from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score,balanced_accuracy_score, classification_report,confusion_matrix

# Used to get list of lidar dates
# old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

# g22_davis_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_davis_mask/'

# ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
# peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'
# mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask/mask{}/'
# mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//multiclass_model_mask_nohp/mask{}/'

# output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//{}_model_fig/figs{}/'
# output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//multiclass_model_fig/figs{}/'

# mpl2nc_dir = 'S://atmos_disk//Luke_W_480//TAN2302_MPL//mpl2nc/'
# alcf_dir = 'S://atmos_disk//Luke_W_480//TAN2302_MPL//alcf//lidar/'

ceil_mask_dir = 'S://atmos_disk//Luke_W_480//TAN2302_CHM15k//alcf/lidar/'
ref_mask_dir = 'S://atmos_disk//Luke_W_480//TAN2302_MPL//vdr_processed/'
mrr_dir = 'S://atmos_disk//Luke_W_480//TAN2302_MRR//mrr2c/' 

output_dir = 'S://atmos_disk//Luke_W_480//TAN2302_MPL//figs//panel_figs/'


cmap1 = colors.ListedColormap([[0,0,0,0],[0.03137254901960784,0.18823529411764706,0.4196078431372549,1]])
cmap2 = colors.ListedColormap([[0,0,0,0],[0.8,0,0,1]])
cmap3 = colors.ListedColormap([[0,0,0,0],[0,0,1,1]])
cmap4 = colors.ListedColormap([[0,0,0,0],[0.8,0.8,0,1]])
cmap5 = colors.ListedColormap([[0,0,0,0],[0.7,0.7,0.7,1]])

c1 = (0.03137254901960784,0.18823529411764706,0.4196078431372549,1)
c2 = (0.8,0,0,1)    
c3 = (0,0,1)
c4 = (0.8,0.8,0)
c5 = (0.7,0.7,0.7)

""" Psuedo Code:

For each date:
    Open ALCF output, extract backscatter + cloud mask
    Open reference mask, extract grid + profile view
    Open model classified mask, extract grid + profile view

"""

def make_figure(date):
    print('Plotting: ',date)
    
    try:
        ref_mask_path = ref_mask_dir+str(date)+'.nc'
        d_ref = ds.read(ref_mask_path)
            
        mpl_cloud_mask = ds.var(d_ref,'cloud_mask')
        mpl_beta = ds.var(d_ref,'backscatter')
        mpl_vdr = ds.var(d_ref,'vdr')
        mpl_vdr_cm = ds.var(d_ref,'vdr_cm')
        mpl_LCC_ij = ds.var(d_ref,'LCC_ij')
        mpl_LCC_j = ds.var(d_ref,'LCC_j')
        mpl_ICC_ij = ds.var(d_ref,'ICC_ij')
        mpl_ICC_j = ds.var(d_ref,'ICC_j')
    except OSError as e:
        print(e)
        mpl_cloud_mask = np.full((288,300),np.nan)
        mpl_beta = np.full((288,300),np.nan)
        mpl_vdr = np.full((288,300),np.nan)
        mpl_vdr_cm = np.full((288,300),np.nan)
        mpl_LCC_ij = np.full((288,300),np.nan)
        mpl_LCC_j = np.full((288,300),np.nan)
        mpl_ICC_ij = np.full((288,300),np.nan)
        mpl_ICC_j = np.full((288,300),np.nan)
        
    try:    
        alcf_format = '{}-{}-{}T000000.nc'.format(date[:4],date[4:6],date[6:8])
        ceil_mask_path = ceil_mask_dir+alcf_format
        d_ceil = ds.read(ceil_mask_path)
            
        ceil_beta = ds.var(d_ceil,'backscatter') 
    except OSError as e:
        print(e)
        ceil_beta = np.full((288,300),np.nan) 
        
    
    try:    
        mrr_path = mrr_dir+str(date)[4:]+'.nc'
        d_mrr = ds.read(mrr_path,['attenuated_radar_reflectivity', 'rain_rate'])
        
        mrr_reflect = ds.var(d_mrr,'attenuated_radar_reflectivity') 
        mrr_rr = ds.var(d_mrr,'rain_rate') 
    except OSError as e:
        print(e)
        mrr_reflect = np.full((288,300),np.nan)  
        mrr_rr = np.full((288,300),np.nan) 
    
    ### Plot figure
    
    gridspec_kw = {
        'height_ratios':[3,3,3,3,2],
        # 'hspace':0.2
        }
    
    fig, axes = plt.subplots(5,figsize=(15,20), gridspec_kw=gridspec_kw)
    (ax1, ax2, ax3, ax4, ax5) = axes
    
    ## Fig 1 - CHM15k backscatter
    
    Im1=ax1.imshow(ceil_beta[:,:100].T,
       	extent=(0, 288, 0, 5),
       	aspect='auto',
       	origin='lower',
        interpolation='None',
        cmap='RdYlBu_r',
       	norm=colors.LogNorm(vmin=1e-8,vmax=1e-4),
        
    )
    
    cax1 = ax1.inset_axes([1.02,0,0.02,1],transform=ax1.transAxes)
    
    fig.colorbar(Im1,
    		extend='neither',
            cax=cax1,
        ).set_label(label='Att. vol. \nbackscattering coef. \n( m$^{-1}$sr$^{-1}$)', size=15, labelpad=6)
    
    ax1.set_ylabel('Altitude (km)', fontsize=15)
    ax1.set_xticklabels([])
    ax1.set_xticks(np.arange(36,288,36))
    ax1.xaxis.set_minor_locator(MultipleLocator(12))
    cax1.tick_params(labelsize=13)
    
    
    
    ## Fig 2 - MPL backscatter
    
    Im2=ax2.imshow(mpl_beta[:,:100].T,
       	extent=(0, 288, 0, 5),
       	aspect='auto',
       	origin='lower',
        interpolation='None',
        cmap='RdYlBu_r',
       	norm=colors.LogNorm(vmin=1e-8,vmax=1e-4),
        
    )
    
    cax2 = ax2.inset_axes([1.02,0,0.02,1],transform=ax2.transAxes)
    
    fig.colorbar(Im2,
    		extend='neither',
            cax=cax2,
        ).set_label(label='Att. vol. \nbackscattering coef. \n( m$^{-1}$sr$^{-1}$)', size=15, labelpad=6)
    
    ax2.set_ylabel('Altitude (km)', fontsize=15)
    ax2.set_xticklabels([])
    ax2.set_xticks(np.arange(36,288,36))
    ax2.xaxis.set_minor_locator(MultipleLocator(12))
    cax2.tick_params(labelsize=13)
    

    
    ## Fig 3 - VDR raw data
        
    # Im3=ax3.imshow(mpl_vdr[:].T,
    Im3=ax3.imshow(mpl_vdr[:,:100].T,             
       	extent=(0, 288, 0, 5),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap='RdYlBu'
    )
    
    ax3.set_xticklabels([])
    ax3.set_xticks(np.arange(36,288,36))
    ax3.xaxis.set_minor_locator(MultipleLocator(12))
    ax3.set_ylabel('Altitude (km)', fontsize=15)
    
    cax3 = ax3.inset_axes([1.02,0,0.02,1],transform=ax3.transAxes)
    
    fig.colorbar(Im3,
    		extend='neither',
            cax=cax3,
        ).set_label(label='VDR', size=15, labelpad=6)
    
    ax3.tick_params(labelsize=13)
    

    
    Im4=ax4.imshow(mrr_reflect[:,:-3].T,             
       	extent=(0, 288, 0, 3.36),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        # vmax=25,
        interpolation='None',
        cmap='Blues',
    )
    
    ax4.set_xticklabels([])
    ax4.set_xticks(np.arange(36,288,36))
    ax4.xaxis.set_minor_locator(MultipleLocator(12))
    ax4.set_ylabel('Altitude (km)', fontsize=15)
    
    cax4 = ax4.inset_axes([1.02,0,0.02,1],transform=ax4.transAxes)
    
    fig.colorbar(Im4,
    		extend='neither',
            cax=cax4,
        ).set_label(label='Att. radar relfectivity\n(dBZ)', size=15, labelpad=6)
    
    ax4.tick_params(labelsize=13)
    
    
    ax5.fill_between(np.linspace(0,288, len(mrr_rr[:,0])),mrr_rr[:,0], color='r')
    ax5.margins(x=0,)
    ax5.set_ylim(0,1.7)
    ax5.set_ylabel('Precipitation\n(mm/hr)',fontsize=15)
        
    
    ax5.set_xticklabels([])
    ax5.set_xticks(np.arange(36,288,36))
    ax5.xaxis.set_minor_locator(MultipleLocator(12))   
    
    # ax8.tick_params(axis='both',which='major',labelsize=20)
    # ax8.tick_params(axis='both',which='minor',labelsize=20)
   
    ax5.set_xlabel('Time (UTC)', fontsize=20)
    
    ax5.set_xticklabels([ '03:00',
                            '06:00',
                            '09:00',
                            '12:00',
                            '15:00',
                            '18:00',
                            '21:00'], fontsize=15)
    
    
      
    
    # ax1.set_title('Backscatter',fontsize=20)
    # ax2.set_title('Volume Depolarization Ratio (VDR)',fontsize=20)
    # ax3.set_title('VDR Reference Cloud Mask',fontsize=20,transform=ax3.transAxes,x=0.5,y=0.88)
    # ax5.set_title('G22-Davis Cloud Mask',fontsize=20,transform=ax5.transAxes,x=0.5,y=0.88)
    
    ddate = date[:4]+'-'+date[4:6]+'-'+date[6:]
    ax1.text(0.5,1.2,ddate,fontsize=30,transform=ax1.transAxes,ha='center')
    
    letters='abcdefgh'
    
    instruments = ['CHM15k','MPL','MPL','MRR-2','MRR-2']
    
    props = dict(facecolor='white', alpha=0.9, linewidth=0)
    
    for i,ax in enumerate(axes):
        ax.tick_params(axis='y', labelsize=13)
        ax.text(-0.05,1,'{}'.format(letters[i]),fontsize=25,transform=ax.transAxes,ha='right',va='top',weight='semibold')
        ax.text(0.03,0.9,'{}'.format(instruments[i]),fontsize=22,transform=ax.transAxes,ha='left',va='top',weight='semibold',bbox=props)
       

    

    

    plt.tight_layout()

    output_path = output_dir+str(date)+'_5km.png'
    
    print('Saving to',output_path)
    plt.savefig(output_path,dpi=300)
    plt.show()
        
    



good_dates = ['20230131',]# '20230115']#,'20230201','20230202']


def main():

    dates = ['2023'+file[-7:-3]for file in glob.glob(mrr_dir+'*')] # MRR dates
    # dates = [file[-20:-10].replace('-','') for file in glob.glob(ceil_mask_dir+'*')] # CHM15k dates
    unprocessed_dates = []
    for date in dates:
        if not os.path.exists(output_dir+str(date)+'.png'):
            unprocessed_dates+=[date]

    
    for date in good_dates[:]:
        print('Processing', date)
        start_time = time.time()
        try:
            make_figure(date)
        except FileNotFoundError:
            print(date,'not found')
        print("--- %s seconds ---" % (time.time() - start_time))
        
    return dates
        

if __name__ == "__main__":
    main()
