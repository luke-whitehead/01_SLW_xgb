# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:18:22 2023

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
old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

g22_davis_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_davis_mask/'

ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'
# mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask/mask{}/'
mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//multiclass_model_mask_nohp/mask{}/'

# output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//{}_model_fig/figs{}/'
output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//multiclass_model_fig/figs{}/'



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

def make_figure(date,TARGET):
    print('Plotting: ',date)
    
    ref_mask_path = ref_mask_dir.format(date[:-2])+str(date)+'.nc'
    d_ref = ds.read(ref_mask_path, ['vdr','vdr_cm'])
    
    vdr = ds.var(d_ref,'vdr')
    vdr_cm = ds.var(d_ref,'vdr_cm')
    
    print('Opening peak dataset')
    peaks_fpath = peaks_dir+'peaks_df_{}.csv'.format('2e-5')
    peaks_df = pd.read_csv(peaks_fpath, index_col=[0]).query('date=={}'.format(date))  
    
    # Optional step to tidy appearance of vdr data
    # vdr = np.where((vdr>1), np.nan, vdr)
    # vdr = np.where((vdr<0), np.nan, vdr)
    
    # mod_mask_path = mod_mask_dir.format(TARGET,date[:-2])+str(date)+'.nc'
    mod_mask_path = mod_mask_dir.format(date[:-2])+str(date)+'.nc'

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
    
    # ref_
    
    Z_ij = ds.var(d_mod,'Z_ij')
    Z_j_ICC = ds.var(d_mod,'Z_j_ICC')
    Z_j_SLCC = ds.var(d_mod,'Z_j_SLCC')
    Z_j_WLCC = ds.var(d_mod,'Z_j_WLCC')

    # Open G22-Davis mask
    model_mask_load_ag = np.load(g22_davis_mask_dir+date+'.npz')
    g22davis_liq = model_mask_load_ag['grid']
    model_mask_profiles_ag = model_mask_load_ag['profile']
    
    g22davis_ice = np.where((alcf_cloud_mask >= g22davis_liq),alcf_cloud_mask-g22davis_liq,alcf_cloud_mask)
    g22chch_ice  = np.where((alcf_cloud_mask >= Z_ij), alcf_cloud_mask-Z_ij, alcf_cloud_mask)


    ## Selecting only time steps with peaks
    unique_ts = np.unique(peaks_df['peak_time_idx'])
    print(str(len(unique_ts))+' time steps with peaks detected')
    
    
    # chch_bal_acc = balanced_accuracy_score(SLCC_j,np.where(np.isnan(Z_j),0,Z_j))
    
    chch_bal_acc = balanced_accuracy_score(SLCC_j,(Z_j_SLCC==1))
    print('chch_bal_acc',str(chch_bal_acc))
    chch_bal_acc_pkts = balanced_accuracy_score(SLCC_j[unique_ts],(Z_j_SLCC[unique_ts]==1))
    print('chch_bal_acc_pkts',str(chch_bal_acc_pkts))
        
    SLCC_j_ts = np.full(np.shape(LCC_j),np.nan)
    SLCC_j_ts[unique_ts]=SLCC_j[unique_ts]

    
    davis_bal_acc = balanced_accuracy_score(SLCC_j,model_mask_profiles_ag)
    print('davis_bal_acc',str(davis_bal_acc))
    davis_bal_acc_pkts = balanced_accuracy_score(SLCC_j[unique_ts],model_mask_profiles_ag[unique_ts])
    print('davis_bal_acc_pkts',str(davis_bal_acc_pkts))
    
    # chch_recall= recall_score(SLCC_j,Z_j)
    # chch_prec= precision_score(SLCC_j,Z_j)
    # chch_f1= f1_score(SLCC_j,Z_j)
    # chch_acc = accuracy_score(SLCC_j,Z_j)
    # chch_bal_acc = balanced_accuracy_score(SLCC_j,Z_j)
    
    ### Plot figure
    
    gridspec_kw = {
        'height_ratios':[8,8,8,1,8,1,8,1],
        # 'hspace':0.2
        }
    
    fig, axes = plt.subplots(8,figsize=(15,25), gridspec_kw=gridspec_kw)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7,ax8) = axes
    
    ## Fig 1 - ALCF backscatter
    
    Im1=ax1.imshow(alcf_backscatter[:].T,
       	extent=(0, 288, 0, 15),
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
    
    CS=ax1.contour(temp[:].T,
                levels=[-60,-50,-40,-30,-20,-10,0],
                colors='k',
                alpha=0.6,
                linestyles='dashed',
                extent=(0, 288, 0, 15),
                )
    
    ax1.clabel(CS,inline=True,fontsize=12,fmt='%.0f')
    
    ## Fig 2 - VDR raw data
        
    # Im2=ax2.imshow(vdr[:].T,
    Im2=ax2.imshow(vdr_cm[:].T,             
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap='RdYlBu'
    )
    
    ax2.set_xticklabels([])
    ax2.set_xticks(np.arange(36,288,36))
    ax2.xaxis.set_minor_locator(MultipleLocator(12))
    ax2.set_ylabel('Altitude (km)', fontsize=15)
    
    cax2 = ax2.inset_axes([1.02,0,0.02,1],transform=ax2.transAxes)
    
    fig.colorbar(Im2,
    		extend='neither',
            cax=cax2,
        ).set_label(label='VDR', size=15, labelpad=6)
    
    cax2.tick_params(labelsize=13)
    
    
    

    ## Fig 3 - Reference (VDR) mask
    
    Im3=ax3.imshow(ICC_ij.T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap4
    )
    
    Im3=ax3.imshow(SLCC_ij.T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap2
    )
    
    Im3=ax3.imshow(WLCC_ij.T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap3
    )
    
    ax3.set_xticklabels([])
    ax3.set_xticks(np.arange(36,288,36))
    ax3.xaxis.set_minor_locator(MultipleLocator(12))
    ax3.set_ylabel('Altitude (km)', fontsize=15)
    
    ax3.contour(temp[:].T,
                levels=[-40,0],
                colors='k',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, 15),
                )
    
    
    colour_labels = ['SLCC', 'WLCC', 'ICC']
    patch_colors = [ c2, c3, c4]
    patches_colours = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(3) ]
    
    patch_linestyles = ['dashed', 'solid']
    linestyle_labels = ['-40 \N{DEGREE SIGN}C', '0 \N{DEGREE SIGN}C']
    patches_lines = [mlines.Line2D([],[],color='k',linestyle=patch_linestyles[i], label=linestyle_labels[i]) for i in range(2)] 
    patches = patches_colours + patches_lines
    ax3.legend(handles=patches, bbox_to_anchor=(1.025, 0.3), loc=6, fontsize=13)#,borderaxespad=0. )
    
    
        
    ## Fig 4 - Reference mask profile view
    
    # Used to show idcs of 
    # Im4=ax4.imshow([~np.isnan(SLCC_j_ts)],
    #    	extent=(0, 288, 0, 1),
    #    	aspect='auto',
    #    	origin='lower',
    #     # vmin=0,
    #     # vmax=1,
    #     interpolation='None',
    #     cmap=cmap4
    # )
    Im4=ax4.imshow([SLCC_j],
       	extent=(0, 288, 0, 1),
       	aspect='auto',
       	origin='lower',
        # vmin=0,
        # vmax=1,
        interpolation='None',
        cmap=cmap2
    )


    # unique_ts
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_xticks(np.arange(36,288,36))
    ax4.set_yticks([])
    ax4.xaxis.set_minor_locator(MultipleLocator(12))    
    
    colour_labels = ['SLCC']
    patch_colors = [ c2]
    patches = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(1) ]
    
    ax4.legend(handles=patches, bbox_to_anchor=(1.025, 0.5), loc=6, fontsize=13)
    
    
    

    ## Fig 5 - G22-Davis

    # Im5=ax5.imshow((~np.isnan(Z_ij))[:].T,
    Im5=ax5.imshow(g22davis_ice[:].T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap4
    )
    
    Im5=ax5.imshow(g22davis_liq.T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap2
    )
    
    
    ax5.set_xticklabels([])
    ax5.set_xticks(np.arange(36,288,36))
    ax5.xaxis.set_minor_locator(MultipleLocator(12))
    ax5.set_ylabel('Altitude (km)', fontsize=15)
    

    
    colour_labels = ['SLCC','Non-SLCC Peaks']
    patch_colors = [ c2,c4]
    patches_two = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(2) ]
    
    ax5.legend(handles=patches_two, bbox_to_anchor=(1.025, 0.1), loc=6, fontsize=13)#,borderaxespad=0. )
    
    
        
    ## Fig 6 - G22-Davis profile view
    
    Im6=ax6.imshow([np.any(((Z_j_ICC==1),(Z_j_SLCC==1),(Z_j_WLCC==1)),axis=0)],
       	extent=(0, 288, 0, 1),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap5
    )
    
    Im6=ax6.imshow([model_mask_profiles_ag],
       	extent=(0, 288, 0, 1),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap2
    )
    
    

    
    colour_labels = ['SLCC', 'Non-SLCC Peaks']
    patch_colors = [ c2, c5]
    patches_one = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(2) ]
    
    ax6.legend(handles=patches_one, bbox_to_anchor=(1.025, 0.5), loc=6, fontsize=13)
    
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])
    ax6.set_xticks(np.arange(36,288,36))
    ax6.set_yticks([])
    ax6.xaxis.set_minor_locator(MultipleLocator(12))   
    

    

    

    ## Fig 7 - G22-Chch mask
    #Used print(plt.cm.datad["Blues"]) to get RGB values of Blues, RdYlGn cmaps

    #ICC
    Im7=ax7.imshow(g22chch_ice[:].T,
    # Im7=ax7.imshow((np.isnan(Z_ij))[:].T,
    # Im7=ax7.imshow((Z_ij==0)[:].T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap4
    )
    
    #SLCC
    Im7=ax7.imshow((Z_ij==1).T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap2
    )
    
    #WLCC
    Im7=ax7.imshow((Z_ij==2).T,
       	extent=(0, 288, 0, 15),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap3
    )
    
    ax7.set_xticklabels([])
    ax7.set_xticks(np.arange(36,288,36))
    ax7.xaxis.set_minor_locator(MultipleLocator(12))
    ax7.set_ylabel('Altitude (km)', fontsize=15)
    
    colour_labels = ['SLCC','WLCC','ICC']
    patch_colors = [ c2,c3, c4]
    patches_two = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(3) ]
    
    

    ax7.legend(handles=patches_two, bbox_to_anchor=(1.025, 0.1), loc=6, fontsize=13)#,borderaxespad=0. )
    
    
        
    ## Fig 8 - G22-Chch profile view
        
    #ICC
    Im8=ax8.imshow([np.any(((Z_j_ICC==1),(Z_j_SLCC==1),(Z_j_WLCC==1)),axis=0)],
       	extent=(0, 288, 0, 1),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap5
    )
    
    #SLCC
    Im8=ax8.imshow([Z_j_SLCC==1],
       	extent=(0, 288, 0, 1),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap2
    )
    


    ax8.legend(handles=patches_one, bbox_to_anchor=(1.025, 0.5), loc=6, fontsize=13)
    
    ax8.set_xticklabels([])
    ax8.set_yticklabels([])
    ax8.set_xticks(np.arange(36,288,36))
    ax8.set_yticks([])
    ax8.xaxis.set_minor_locator(MultipleLocator(12))   
    
    # ax8.tick_params(axis='both',which='major',labelsize=20)
    # ax8.tick_params(axis='both',which='minor',labelsize=20)
   
    ax8.set_xlabel('Time (UTC)', fontsize=20)
    
    ax8.set_xticklabels([ '03',
                            '06',
                            '09',
                            '12',
                            '15',
                            '18',
                            '21'], fontsize=15)
    
    
    
    
    ax1.xaxis.tick_top()
    ax1.set_xticklabels([ '03',
                           '06',
                           '09',
                           '12',
                           '15',
                           '18',
                           '21'], fontsize=15)
    
    
    # ax1.set_title('Backscatter',fontsize=20)
    # ax2.set_title('Volume Depolarization Ratio (VDR)',fontsize=20)
    ax3.set_title('VDR Reference Cloud Mask',fontsize=20,transform=ax3.transAxes,x=0.5,y=0.88)
    ax5.set_title('G22-Davis Cloud Mask',fontsize=20,transform=ax5.transAxes,x=0.5,y=0.88)
    ax7.set_title('G22-Christchurch Cloud Mask',fontsize=20,transform=ax7.transAxes,x=0.5,y=0.88)
    
    ddate = date[:4]+'-'+date[4:6]+'-'+date[6:]
    # ax1.text(0.5,1.2,'Z_'+TARGET+'   '+ddate,fontsize=30,transform=ax1.transAxes,ha='center')
    ax1.text(0.5,1.2,ddate,fontsize=30,transform=ax1.transAxes,ha='center')
    
    letters='abcdefgh'
    
    for i,ax in enumerate(axes):
        ax.tick_params(axis='y', labelsize=13)
        ax.text(-0.05,1,'({})'.format(letters[i]),fontsize=25,transform=ax.transAxes,ha='right',va='top',weight='semibold')
        
    # davis_recall= recall_score(SLCC_j,model_mask_profiles_ag)
    # davis_prec= precision_score(SLCC_j,model_mask_profiles_ag)
    # davis_f1= f1_score(SLCC_j,model_mask_profiles_ag)
    # davis_acc = accuracy_score(SLCC_j,model_mask_profiles_ag)
    # davis_bal_acc = balanced_accuracy_score(SLCC_j,model_mask_profiles_ag)
    
    # chch_recall= recall_score(SLCC_j,Z_j)
    # chch_prec= precision_score(SLCC_j,Z_j)
    # chch_f1= f1_score(SLCC_j,Z_j)
    # chch_acc = accuracy_score(SLCC_j,Z_j)
    # chch_bal_acc = balanced_accuracy_score(SLCC_j,Z_j)
    
    

    
    
    # day_acc_liq = recall_score(ref_liq_prof,model_mask_profiles)
    
    # ax5.text(1.03,0.8,'Recall={:.2f}'.format(davis_recall),fontsize=20,transform=ax5.transAxes)
    # ax5.text(1.03,0.7,'Precision={:.2f}'.format(davis_prec),fontsize=20,transform=ax5.transAxes)
    # ax5.text(1.03,0.6,'F1={:.2f}'.format(davis_f1),fontsize=20,transform=ax5.transAxes)
    # ax5.text(1.03,0.5,'Acc={:.2f}'.format(davis_acc),fontsize=20,transform=ax5.transAxes)
    ax5.text(1.03,0.8,'SLCC\nBalanced\nAccuracy:',fontsize=17,transform=ax5.transAxes)
    ax5.text(1.03,0.6,'All profiles\n   ={:.2f}'.format(davis_bal_acc),fontsize=18,transform=ax5.transAxes)
    ax5.text(1.03,0.3,'Profiles with \npeaks\n   ={:.2f}'.format(davis_bal_acc_pkts),fontsize=18,transform=ax5.transAxes)
   
    # ax7.text(1.03,0.8,'Recall={:.2f}'.format(chch_recall),fontsize=20,transform=ax7.transAxes)
    # ax7.text(1.03,0.7,'Precision={:.2f}'.format(chch_prec),fontsize=20,transform=ax7.transAxes)
    # ax7.text(1.03,0.6,'F1={:.2f}'.format(chch_f1),fontsize=20,transform=ax7.transAxes)
    # ax7.text(1.03,0.5,'Acc={:.2f}'.format(chch_acc),fontsize=20,transform=ax7.transAxes)
    ax7.text(1.03,0.8,'SLCC\nBalanced\nAccuracy:',fontsize=17,transform=ax7.transAxes)
    ax7.text(1.03,0.6,'All profiles\n   ={:.2f}'.format(chch_bal_acc),fontsize=18,transform=ax7.transAxes)
    ax7.text(1.03,0.3,'Profiles with \npeaks\n   ={:.2f}'.format(chch_bal_acc_pkts),fontsize=18,transform=ax7.transAxes)
    # ax7.text(1.03,0.8,'Balanced\nAccuracy={:.2f}'.format(chch_bal_acc),fontsize=20,transform=ax7.transAxes)
    # ax7.text(1.03,0.5,'Balanced\nAccuracy={:.2f}'.format(chch_bal_acc_pkts),fontsize=19,transform=ax7.transAxes)
    
    plt.tight_layout()
    
    # return peaks_df
    
    
    # save_fig_path = 'figures/panel_fig_ag_compare/'
    output_path = output_dir.format(date[:-2])+str(date)+'.png'
    
    print('Saving to',output_path)
    # plt.savefig(output_path,dpi=300)
    # plt.show()
        
    



good_dates = ['20210518', '20210605']#,'20210521','20210526','20210528','20210530']


def main():
    TARGET='SLCC_ij'
    
    dates = [file[-20:-10].replace('-','') for file in glob.glob(old_alcf_dir+'*')]
    unprocessed_dates = []
    for date in dates:
        if not os.path.exists(output_dir.format(date[:-2])+str(date)+'.png'):
            unprocessed_dates+=[date]

    
    for date in good_dates:
        print('Processing', date)
        start_time = time.time()
        try:
            make_figure(date,TARGET)
        except FileNotFoundError:
            print(date,'not found')
        print("--- %s seconds ---" % (time.time() - start_time))
        

if __name__ == "__main__":
    main()
