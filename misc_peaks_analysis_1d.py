# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:06:26 2023

@author: lwh41
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
import scipy

import matplotlib.patches as mpatches
import matplotlib.lines as mlines


from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'

peaks_fpath = peaks_dir+'peaks_df_{}.csv'.format('2e-5')

peaks_df = pd.read_csv(peaks_fpath, index_col=[0])

peaks_df['peak_alt'] = peaks_df['peak_alt']/1000
# peaks_df['peak_backscatter'] = peaks_df['peak_backscatter']*1000

peaks_df = peaks_df.query('amps_flag==1')

# peaks_df = peaks_df.query('peak_order==0')


# peaks_df = peaks_df.query('peak_alt<=0.5')
# peaks_df = peaks_df.query('peak_alt>=0.5 and peak_alt<=1')
# peaks_df = peaks_df.query('peak_alt>=1 and peak_alt<=1.5')
# peaks_df = peaks_df.query('peak_alt>=1.5 and peak_alt<=2')
# peaks_df = peaks_df.query('peak_alt>=2')


# print(str(len(peaks_df))+' peaks with altitude greater than 1 km')

# Using _ij
LCC_peaks = peaks_df.query('LCC_ij == 1')
noLCC_peaks = peaks_df.query('LCC_ij == 0')

# SLCC_peaks = peaks_df.query('SLCC_ij == 1')
SLCC_peaks = peaks_df.query('SLCC_ij == 1')
noSLCC_peaks = peaks_df.query('SLCC_ij == 0')

# WLCC_peaks = peaks_df.query('WLCC_ij == 1')
WLCC_peaks = peaks_df.query('WLCC_ij == 1')
noWLCC_peaks = peaks_df.query('WLCC_ij == 0')

ICC_peaks = peaks_df.query('ICC_ij == 1')
noICC_peaks = peaks_df.query('ICC_ij == 0')

# print(str(len(SLCC_peaks))+' of which are SLCC')
# print(str(len(WLCC_peaks))+' of which are WLCC')

# Using _j
# LCC_peaks = peaks_df.query('LCC_j == 1')
# noLCC_peaks = peaks_df.query('LCC_j == 0')

# SLCC_peaks = peaks_df.query('SLCC_j == 1')
# noSLCC_peaks = peaks_df.query('SLCC_j == 0')

# WLCC_peaks = peaks_df.query('WLCC_j == 1')
# noWLCC_peaks = peaks_df.query('WLCC_j == 0')

# ICC_peaks = peaks_df.query('ICC_j == 1')
# noICC_peaks = peaks_df.query('ICC_j == 0')

var = 'peak_backscatter'

q2_SLCC = np.nanmedian(SLCC_peaks[var])
q2_WLCC = np.nanmedian(WLCC_peaks[var])
q2_ICC = np.nanmedian(ICC_peaks[var])
q2_LCC = np.nanmedian(LCC_peaks[var])

# # diff = abs(q2_SLCC-q2_WLCC)


# fig, (ax1) = plt.subplots(1,1)

# c1 = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
# c2 = (1.0, 0.4980392156862745, 0.054901960784313725)
# c3 = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
# c4 = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)

c1 = (0.03137254901960784,0.18823529411764706,0.4196078431372549,1)
c2 = (0.8,0,0,1)    
c3 = (0,0,1)
c4 = (0.8,0.8,0)

# ax1 = sns.distplot(LCC_peaks[var],ax=ax1,bins=100,label='LCC',color=c2)
# ax1 = sns.distplot(ICC_peaks[var],ax=ax1,bins=100,label='ICC',color=c4)
# ax1 = sns.distplot(SLCC_peaks[var],ax=ax1,bins=100,label='SLCC',color=c1)
# ax1 = sns.distplot(WLCC_peaks[var],ax=ax1,bins=100,label='WLCC',color=c3)

# ax1.legend()
# ax1.set_xlim(0,0.3)

# plt.axvline(q2_LCC,c=c2,linestyle='--')
# plt.axvline(q2_ICC,c=c4,linestyle='--')
# plt.axvline(q2_SLCC,c=c1,linestyle='--')
# plt.axvline(q2_WLCC,c=c3,linestyle='--')


# peaks_df = peaks_df_all[peaks_df_all['date'].isin(allowed_dates_pd)]



# vars_to_drop = ['date','peak_time_idx', 'peak_alt_idx',
#                             'ref_slw', 'ref_wliq','ref_liq','ref_ice',
#                             'peak_vdr','peak_vdr_mask','peak_vdr_mean',
#                             # 'lr',
#                             # 'lr_grad_abs',
#                             'num_peaks','peak_order',
#                             # 'model_result','model_result_ag'
#                             # 'peak_temp'
#                             ]

#### ------------------------------------------------------------

## Plotting peak properties

# LCC vs ICC

fig, axes = plt.subplots(3,3, figsize=(18,15))

var_list = ['peak_backscatter', 'peak_width', 'peak_width_height',
        'peak_prominence', 'peak_alt', 'peak_temp', 'num_peaks','peak_order',]

units_list = ['$\\beta$ Height (m$^{-1}$sr$^{-1}$)', 'Width', ' Width Height (m$^{-1}$sr$^{-1}$)',
        'Prominence (m$^{-1}$sr$^{-1}$)', 'Altitude (km)', 'Temperature ($\\degree$C)', 'n Peaks','Peak Order',]

letters='abcdefgh'

for i,var in enumerate(var_list):
    # print(var)
    
    ax = axes.flatten()[i]
    
    ax.tick_params(axis='both',which='major',labelsize=16)
    ax.tick_params(axis='both',which='minor',labelsize=16)
    
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,5))
    
    if var in ['num_peaks','peak_order']:
        sns.distplot(LCC_peaks[var],label='LCC',ax=ax, color=c2,bins=30,hist=False)
        sns.distplot(ICC_peaks[var],label='ICC',ax=ax, color=c4,bins=30,hist=False)    
    else:
        sns.distplot(LCC_peaks[var],label='LCC',ax=ax, color=c2,bins=30)
        sns.distplot(ICC_peaks[var],label='ICC',ax=ax, color=c4,bins=30)    
    
        ax.axvline(np.nanmedian(LCC_peaks[var]),c=c2,linestyle='--', label='LCC Median')
        ax.axvline(np.nanmedian(ICC_peaks[var]),c=c4,linestyle='--', label='ICC Median')
    
    unit = units_list[i]
    ax.set_xlabel(unit,fontsize=20,labelpad=2)
    # ax.set_title(var,fontsize=20)
    
    ax.ticklabel_format(axis='both',style='sci',scilimits=(-2,2))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
        
    
    if i in [0,3,6]:
        ax.set_ylabel('Density', fontsize=15)
    else:
        ax.set_ylabel(' ')
            
    ax.text(-0.13,1,'({})'.format(letters[i]),fontsize=23,transform=ax.transAxes,ha='right',va='top',weight='semibold')
        
 
# print(axes[0,2].get_xlim())
axes[0,0].set_xlim(-8.789452652380182e-05, 0.0009970278532549434) #height
# axes[0,0].set_xlim(-8.789452652380182e-02, 0.9970278532549434) #height

axes[0,2].set_xlim(-8.789452652380182e-05, 0.0006970278532549434) #width height
axes[1,0].set_xlim(-8.789452652380182e-05, 0.0009970278532549434) #prom

axes[0,1].set_xlim(-0.6,20.2) # width
# axes[1,1].set_xlim(-1,9) # altitude

axes[2,2].axis('off')

colour_labels = ['LCC','ICC']
patch_colors = [ c2,c4]
patches_colours = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(2) ]

linestyle_labels = ['LCC Median', 'ICC Median']
patches_lines = [mlines.Line2D([],[],color=patch_colors[i],linestyle='dashed', label=linestyle_labels[i]) for i in range(2)] 
patches = [patches_colours[0]] + [patches_lines[0]] + [patches_colours[1]] + [patches_lines[1]]
# patches = patches_colours + patches_lines
axes[2,2].legend(handles=patches, loc='center', fontsize=20)#,borderaxespad=0. )
    
# plt.text(0.2,0.65,'# Liquid peaks   =    '+ str(len(LCC_peaks)),transform=axes[2,2].transAxes,fontsize=17)
# plt.text(0.2,0.55,'# Ice peaks = '+ str(len(ICC_peaks)),transform=axes[2,2].transAxes,fontsize=17)
plt.tight_layout()

output_fpath = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//feature_dist/kde_all_update'
plt.savefig(output_fpath,dpi=300)

#### ------------------------------------------------------------

##### SLCC vs WLCC

# fig, axes = plt.subplots(3,3, figsize=(18,15))

# var_list = ['peak_backscatter', 'peak_width', 'peak_width_height',
#         'peak_prominence', 'peak_alt', 'peak_temp', 'num_peaks','peak_order',]

# units_list = ['$\\beta$ Height (m$^{-1}$sr$^{-1}$)', 'Width', ' Width Height (m$^{-1}$sr$^{-1}$)',
#         'Prominence (m$^{-1}$sr$^{-1}$)', 'Altitude (km)', 'Temperature ($\\degree$C)', 'n Peaks','Peak Order',]

# letters='abcdefgh'

# for i,var in enumerate(var_list):
#     # print(var)
    
#     ax = axes.flatten()[i]
    
#     ax.tick_params(axis='both',which='major',labelsize=16)
#     ax.tick_params(axis='both',which='minor',labelsize=16)
    
#     ax.ticklabel_format(axis='x',style='sci',scilimits=(0,5))
    
#     if var in ['num_peaks','peak_order']:
#         sns.distplot(SLCC_peaks[var],label='SLCC',ax=ax, color=c2,bins=30,hist=False)
#         sns.distplot(WLCC_peaks[var],label='WLCC',ax=ax, color=c3,bins=30,hist=False)    
#     else:
#         sns.distplot(SLCC_peaks[var],label='SLCC',ax=ax, color=c2,bins=30)
#         sns.distplot(WLCC_peaks[var],label='WLCC',ax=ax, color=c3,bins=30)    
    
#         ax.axvline(np.nanmedian(SLCC_peaks[var]),c=c2,linestyle='--', label='SLCC Median')
#         ax.axvline(np.nanmedian(WLCC_peaks[var]),c=c3,linestyle='--', label='WLCC Median')
    
#     unit = units_list[i]
#     ax.set_xlabel(unit,fontsize=20,labelpad=2)
#     # ax.set_title(var,fontsize=20)
    
#     ax.ticklabel_format(axis='both',style='sci',scilimits=(-2,2))
    
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
        
    
#     if i in [0,3,6]:
#         ax.set_ylabel('Density', fontsize=15)
#     else:
#         ax.set_ylabel(' ')
            
#     ax.text(-0.13,1,'({})'.format(letters[i]),fontsize=23,transform=ax.transAxes,ha='right',va='top',weight='semibold')
        
 
# # print(axes[0,2].get_xlim())
# axes[0,0].set_xlim(-8.789452652380182e-05, 0.0009970278532549434) #height
# # axes[0,0].set_xlim(-8.789452652380182e-02, 0.9970278532549434) #height

# axes[0,2].set_xlim(-8.789452652380182e-05, 0.0006970278532549434) #width height
# axes[1,0].set_xlim(-8.789452652380182e-05, 0.0009970278532549434) #prom

# axes[0,1].set_xlim(-0.2,8.2) # width
# axes[1,1].set_xlim(-1,9) # altitude

# axes[2,2].axis('off')

# colour_labels = ['SLCC','WLCC']
# patch_colors = [ c2,c3]
# patches_colours = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(2) ]

# linestyle_labels = ['SLCC Median', 'WLCC Median']
# patches_lines = [mlines.Line2D([],[],color=patch_colors[i],linestyle='dashed', label=linestyle_labels[i]) for i in range(2)] 
# patches = [patches_colours[0]] + [patches_lines[0]] + [patches_colours[1]] + [patches_lines[1]]
# # patches = patches_colours + patches_lines
# axes[2,2].legend(handles=patches, loc='center', fontsize=20)#,borderaxespad=0. )
    
# # plt.text(0.2,0.65,'# Liquid peaks   =    '+ str(len(LCC_peaks)),transform=axes[2,2].transAxes,fontsize=17)
# # plt.text(0.2,0.55,'# Ice peaks = '+ str(len(ICC_peaks)),transform=axes[2,2].transAxes,fontsize=17)
# plt.tight_layout()

# output_fpath = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//feature_dist/kde_all_update'
# # plt.savefig(output_fpath,dpi=300)



#### ------------------------------------------------------------

### NEW - median of WLCC vs SLCC peak heights as func of alt

# peaks_df = pd.read_csv(peaks_fpath, index_col=[0])

# peaks_df['peak_alt'] = peaks_df['peak_alt']/1000
# # peaks_df['peak_backscatter'] = peaks_df['peak_backscatter']*1000

# peaks_df = peaks_df.query('amps_flag==1')

# q2_SLCC_all = np.empty(len(np.unique(peaks_df['peak_alt'])))
# q2_WLCC_all = np.empty(len(np.unique(peaks_df['peak_alt'])))
# q2_ICC_all = np.empty(len(np.unique(peaks_df['peak_alt'])))
# q2_LCC_all = np.empty(len(np.unique(peaks_df['peak_alt'])))

# print(np.unique(peaks_df['peak_alt']))

# for idx, alt in enumerate(np.unique(peaks_df['peak_alt'])):
#     # print(alt)
#     peaks_df_alt = peaks_df.query('peak_alt=={}'.format(alt))
    
#     SLCC_peaks = peaks_df_alt.query('SLCC_ij == 1')
#     WLCC_peaks = peaks_df_alt.query('WLCC_ij == 1')
#     ICC_peaks = peaks_df_alt.query('ICC_ij == 1')
#     LCC_peaks = peaks_df_alt.query('LCC_ij == 1')

#     var = 'peak_backscatter'
#     q2_SLCC = np.nanmedian(SLCC_peaks[var])
#     q2_WLCC = np.nanmedian(WLCC_peaks[var])
#     q2_ICC = np.nanmedian(ICC_peaks[var])
#     q2_LCC = np.nanmedian(LCC_peaks[var])
    
#     q2_SLCC_all[idx]= q2_SLCC
#     q2_WLCC_all[idx]= q2_WLCC
#     q2_ICC_all[idx]= q2_ICC
#     q2_LCC_all[idx]= q2_LCC

# plt.scatter(q2_SLCC_all,np.unique(peaks_df['peak_alt']),color=c1)
# plt.scatter(q2_WLCC_all,np.unique(peaks_df['peak_alt']),color=c2)
    