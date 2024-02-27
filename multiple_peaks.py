# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:36:01 2022

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

## Open Peak dataset
# peaks_df = pd.read_csv(data_dir+'MPL_peak_df_v2.csv', index_col=[0])

# peaks_df['peak_backscatter'] = peaks_df['peak_backscatter']*1000

# Does no LCC separation (so looks at all peaks)
prim_LCC_cond = 'True'
sec_LCC_cond = 'True'

# As per AG:
# prim_LCC_cond = 'LCC_j==1'
# sec_LCC_cond = 'peak_width < 4 and peak_width_height > 40e-6 and peak_prominence > 60e-6'

# Experimental conditions
# prim_LCC_cond = 'SLCC_j==1'
# sec_LCC_cond = 'SLCC_j==1'

# peaks_df = peaks_df.query('peak_alt>=1.5 and peak_alt<=2')
# peaks_df = peaks_df.query('peak_alt<=0.5')
# peaks_df = peaks_df.query('peak_alt>=2')

# primary_peaks = peaks_df.query('peak_order == 0 and '+prim_LCC_cond)
# secondary_peaks = peaks_df.query('peak_order > 0 and '+sec_LCC_cond)
primary_peaks = peaks_df.query('peak_order == 0 and '+prim_LCC_cond)
secondary_peaks = peaks_df.query('peak_order > 0 and '+sec_LCC_cond)

# secondary_peaks['peak_backscatter'] = secondary_peaks['peak_backscatter']*3

q2_prim = np.median(primary_peaks['peak_backscatter'])
q2_sec = np.median(secondary_peaks['peak_backscatter'])

q2_prim_mad = scipy.stats.median_abs_deviation(primary_peaks['peak_backscatter'])
q2_sec_mad = scipy.stats.median_abs_deviation(secondary_peaks['peak_backscatter'])

diff = q2_prim-q2_sec

print('Primary peaks: Median = {:.3e} with {:.3e} MAD'.format(q2_prim,q2_prim_mad))
print('Secondary peaks: Median = {:.3e} with {:.3e} MAD'.format(q2_sec,q2_sec_mad))

print('Difference = {:.3e}'.format(diff))

c1 = (0.03137254901960784,0.18823529411764706,0.4196078431372549,1)
c2 = (0.8,0,0,1)    
c3 = (0,0,1)
c4 = (0.8,0.8,0)

from matplotlib import cm
c5 = cm.get_cmap('RdYlBu_r', 10)(8)
c6 = cm.get_cmap('RdYlBu_r', 10)(2)


# primary_peaks['peak_backscatter'] = primary_peaks['peak_backscatter']/1000
# secondary_peaks['peak_backscatter'] = secondary_peaks['peak_backscatter']/1000


unit='$\\beta$ Height (m$^{-1}$sr$^{-1}$)'

# letters='abcdefgh'

# print(var)
fig, (ax) = plt.subplots(1,1,figsize=(10,6))

ax.tick_params(axis='both',which='major',labelsize=16)
ax.tick_params(axis='both',which='minor',labelsize=16)

ax.ticklabel_format(axis='x',style='sci',scilimits=(0,5))

# kde_kws={'clip':(0,5e-4)}
kde_kws={'cut':0}
# kde_kws={}

bin_min = np.min(primary_peaks['peak_backscatter'])
bin_max = np.max(primary_peaks['peak_backscatter'])
# hist_kws = {'binrange':(bin_min,bin_max)}
hist_kws = {}
line_kws = dict(linewidth=0)

# sns.distplot(primary_peaks['peak_backscatter'],label='Primary',ax=ax,bins=30,color=c5,kde=True,hist=True,kde_kws=kde_kws, hist_kws=hist_kws)
# sns.distplot(secondary_peaks['peak_backscatter'],label='Secondary',ax=ax,bins=30,color=c6,kde=True,hist=True,kde_kws=kde_kws, hist_kws=hist_kws)    

sns.histplot(primary_peaks['peak_backscatter'],label='Primary',ax=ax,bins=30,
             color=c5,kde=True,alpha=0.4, binrange=(bin_min,bin_max), 
             stat='proportion', linewidth=0.1, kde_kws=dict(cut=0))
sns.histplot(secondary_peaks['peak_backscatter'],label='Secondary',ax=ax,bins=30,
             color=c6,kde=True,alpha=0.4, binrange=(bin_min,bin_max), 
             stat='proportion', linewidth=0.1, kde_kws=dict(cut=0))   

ax.axvline(q2_prim,c=c5, label='SLCC Median',linestyle='--')
ax.axvline(q2_sec,c=c6, label='WLCC Median',linestyle='--')

# ax.axvline(q2_prim-q2_prim_mad,c=c2,linestyle='--')
# ax.axvline(q2_sec-q2_sec_mad,c=c4,linestyle='--')

# ax.axvline(q2_prim+q2_prim_mad,c=c2,linestyle='--')
# ax.axvline(q2_sec+q2_sec_mad,c=c4,linestyle='--')

# unit = units_list[i]
ax.set_xlabel(unit,fontsize=20,labelpad=2)
# ax.set_title(var,fontsize=20)

ax.ticklabel_format(axis='both',style='sci',scilimits=(-2,2))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
    
ax.set_ylabel('Density', fontsize=15)

n_prim = len(primary_peaks['peak_backscatter'])
n_sec = len(secondary_peaks['peak_backscatter'])

# colour_labels = ['Primary (n={})'.format(n_prim),'Secondary (n={})'.format(n_sec)]
colour_labels = ['Primary (n=35,686)','Secondary (n=6,705)']

patch_colors = [ c5,c6]
patches_colours = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(2) ]

# linestyle_labels = ['$Q2_{\mathrm{prime}}$ = %.1e'%(q2_prim), '$Q2_{\mathrm{second}}$ = %.1e'%(q2_sec)]
linestyle_labels = [r'$Q2_{\mathrm{prime}}=1.1\times10^{-4}\mathrm{m}^{-1}\mathrm{sr}^{-1}$',
                    r'$Q2_{\mathrm{second}}=4.4\times10^{-5}\mathrm{m}^{-1}\mathrm{sr}^{-1}$']

patches_lines = [mlines.Line2D([],[],color=patch_colors[i],linestyle='--', label=linestyle_labels[i]) for i in range(2)] 
patches = [patches_colours[0]] + [patches_lines[0]] + [patches_colours[1]] + [patches_lines[1]]
# patches = patches_colours + patches_lines
ax.legend(handles=patches, loc='upper right', fontsize=16)#,borderaxespad=0. )  
# ax.text(-0.13,1,'{}'.format(letters[i]),fontsize=25,transform=ax.transAxes,ha='right',va='top',weight='semibold')
        
ax.set_xlim(-3e-5,9e-4)

plt.tight_layout()

output_fpath = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//feature_dist/prim_sec_pks_beta_update'
plt.savefig(output_fpath,dpi=300)
