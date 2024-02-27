# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 23:28:50 2023

@author: lwh41
"""

import os
import sys
import logging
import traceback
import copy
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import date2num, AutoDateFormatter, AutoDateLocator
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm, LogNorm
import matplotlib.lines as mlines
import aquarius_time as aq
import ds_format as ds
# from alcf import misc, algorithms
# from alcf.lidars import LIDARS

 


COLORS = [
	'#0084c8',
	'#dc0000',
	'#009100',
	'#ffc022',
	'#ba00ff',
    'k',
]

LINESTYLE = 'solid'

ALPHA = 1

VARIABLES = [
	'time',
	'time_bnds',
	'backscatter',
	'backscatter_sd',
	'backscatter_mol',
	'backscatter_sd_full',
	'backscatter_sd_hist',
	'zfull',
	'lr',
	'cl',
	'clt',
	'n',
	'cloud_mask',
	'backscatter_hist',
	'backscatter_full',
	'altitude',
	'clw',
	'cli',
]


def plot_bar(dd, 
             subcolumn=0,
             labels=None):
    bars = []
    for i, d in enumerate(dd):
        cl = d['cl'][:,subcolumn] \
			if len(d['cl'].shape) == 2 \
			else d['cl']
        clt = d['clt'][subcolumn] \
			if len(d['clt'].shape) == 1 \
			else d['clt']
        n = d['n']
        bars.append(clt)
    xlab = (labels if labels is not None else np.arange(len(bars)))
    print(xlab)
    print(bars)
    plt.bar(xlab, bars)
        
        

def plot_legend(*args, theme='light', **kwargs):
    legend = plt.legend(*args, fontsize=10, **kwargs)
    f = legend.get_frame()
    f.set_facecolor('k' if theme == 'light' else 'white')
    f.set_linewidth(0)
    f.set_alpha(0.1 if theme == 'light' else 0.9)



def plot_cloud_occurrence(dd,
                          colors=COLORS,
                          linestyle=LINESTYLE,
                          alpha=ALPHA,
                          lw=None,
                          labels=None,
                          subcolumn=0,
                          xlim=[0., 75.],
                          zlim=[0., 10500],
                          **kwargs):
    for i, d in enumerate(dd):
        zfull = d['zfull']
        cl = d['cl'][:,subcolumn] \
			if len(d['cl'].shape) == 2 \
			else d['cl']
        clt = d['clt'][subcolumn] \
			if len(d['clt'].shape) == 1 \
			else d['clt']
        n = d['n']
        # s = np.sqrt(((cl)*(100-cl))/n)
        # st = np.sqrt(((clt)*(100-clt))/n)
        label = (labels[i] if labels is not None else '')
        # label += ' | CF: %d%% +- %f%% | %d profiles' % (clt,st,n)
        label += ' | CF: %d%%' % (clt)
        # label += ' | CF: %d%% | %3d days' % (clt,n/288)

        plt.plot(cl, 1e-3*zfull,
 			color=(colors[i] if type(colors) is list else colors),
			linestyle=(linestyle[i] if type(linestyle) is list else linestyle),
			lw=lw,
			label=label,
            alpha=(alpha[i] if type(alpha) is list else alpha),
		)
        
#         plt.plot(cl+s, 1e-3*zfull,
#  			color=(colors[i] if type(colors) is list else colors),
# 			linestyle='dashed',
# 			lw=lw,
# # 			label=label,
#             # alpha=(alpha[i] if type(alpha) is list else alpha),
# 		)
#         plt.plot(cl-s, 1e-3*zfull,
#  			color=(colors[i] if type(colors) is list else colors),
# 			linestyle='dashed',
# 			lw=lw,
# # 			label=label,
#             # alpha=(alpha[i] if type(alpha) is list else alpha),
# 		)
        
    
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(zlim[0]*1e-3, zlim[1]*1e-3)
    plt.xlabel('Cloud occurrence (%)')
    plt.ylabel('Altitude (km)')
    
    # plt.xscale('log')
    
    if labels is not None:
        plot_legend()
        
        
        

def plot_cloud_occurrence_diff(dd,
	colors=COLORS,
	linestyle=LINESTYLE,
    alpha=ALPHA,
	lw=None,
	labels=None,
	subcolumn=0,
	xlim=[0., 75.],
	zlim=[0., 15000],
	**kwargs
):
# 	for i, d in enumerate(dd):
    d = dd[0]
    zfull = d['zfull']
    cld = d['cl'][:,subcolumn] \
		if len(d['cl'].shape) == 2 \
		else d['cl']
    cltd = d['clt'][subcolumn] \
		if len(d['clt'].shape) == 1 \
		else d['clt']
        
    e = dd[1]
    zfull = e['zfull']
    cle = e['cl'][:,subcolumn] \
		if len(e['cl'].shape) == 2 \
		else e['cl']
    clte = e['clt'][subcolumn] \
		if len(e['clt'].shape) == 1 \
		else e['clt']
# 		n = d['n']
# 		label = (labels[i] if labels is not None else '')
# 		label += ' | CF: %d%% | %d days' % (clt,n/288)

    cl = np.where(np.logical_or(cle==0,cld==0),np.nan,cld/cle)
        
    print(cl)
    plt.plot(cl, 1e-3*zfull,
 		 # color=(colors[i] if type(colors) is list else colors),
# 		linestyle=(linestyle[i] if type(linestyle) is list else linestyle),
# 		lw=lw,
# 		label=label,
        # alpha=(alpha[i] if type(alpha) is list else alpha),
		)
    plt.vlines(1, zlim[0]*1e-3, zlim[1]*1e-3, linestyle='dashed')
    
    # plt.xlim(xlim[0], xlim[1])
    plt.ylim(zlim[0]*1e-3, zlim[1]*1e-3)
    # plt.xlabel('Cloud occurrence (%)')
    plt.ylabel('Altitude (km)')
    
# 	pcen=np.percentile(cl, 50, interpolation="nearest")
# 	i_near=abs(cl-pcen).argmin()
# 	plt.hlines(i_near,0,75)
# 	i_near=abs(cl-pcen).argmin()


# 	if labels is not None:
# 		plot_legend()
        
        


## Tangaroa

chm15k_dir = 'S://atmos_disk//Luke_W_480//TAN2302_CHM15k//alcf/'
mpl_dir = 'S://atmos_disk//Luke_W_480//TAN2302_MPL//alcf/'
# mrr_dir = 'S://atmos_disk//Luke_W_480//TAN2302_MRR//mrr2c/'
output = 'S://atmos_disk//Luke_W_480//TAN2302_MPL//figs//' 

chm15k_path = chm15k_dir+'stats/all.nc'
mpl_path = mpl_dir+'stats/all.nc'
dd = [ds.read(chm15k_path, VARIABLES), ds.read(mpl_path, VARIABLES)]
labels=['CHM15k', 'MiniMPL']
# dd = [ds.read(chm15k_path, VARIABLES)]
# fig = plt.figure(figsize=[5, 5])
# plot_cloud_occurrence(dd, labels=['CHM15k'])
title = 'TAN2302 (2023-01-14 to 2023-02-22)'
# plt.title(title)

# fig = plt.figure(figsize=[5, 5])
# plot_cloud_occurrence_diff(dd, labels=['CHM15k', 'MiniMPL'])




## Christchurch
 
chch05_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202105//stats/all.nc'
chch06_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202106//stats/all.nc'
chch07_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202107//stats/all.nc'
chch08_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202108//stats/all.nc'
chch09_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202109//stats/all.nc'
chch10_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202110//stats/all.nc'
chch11_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202111//stats/all.nc'
chch12_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202112//stats/all.nc'
chch01_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//alcf202201//stats/all.nc'

chch05 = ds.read(chch05_path, VARIABLES)
chch06 = ds.read(chch06_path, VARIABLES)
chch07 = ds.read(chch07_path, VARIABLES)
chch08 = ds.read(chch08_path, VARIABLES)
chch09 = ds.read(chch09_path, VARIABLES)
chch10 = ds.read(chch10_path, VARIABLES)
chch11 = ds.read(chch11_path, VARIABLES)
chch12 = ds.read(chch12_path, VARIABLES)
chch01 = ds.read(chch01_path, VARIABLES)

# dd = [chch05, 
#         chch06,
#         chch07, 
#         chch08, 
#         chch09, 
#         chch10, 
#         chch11,
#         chch12,
#         chch01
#       ]
# labels=['Christchurch (2021-05)',
#         'Christchurch (2021-06)',
#         'Christchurch (2021-07)',
#         'Christchurch (2021-08)',
#         'Christchurch (2021-09)',
#         'Christchurch (2021-10)',
#         'Christchurch (2021-11)',
#         'Christchurch (2021-12)',
#         'Christchurch (2022-01)',
#         ]
# title = 'Christchurch Monthly'

chch_mjj_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A_month//alcf_mjj//stats/all.nc'
chch_aso_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A_month//alcf_aso//stats/all.nc'
chch_ndj_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A_month//alcf_ndj//stats/all.nc'

chch_mjj = ds.read(chch_mjj_path, VARIABLES)
chch_aso = ds.read(chch_aso_path, VARIABLES)
chch_ndj = ds.read(chch_ndj_path, VARIABLES)


chch_all_path= 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A_all//alcf_all//stats/all.nc'
chch_all = ds.read(chch_all_path, VARIABLES)

# dd = [chch_all,
#       chch_mjj, 
#         chch_aso,
#         chch_ndj, 

#       ]
# labels=['Christchurch (all)',
#         'Christchurch MJJ  ',
#         'Christchurch ASO',
#         'Christchurch NDJ ',
#         ]
# title = 'Christchurch Seasonally'
# dd = [chch_all
#       ]

# labels = ['Christchurch',
#     ]

## GoSouth + Invercargill
 
gs_dir = 'S://atmos_disk//Luke_W_480//GoSouth_MPL//alcf/'
inv_all_dir = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf/'
mrr_dir = 'S://atmos_disk//Luke_W_480//SoSouth_MRR//mrr2c/'
# output = 'S://atmos_disk//Luke_W_480//GoSouth_MPL//figs//'

inv05_path = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf202205//stats/all.nc'
inv06_path = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf202206//stats/all.nc'
inv07_path = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf202207//stats/all.nc'
inv08_path = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf202208//stats/all.nc'
inv09_path = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf202209//stats/all.nc'
inv10_path = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf202210//stats/all.nc'
inv11_path = 'S://atmos_disk//Luke_W_480//Invercargill_MPL//alcf202211//stats/all.nc'

inv05 = ds.read(inv05_path, VARIABLES)
inv06 = ds.read(inv06_path, VARIABLES)
inv07 = ds.read(inv07_path, VARIABLES)
inv08 = ds.read(inv08_path, VARIABLES)
inv09 = ds.read(inv09_path, VARIABLES)
inv10 = ds.read(inv10_path, VARIABLES)
inv11 = ds.read(inv11_path, VARIABLES)


## GoSouth
gs_path = gs_dir+'stats/all.nc'
gs = ds.read(gs_path, VARIABLES)

inv_all_path = inv_all_dir+'stats/all.nc'
inv_all = ds.read(inv_all_path, VARIABLES)

# dd = [gs]
# labels=['GoSouth (2022-11-03 to 2022-11-30)']

# dd = [inv_all, gs]
# labels=['Invercargill', 'GoSouth    ']
# title = 'Invercargill & GoSouth'


# dd = [inv05, 
#         inv06,
#         inv07, 
#         inv08, 
#         inv09, 
#         inv10, 
#         # inv11
#       ]
# labels=['Invercargill (2022-05)',
#         'Invercargill (2022-06)',
#         'Invercargill (2022-07)',
#         'Invercargill (2022-08)',
#         'Invercargill (2022-09)',
#         'Invercargill (2022-10)',
#         # 'Invercargill (2022-11)'
#         ]
# title = 'Invercargill Monthly'


xlim=[0., 65.]
zlim=[0., 10000]

fig = plt.figure(figsize=[6,6])
plot_cloud_occurrence(dd, labels=labels, xlim=xlim,	zlim=zlim)
plt.title(title)

# fig = plt.figure(figsize=[6,6])
# plot_bar(dd, labels=labels)

## Savefig
plt.savefig(output+'tan_cloud_occurrence', bbox_inches='tight', dpi=300)
# 