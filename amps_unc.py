# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:16:02 2022

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
import ds_format as ds
import time

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

data_dir = 'C://Local//lwh41//DATA//'
alcf_dir = data_dir+'alcf_output/'
amps_dir = data_dir+'AMPS_NC/'
reference_mask_dir = data_dir+'vdr_reference_mask_v2/'
mpl_pre_processed_dir = data_dir+'mpl_pre_processed/'
mpl2nc_dir = data_dir+'mpl2nc/'


# save_fig_path = 'figures/amps_unc/'

# Used to get list of lidar dates
old_alcf_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_alcf_output//lidar/'

g22_davis_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_g22_davis_mask/'

ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'
# mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//{}_model_mask/mask{}/'
mod_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//C//multiclass_model_mask_nohp/mask{}/'

# output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//{}_model_fig/figs{}/'
output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//multiclass_model_fig/figs{}/'

save_fig_path = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//amps_unc/'

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

def make_figure(date,loaded_temps_all, y_max = 10):
    print('Plotting: ',date)
        
    
    loaded_temps_mean = np.mean(loaded_temps_all, axis=0)
    # loaded_temps_std = np.std(loaded_temps_all, axis=0)

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
    

    
    z = int(300*y_max/15)



    ### Plot figure
    
    gridspec_kw = {
        'height_ratios':[1,1],
        # 'hspace':0.2
        }
    
    fig, axes = plt.subplots(2,figsize=(15,12), gridspec_kw=gridspec_kw)
    (ax1, ax2) = axes
    
    ## Fig 1 - ALCF backscatter
    
    Im1=ax1.imshow(alcf_backscatter[:,:z].T,
       	extent=(0, 288, 0, y_max),
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
    
    ax1.set_ylabel('Altitude (km)', fontsize=20)
    ax1.set_xticklabels([])
    ax1.set_xticks(np.arange(36,288,36))
    ax1.xaxis.set_minor_locator(MultipleLocator(12))
    cax1.tick_params(labelsize=14)
        
    # CS=ax1.contour(temp[:,:100].T,
    #             levels=[-60,-50,-40,-30,-20,-10,0],
    #             colors='k',
    #             alpha=0.6,
    #             linestyles='solid',
    #             extent=(0, 288, 0, 5),
    #             )
    
    # ax1.clabel(CS,inline=True,fontsize=12,fmt='%.0f')
        
    ax1.contour(loaded_temps_all[0][:,:z].T,
                levels=[-40,0],
                colors='grey',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )
    
    ax1.contour(loaded_temps_all[1][:,:z].T,
                levels=[-40,0],
                colors='grey',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )
    
    ax1.contour(loaded_temps_all[2][:,:z].T,
                levels=[-40,0],
                colors='grey',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )
    
    ax1.contour(loaded_temps_mean[:,:z].T,
                levels=[-40,0],
                colors='k',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )





    ## Fig 3 - Reference (VDR) mask
    
    Im2=ax2.imshow(ICC_ij[:,:z].T,
       	extent=(0, 288, 0, y_max),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap4
    )
    
    Im2=ax2.imshow(SLCC_ij[:,:z].T,
       	extent=(0, 288, 0, y_max),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap2
    )
    
    Im2=ax2.imshow(WLCC_ij[:,:z].T,
       	extent=(0, 288, 0, y_max),
       	aspect='auto',
       	origin='lower',
        vmin=0,
        vmax=1,
        interpolation='None',
        cmap=cmap3
    )
    
    ax2.set_xticklabels([])
    ax2.set_xticks(np.arange(36,288,36))
    ax2.xaxis.set_minor_locator(MultipleLocator(12))
    ax2.set_ylabel('Altitude (km)', fontsize=20)
    
    # ax2.contour(temp[:,:z].T,
    #             levels=[-38,0],
    #             colors='r',
    #             linestyles=['dashed','solid'],
    #             extent=(0, 288, 0, y_max),
    #             linewidths=2,
    #             )
    
    
    colour_labels = ['SLCC', 'WLCC', 'ICC']
    patch_colors = [ c2, c3, c4]
    patches_colours = [ mpatches.Patch(color=patch_colors[i], label=colour_labels[i] ) for i in range(3) ]
    
    # patch_linestyles = ['dashed', 'solid']
    # linestyle_labels = ['-38 \N{DEGREE SIGN}C', '0 \N{DEGREE SIGN}C']
    # patches_lines = [mlines.Line2D([],[],color='k',linestyle=patch_linestyles[i], label=linestyle_labels[i]) for i in range(2)] 
    patches = patches_colours #+ patches_lines
    ax2.legend(handles=patches, bbox_to_anchor=(1.025, 0.5), loc=6, fontsize=15)#,borderaxespad=0. )

    
    ax2.contour(loaded_temps_all[0][:,:z].T,
                levels=[-40,0],
                colors='grey',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )
    
    ax2.contour(loaded_temps_all[1][:,:z].T,
                levels=[-40,0],
                colors='grey',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )
    
    ax2.contour(loaded_temps_all[2][:,:z].T,
                levels=[-40,0],
                colors='grey',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )
    
    ax2.contour(loaded_temps_mean[:,:z].T,
                levels=[-40,0],
                colors='k',
                linestyles=['dashed','solid'],
                extent=(0, 288, 0, y_max),
                linewidths=2,
                )
    
        
    
   
    ax2.set_xlabel('Time (UTC)', fontsize=20)
    
    ax2.set_xticklabels([ '03',
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
    
    
    ddate = date[:4]+'-'+date[4:6]+'-'+date[6:]
    ax1.text(0.5,1.2,ddate,fontsize=30,transform=ax1.transAxes,ha='center')
    
    letters='abcdefgh'
    
    for i,ax in enumerate(axes):
        ax.tick_params(axis='y', labelsize=15)
        ax.text(-0.05,1,'{}'.format(letters[i]),fontsize=25,transform=ax.transAxes,ha='right',va='top',weight='semibold')

    
    
    plt.tight_layout()
    
    print('Saving', date)
    # plt.savefig(save_fig_path+date, dpi=300)
    
        
    

def process(date,site_temp,iso_lvls):
        
    ### Opens MPL ALCF output

    alcf_format = '{}-{}-{}T000000.nc'.format(date[:4],date[4:6],date[6:8])
    alcf_ds = Dataset(alcf_dir+alcf_format)
    alcf_cloud_mask = alcf_ds.variables['cloud_mask']
    alcf_backscatter = alcf_ds['backscatter']
    zfull = alcf_ds.variables['zfull'][:] #size 300 
    
    
    ## Interpolating AMPS temperature grid over MPL grid
    x = np.linspace(0,287, 8)
    y = np.linspace(0,999,21)
    X,Y = np.meshgrid(x,y)
    
    Z = site_temp
    
    x2 = np.linspace(0,287,288)
    y2 = np.linspace(0,999,1000)
    X2, Y2 = np.meshgrid(x2, y2)
        
    f = interpolate.interp2d(y, x, Z, kind='cubic')
    Z2 = f(y2, x2)
    
    site_temp_interpol = np.flip(Z2,axis=1)
    
    ## Interpolation of AMPS iso Levels
    xx = np.linspace(0,999,21)
    yy = iso_lvls.values
    
    xx_new = np.linspace(0,999,1000)
    
    f = interpolate.interp1d(xx, yy)
    pres_levels_interpolate = np.flip(f(xx_new))
    
    def pres2alt(pres):
        """
        pressure: units hPa
        alt: units m
        """
        
        scale_height = 7000# use tsurface
        sfc_press = 1013
        
        alt = -scale_height*np.log(pres/sfc_press)
        return alt#.replace([np.inf, -np.inf], np.nan).dropna()
    
    alt_levels = pres2alt(pres_levels_interpolate)
    
    
    def find_nearest(array, value):
        ### Inputs an array 'a' and value 'x'; returns the closest element of a to x
        array = np.asarray(array)
        diffs = (np.abs(array - value))
        min_diff = diffs.min()
        idx= diffs.argmin()
        return idx,min_diff

    
    temps_grid = np.full(np.shape(alcf_backscatter),-999.)
    
    for time_idx, profile in enumerate(alcf_cloud_mask):
        
        for alt_idx, mask in enumerate(profile):
            alt = zfull[alt_idx]
            amps_alt_idx,_ = find_nearest(alt_levels,alt)
            
            temp = site_temp_interpol[time_idx, amps_alt_idx]
            temps_grid[time_idx,alt_idx] = temp
            
            
    alcf_ds.close()   
    
    return temps_grid
            
    

def amps_var(date,y_max=10):
    
    print('Date:',date,'loading')
   
    
    temps_grid_all = []
    
    for i in [0,1,2]:
        date_x = str(int(date)-i)
        
        print('Date:',date,'loading forecast ',i, date_x)
            
        amps_filepath= amps_dir+"{0}00/{0}00_WRF_NZ_f*.nc".format(date_x)
        amps_ds = xr.open_mfdataset(amps_filepath,combine='nested',concat_dim='time',drop_variables=['A_PCP_GDS5_SFC_acc3h','A_PCP_GDS5_SFC_acc'])
        
        temps=amps_ds.variables['TMP_GDS5_ISBL']#temp as function of pressure level (altitude)
        
        temps_00 = temps[0:4]
        temps_01 = temps[4:8]
        temps_02 = temps[8:12]
        temps_03 = temps[12:16]
        temps_04 = temps[16:20]
        temps_05 = temps[20:24]
        temps_06 = temps[24:]
        
        temps_10 = xr.Variable.concat([temps_00, temps_01],dim='time')
        temps_20 = xr.Variable.concat([temps_02, temps_03],dim='time')
        temps_30 = xr.Variable.concat([temps_04, temps_05],dim='time')
        
        temps_all = [temps_10,temps_20,temps_30]
        forecast_temps = temps_all[i]
        
        site_temp = forecast_temps[:,:,228,257]-273
        
        iso_lvls=amps_ds.variables['lv_ISBL2']#pressure levels (ie altitudes) 

    
        temps_grid = process(date_x,site_temp,iso_lvls)
        
        temps_grid_all.append(temps_grid)
        
        
    print('Date:',date,'plotting')
    make_figure(date,temps_grid_all,y_max)

 

file_list = glob.glob(mpl2nc_dir+'*')
date_list_all = [file[-15:-7] for file in file_list ]
date_list = np.unique(np.array(date_list_all))

# date_list = [date for date in date_list if int(date) >= 20211101]

# date_list = ['20210830','20211011','20210707']
# date_list = ['20211011'] 
date_list = ['20211103']

for date in date_list:
    
    amps_filepath= amps_dir+"{0}00/{0}00_WRF_NZ_f*.nc".format(date)
    
    start_time = time.time()
    try:
        amps_var(date,y_max=4.5)
    # except OSError:
        # print(date,'OSError')
    except ValueError as e:
        print(e)
    except OSError as e:
        print(e)
    except KeyError as e:
        print(e)
    print("--- %s seconds ---" % (time.time() - start_time))
        

