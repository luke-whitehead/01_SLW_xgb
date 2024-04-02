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

import matplotlib.patches as mpatches

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import matplotlib.gridspec as gridspec

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        # plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


print('Opening peak dataset')
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'

peaks_fpath = peaks_dir+'peaks_df_{}.csv'.format('2e-5')

peaks_df = pd.read_csv(peaks_fpath, index_col=[0])


# peaks_df['peak_backscatter'] = peaks_df['peak_backscatter']*1000

# Using _ij
LCC_peaks = peaks_df.query('LCC_ij == 1')
noLCC_peaks = peaks_df.query('LCC_ij == 0')

SLCC_peaks = peaks_df.query('SLCC_ij == 1')
noSLCC_peaks = peaks_df.query('SLCC_ij == 0')

WLCC_peaks = peaks_df.query('WLCC_ij == 1')
noWLCC_peaks = peaks_df.query('WLCC_ij == 0')

ICC_peaks = peaks_df.query('ICC_ij == 1')
noICC_peaks = peaks_df.query('ICC_ij == 0')

# Using _j
# LCC_peaks = peaks_df.query('LCC_j == 1')
# noLCC_peaks = peaks_df.query('LCC_j == 0')

# SLCC_peaks = peaks_df.query('SLCC_j == 1')
# noSLCC_peaks = peaks_df.query('SLCC_j == 0')

# WLCC_peaks = peaks_df.query('WLCC_j == 1')
# noWLCC_peaks = peaks_df.query('WLCC_j == 0')

# ICC_peaks = peaks_df.query('ICC_j == 1')
# noICC_peaks = peaks_df.query('ICC_j == 0')

var = 'peak_width'

q2_SLCC = np.nanmedian(SLCC_peaks[var])
q2_WLCC = np.nanmedian(WLCC_peaks[var])
q2_ICC = np.nanmedian(ICC_peaks[var])
q2_LCC = np.nanmedian(LCC_peaks[var])

# diff = abs(q2_SLCC-q2_WLCC)


# fig, (ax1) = plt.subplots(1,1)

# c1 = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
# c2 = (1.0, 0.4980392156862745, 0.054901960784313725)
# c3 = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
# c4 = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)

# ax1 = sns.distplot(LCC_peaks[var],ax=ax1,bins=100,label='LCC',color=c1)
# ax1 = sns.distplot(ICC_peaks[var],ax=ax1,bins=100,label='ICC',color=c2)
# ax1 = sns.distplot(SLCC_peaks[var],ax=ax1,bins=100,label='SLCC',color=c3)
# ax1 = sns.distplot(WLCC_peaks[var],ax=ax1,bins=100,label='WLCC',color=c4)

# ax1.legend()
# # ax1.set_xlim(0,1)

# # plt.axvline(q2_LCC,c=c1,linestyle='--')
# # plt.axvline(q2_ICC,c=c2,linestyle='--')
# plt.axvline(q2_SLCC,c=c3,linestyle='--')
# plt.axvline(q2_WLCC,c=c4,linestyle='--')

### Joint Plot for all days

# k_0 = sns.jointplot(y='peak_backscatter',x='peak_width',data=peaks_df,hue='LCC_ij',
#                   kind='kde',
#                   # alpha=0.1,
#                     levels=[0.2,0.4,0.6,0.8],
#                     # height=7,ratio=3,space=0,
#                     # fill=True,
#                     # ratio=3,
#                     xlim=[0,25],
#                     ylim=[0,0.0009],
#                     hue_order=[False,True],
#                     # marginal_kws=dict(common_norm=True),
#                   # marginal_ticks=True,
#                   )

# k_0.ax_marg_x.remove()
# k_0.ax_marg_y.remove()
# print('ax = k_0.ax_joint')
# ax = k_0.ax_joint
# plt.text(0.6,0.65,'All days',transform=ax.transAxes)
# plt.text(0.5,0.55,'# Liquid peaks = '+ str(len(LCC_peaks)),transform=ax.transAxes)
# plt.text(0.5,0.6,'# Non-Liquid peaks = '+ str(len(noLCC_peaks)),transform=ax.transAxes)


# k_1 = sns.jointplot(x='peak_temp',y='peak_alt',data=peaks_df,hue='LCC_ij',
#                   kind='kde',
#                   # alpha=0.05,
#                     levels=[0.2,0.4,0.6,0.8],
#                     # height=7,ratio=3,space=0,
#                     # fill=True,
#                     ratio=3,
#                     # ylim=[0,20],
#                     # ylim=[0,0.00025],
#                     hue_order=[False,True],
#                     marginal_kws=dict(common_norm=True),
#                   marginal_ticks=True,
#                   )

# k_2 = sns.jointplot(y='peak_alt',x='peak_width',data=peaks_df,hue='LCC_ij',
#                   kind='kde',
#                   # alpha=0.1,
#                     levels=[0.2,0.4,0.6,0.8],
#                     # height=7,ratio=3,space=0,
#                     # fill=True,
#                     ratio=3,
#                     xlim=[0,20],
#                     # ylim=[0,0.00025],
#                     hue_order=[False,True],
#                     marginal_kws=dict(common_norm=True),
#                   marginal_ticks=True,
#                   )

# k_3 = sns.jointplot(y='num_peaks',x='peak_width',data=peaks_df,hue='LCC_ij',
#                   kind='kde',
#                   # alpha=0.1,
#                     levels=[0.2,0.4,0.6,0.8],
#                     # height=7,ratio=3,space=0,
#                     # fill=True,
#                     ratio=3,
#                     xlim=[0,20],
#                     # ylim=[0,0.00025],
#                     hue_order=[False,True],
#                     marginal_kws=dict(common_norm=True),
#                   marginal_ticks=True,
#                   )

# print('Trying k_0.plot_joint')
# try:
    # k_0.plot_joint(sns.kdeplot,levels=[0.2,0.4,0.6,0.8])
    # k_1.plot_joint(sns.kdeplot,levels=[0.2,0.4,0.6,0.8])
    # k_2.plot_joint(sns.kdeplot,levels=[0.2,0.4,0.6,0.8])
    # k_3.plot_joint(sns.kdeplot,levels=[0.2,0.4,0.6,0.8])
# except IndexError:
    # print(' could not plot 2d KDE')
    
    
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,gridspec_kw={'hspace':0.2})
    
print('Plotting joint plot 1')

ax1 = sns.kdeplot(data=peaks_df,
                  x='peak_width',
                  y='peak_backscatter',
                  hue='LCC_ij',
                  ax=ax1
                  )

print('Plotting joint plot 2')

ax2 = sns.kdeplot(data=peaks_df,
                  x='peak_alt',
                  y='peak_backscatter',
                  hue='LCC_ij',
                  ax=ax2
                  )

print('Plotting joint plot 3')

ax3 = sns.kdeplot(data=peaks_df,
                  x='num_peaks',
                  y='peak_backscatter',
                  hue='LCC_ij',
                  ax=ax3
                  )

print('Plotting joint plot 4')

ax4 = sns.kdeplot(data=peaks_df,
                  x='peak_temp',
                  y='peak_backscatter',
                  hue='LCC_ij',
                  ax=ax4
                  )

plt.tight_layout()



# fig = plt.figure(figsize=(5,5))
# gs = gridspec.GridSpec(2, 2)

# mg0 = SeabornFig2Grid(ax1, fig, gs[0])
# mg1 = SeabornFig2Grid(ax2, fig, gs[1])
# mg2 = SeabornFig2Grid(ax3, fig, gs[3])
# mg3 = SeabornFig2Grid(ax4, fig, gs[2])

# gs.tight_layout(fig)
#gs.update(top=0.7)





# plt.show()
