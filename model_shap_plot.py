# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:12:51 2023

@author: lwh41
"""

import matplotlib.gridspec as gridspec
import glob
import numpy as np
import os.path
import xarray as xr
from scipy import interpolate
import scipy.signal as signal
from netCDF4 import Dataset
from datetime import datetime
import time
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold,GroupKFold,LeaveOneGroupOut,StratifiedGroupKFold
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict,permutation_test_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score,classification_report,confusion_matrix
import pickle
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'

print('Preparing peak dataset')
peaks_fpath = peaks_dir+'peaks_df_{}.csv'.format('2e-5')
peaks_df = pd.read_csv(peaks_fpath, index_col=[0])  