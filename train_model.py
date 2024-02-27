# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:38:28 2022

@author: lwh41

Opens the peak dataset and trains XGB models to classify peaks with target 
variable SLCC and LCC, using the reference VDR mask to train. Saves the model.
"""

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
import matplotlib.colors as colors

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold,GroupKFold,LeaveOneGroupOut,StratifiedGroupKFold
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict,permutation_test_score
from sklearn.metrics import make_scorer,accuracy_score,recall_score,precision_score,f1_score, balanced_accuracy_score,roc_auc_score,classification_report,confusion_matrix, matthews_corrcoef,ConfusionMatrixDisplay
import pickle
import shap
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
# import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec


## Update 14.3.2023
ref_mask_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//A//processed{}/'
peaks_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//peaks/'

# output_dir = 'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//B//models/'


# Opens the peak dataset
print('Preparing peak dataset')
peaks_fpath = peaks_dir+'peaks_df_{}.csv'.format('2e-5')
peaks_df = pd.read_csv(peaks_fpath, index_col=[0])      

peaks_df['peak_alt']= peaks_df['peak_alt']/1000                      
                            
# Exclude case study dates from training dataset                           
exclusion_dates = [
    20210518,
    20210605,
    ]
                     
peaks_training = peaks_df[~peaks_df['date'].isin(exclusion_dates)]
peaks_training_SLCC = peaks_training.query('amps_flag==1') # Required if training using SLCC

peaks_training_LCC = peaks_training.reset_index(drop=True)
peaks_training_SLCC = peaks_training_SLCC.reset_index(drop=True)

## Select input and target variables for model training
VARS = [
    'peak_backscatter', 
    'peak_width',
    'peak_width_height',
    'peak_prominence',
    'peak_alt',
    'num_peaks',
    'peak_order',
    'peak_temp',
    ]

print(VARS)

X_LCC = peaks_training_LCC[VARS]    
X_SLCC = peaks_training_SLCC[VARS]   

# Selecting target for binary classification
TARGET_LCC = 'LCC_ij'
TARGET_SLCC = 'SLCC_ij' # Need to uncomment peaks_training.query('amps_flag==1') above

Y_LCC = peaks_training_LCC[TARGET_LCC].astype(int)                       
Y_SLCC = peaks_training_SLCC[TARGET_SLCC].astype(int)        

# Selecting target for multiclass classification
TARGET_multi = peaks_training_SLCC[['ICC_ij','SLCC_ij','WLCC_ij']].idxmax(axis=1)
peaks_training_SLCC['TargetStr']= TARGET_multi 
peaks_training_SLCC['TargetInt']= peaks_training_SLCC['TargetStr'].astype('category').cat.codes  

Y_multi =  peaks_training_SLCC['TargetInt']           
                    
# Defining groups for the different training datasets        
groupsLCC = peaks_training_LCC['date'].astype('str').str[:-2]
groupsSLCC = peaks_training_SLCC['date'].astype('str').str[:-2]
groups_multi = groupsSLCC


## Setting up XG Boost
print('Setting up XGB')
# skf = StratifiedKFold(n_splits=3)
# gkf = GroupKFold(n_splits=3)
# logo = LeaveOneGroupOut() #Same as doing gkf (n=9)
sgkf = StratifiedGroupKFold(n_splits=3)

cv = sgkf

model_LCC = xgb.XGBClassifier(objective='binary:logistic',
                      booster='gbtree', n_jobs=-1,
                      use_label_encoder=False,verbosity=0)

model_SLCC = xgb.XGBClassifier(objective='binary:logistic',
                      booster='gbtree', n_jobs=-1,
                      use_label_encoder=False,verbosity=0)

model_multi = xgb.XGBClassifier(objective='multi:softprob',
                      booster='gbtree', n_jobs=-1,
                      use_label_encoder=False,verbosity=0)

hyperparameter_space = {
                        'max_depth':[3,6,9], 
                        'learning_rate':[0.01,0.02,0.1,0.3,0.5],
                        # 'gamma':[0,3,6],
                        # 'reg_lambda':[0,1.0,3.0,10.0],
                        # 'scale_pos_weight':[1,3,5]
                        }


# gs = GridSearchCV(model_multi, param_grid=hyperparameter_space, 
#                   scoring="balanced_accuracy",verbose=4,
#                   n_jobs=-1, cv=cv,return_train_score=True)

# print('StratifiedGroupKFold, k=3')
# print('TARGET =',TARGET)

# print('Full dataset size: = {}'.format(len(X)))
# print('Full dataset class ratio: = {:.3f}'.format(Y.mean()))

# print('Starting grid search')
# start_time = time.time()
# gs.fit(X_SLCC,Y_multi,groups=groups_multi)
# print(gs.best_params_)
# print(gs.best_score_)
# best_model = gs.best_estimator_
# print("--- %s seconds ---" % (time.time() - start_time))

# model_multi = best_model


# If not doing grid search:
# print('Fitting model')

# params_LCC_ij = {'learning_rate': 0.1, 'max_depth': 3} # For LCC_ij
# params_LCC_j = {'learning_rate': 0.02, 'max_depth': 6} # For LCC_j
# params_SLCC_ij = {'learning_rate': 0.02, 'max_depth': 6} # For SLCC_ij
# params_SLCC_j = {'learning_rate': 0.02, 'max_depth': 9} # For SLCC_j

# params_LCC_all = {'LCC_ij':params_LCC_ij,'LCC_j':params_LCC_j}
# params_SLCC_all = {'SLCC_ij':params_SLCC_ij,'SLCC_j':params_SLCC_j}

# params_multi = {'learning_rate': 0.1, 'max_depth': 3}

# params_LCC = params_LCC_all[TARGET_LCC]
# params_SLCC = params_SLCC_all[TARGET_SLCC]

# # print('Yes hyperparams')
# # model_LCC.set_params(**params_LCC)
# # model_SLCC.set_params(**params_SLCC)

# model_multi.set_params(**params_multi)


start_time = time.time()

# # model_LCC.fit(X_LCC,Y_LCC)
# # model_SLCC.fit(X_SLCC,Y_SLCC)
model_multi.fit(X_SLCC, Y_multi)

print("--- %s seconds ---" % (time.time() - start_time))

## --------------------------------------------------------------------------


# plt.figure()
# sns.displot(data=peaks_training_SLCC, x='peak_temp',hue='TargetInt',kind='kde')



## --------------------------------------------------------------------------

# Save the model to file

# print('Saving model to file...')

# model_LCC_save_name = 'models/X_'+TARGET_LCC+'.json'
# model_SLCC_save_name = 'models/X_'+TARGET_SLCC+'.json'
# model_multi_save_name = 'models/multi_ij_no_hp.json'

# best_model.save_model(output_dir+'model.json') #Doesn't work because S:
# model_LCC.save_model(model_LCC_save_name)
# model_SLCC.save_model(model_SLCC_save_name)
# model_multi.save_model(model_multi_save_name)




## --------------------------------------------------------------------------

# 3-fold group cross validation, more detail

# for idx, (train, test) in enumerate(cv.split(X, Y,groups=groups)):
#     print('CV fold', idx)
#     print('Train dataset class ratio = {:.3f}'.format(Y[train].mean()))
#     print('Test dataset class ratio = {:.3f}'.format(Y[test].mean()))
#     print('Train/Test dataset size = {}/{}'.format(len(train),len(test)))

#     best_model.fit(X.iloc[train], Y[train])
#     Y_pred = best_model.predict(X.iloc[test])
#     print(idx, classification_report(Y[test], Y_pred))
    
#     print(confusion_matrix(Y[test], Y_pred,normalize='all'))
    
## --------------------------------------------------------------------------

# Permutation significance testing

# print('Starting perm test')

# start_time = time.time()

# n_permutations=100
# score, perm_scores, pvalue = permutation_test_score(model, X, Y, scoring="balanced_accuracy", cv=cv, groups=groups, n_permutations=n_permutations,verbose=3,n_jobs=-1)
# print(score, perm_scores, pvalue)
# min_p = 1/(n_permutations+1)

# print("--- %s seconds ---" % (time.time() - start_time))

# fig, ax = plt.subplots()

# ax.hist(perm_scores, bins=20, density=True,range=(0,1))
# ax.axvline(score, ls="--", color="r")
# score_label = f"Score: {score:.4f}\n(p-value: {pvalue:.4f})\nmin p_val: {min_p:.4f}"
# ax.text(0.1, 0.8, score_label, fontsize=12, transform=ax.transAxes)
# ax.set_xlabel("Accuracy score")
# ax.set_xlim(0,1)
# _ = ax.set_ylabel("Density")

# plt.show()

## --------------------------------------------------------------------------
## Plotting train/test folds

# import seaborn as sns
# import matplotlib.pyplot as plt

# fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,figsize=(10,5))

# axes = [(ax1, ax2), (ax3, ax4), (ax5, ax6)]

# c1 = (0.03137254901960784,0.18823529411764706,0.4196078431372549,1)
# c2 = (0.6470588235294118,0.0,0.14901960784313725,1)    
# c3 = (0.47058823529411764, 0.7764705882352941, 0.4745098039215686)

# for idx, (train, test) in enumerate(cv.split(X, Y,groups=groups)):
#     ax_A, ax_B = axes[idx]

#     ax_A.invert_yaxis()
#     sns.rugplot(train, height=1, ax=ax_A, color=c1,alpha=0.5,lw=0.1)
#     ax_A.invert_yaxis()
#     ax_A.set_xlim(0,len(peaks_training))
#     # ax_A.set_xlim(25000,30000)
    
#     ax_A.set_xticklabels([])
#     ax_A.set_xticks([])
#     ax_A.set_yticklabels([])
#     ax_A.set_yticks([])

#     sns.rugplot(test, height=1, ax=ax_B, color=c2,alpha=0.5,lw=0.1)
#     ax_B.set_xlim(0,len(peaks_training))
#     # ax_B.set_xlim(25000,30000)
    
#     if ax_B != ax6:
#         ax_B.set_xticklabels([])
#         ax_B.set_xticks([])
#     else:
#         ax_B.set_xticklabels(np.unique(groups, return_index=True)[0])
#         ax_B.set_xticks(np.unique(groups, return_index=True)[1])
#     ax_B.set_yticklabels([])
#     ax_B.set_yticks([])
    
#     ax_A.text(1.02,0.4,'Train',transform=ax_A.transAxes)
#     ax_B.text(1.02,0.4,'Test',transform=ax_B.transAxes)
    
    
# ax6.set_xlabel('Month')

# ax1.text(-0.1,-0.2,'Fold 1',transform=ax1.transAxes)
# ax3.text(-0.1,-0.2,'Fold 2',transform=ax3.transAxes)
# ax5.text(-0.1,-0.2,'Fold 3',transform=ax5.transAxes)
# # ax7.text(-0.1,-0.2,'Fold 4',transform=ax7.transAxes)
# # ax9.text(-0.1,-0.2,'Fold 5',transform=ax9.transAxes)
   
# plt.show()

## --------------------------------------------------------------------------

# Printing accuracy scores:
# print('Recall')
# scores_recall = cross_val_score(best_model, X, Y, cv=cv, scoring='recall',groups=groups)
# print(scores_recall, '{:.3f}, {:.3f}'.format(np.mean(scores_recall),np.std(scores_recall)))
# print('Precision')
# scores_precision = cross_val_score(best_model, X, Y, cv=cv, scoring='precision',groups=groups)
# print(scores_precision, '{:.3f}, {:.3f}'.format(np.mean(scores_precision),np.std(scores_precision)))
# print('F1 - LCC')
# scores_f1 = cross_val_score(model_LCC, X_LCC, Y_LCC, cv=cv, scoring='f1',groups=groupsLCC)
# print(scores_f1, '{:.3f}, {:.3f}'.format(np.mean(scores_f1),np.std(scores_f1)))
# print('F1 - SLCC')
# scores_f1 = cross_val_score(model_SLCC, X_SLCC, Y_SLCC, cv=cv, scoring='f1',groups=groupsSLCC)
# print(scores_f1, '{:.3f}, {:.3f}'.format(np.mean(scores_f1),np.std(scores_f1)))
# print('Accuracy')
# scores_acc = cross_val_score(best_model, X, Y, cv=cv, scoring='accuracy',groups=groups)
# print(scores_acc, '{:.3f}, {:.3f}'.format(np.mean(scores_acc),np.std(scores_acc)))
# print('Balanced Accuracy - LCC')
# scores_bal = cross_val_score(model_LCC, X_LCC, Y_LCC, cv=cv, scoring='balanced_accuracy',groups=groupsLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))
# print('Balanced Accuracy - SLCC')
# scores_bal = cross_val_score(model_SLCC, X_SLCC, Y_SLCC, cv=cv, scoring='balanced_accuracy',groups=groupsSLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))
# print('ROC AUC')
# scores_roc = cross_val_score(best_model, X, Y, cv=cv, scoring='roc_auc',groups=groups)
# print(scores_roc, '{:.3f}, {:.3f}'.format(np.mean(scores_roc),np.std(scores_roc)))


# matthews_corrcoef
# print('MCC Accuracy - LCC')
# scores_bal = cross_val_score(model_LCC, X_LCC, Y_LCC, cv=cv, scoring='matthews_corrcoef',groups=groupsLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))
# print('MCC Accuracy - SLCC')
# scores_bal = cross_val_score(model_SLCC, X_SLCC, Y_SLCC, cv=cv, scoring='matthews_corrcoef',groups=groupsSLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))

## --------------------------------------------------------------------------

## Confusion matrix

class_names = ['ICC', 'SLCC', 'WLCC']
# disp = plot_confusion_matrix(model_multi, X_SLCC, Y_multi, display_labels=class_names, cmap=plt.cm.Blues, xticks_rotation='vertical')

# disp = ConfusionMatrixDisplay.from_estimator(
#         model_multi,
#         X_SLCC,
#         Y_multi,
#         display_labels=class_names,
#         cmap=plt.cm.Blues,
#         normalize='true',
#     )

# plt.show()

# print('Balanced Accuracy - Multi')
# scores_bal = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring='balanced_accuracy',groups=groupsSLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))

# print('Recall - Multi')
# scores_bal = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring='recall_macro',groups=groupsSLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))

# print('F1 - Multi')
# scores_bal = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring='f1_macro',groups=groupsSLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))

# print('MCC - Multi')
# scores_bal = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring='matthews_corrcoef',groups=groupsSLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))

# print(classification_report(Y_multi, model_multi.predict(X_SLCC))) # Does class report on all peaks in training set



### Testing the model on dates that were excluded from training
# testing_dates = [
#     20210518,
#     20210605,
#     ]
                     
# peaks_testing = peaks_df[peaks_df['date'].isin(testing_dates)]
# peaks_testing_SLCC = peaks_testing.query('amps_flag==1') # Required if training using SLCC

# peaks_testing_SLCC = peaks_testing_SLCC.reset_index(drop=True)

# ## Select input and target variables for model training
# VARS = [
#     'peak_backscatter', 
#     'peak_width',
#     'peak_width_height',
#     'peak_prominence',
#     'peak_alt',
#     'num_peaks',
#     'peak_order',
#     'peak_temp',
#     ]

# X_SLCC = peaks_testing_SLCC[VARS]   

# # Selecting target for binary classification
# TARGET_SLCC = 'SLCC_ij' # Need to uncomment peaks_training.query('amps_flag==1') above

# Y_SLCC = peaks_testing_SLCC[TARGET_SLCC].astype(int)        

# # Selecting target for multiclass classification
# TARGET_multi = peaks_testing_SLCC[['ICC_ij','SLCC_ij','WLCC_ij']].idxmax(axis=1)
# peaks_testing_SLCC['TargetStr']= TARGET_multi 
# peaks_testing_SLCC['TargetInt']= peaks_testing_SLCC['TargetStr'].astype('category').cat.codes  

# Y_multi =  peaks_testing_SLCC['TargetInt']        

# print(classification_report(Y_multi, model_multi.predict(X_SLCC)))

## --------------------------------------------------------------------------

# Getting more detailed accuracy scores of the 3-fold CV

# for idx, (train, test) in enumerate(cv.split(X_SLCC, Y_multi,groups=groups_multi)):
#     model_multi.fit(X_SLCC.loc[train], Y_multi[train])
#     y_pred = model_multi.predict(X_SLCC.loc[test])
#     print()
    # print(idx, classification_report(Y_multi[test], y_pred))
    
#     # print(recall_score(Y_multi[test], y_pred,average=None))
#     # print(f1_score(Y_multi[test], y_pred,average=None,labels=[0]))

# print('Balanced Accuracy - Multi')
# scores_bal = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring='balanced_accuracy',groups=groupsSLCC)
# print(scores_bal, '{:.3f}, {:.3f}'.format(np.mean(scores_bal),np.std(scores_bal)))
    
# print('F1 - Class 0 (ICC)')
# scores_f1_0 = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring=make_scorer(f1_score,average=None,labels=[0]),groups=groupsSLCC)
# print(scores_f1_0, '{:.3f}, {:.3f}'.format(np.mean(scores_f1_0),np.std(scores_f1_0)))

# print('F1 - Class 1 (SLCC)')
# scores_f1_1 = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring=make_scorer(f1_score,average=None,labels=[1]),groups=groupsSLCC)
# print(scores_f1_1, '{:.3f}, {:.3f}'.format(np.mean(scores_f1_1),np.std(scores_f1_1)))     

# print('F1 - Class 2 (WLCC)')
# scores_f1_2 = cross_val_score(model_multi, X_SLCC, Y_multi, cv=cv, scoring=make_scorer(f1_score,average=None,labels=[2]),groups=groupsSLCC)
# print(scores_f1_2, '{:.3f}, {:.3f}'.format(np.mean(scores_f1_2),np.std(scores_f1_2)))     
    
    
#     disp = ConfusionMatrixDisplay.from_estimator(
#         model_multi,
#         X_SLCC.loc[test],
#         Y_multi[test],
#         display_labels=class_names,
#         cmap=plt.cm.Blues,
#         normalize='true',
#     )

#     plt.show()
    

## --------------------------------------------------------------------------
# from sklearn.pipeline import Pipeline
# import joblib

# # Opens AG model (old)
# model_ag_load = xgb.XGBClassifier();
# model_ag_load.load_model("models/AG_XGBoost.json")
# std = joblib.load('models/AG_scaler.save') 

# pipe = Pipeline([('standardscaler',std),('xgbclassifier',model_ag_load)])

# new_cols = ['peak_order','num_peaks','peak_prominence','peak_backscatter','peak_width','peak_width_height','peak_temp','peak_alt']
# X_ag = X_SLCC[new_cols]

# y_pred_ag = pipe.predict(X_ag)
# recall_ag = recall_score(Y_multi, y_pred_ag, average=None)

# disp = ConfusionMatrixDisplay.from_estimator(
#         pipe,
#         X_ag,
#         peaks_training_SLCC[TARGET_SLCC].astype(int) ,
#         # display_labels=class_names,
#         cmap=plt.cm.Blues,
#         normalize='true',
#     )

# plt.show()

# print(recall_score(peaks_training_SLCC[TARGET_SLCC].astype(int), y_pred_ag, average=None))


## --------------------------------------------------------------------------

# Feature analysis

# sorted_idx = best_model.feature_importances_.argsort()
# plt.barh(X.columns[sorted_idx], best_model.feature_importances_[sorted_idx])
# plt.title(TARGET)
# plt.show()


# SHAP Values



## explainer = shap.TreeExplainer(best_model)
## shap_values = explainer.shap_values(X)
## shap.summary_plot(shap_values,X)



# print('Getting SHAP values')
explainer_multi = shap.Explainer(model_multi)
shap_values_multi = explainer_multi(X_SLCC)

# explainer_LCC = shap.Explainer(model_LCC)
# shap_values_LCC = explainer_LCC(X_LCC)

# explainer_SLCC = shap.Explainer(model_SLCC)
# shap_values_SLCC = explainer_SLCC(X_SLCC)

# # shap_interaction = explainer.shap_interaction_values(X)

# print('Plotting SHAP')


cmap='coolwarm'
c1 = (0.03137254901960784,0.18823529411764706,0.4196078431372549,1)
c_slcc = (0.8,0,0,1)    #SLCC
c_wlcc = (0,0,1)    #WLCC
c_icc = (0.8,0.8,0)    #ICC
c_arr = [c_icc,c_slcc,c_wlcc]

# X = X_SLCC
# TARGET = TARGET_SLCC
# shap_values = shap_values_multi[:,:,:]

#### ----------------------------------
## Makes beeswarm plot, saves to file

# shap.summary_plot(shap_values,X, cmap=cmap,show=False,feature_names=X.columns)


# plt.tight_layout()
# plt.savefig(f'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//SHAP_figs/{TARGET}_beeswarm.png',dpi=200)

#### ----------------------------------
## Makes SHAP plot for multiclass, one bar showing all

# shap_values = shap.TreeExplainer(model_multi).shap_values(X_SLCC)

# fig = plt.figure(figsize=(10,7))

# ax1 = fig.add_subplot(111)

feat_names = ['Peak Height','Width','Width Height','Prominence','Altitude','n Peaks', 'Peak Order', 'Temperature']
# class_names = ['ICC', 'SLCC', 'WLCC']

# color_lamb = lambda i: c_arr[i]
# shap.summary_plot(shap_values, X_SLCC,feature_names=feat_names,class_names=class_names,show=False,plot_size=None,color=color_lamb)

# ax1.legend(loc=7, fontsize=16,frameon=False)#,borderaxespad=0. )

# for fc in ax1.get_children()[:-1]:
#     if isinstance(fc,mpl.lines.Line2D):
#         fc.remove()
#     if isinstance(fc,mpl.patches.Rectangle):
#         fc.set(lw=0)
        

# ax1.spines['left'].set_visible(True)
# # ax1.set_xlim(-0.1,10.1)
# ax1.tick_params(axis='both',labelsize=15)

# plt.tight_layout()
# # plt.savefig('S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//SHAP_figs/multi_class_bar.png',dpi=250)


# ## Makes SHAP plot for multiclass, multiple beeswarm

# fig = plt.figure(figsize=(10,14))

# ax1 = fig.add_subplot(311)

# shap.summary_plot(shap_values_multi[:,:,0].values,X_SLCC.values,feature_names=feat_names, cmap=cmap,show=False,plot_size=None)
# # shap.summary_plot(shap_values_multi[:,:,0].values,X_SLCC.values, cmap=cmap,show=False,feature_names=X_SLCC.columns)

# ax2= fig.add_subplot(312,sharex=ax1)
# shap.summary_plot(shap_values_multi[:,:,1].values,X_SLCC.values,feature_names=feat_names, cmap=cmap,show=False,plot_size=None)

# ax3= fig.add_subplot(313,sharex=ax1)
# shap.summary_plot(shap_values_multi[:,:,2].values,X_SLCC.values,feature_names=feat_names, cmap=cmap,show=False,plot_size=None)

# ax1.set_xlabel('')
# ax1.spines['bottom'].set_visible(False)
# ax1.tick_params(axis='x',bottom=False,labelbottom=False)

# ax2.set_xlabel('')
# ax2.spines['bottom'].set_visible(False)
# ax2.tick_params(axis='x',bottom=False,labelbottom=False)

# ax3.tick_params(axis='x',labelsize=14)

# ax1.text(-0.05,0.97,'a - ICC',transform=ax1.transAxes,ha='right',va='bottom',fontsize=20)
# ax2.text(-0.05,0.97,'b - SLCC',transform=ax2.transAxes,ha='right',va='bottom',fontsize=20)
# ax3.text(-0.05,0.97,'c - WLCC',transform=ax3.transAxes,ha='right',va='bottom',fontsize=20)

# ax1.tick_params(axis='y', labelsize=14)
# ax2.tick_params(axis='y', labelsize=14)
# ax3.tick_params(axis='y', labelsize=14)

# plt.tight_layout()
# # plt.savefig('S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//SHAP_figs/multi_class_beeswarm.png',dpi=250)


#### ----------------------------------


## Makes scatter plots, saves to file

# for i in range(3):
    
#     shap_values = shap_values_multi[:,:,i]
    
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,16),gridspec_kw={'hspace':0.2,'wspace':0.25})
    
#     scatter_colour = shap_values
    
#     shap.plots.scatter(shap_values[:,"peak_backscatter"], ax=axes[0, 0],show=False,color=scatter_colour,cmap=cmap)
#     axes[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
#     shap.plots.scatter(shap_values[:,"peak_width"], ax=axes[0, 1],show=False,color=scatter_colour,cmap=cmap)
    
#     shap.plots.scatter(shap_values[:,"peak_width_height"], ax=axes[0, 2],show=False,color=scatter_colour,cmap=cmap)
#     axes[0, 2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
#     shap.plots.scatter(shap_values[:,"peak_prominence"], ax=axes[1, 0],show=False,color=scatter_colour,cmap=cmap)
#     axes[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
#     shap.plots.scatter(shap_values[:,"peak_alt"], ax=axes[1, 1],show=False,color=scatter_colour,cmap=cmap)
    
#     shap.plots.scatter(shap_values[:,"peak_temp"], ax=axes[1, 2],show=False,color=scatter_colour,cmap=cmap)
#     shap.plots.scatter(shap_values[:,"num_peaks"], ax=axes[2, 0],show=False,color=scatter_colour,cmap=cmap)
#     shap.plots.scatter(shap_values[:,"peak_order"], ax=axes[2, 1],show=False,color=scatter_colour,cmap=cmap)
    
#     axes[2, 2].axis('off')
    
#     plt.tight_layout()
#     plt.show()
# plt.savefig(f'S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//SHAP_figs/{TARGET}_scatters.png',dpi=200)


#### ----------------------------------
## Makes beeswarm plot, subplots

# fig = plt.figure(figsize=(10,10))

# ax1 = fig.add_subplot(211)

# # feat_names = ['Peak Height','Width','Width Height','Prominence','Altitude','n Peaks', 'Peak Order', 'Temperature']
# feat_names = X_LCC.columns

# shap.summary_plot(shap_values_LCC,X_LCC,feature_names=feat_names, cmap=cmap,show=False,plot_size=None)

# ax2= fig.add_subplot(212,sharex=ax1)
# shap.summary_plot(shap_values_SLCC,X_SLCC,feature_names=feat_names,cmap=cmap,show=False,plot_size=None)

# ax1.set_xlabel('')
# ax1.spines['bottom'].set_visible(False)
# ax1.tick_params(axis='x',bottom=False,labelbottom=False)

# ax2.tick_params(axis='x',labelsize=14)

# ax1.text(-0.05,0.97,'a - $X_1$',transform=ax1.transAxes,ha='right',va='bottom',fontsize=20)

# ax2.text(-0.05,0.97,'b - $X_2$',transform=ax2.transAxes,ha='right',va='bottom',fontsize=20)

# ax1.tick_params(axis='y', labelsize=14)
# ax2.tick_params(axis='y', labelsize=14)

# plt.tight_layout()

#plt.savefig('S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//SHAP_figs/xLCC_ij_multi_beeswarm_with_hp.png',dpi=200)



#### ----------------------------------
## Makes scatter plots with reduced rows, saves to file

shap_values_ICC = shap_values_multi[:,:,0]
shap_values_LCC = shap_values_multi[:,:,1]
shap_values_SLCC = shap_values_multi[:,:,2]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,18),gridspec_kw={'hspace':0.5,'wspace':0.25})

# scatter_colour_ICC = shap_values_ICC[:,'peak_width']
# scatter_colour_LCC = shap_values_LCC[:,'peak_width']
# scatter_colour_SLCC = shap_values_SLCC[:,'peak_width']

c = plt.get_cmap(cmap)
scatter_colour_ICC = (0.8,0.8,0,1)    #Ice/Non-SLW
scatter_colour_LCC = c(0.99)
scatter_colour_SLCC = c(0.01)


shap.plots.scatter(shap_values_ICC[:,"peak_width"], ax=axes[0, 0],show=False,color=scatter_colour_ICC,cmap=cmap)
shap.plots.scatter(shap_values_LCC[:,"peak_width"], ax=axes[1, 0],show=False,color=scatter_colour_LCC,cmap=cmap)
shap.plots.scatter(shap_values_SLCC[:,"peak_width"], ax=axes[2, 0],show=False,color=scatter_colour_SLCC,cmap=cmap)
axes[0, 0].set_xlabel('Width',fontsize=15)
axes[1, 0].set_xlabel('Width',fontsize=15)
axes[2, 0].set_xlabel('Width',fontsize=15)

shap.plots.scatter(shap_values_ICC[:,"peak_alt"], ax=axes[0, 1],show=False,color=scatter_colour_ICC,cmap=cmap)
shap.plots.scatter(shap_values_LCC[:,"peak_alt"], ax=axes[1, 1],show=False,color=scatter_colour_LCC,cmap=cmap)
shap.plots.scatter(shap_values_SLCC[:,"peak_alt"], ax=axes[2, 1],show=False,color=scatter_colour_SLCC,cmap=cmap)
axes[0, 1].set_xlabel('Altitude (km)',fontsize=15)
axes[1, 1].set_xlabel('Altitude (km)',fontsize=15)
axes[2, 1].set_xlabel('Altitude (km)',fontsize=15)

# axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

shap.plots.scatter(shap_values_ICC[:,"peak_temp"], ax=axes[0, 2],show=False,color=scatter_colour_ICC,cmap=cmap)
shap.plots.scatter(shap_values_LCC[:,"peak_temp"], ax=axes[1, 2],show=False,color=scatter_colour_LCC,cmap=cmap)
shap.plots.scatter(shap_values_SLCC[:,"peak_temp"], ax=axes[2, 2],show=False,color=scatter_colour_SLCC,cmap=cmap)
axes[0, 2].set_xlabel('Temperature ($\\degree$C)',fontsize=15)
axes[1, 2].set_xlabel('Temperature ($\\degree$C)',fontsize=15)
axes[2, 2].set_xlabel('Temperature ($\\degree$C)',fontsize=15)



letters='abcdefghi'
for i, ax in enumerate(axes.flatten()):
    ax.text(-0.1,1,'({})'.format(letters[i]),fontsize=25,transform=ax.transAxes,ha='right',va='bottom',weight='semibold')
    ax.tick_params(axis='both',which='both',labelsize=14)
    ax.set_ylabel(ax.get_ylabel(),fontsize=14)
    


axes[0, 1].text(0.5,1.12,'ICC',transform=axes[0, 1].transAxes,ha='center',va='bottom',fontsize=25,weight='semibold')
axes[1,1].text(0.5,1.12,'SLCC',transform=axes[1, 1].transAxes,ha='center',va='bottom',fontsize=25,weight='semibold')
axes[2, 1].text(0.5,1.12,'WLCC',transform=axes[2, 1].transAxes,ha='center',va='bottom',fontsize=25,weight='semibold')


plt.tight_layout()
# plt.savefig('S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//SHAP_figs/multi_class_scatters_update.png',dpi=300)


#### ----------------------------------

# Correlation plot

# PLotting feature correlations

# feat_names_sub = feat_names
# feat_names_sub[0] = 'Peak\nHeight'
# feat_names_sub[2] = 'Width\nHeight'
# feat_names_sub[6] = 'Peak\nOrder'

# def correlation_heatmap(df):
#     correlations = df.corr()[1:].drop('peak_temp',axis=1)
    
#     mask = ~np.tril(np.ones_like(correlations, dtype=bool))
    
#     fig, ax = plt.subplots(figsize=(10,10))
#     ax=sns.heatmap(correlations,mask=mask,vmax=1.0,vmin=-1.0, center=0, fmt='.2f', cmap='RdBu',
#                 square=True,linewidths=1,annot=True, ax=ax,
#                 cbar_kws={'shrink':.70,'label':'Correlation Coefficient','ticks':[-1,-0.5,0,0.5,1]},
#                 xticklabels=feat_names_sub[:-1],yticklabels=feat_names_sub[1:]
#                 )
    
#     ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    
#     # plt.savefig('S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//D//feature_dist/corr.png',dpi=200,bbox_inches='tight')
    
#     return ax

    
# ax = correlation_heatmap(peaks_training_LCC[VARS])


#### -----------------------------------------

# # Plot of backscatter histograms
# plt.hist([peaks_training['peak_alt'],peaks_training.query('{}==1'.format(TARGET))['peak_alt']],density=True)
# plt.axvline(np.median(peaks_training['peak_alt']),c='k',linestyle='--')
# plt.axvline(np.median(peaks_training.query('{}==1'.format(TARGET))['peak_alt']),c='r',linestyle='--')




## SHAP experimentation

# def annotate_axes(fig):
#     for i, ax in enumerate(fig.axes):
#         ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
#         ax.tick_params(labelbottom=False, bottom=False,labelleft=False,left=False)

# fig = plt.figure(figsize = (20,25))

# gs1 = GridSpec(1, 1, top=0.98, bottom=0.6, left=0.3, right=0.9,wspace=0.05,figure=fig)
# ax1 = fig.add_subplot(gs1[0,0])
# shap.plots.beeswarm(shap_values, show=False)

# gs2 = GridSpec(2,3, top=0.48, bottom=0.05, wspace=0.05,hspace=0.05,figure=fig)
# ax2 = fig.add_subplot(gs2[0,0])
# shap.dependence_plot(("peak_prominence", "peak_width"),shap_interaction, X,display_features=X, ax=ax2,show=False)

# ax3 = fig.add_subplot(gs2[0,1])
# ax4 = fig.add_subplot(gs2[0,2])
# ax5 = fig.add_subplot(gs2[1,0])
# ax6 = fig.add_subplot(gs2[1,1])
# ax7 = fig.add_subplot(gs2[1,2])


# shap.dependence_plot(("peak_prominence", "peak_width"),shap_interaction, X,display_features=X, ax=axes[0, 1],show=False)

# shap.dependence_plot('age', shap_values[1], X_train, ax=axes[0, 0], show=False)
# shap.dependence_plot('income', shap_values[1], X_train, ax=axes[0, 1], show=False)
# shap.dependence_plot('score', shap_values[1], X_train, ax=axes[1, 0], show=False)
# plt.sca(axes[1, 0])
# shap.summary_plot(shap_values,X,show=False)
# plt.sca(axes[1, 1])
# shap.plots.beeswarm(shap_values, show=False)
# plt.show()

# gs1.tight_layout()
# annotate_axes(fig)

# shap.plots.beeswarm(shap_values, show=False)


# shap.plots.scatter(shap_values[:,"peak_temp"], color=shap_values)
# shap.plots.force(shap_values[0],matplotlib=True)
# shap.plots.waterfall(shap_values[0]) 
# shap.summary_plot(shap_values,X)









## --------------------------------------------------------------------------



### Trains the model on train/test subset
# model.fit(X, Y)

# y_pred = best_model.predict(X_val)
# y_val_pred = model.predict(X_val)
# print('Validation subset:')
# print(classification_report(y_val, y_val_pred))

# Applies the trained model to entire peaks dataset
# peaks_df_full_result = model.predict(peaks_df_full_X)
# peaks_df_result = peaks_df.assign(model_result=peaks_df_full_result)


# model.save_model('S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_models//g22_chch.json') #Doesn't work
# model.save_model('g22_chch_slw.json')

# pickle.dump(model, open('S://atmos_disk//Luke_W_480//MPL_Christchurch_2021//cal_models//g22_chch.pickle.dat', 'wb'))




    



# X_corr = train_test_subset.drop(['date','peak_time_idx', 'peak_alt_idx',
#                              'profile_liq_bool','profile_ice_bool',
#                             # 'peak_vdr',
#                             # 'peak_alt','peak_temp'
#                             ], axis=1)

# ## PLotting feature correlations
# def correlation_heatmap(train):
#     correlations = train.corr()
    
#     fig, ax = plt.subplots(figsize=(10,10))
#     sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap='RdBu',
#                 square=True,linewidths=.5,annot=True,cbar_kws={'shrink':.70})
#     plt.show()

    
# correlation_heatmap(X_corr)



# for var in peaks_df_full_X.columns[:]:

#     palette = itertools.cycle(sns.color_palette())
    
#     fig, (ax1) = plt.subplots(1,1, figsize=(10, 7), sharex=True)
    
#     c=next(palette)
#     ax1 = sns.distplot(peaks_df.query("ref_liq == 1")[var],label='Liquid peaks',ax=ax1, color=c,bins=50)
#     ax1.set_ylabel('Density')
    
#     c=next(palette)
#     ax1 = sns.distplot(peaks_df.query("ref_liq == 0")[var],label='Non-Liquid peaks',ax=ax1, color=c,bins=50)
#     # ax3 = sns.distplot(peaks_df.query("model_result == 1")[var],label='SLW peaks',ax=ax3, color=c,bins=1000)
#     # c=next(palette)
#     # ax2 = sns.distplot(peaks_df.query("ref_liq == 0")[var],label='Non-Liquid peaks',ax=ax2, color=c)
#     # ax3 = sns.distplot(peaks_df.query("model_result == 0")[var],label='Non-SLW peaks',ax=ax3, color=c,bins=1000)
#     # ax2.set_ylabel('Density')
    
#     ax1.set_xlabel('')
#     # ax2.set_xlabel('')
#     ax1.set_title(var, fontsize=18)
#     # ax2.set_title('VDR Reference Mask Classification')

    
    
#     if var == 'peak_temp':
#         ax1.legend(loc=2)
#         # ax2.legend(loc=2)
#     else:
#         ax1.legend(loc=1)
#         # ax2.legend(loc=1)
        
        
#     # if var == 'lr' or var == 'lr_grad_abs':
#     #     plt.xlim(-100,100)
    
#     plt.tight_layout()
#     plt.show()

## Plotting peak properties

# peaks_df_full_X= peaks_df.drop(vars_to_drop, axis=1)

# fig, axes = plt.subplots(3,3, figsize=(18,15))

# vars_x = ['peak_backscatter', 'peak_width', 'peak_width_height',
#        'peak_prominence', 'num_peaks', 'peak_order', 'peak_temp',
#                             'peak_alt']

# for i,var in enumerate(vars_x):
#     print(var)
    
#     ax = axes.flatten()[i]
    
#     ax.tick_params(axis='both',which='major',labelsize=13)
#     ax.tick_params(axis='both',which='minor',labelsize=13)
    
#     ax.ticklabel_format(axis='x',style='sci',scilimits=(0,5))

#     palette = itertools.cycle(sns.color_palette())
  
    
#     c=next(palette)
#     sns.distplot(peaks_df.query("ref_liq == 0")[var],label='Non-Liquid peaks',ax=ax, color=c,bins=40)
    
#     c=next(palette)
#     sns.distplot(peaks_df.query("ref_liq == 1")[var],label='Liquid peaks',ax=ax, color=c,bins=40)
#     ax.set_ylabel('Density')

#     ax.set_xlabel('')
#     ax.set_title(var,fontsize=20)
    
#     if var == 'peak_width':
#         ax.set_xlim(0,25)
        
# axes[2,2].axis('off')