#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:48:58 2020

@author: leonardo
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#ROC AUC RSoKF 2


name_clf ='SVM_linear_STDS'

abbreviation_best_HP = 'svm_lin_STD'
abbreviation_def_HP = 'svm_linear'


#score best HP
path_summary_best_HP_scores_RSoKF_2 = '/home/leonardo/Scrivania/result_brats/05_30/score_ROC_AUC_optimization_using_all_HP_sets_USING_MEAN_05_30/data_without_HISTO_SPATIAL_INTENSITY/summary_scores_all_HP_sets_without_HISTO_SPATIAL_INTENSITY_RSoKF_2.csv'


data_best_HP = pd.read_csv(path_summary_best_HP_scores_RSoKF_2, index_col=0) 
data_roc_auc_best_HP = data_best_HP[['ROC_AUC_TEST_MEAN', 'ROC_AUC_TEST_STD']]
data_roc_auc_best_HP_ADA = data_roc_auc_best_HP.loc[data_roc_auc_best_HP.index.str.contains(abbreviation_best_HP)]
data_roc_auc_best_HP_ADA_ordered = data_roc_auc_best_HP_ADA.iloc[np.r_[0:len(data_roc_auc_best_HP_ADA) - 2, -1, -2]]


x = np.array([1, 2, 3, 4])
y = data_roc_auc_best_HP_ADA_ordered['ROC_AUC_TEST_MEAN'] 
e = data_roc_auc_best_HP_ADA_ordered['ROC_AUC_TEST_STD']


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.errorbar(x, y, e, linestyle='None', capsize=5, marker='.')
ax1.set_xticks(x)
ax1.set_xticklabels(['Anova', 'Mutual info', 'PCA', 'None'])
ax1.set_yticks(np.arange(0.5, 1.25, step=0.1))
ax1.set_xlabel('Dimensionality reduction algorithm', labelpad=10)
ax1.set_ylabel('ROC AUC score')
ax1.set_title(f'ROC AUC score for \n optimized pipeline relative to \n {name_clf} classifier', pad=10)





#score default HP
path_summary_def_HP_scores_RSoKF_2 = '/home/leonardo/Scrivania/result_brats/05_30/score_ROC_AUC_default_HP_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_default_HP/summary_scores_default_HP_without_HISTO_SPATIAL_INTENSITY_RSoKF_2.csv'


data_def_HP = pd.read_csv(path_summary_def_HP_scores_RSoKF_2, index_col=0) 
data_roc_auc_def_HP = data_def_HP[['ROC_AUC_TEST_MEAN', 'ROC_AUC_TEST_STD']]
data_roc_auc_def_HP_ADA = data_roc_auc_def_HP.loc[data_roc_auc_def_HP.index.str.contains(abbreviation_def_HP)]
data_roc_auc_def_HP_ADA_ordered = data_roc_auc_def_HP_ADA.iloc[np.r_[0, 2, 3, 1]]




x_2 = np.array([1, 2, 3, 4])
y_2 = data_roc_auc_def_HP_ADA['ROC_AUC_TEST_MEAN'] 
e_2 = data_roc_auc_def_HP_ADA['ROC_AUC_TEST_STD']


ax2.errorbar(x_2, y_2, e_2, linestyle='None', capsize=5, marker='.', color='r')
ax2.set_xticks(x_2)
ax2.set_xticklabels(['MMS', 'RBTS', 'STDS', 'NONE'])
ax2.set_yticks(np.arange(0.5, 1.25, step=0.1))
ax2.set_xlabel('Scaler', labelpad=10)
#ax2.set_ylabel('ROC AUC score')
ax2.set_title(f'ROC AUC score for \n {name_clf} classifier with \n default hyper-parameters', pad=10)







fig.set_figwidth(7)
fig.set_figheight(6)


#create folder and save


outname = f'compare_score_features_best_and_def_HP_{name_clf}.png'

outdir = '/home/leonardo/Scrivania/scrittura_TESI/img/original/without_HISTO_SPATIAL_INTENSITY/2_subplot_best_def_HP'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()

