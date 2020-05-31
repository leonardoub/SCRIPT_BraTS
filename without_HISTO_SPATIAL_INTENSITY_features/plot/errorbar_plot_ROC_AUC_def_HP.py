#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:04:26 2020

@author: leonardo
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#ROC AUC RSoKF 2


name_clf ='SVM_linear'

abbreviation = 'svm_lin'

#score default HP
path_summary_def_HP_scores_RSoKF_2 = '/home/leonardo/Scrivania/result_brats/05_30/score_ROC_AUC_default_HP_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_default_HP/summary_scores_default_HP_without_HISTO_SPATIAL_INTENSITY_RSoKF_2.csv'

data_def_HP = pd.read_csv(path_summary_def_HP_scores_RSoKF_2, index_col=0) 
data_roc_auc_def_HP = data_def_HP[['ROC_AUC_TEST_MEAN', 'ROC_AUC_TEST_STD']]
data_roc_auc_def_HP_ADA = data_roc_auc_def_HP.loc[data_roc_auc_def_HP.index.str.contains(abbreviation)]

data_roc_auc_def_HP_ADA_ordered = data_roc_auc_def_HP_ADA.iloc[np.r_[0, 2, 3, 1]]



x = np.array([1, 2, 3, 4])
y = data_roc_auc_def_HP_ADA['ROC_AUC_TEST_MEAN'] 
e = data_roc_auc_def_HP_ADA['ROC_AUC_TEST_STD']


fig, ax1 = plt.subplots()


ax1.errorbar(x, y, e, linestyle='None', capsize=5, marker='.', color='r')

ax1.set_xticks(x)
ax1.set_xticklabels(['MMS', 'RBTS', 'STDS', 'NONE'])
ax1.set_yticks(np.arange(0.5, 1.25, step=0.1))

ax1.set_xlabel('Scaler', labelpad=10)
ax1.set_ylabel('ROC AUC score')

ax1.set_title(f'ROC AUC score for {name_clf} classifier with default hyper-parameters', pad=25)


fig.set_figwidth(7)
fig.set_figheight(5)

#create folder and save


outname = f'compare_score_features_def_HP_{abbreviation}.png'

outdir = '/home/leonardo/Scrivania/scrittura_TESI/img/original/without_HISTO_SPATIAL_INTENSITY/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()



