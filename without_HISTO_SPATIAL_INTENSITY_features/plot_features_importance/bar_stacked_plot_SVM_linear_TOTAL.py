#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:50:04 2020

@author: leonardo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:15:31 2020

@author: leonardo
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter



scaler_list=['MMS', 'STDS', 'RBTS']

GENERAL_TOT_LIST_NAME_MMS= []

tot_list_name = []

scaler = 'MMS'
    
for i in range(1,6):

    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_svm_lin/features_importance_for_svm_lin_scaler_{scaler}_dim_red_NONE_BEST_HP_RSoKF_{2*i}.csv'
  
    data = pd.read_csv(path) 
    data_1 = data.drop(['Unnamed: 0'], axis=1)  


    name_feat_TR_set_1 = list(data['CLF_best_HP_FOLD_1_FEATURES'])
    name_feat_TR_set_2 = list(data['CLF_best_HP_FOLD_2_FEATURES'])
    name_feat_TR_set_3 = list(data['CLF_best_HP_FOLD_3_FEATURES'])
    name_feat_TR_set_4 = list(data['CLF_best_HP_FOLD_4_FEATURES'])
    name_feat_TR_set_5 = list(data['CLF_best_HP_FOLD_5_FEATURES'])

#creare lista tutti nomi
#unire liste eliminando gli elementi ripetuti
    
    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
    GENERAL_TOT_LIST_NAME_MMS.append(tot_list_name)
        
    
    
GENERAL_TOT_LIST_NAME_flat_MMS = [item for sublist in GENERAL_TOT_LIST_NAME_MMS for item in sublist]




D_MMS = Counter(GENERAL_TOT_LIST_NAME_flat_MMS)



df = pd.DataFrame.from_dict(D_MMS, orient='index', columns = ['votes'])
  

df_select_votes = df.loc[df['votes'] >= 15]

    
fig, ax1 = plt.subplots()

df_select_votes.plot(ax=ax1, kind='bar', stacked=False, width=0.5, fontsize=6)


#ax1.set_xticklabels(ax1.get_xticklabels(), rotation=85)
ax1.set_yticks(np.arange(0, 26, 1))

ax1.set_xlabel('Features', labelpad=0)
ax1.set_ylabel('Votes', labelpad=10)






##############################
###############################
#
#GENERAL_TOT_LIST_NAME_RBTS= []
#
#tot_list_name = []
#
#scaler = 'RBTS'
#    
#for i in range(1,6):
#
#    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_svm_lin/features_importance_for_svm_lin_scaler_{scaler}_dim_red_NONE_BEST_HP_RSoKF_{2*i}.csv'
#  
#    data = pd.read_csv(path) 
#    data_1 = data.drop(['Unnamed: 0'], axis=1)  
#
#
#    name_feat_TR_set_1 = list(data['CLF_best_HP_FOLD_1_FEATURES'])
#    name_feat_TR_set_2 = list(data['CLF_best_HP_FOLD_2_FEATURES'])
#    name_feat_TR_set_3 = list(data['CLF_best_HP_FOLD_3_FEATURES'])
#    name_feat_TR_set_4 = list(data['CLF_best_HP_FOLD_4_FEATURES'])
#    name_feat_TR_set_5 = list(data['CLF_best_HP_FOLD_5_FEATURES'])
#
##creare lista tutti nomi
##unire liste eliminando gli elementi ripetuti
#    
#    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
#    GENERAL_TOT_LIST_NAME_RBTS.append(tot_list_name)
#        
#    
#    
#GENERAL_TOT_LIST_NAME_flat_RBTS = [item for sublist in GENERAL_TOT_LIST_NAME_RBTS for item in sublist]
#
#D_RBTS = Counter(GENERAL_TOT_LIST_NAME_flat_RBTS)
#
#
##############################
###############################
#
#GENERAL_TOT_LIST_NAME_STDS= []
#
#tot_list_name = []
#
#scaler = 'STDS'
#    
#for i in range(1,6):
#
#    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_svm_lin/features_importance_for_svm_lin_scaler_{scaler}_dim_red_NONE_BEST_HP_RSoKF_{2*i}.csv'
#  
#    data = pd.read_csv(path) 
#    data_1 = data.drop(['Unnamed: 0'], axis=1)  
#
#
#    name_feat_TR_set_1 = list(data['CLF_best_HP_FOLD_1_FEATURES'])
#    name_feat_TR_set_2 = list(data['CLF_best_HP_FOLD_2_FEATURES'])
#    name_feat_TR_set_3 = list(data['CLF_best_HP_FOLD_3_FEATURES'])
#    name_feat_TR_set_4 = list(data['CLF_best_HP_FOLD_4_FEATURES'])
#    name_feat_TR_set_5 = list(data['CLF_best_HP_FOLD_5_FEATURES'])
#
##creare lista tutti nomi
##unire liste eliminando gli elementi ripetuti
#    
#    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
#    GENERAL_TOT_LIST_NAME_STDS.append(tot_list_name)
#        
#    
#    
#GENERAL_TOT_LIST_NAME_flat_STDS = [item for sublist in GENERAL_TOT_LIST_NAME_STDS for item in sublist]
#
#    
#
#D_STDS = Counter(GENERAL_TOT_LIST_NAME_flat_STDS)
