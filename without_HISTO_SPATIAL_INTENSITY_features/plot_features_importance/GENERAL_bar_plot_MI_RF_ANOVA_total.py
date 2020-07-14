#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:25:25 2020

@author: leonardo
"""




import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
import seaborn as sns


#MI
GENERAL_TOT_LIST_NAME_MI = []
tot_list_name = []

for i in range(1,6):

    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /MUTUAL_INFO_best_features/AdaBoost/AdaBoost_features_selected_MUTUAL_INFO_RSoKF_{2*i}.csv'

  
    data = pd.read_csv(path) 
    data_1 = data.drop(['Unnamed: 0'], axis=1)  


    name_feat_TR_set_1 = list(data['FOLD_1_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_2 = list(data['FOLD_2_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_3 = list(data['FOLD_3_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_4 = list(data['FOLD_4_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_5 = list(data['FOLD_5_MUTUAL_INFO_FEATURES'])[:20]
   
    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
    GENERAL_TOT_LIST_NAME_MI.append(tot_list_name)
        
GENERAL_TOT_LIST_NAME_flat_MI = [item for sublist in GENERAL_TOT_LIST_NAME_MI for item in sublist]

D_MI = Counter(GENERAL_TOT_LIST_NAME_flat_MI)
df_MI = pd.DataFrame.from_dict(D_MI, orient='index', columns = ['votes_MI'])
df_select_votes_MI = df_MI.loc[df_MI['votes_MI'] >= 13]

####################################################################################
####################################################################################

#RF
GENERAL_TOT_LIST_NAME_RF = []
tot_list_name = []

for i in range(1,6):

    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_RandomForest/features_importance_for_RandomForest_scaler_VARIABLE_dim_red_NONE_BEST_HP_RSoKF_{2*i}.csv'
    
    data = pd.read_csv(path) 
    data_1 = data.drop(['Unnamed: 0'], axis=1)  

    name_feat_TR_set_1 = list(data['CLF_best_HP_FOLD_1_FEATURES'])
    name_feat_TR_set_2 = list(data['CLF_best_HP_FOLD_2_FEATURES'])
    name_feat_TR_set_3 = list(data['CLF_best_HP_FOLD_3_FEATURES'])
    name_feat_TR_set_4 = list(data['CLF_best_HP_FOLD_4_FEATURES'])
    name_feat_TR_set_5 = list(data['CLF_best_HP_FOLD_5_FEATURES'])

    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
    GENERAL_TOT_LIST_NAME_RF.append(tot_list_name)
      
GENERAL_TOT_LIST_NAME_flat_RF = [item for sublist in GENERAL_TOT_LIST_NAME_RF for item in sublist]

D_RF = Counter(GENERAL_TOT_LIST_NAME_flat_RF)
df_RF = pd.DataFrame.from_dict(D_RF, orient='index', columns = ['votes_RF'])
df_select_votes_RF = df_RF.loc[df_RF['votes_RF'] >= 13]

####################################################################################
####################################################################################

#SVM

GENERAL_TOT_LIST_NAME_SVM= []
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

    
    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
    GENERAL_TOT_LIST_NAME_SVM.append(tot_list_name)
        
    
    
GENERAL_TOT_LIST_NAME_flat_SVM = [item for sublist in GENERAL_TOT_LIST_NAME_SVM for item in sublist]

D_SVM = Counter(GENERAL_TOT_LIST_NAME_flat_SVM)
df_SVM = pd.DataFrame.from_dict(D_SVM, orient='index', columns = ['votes_SVM'])
df_select_votes_SVM = df_SVM.loc[df_SVM['votes_SVM'] >= 13]

####################################################################################
####################################################################################


df_tot = pd.concat([df_select_votes_MI, df_select_votes_RF, df_select_votes_SVM], axis=1, sort=False)

sns.countplot(df_tot)




