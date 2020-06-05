#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:43:16 2020

@author: leonardo
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter

scaler_svml_list = ['MMS', 'RBTS', 'STDS']

for i in range(1,6):
        
    for scaler_svml in scaler_svml_list: 
        
        path_summary_ANOVA_features = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /ANOVA_best_features/AdaBoost/ANOVA_summary/summary_feature_ANOVA_RSoKF_{2*i}.csv'
        data_feat_ANOVA = pd.read_csv(path_summary_ANOVA_features, index_col=0) 
        
        path_summary_MI_features = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /MUTUAL_INFO_best_features/AdaBoost/MUTUAL_INFO_summary/summary_feature_MUTUAL_INFO_RSoKF_{2*i}.csv'
        data_feat_MI = pd.read_csv(path_summary_MI_features, index_col=0) 
        
        path_summary_RF_features = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_RandomForest/RF_summary/summary_feature_RF_RSoKF{2*i}.csv'
        data_feat_RF = pd.read_csv(path_summary_RF_features, index_col=0) 
        
        path_summary_svml_MMS_features = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_svm_lin/svml_summary_{scaler_svml}/summary_feature_svml_{scaler_svml}_RSoKF_{2*i}.csv'
        data_feat_svml_MMS = pd.read_csv(path_summary_svml_MMS_features, index_col=0) 
        
        
        
        #x_A = np.array([1, 2, 3, 4, 5])
        y_A = data_feat_ANOVA['total_score']
        y_A_selected = list(y_A.iloc[0:15].index)
        
        
        #x_MI = np.array([1, 2, 3, 4, 5])
        y_MI = data_feat_MI['total_score']
        y_MI_selected = list(y_MI.iloc[0:15].index)
        
        
        #x_RF = np.array([1, 2, 3, 4, 5])
        y_RF = data_feat_RF['total_score']
        y_RF_selected = list(y_RF.iloc[0:15].index) 
        
        
        #x_svml_MMS = np.array([1, 2, 3, 4, 5])
        y_svml_MMS = data_feat_svml_MMS['total_score']
        y_svml_MMS_selected = list(y_svml_MMS.iloc[0:15].index) 
        
        
        
        #unire liste eliminando gli elementi ripetuti
        
        tot_list = y_A_selected + y_MI_selected + y_RF_selected + y_svml_MMS_selected
        tot_list_no_duplicates = list(set(tot_list))
        N = len(tot_list_no_duplicates)
        
        
        dict_features = {k:0 for k in tot_list}
        dict_ANOVA = {k:0 for k in tot_list}
        dict_MI = {k:0 for k in tot_list}
        dict_RF = {k:0 for k in tot_list}
        dict_svml_MMS = {k:0 for k in tot_list}
        
        
        D = Counter(tot_list)
        
        
        for k in dict_features.keys():
            if k in y_A_selected:
                dict_ANOVA[k] = 1
                
            if k in y_MI_selected:
                dict_MI[k] = 1
        
            if k in y_RF_selected:
                dict_RF[k] = 1
                
            if k in y_svml_MMS_selected:
                dict_svml_MMS[k] = 1        
        
        
        
        
        
        #CREANDO UN DATAFRAME E USANDO PANDAS
        
        my_dict = {'Anova':dict_ANOVA,
                   'Mutual_Information': dict_MI,
                   'Random_Forest':dict_RF,
                   f'SVM_lin_{scaler_svml}':dict_svml_MMS}
        
        
        df = pd.DataFrame(my_dict)
        
        fig, ax1 = plt.subplots()
        
        df.plot(ax=ax1, kind='bar', stacked=True, width=0.7, fontsize=6)
        
        ax1.set_yticks(np.arange(0, 5, 1))
        
        ax1.set_xlabel('Features', labelpad=0)
        ax1.set_ylabel('Votes', labelpad=10)
        
        ax1.set_title(f'Features importance RSoKF {2*i}')
        
        
        #fig.show()
        
        
        fig.set_figwidth(8)
        fig.set_figheight(6)
        
        
        plt.subplots_adjust(left=0.125, bottom=0.35, right=0.9, top=0.95, wspace=0, hspace=0)
        
        
        #create folder and save
        
        outname = f'general_rank_features_importance_with_svml_{scaler_svml}.png'
        
        outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original/without_HISTO_SPATIAL_INTENSITY/RSoKF_{2*i}/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        fullname = os.path.join(outdir, outname)    
        
        plt.savefig(fullname)
        plt.close()
    
    
    
    
    
    




#TROPPO COMPLICATO CON MATPLOTLIB
#h_ANOVA = list(dict_ANOVA.values())
#h_MI = list(dict_MI.values())
#h_RF = list(dict_RF.values())
#h_svml_MMS = list(dict_svml_MMS.values())
#
#
#ind = np.arange(N)
#
#fig, ax1 = plt.subplots()
#
#
#ax1.bar(ind, h_ANOVA, width=0.5)
#
#ax1.bar(ind, h_MI, width=0.5, bottom=h_ANOVA)
#
#ax1.bar(ind, h_RF, width=0.5, bottom=h_MI)
#
#ax1.bar(ind, h_svml_MMS, width=0.5, bottom=h_RF)

