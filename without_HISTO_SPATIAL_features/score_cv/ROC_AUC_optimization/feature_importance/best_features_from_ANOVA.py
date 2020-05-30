#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:57:09 2020

@author: leonardo
"""



import os
import pandas as pd


for j in range(1,6):

    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL/score_ROC_AUC_optimization /ANOVA_best_features/AdaBoost/AdaBoost_features_selected_ANOVA_RSoKF_{2*j}.csv'

  
    data = pd.read_csv(path) 
    data_1 = data.drop(['Unnamed: 0'], axis=1)  

    
    TOT_DICT={}
    
    for i in range(1, 6):
    
        for name_feat, value in zip(data_1[f'FOLD_{i}_ANOVA_FEATURES'], data_1[f'FOLD_{i}_p_value']):
            
            if name_feat in TOT_DICT:   
                TOT_DICT[name_feat].append(value)
                
            else:
                TOT_DICT[name_feat] = [value]
    
    
    TOT_DICT_SUM ={k:sum(v) for k,v in TOT_DICT.items()}
    
    
    df_rank_best_feat = pd.DataFrame.from_dict(TOT_DICT_SUM, orient='index', columns=['total_score'] )
    
    df_rank_best_feat_sorted = df_rank_best_feat.sort_values(by='total_score', ascending=False)

   
    outname = f'summary_feature_RF_RSoKF_{2*j}.csv'
    
    outdir = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL/score_ROC_AUC_optimization /best_feature_importances_for_RandomForest/RF_summary/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    fullname = os.path.join(outdir, outname)    
    df_rank_best_feat_sorted.to_csv(fullname)


    df_rank_best_feat_sorted.to_csv























#
#
###PROTOTIPO PER UN SINGOLO FILE DI BEST FEATURES RF CHE HO PROVATO POI A GENERALIZZARE CREANDO LO SPRIPT SOPRA
#
#path = '/home/leonardo/Scrivania/result_brats/important_features_05_24/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_RandomForest/features_importance_for_RandomForest_scaler_VARIABLE_dim_red_NONE_BEST_HP_RSoKF_2.csv'
#
#
#data = pd.read_csv(path) 
#data_1 = data.drop(['Unnamed: 0'], axis=1)  
#
#
#
#TOT_DICT={}
#
#for i in range(1, 6):
#
#    for name_feat, value in zip(data_1[f'CLF_best_HP_FOLD_{i}_FEATURES'], data_1[f'FOLD_{i}_value']):
#        
#        if name_feat in TOT_DICT:   
#            TOT_DICT[name_feat].append(value)
#            
#        else:
#            TOT_DICT[name_feat] = [value]
#
#
#TOT_DICT_SUM ={k:sum(v) for k,v in TOT_DICT.items()}
#
#
#df_rank_best_feat = pd.DataFrame.from_dict(TOT_DICT_SUM, orient='index', columns=['total_score'] )
#
#df_rank_best_feat_sorted = df_rank_best_feat.sort_values(by='total_score', ascending=False)
#
#
#




