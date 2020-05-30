#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:40:31 2020

@author: leonardo
"""



import os
import pandas as pd


for j in range(1,6):

    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /MUTUAL_INFO_best_features/AdaBoost/AdaBoost_features_selected_MUTUAL_INFO_RSoKF_{2*j}.csv'

  
    data = pd.read_csv(path) 
    data_1 = data.drop(['Unnamed: 0'], axis=1)  

    
    TOT_DICT={}
    
    for i in range(1, 6):
    
        for name_feat, value in zip(data_1[f'FOLD_{i}_MUTUAL_INFO_FEATURES'], data_1[f'FOLD_{i}_value']):
            
            if name_feat in TOT_DICT:   
                TOT_DICT[name_feat].append(value)
                
            else:
                TOT_DICT[name_feat] = [value]
    
    
    TOT_DICT_SUM ={k:sum(v) for k,v in TOT_DICT.items()}
    
    
    df_rank_best_feat = pd.DataFrame.from_dict(TOT_DICT_SUM, orient='index', columns=['total_score'] )
    
    df_rank_best_feat_sorted = df_rank_best_feat.sort_values(by='total_score', ascending=False)

   
    outname = f'summary_feature_MUTUAL_INFO_RSoKF_{2*j}.csv'
    
    outdir = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /MUTUAL_INFO_best_features/AdaBoost/MUTUAL_INFO_summary/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    fullname = os.path.join(outdir, outname)    
    df_rank_best_feat_sorted.to_csv(fullname)


    df_rank_best_feat_sorted.to_csv





















