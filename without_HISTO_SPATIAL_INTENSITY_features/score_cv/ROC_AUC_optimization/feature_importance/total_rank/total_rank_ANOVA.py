#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:24:53 2020

@author: leonardo
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


TOT_DICT={}


for i in range(1,6):

    path_summary_ANOVA_features = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /ANOVA_best_features/AdaBoost/ANOVA_summary/summary_feature_ANOVA_RSoKF_{2*i}.csv'
    data_feat_ANOVA = pd.read_csv(path_summary_ANOVA_features, index_col=0) 
   
   
    for name_feat, value in zip(data_feat_ANOVA.index, data_feat_ANOVA[f'total_score']):
            
        if name_feat in TOT_DICT:   
            TOT_DICT[name_feat].append(value)
                
        else:
            TOT_DICT[name_feat] = [value]
    
    

TOT_DICT_SUM ={k:sum(v) for k,v in TOT_DICT.items()}


    
df_tot_rank_best_feat = pd.DataFrame.from_dict(TOT_DICT_SUM, orient='index', columns=['total_score'] )

df_tot_rank_best_feat_sorted = df_tot_rank_best_feat.sort_values(by='total_score', ascending=True)

   
outname = f'rank_tot_feature_ANOVA.csv'

outdir = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /ANOVA_best_features/AdaBoost/ANOVA_rank_tot'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    
df_tot_rank_best_feat_sorted.to_csv(fullname)


df_tot_rank_best_feat_sorted.to_csv



