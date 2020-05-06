#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:48:12 2020

@author: leonardo
"""



import os
import pandas as pd


path_ANOVA_features = '/home/leonardo/Scrivania/result_brats/result_CV/nested_CV/data_without_HISTO_SPATIAL/ROC_AUC_optimization/*/*features_selected_ANOVA'


import glob
for folder in sorted(glob.glob(path_ANOVA_features)):
    print(folder)
    
    D={'FOLD_1_FEATURES':[], 'FOLD_1_pvalue':[],
       'FOLD_2_FEATURES':[], 'FOLD_2_pvalue':[],
       'FOLD_3_FEATURES':[], 'FOLD_3_pvalue':[],
       'FOLD_4_FEATURES':[], 'FOLD_4_pvalue':[],
       'FOLD_5_FEATURES':[], 'FOLD_5_pvalue':[]}
    
    
    file_list = []
    
    for i, file in enumerate(sorted(os.listdir(folder)) ,1):
        
        file_path = os.path.join(folder, file)
        
        file_list.append(file_path)
        
        data = pd.read_csv(file_path) 
        D[f'FOLD_{i}_FEATURES'].append(data['features'])
        D[f'FOLD_{i}_pvalue'].append(data['p_value'])

    df = pd.DataFrame.from_dict(D)
        