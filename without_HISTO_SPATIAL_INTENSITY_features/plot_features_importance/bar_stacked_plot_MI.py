#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:11:33 2020

@author: leonardo
"""



import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter


for i in range(1,6):

    path = f'/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /MUTUAL_INFO_best_features/AdaBoost/AdaBoost_features_selected_MUTUAL_INFO_RSoKF_{2*i}.csv'

  
    data = pd.read_csv(path) 
    data_1 = data.drop(['Unnamed: 0'], axis=1)  


    name_feat_TR_set_1 = list(data['FOLD_1_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_2 = list(data['FOLD_2_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_3 = list(data['FOLD_3_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_4 = list(data['FOLD_4_MUTUAL_INFO_FEATURES'])[:20]
    name_feat_TR_set_5 = list(data['FOLD_5_MUTUAL_INFO_FEATURES'])[:20]

    #creare lista tutti nomi
    #unire liste eliminando gli elementi ripetuti
    
    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
    tot_list_name_no_duplicates = list(set(tot_list_name))
    N = len(tot_list_name_no_duplicates)



  
    
    dict_features = {k:0 for k in tot_list_name}
    dict_feat_TR_set_1 = {k:0 for k in tot_list_name}
    dict_feat_TR_set_2 = {k:0 for k in tot_list_name}
    dict_feat_TR_set_3 = {k:0 for k in tot_list_name}
    dict_feat_TR_set_4 = {k:0 for k in tot_list_name}
    dict_feat_TR_set_5 = {k:0 for k in tot_list_name}
    
    
#    D = Counter(tot_list_name)
    
    
    for k in dict_features.keys():
        
        if k in name_feat_TR_set_1:
            dict_feat_TR_set_1[k] = 1
            
        if k in name_feat_TR_set_2:
            dict_feat_TR_set_2[k] = 1
            
        if k in name_feat_TR_set_3:
            dict_feat_TR_set_3[k] = 1
            
        if k in name_feat_TR_set_4:
            dict_feat_TR_set_4[k] = 1     
            
        if k in name_feat_TR_set_5:
            dict_feat_TR_set_5[k] = 1    

    
    
    
    
    #CREANDO UN DATAFRAME E USANDO PANDAS
    
    my_dict = {'Train_set_1':dict_feat_TR_set_1,
               'Train_set_2':dict_feat_TR_set_2,
               'Train_set_3':dict_feat_TR_set_3,
               'Train_set_4':dict_feat_TR_set_4,
               'Train_set_5':dict_feat_TR_set_5}
    
    
    df = pd.DataFrame(my_dict)
    
    fig, ax1 = plt.subplots()
    
    df.plot(ax=ax1, kind='bar', stacked=True, width=0.7, fontsize=6)
    
    
#    ax1.set_xticklabels(rotation=45)
    ax1.set_yticks(np.arange(0, 6, 1))
    
    ax1.set_xlabel('Features', labelpad=0)
    ax1.set_ylabel('Votes', labelpad=10)
    
    #ax1.set_title(f'Features importance mutual information RSoKF {2*i}')
    
    
    #fig.show()
    
    
    fig.set_figwidth(8)
    fig.set_figheight(6)
    
    
    plt.subplots_adjust(left=0.125, bottom=0.35, right=0.9, top=0.95, wspace=0, hspace=0)
    
    
    #create folder and save
    
    outname = f'bar_stacked_plot_MI.png'
    
    outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/BRATS/without_HISTO_SPATIAL_INTENSITY/RSoKF_{2*i}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    fullname = os.path.join(outdir, outname)    
    
    plt.savefig(fullname)
    plt.close()






















#
#    
#    TOT_DICT={}
#    
#    for i in range(1, 6):
#    
#        for name_feat, value in zip(data_1[f'FOLD_{i}_ANOVA_FEATURES'], data_1[f'FOLD_{i}_p_value']):
#            
#            if name_feat in TOT_DICT:   
#                TOT_DICT[name_feat].append(value)
#                
#            else:
#                TOT_DICT[name_feat] = [value]
#    
#    
#    
#    
#    
#    
#    TOT_DICT_SUM ={k:len(v) for k,v in TOT_DICT.items()}
#    
#    
#    df_rank_best_feat = pd.DataFrame.from_dict(TOT_DICT_SUM, orient='index', columns=['total_score'] )
#    
#    df_rank_best_feat_sorted = df_rank_best_feat.sort_values(by='total_score', ascending=True)
#
#   