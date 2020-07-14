#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:45:20 2020

@author: leonardo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:28:07 2020

@author: leonardo
"""



import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter



GENERAL_TOT_LIST_NAME = []

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

    #creare lista tutti nomi
    #unire liste eliminando gli elementi ripetuti
    
    tot_list_name = name_feat_TR_set_1 + name_feat_TR_set_2 + name_feat_TR_set_3 + name_feat_TR_set_4 + name_feat_TR_set_5
    GENERAL_TOT_LIST_NAME.append(tot_list_name)
    
    
    
GENERAL_TOT_LIST_NAME_flat = [item for sublist in GENERAL_TOT_LIST_NAME for item in sublist]


D = Counter(GENERAL_TOT_LIST_NAME_flat)


df = pd.DataFrame.from_dict(D, orient='index', columns = ['votes'])
  

df_select_votes = df.loc[df['votes'] >= 15]

    
fig, ax1 = plt.subplots()

df_select_votes.plot(ax=ax1, kind='bar', stacked=False, width=0.5, fontsize=6)


#ax1.set_xticklabels(ax1.get_xticklabels(), rotation=85)
ax1.set_yticks(np.arange(0, 26, 1))

ax1.set_xlabel('Features', labelpad=0)
ax1.set_ylabel('Votes', labelpad=10)
  
#    
#dict_features = {k:0 for k in tot_list_name}
#dict_feat_TR_set_1 = {k:0 for k in tot_list_name}
#dict_feat_TR_set_2 = {k:0 for k in tot_list_name}
#dict_feat_TR_set_3 = {k:0 for k in tot_list_name}
#dict_feat_TR_set_4 = {k:0 for k in tot_list_name}
#dict_feat_TR_set_5 = {k:0 for k in tot_list_name}
#    
#    
#D = Counter(GENERAL_TOT_LIST_NAME_flat)
#    
#    
#    for k in dict_features.keys():
#        
#        if k in name_feat_TR_set_1:
#            dict_feat_TR_set_1[k] = 1
#            
#        if k in name_feat_TR_set_2:
#            dict_feat_TR_set_2[k] = 1
#            
#        if k in name_feat_TR_set_3:
#            dict_feat_TR_set_3[k] = 1
#            
#        if k in name_feat_TR_set_4:
#            dict_feat_TR_set_4[k] = 1     
#            
#        if k in name_feat_TR_set_5:
#            dict_feat_TR_set_5[k] = 1    
#
#    
#    
#    
#    
#    #CREANDO UN DATAFRAME E USANDO PANDAS
#    
#    my_dict = {'Train_set_1':dict_feat_TR_set_1,
#               'Train_set_2':dict_feat_TR_set_2,
#               'Train_set_3':dict_feat_TR_set_3,
#               'Train_set_4':dict_feat_TR_set_4,
#               'Train_set_5':dict_feat_TR_set_5}
#    
#    
#    df = pd.DataFrame(my_dict)
#    
#    fig, ax1 = plt.subplots()
#    
#    df.plot(ax=ax1, kind='bar', stacked=True, width=0.7, fontsize=6)
#    
#    
##    ax1.set_xticklabels(rotation=45)
#    ax1.set_yticks(np.arange(0, 6, 1))
#    
#    ax1.set_xlabel('Features', labelpad=0)
#    ax1.set_ylabel('Votes', labelpad=10)
#    
#    #ax1.set_title(f'Features importance Random Forest RSoKF {2*i}')
#    
#    
#    #fig.show()
#    
#    
#    fig.set_figwidth(8)
#    fig.set_figheight(6)
#    
#    
#    plt.subplots_adjust(left=0.125, bottom=0.35, right=0.9, top=0.95, wspace=0, hspace=0)
#    
#    
#    #create folder and save
#    
#    outname = f'bar_stacked_plot_RF.png'
#    
#    outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/BRATS/without_HISTO_SPATIAL_INTENSITY/RSoKF_{2*i}/'
#    if not os.path.exists(outdir):
#        os.makedirs(outdir)
#    
#    fullname = os.path.join(outdir, outname)    
#    
#    plt.savefig(fullname)
#    plt.close()
#
#
#
#
#
#













