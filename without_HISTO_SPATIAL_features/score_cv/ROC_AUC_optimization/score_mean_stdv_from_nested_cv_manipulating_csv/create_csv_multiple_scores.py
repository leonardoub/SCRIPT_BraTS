#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:48:35 2020

@author: leonardo
"""


import os
import pandas as pd


path = '/home/leonardo/Scrivania/result_brats/result_CV/nested_CV/data_without_HISTO_SPATIAL/ROC_AUC_optimization/*/*.csv'



my_dict = {'ACC_TEST_MEAN': [],
           'ACC_TEST_STD': [], 
           'BAL_ACC_TEST_MEAN': [],
           'BAL_ACC_TEST_STD': [],
           'ROC_AUC_TEST_MEAN': [],
           'ROC_AUC_TEST_STD': []}


clf_list = []

import glob
for name in sorted(glob.glob(path)):
    print(name)
    data = pd.read_csv(name) 
    clf = os.path.split(name)[-1]
    clf = clf[12:-4]
#    my_dict['SCALER'].append(data['SCALER'][0])
    my_dict['ACC_TEST_MEAN'].append(data['test_accuracy_MEAN'][0])
    my_dict['ACC_TEST_STD'].append(data['test_accuracy_STD'][0])
    my_dict['BAL_ACC_TEST_MEAN'].append(data['test_balanced_accuracy_MEAN'][0])
    my_dict['BAL_ACC_TEST_STD'].append(data['test_balanced_accuracy_STD'][0])
    my_dict['ROC_AUC_TEST_MEAN'].append(data['test_ROC_AUC_score_MEAN'][0])
    my_dict['ROC_AUC_TEST_STD'].append(data['test_ROC_AUC_score_STD'][0])
    
    
    clf_list.append(clf)
 
    
              
df = pd.DataFrame(my_dict, index=clf_list)

df.to_csv('/home/leonardo/Scrivania/result_brats/score_ROC_AUC_optimization_using_all_HP_sets_USING_MEAN/summary_scores_all_HP_sets.csv')



#
#a = os.path.split(name)[-1]
#a = a[12:-4]
#
#b = data['accuracy_train_mean'][0]
#
#
#my_dict = {'ACC_TRAIN_MEAN': [data['accuracy_train_mean'][0]],
#           'ACC_TRAIN_STD': [data['accuracy_train_std'][0]], 
#           'ACC_TEST_MEAN': [data['accuracy_test_mean'][0]],
#           'ACC_Test_std': [data['accuracy_test_std'][0]]}
#
#my_dict['ACC_TRAIN_MEAN'].append(3)
#
#c=pd.DataFrame(my_dict, index=[a])
#c.index.name = 'classifier'



#in caso dovessi concatenare delle colonne con dimensioni diverse conviene fare i
#dataframe di ogni colonna e poi concatenarli usando 
#df_tot = pd.concat([df_1, df_2, df_3, df_4, df_5], axis=1)

