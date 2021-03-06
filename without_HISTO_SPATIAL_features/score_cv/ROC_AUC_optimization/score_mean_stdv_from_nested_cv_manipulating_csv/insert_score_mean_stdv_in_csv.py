#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:42:00 2020

@author: leonardo
"""

import pandas as pd

path = '/home/leonardo/Scrivania/result_brats/05_30/result_CV_05_30/nested_CV/data_without_HISTO_SPATIAL/ROC_AUC_optimization/*/RS_*/*.csv'



import glob
for name in glob.glob(path):
    print(name)
    data = pd.read_csv(name) 

    acc_test_mean = data['outer_loop_accuracy_scores'].mean()
    acc_test_std = data['outer_loop_accuracy_scores'].std()


    bal_acc_test_mean = data['outer_loop_balanced_accuracy_scores'].mean()
    bal_acc_test_std = data['outer_loop_balanced_accuracy_scores'].std()
    
    
    ROC_AUC_test_mean = data['outer_loop_roc_auc_scores_predict_proba'].mean()
    ROC_AUC_test_std = data['outer_loop_roc_auc_scores_predict_proba'].std()

    
    
    
    df_test_acc_mean = pd.DataFrame([{'test_accuracy_MEAN':acc_test_mean}])
    df_test_acc_std = pd.DataFrame([{'test_accuracy_STD':acc_test_std}])


    df_test_bal_acc_mean = pd.DataFrame([{'test_balanced_accuracy_MEAN':bal_acc_test_mean}])
    df_test_bal_acc_std = pd.DataFrame([{'test_balanced_accuracy_STD':bal_acc_test_std}])

    
    df_test_ROC_AUC_mean = pd.DataFrame([{'test_ROC_AUC_score_MEAN':ROC_AUC_test_mean}])
    df_test_ROC_AUC_std = pd.DataFrame([{'test_ROC_AUC_score_STD':ROC_AUC_test_std}])


    df = pd.concat([data, df_test_acc_mean, df_test_acc_std, df_test_bal_acc_mean, df_test_bal_acc_std, df_test_ROC_AUC_mean, df_test_ROC_AUC_std], axis=1)

    df.to_csv(name, index=False)


#data = pd.read_csv(name) 

#acc_train_mean = data['accuracy_train'].mean()

#data['accuracy_train'] = acc_train_mean




