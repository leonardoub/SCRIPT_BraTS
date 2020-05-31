#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:06:28 2020

@author: leonardo
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


path_summary_MI_features = '/home/leonardo/Scrivania/result_brats/05_30/important_features_05_30/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /MUTUAL_INFO_best_features/AdaBoost/MUTUAL_INFO_summary/summary_feature_MUTUAL_INFO_RSoKF_2.csv'

data_feat_MI = pd.read_csv(path_summary_MI_features, index_col=0) 


x = np.array([1, 2, 3, 4, 5])
y = data_feat_MI['total_score']
y_selected = y.iloc[0:5] 




fig, ax1 = plt.subplots()
ax1.bar(x, y_selected, width=0.5)




