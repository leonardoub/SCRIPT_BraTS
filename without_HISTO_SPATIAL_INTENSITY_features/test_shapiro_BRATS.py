#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:05:12 2020

@author: leonardo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy




dataset_path = '/home/leonardo/Scrivania/BRATS_data/data_without_NAN_without_HISTO_without_SPATIAL_without_INTENSITY_with_histologies.csv'



df_data = pd.read_csv(dataset_path)

data_1 = df_data.drop(['ID', 'Date'], axis=1)

data = df_data.drop(['Histology', 'ID', 'Date'], axis=1)


#Shapiro test for normality

from scipy.stats import shapiro

p_value_list_train=[]

dict_normal_features = {}
normal_features = []

for column in data.columns:
  stat, p_value = shapiro(data[column])
  p_value_list_train.append(p_value)


  if p_value > 0.05/550:
    print(stat, p_value, column)
    normal_features.append(column)

    dict_normal_features.update({column : p_value})

#130 features su 554 distribuite normalmente