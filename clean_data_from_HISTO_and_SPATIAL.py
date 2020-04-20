#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:46:32 2020

@author: leonardo
"""

import pandas as pd


path_data_without_NAN_without_HISTO = '/home/leonardo/Scrivania/BRATS_data/data_without_NAN_without_HISTO_with_histologies.csv'


#read csv
data_without_NAN_without_HISTO = pd.read_csv(path_data_without_NAN_without_HISTO) 




#TOGLIERE LE FEATURES 'HISTO'

features_list = list(data_without_NAN_without_HISTO.columns)
index_1 = features_list.index('SPATIAL_Frontal')
index_2 = features_list.index('SPATIAL_Brain_stem')


data_without_NAN_without_HISTO_without_SPATIAL = data_without_NAN_without_HISTO.drop(data_without_NAN_without_HISTO.iloc[:, index_1:index_2+1], inplace = False, axis = 1) 
data_without_NAN_without_HISTO_without_SPATIAL.to_csv('/home/leonardo/Scrivania/BRATS_data/data_without_NAN_without_HISTO_without_SPATIAL_with_histologies.csv', index=False)
