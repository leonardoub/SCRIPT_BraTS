#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:20:40 2020

@author: leonardo
"""

import pandas as pd


path_data_without_NAN_without_HISTO = '/home/leonardo/Scrivania/BRATS_data/data_without_NAN_without_HISTO_without_SPATIAL_with_histologies.csv'


#read csv
data_without_NAN_without_HISTO = pd.read_csv(path_data_without_NAN_without_HISTO) 




#TOGLIERE LE FEATURES 'HISTO'

features_list = list(data_without_NAN_without_HISTO.columns)
index_1 = features_list.index('INTENSITY_Mean_ET_T1Gd')
index_2 = features_list.index('INTENSITY_STD_ED_FLAIR')


data_without_NAN_without_HISTO_without_SPATIAL = data_without_NAN_without_HISTO.drop(data_without_NAN_without_HISTO.iloc[:, index_1:index_2+1], inplace = False, axis = 1) 
data_without_NAN_without_HISTO_without_SPATIAL.to_csv('/home/leonardo/Scrivania/BRATS_data/data_without_NAN_without_HISTO_without_SPATIAL_without_INTENSITY_with_histologies.csv', index=False)
