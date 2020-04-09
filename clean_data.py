#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:44:08 2020

@author: leonardo
"""

import pandas as pd



path_GBM = '/home/leonardo/Scrivania/BRATS_data/TCGA_GBM_radiomicFeatures.csv'
path_LGG = '/home/leonardo/Scrivania/BRATS_data/TCGA_LGG_radiomicFeatures.csv'


#read csv
data_GBM = pd.read_csv(path_GBM) 
data_LGG = pd.read_csv(path_LGG) 


features_GBM = list(data_GBM.columns)
features_LGG = list(data_LGG.columns)
##features_GBM==features_LGG hanno le stesse features

#add histology
data_GBM['Histology'] = 'GBM'
data_LGG['Histology'] = 'LGG'


#le features a partire da TGM_Cog_X_2 sono quasi tutte nulle, posso toglierle
#tolgo da TGM_Cog_X_2 fino a TGM_T_6 comprese
index_1=features_GBM.index('TGM_Cog_X_2')
index_2=features_GBM.index('TGM_T_6')


data_GBM=data_GBM.drop(data_GBM.iloc[:, index_1:index_2+1], inplace = False, axis = 1) 
data_LGG=data_LGG.drop(data_LGG.iloc[:, index_1:index_2+1], inplace = False, axis = 1) 



#trovare le features che contengono almeno un Nan
fetaures_with_NAN_GBM = data_GBM.columns[data_GBM.isna().any()].tolist() #sono 0
fetaures_with_NAN_LGG = data_LGG.columns[data_LGG.isna().any()].tolist() #sono 444


#togliere le features che sono nulle per tutti i patterns, in pratica le features che
#fanno parte della tabella ma non sono state raccolte
data_without_features_all_nan_GBM = data_GBM.dropna(axis=1, how='all')
data_without_features_all_nan_LGG = data_LGG.dropna(axis=1, how='all')
##non ce ne sono

#lasciamo solo i patterns che non contengono nessuna Nan
data_patterns_without_NAN_GBM = data_GBM.dropna(axis=0) #sono 102/102
data_patterns_without_NAN_LGG = data_LGG.dropna(axis=0) #sono 44/65



#merging datasetd
data = pd.concat([data_GBM, data_LGG], ignore_index=True)
#data.to_csv('/home/leonardo/Scrivania/BRATS_data/data_with_histologies.csv')


data_without_nan = pd.concat([data_patterns_without_NAN_GBM, data_patterns_without_NAN_LGG], ignore_index=True)
#data_without_nan.to_csv('/home/leonardo/Scrivania/BRATS_data/data_without_NAN_with_histologies.csv')


