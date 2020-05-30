#Cross Validation on SVM for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
import load_data
import save_output
import nested_cv
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
import save_features_selected_MUTUAL_INFO


name_1 = 'svm_lin_MMS'
name_2 = 'svm_lin_RBTS'
name_3 = 'svm_lin_STDS'
dim_reduction = 'MUTUAL_INFO'


#load data

public_data, public_labels = load_data.function_load_data()
tot_features = public_data.columns

#Scalers
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [RobustScaler(), MinMaxScaler()]

#Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, dtype=float))
n_features_to_test = [0.85, 0.9, 0.95]


#SVM
steps = [('scaler', MinMaxScaler()), ('red_dim', SelectPercentile(mutual_info_classif, percentile=10)), ('clf', SVC(kernel='linear', probability=True, random_state=503))]

pipeline = Pipeline(steps)



#MMS
parameteres_1 = [{'scaler':[MinMaxScaler()],
              'clf__C':list(C_range), 'clf__class_weight':[None, 'balanced']}]
              
for j in range(1,6):
    results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres_1, j*2)

    #create and save file best features MUTUAL_INFO
    save_features_selected_MUTUAL_INFO.function_save_features_selected_MUTUAL_INFO(dim_reduction, name_1, tot_features, dict_best_estimators, j*2)

 
    #create folder and save
    save_output.function_save_output(results, dim_reduction, name_1, j*2)



#RBTS
parameteres_2 = [{'scaler':[RobustScaler()], 
              'clf__C':list(C_range), 'clf__class_weight':[None, 'balanced']}]
              
for j in range(1,6):
    results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres_2, j*2)

    #create and save file best features MUTUAL_INFO
    save_features_selected_MUTUAL_INFO.function_save_features_selected_MUTUAL_INFO(dim_reduction, name_2, tot_features, dict_best_estimators, j*2)

 
    #create folder and save
    save_output.function_save_output(results, dim_reduction, name_2, j*2)



#STDS
parameteres_3 = [{'scaler':[StandardScaler()], 
              'clf__C':list(C_range), 'clf__class_weight':[None, 'balanced']}]
              
for j in range(1,6):
    results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres_3, j*2)

    #create and save file best features MUTUAL_INFO
    save_features_selected_MUTUAL_INFO.function_save_features_selected_MUTUAL_INFO(dim_reduction, name_3, tot_features, dict_best_estimators, j*2)

 
    #create folder and save
    save_output.function_save_output(results, dim_reduction, name_3, j*2)
