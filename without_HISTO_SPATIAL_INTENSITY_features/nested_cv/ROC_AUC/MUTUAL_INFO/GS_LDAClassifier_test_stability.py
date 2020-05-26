#Cross Validation on LinearDiscriminantAnalysisClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

name = 'LDAClassifier'
dim_reduction = 'MUTUAL_INFO'

#load data

public_data, public_labels = load_data.function_load_data()
tot_features = public_data.columns

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]


score_func = mutual_info_classif(random_state=5)


#LinearDiscriminantAnalysis
steps = [('scaler', MinMaxScaler()), ('red_dim', SelectPercentile(score_func, percentile=10)), ('clf', LinearDiscriminantAnalysis())]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'clf__solver':['svd']},
              {'scaler':scalers_to_test, 'clf__solver':['lsqr', 'eigen'], 'clf__shrinkage':['auto', None]}]



for j in range(1,6):
    results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres, j*2)

    #create and save file best features MUTUAL_INFO
    save_features_selected_MUTUAL_INFO.function_save_features_selected_MUTUAL_INFO(dim_reduction, name, tot_features, dict_best_estimators, j*2)

 
    #create folder and save
    save_output.function_save_output(results, dim_reduction, name, j*2)

