#Cross Validation on KNeighborsClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
import load_data
import save_output
import nested_cv
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
import save_features_selected_ANOVA

name = 'KNeighbors'
dim_reduction = 'ANOVA'

#load data

public_data, public_labels = load_data.function_load_data()
tot_features = public_data.columns

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
k = np.arange(1,11)

#KNeighborsClassifier
steps = [('scaler', MinMaxScaler()), ('red_dim', SelectPercentile(f_classif, percentile=10)), ('clf', KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'clf__n_neighbors':k, 
              'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]



for j in range(1,6):
    results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres, j*2)

    #create and save file best features ANOVA
    save_features_selected_ANOVA.function_save_features_selected_ANOVA(dim_reduction, name, tot_features, dict_best_estimators, j*2)

    #create folder and save
    save_output.function_save_output(results, dim_reduction, name, j*2)

