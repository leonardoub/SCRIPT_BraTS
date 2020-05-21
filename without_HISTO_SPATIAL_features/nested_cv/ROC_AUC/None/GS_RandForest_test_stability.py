#Cross Validation on RandomForestClassifier for classification

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
from sklearn.ensemble import RandomForestClassifier
import load_data
import save_output
import nested_cv
import save_features_selected_RF

name = 'RandomForest'
dim_reduction = 'NONE'

#load data

public_data, public_labels = load_data.function_load_data()
tot_features = public_data.columns

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

# Designate distributions to sample hyperparameters from 
n_tree = [10, 30, 50, 70, 100, 250]
depth = [10, 25, 50, 75, 100, None]


#RandomForestClassifier
steps = [('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=503))]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test,
            'clf__n_estimators':list(n_tree), 'clf__criterion':['gini', 'entropy'], 
            'clf__max_depth':depth, 'clf__min_samples_split':[2, 5, 10], 
            'clf__min_samples_leaf':[1, 2, 4], 'clf__class_weight':[None, 'balanced']}]



for j in range(1,6):
    results, best_estimators_dict = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres, j*2)

    #save best features svm
    save_features_selected_RF.function_save_features_selected_RF(best_estimators_dict, tot_features, name, 'VARIABLE', 'NONE', 'BEST_HP', j*2)

    #create folder and save
    save_output.function_save_output(results, dim_reduction, name, j*2)
