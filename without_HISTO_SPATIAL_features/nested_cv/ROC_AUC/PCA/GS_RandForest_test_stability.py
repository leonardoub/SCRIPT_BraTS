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

name = 'RandomForest'
dim_reduction = 'PCA'

#load data

public_data, public_labels = load_data.function_load_data()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]


# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
n_tree = [10, 30, 50, 70, 100, 250]
depth = [10, 25, 50, 75, 100, None]

#RandomForestClassifier
steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', RandomForestClassifier(random_state=503))]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':list(n_features_to_test),
                'clf__n_estimators':list(n_tree), 'clf__criterion':['gini', 'entropy'], 
                'clf__max_depth':depth, 'clf__min_samples_split':[2, 5, 10], 
                'clf__min_samples_leaf':[1, 2, 4], 'clf__class_weight':[None, 'balanced']}]

    
results = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, dim_reduction, name)
