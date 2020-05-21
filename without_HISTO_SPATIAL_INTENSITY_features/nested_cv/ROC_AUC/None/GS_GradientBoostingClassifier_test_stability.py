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
from sklearn.ensemble import GradientBoostingClassifier
import load_data
import save_output
import nested_cv

name = 'GradientBoosting'
dim_reduction = 'NONE'

#load data

public_data, public_labels = load_data.function_load_data()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

# Designate distributions to sample hyperparameters from 
n_tree = [10, 30, 50, 70, 100, 250, 500, 1000]
depth = [3, 6, 10, 25, 50, 75]
lr = [0.01, 0.1, 0.50, 1.0]


#RandomForestClassifier
steps = [('scaler', StandardScaler()), ('clf', GradientBoostingClassifier(random_state=503))]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'clf__learning_rate':lr,
            'clf__n_estimators':list(n_tree),  
            'clf__max_depth':depth, 'clf__min_samples_split':[2, 5, 10], 
            'clf__min_samples_leaf':[1, 2, 4]}]



for j in range(1,6):
    results, best_estimators_dict = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres, j*2)

    #create folder and save

    save_output.function_save_output(results, dim_reduction, name, j*2)
