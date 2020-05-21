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
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
import save_features_selected_ANOVA
name = 'GradientBoosting'
dim_reduction = 'ANOVA'

#load data

public_data, public_labels = load_data.function_load_data()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

# Designate distributions to sample hyperparameters from 
n_tree = [10, 30, 50, 70, 100, 250, 500, 1000]
depth = [3, 6, 10, 25, 50, 75]
lr = [0.01, 0.1, 0.50, 1.0]
n_features_to_test = [10]


#RandomForestClassifier
steps = [('scaler', StandardScaler()), ('red_dim', SelectPercentile(f_classif, percentile=10)), ('clf', GradientBoostingClassifier(random_state=503))]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'clf__learning_rate':lr,
            'clf__n_estimators':list(n_tree),  
            'clf__max_depth':depth, 'clf__min_samples_split':[2, 5, 10], 
            'clf__min_samples_leaf':[1, 2, 4]}]



results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, dim_reduction, name)
