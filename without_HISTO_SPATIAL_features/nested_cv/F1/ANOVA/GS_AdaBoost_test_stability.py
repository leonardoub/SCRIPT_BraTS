#Cross Validation on AdaBoostClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import load_data
import save_output
import nested_cv
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
import save_features_selected_ANOVA

name = 'AdaBoost'
dim_reduction = 'ANOVA'

#load data

public_data, public_labels = load_data.function_load_data()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

# Designate distributions to sample hyperparameters from 
n_features_to_test = [10]
n_estimators = [10, 30, 50, 70, 100, 150]
lr = [0.001, 0.01, 0.1, 0.50, 1.0]

#AdaBoost
steps = [('scaler', MinMaxScaler()), ('red_dim', SelectPercentile(f_classif, percentile=10)), ('clf', AdaBoostClassifier(random_state=503))]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 
                     'clf__base_estimator': [DecisionTreeClassifier(max_depth = j) for j in range(1,6)],                     
                     'clf__n_estimators':n_estimators, 'clf__learning_rate':lr, 'clf__algorithm':['SAMME', 'SAMME.R']}]


results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres)

#create and save file best features ANOVA

tot_features = public_data.columns
save_features_selected_ANOVA.function_save_features_selected_ANOVA(dim_reduction, name, tot_features, dict_best_estimators)

#create folder and save

save_output.function_save_output(results, dim_reduction, name)

