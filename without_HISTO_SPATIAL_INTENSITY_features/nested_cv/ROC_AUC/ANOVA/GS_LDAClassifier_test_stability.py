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
from sklearn.feature_selection import f_classif
import save_features_selected_ANOVA

name = 'LDAClassifier'
dim_reduction = 'ANOVA'

#load data

public_data, public_labels = load_data.function_load_data()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

#LinearDiscriminantAnalysis
steps = [('scaler', MinMaxScaler()), ('red_dim', SelectPercentile(f_classif, percentile=10)), ('clf', LinearDiscriminantAnalysis())]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'clf__solver':['svd']},
              {'scaler':scalers_to_test, 'clf__solver':['lsqr', 'eigen'], 'clf__shrinkage':['auto', None]}]


results, dict_best_estimators = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres)

#create and save file best features ANOVA

tot_features = public_data.columns
save_features_selected_ANOVA.function_save_features_selected_ANOVA(dim_reduction, name, tot_features, dict_best_estimators)

#create folder and save

save_output.function_save_output(results, dim_reduction, name)

