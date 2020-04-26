#Cross Validation on RadiusNeighborsClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
import load_data
import save_output
import nested_cv


name = 'RadiusNeighbors'
dim_reduction = 'NONE'

#load data

public_data, public_labels = load_data.function_load_data()

encoder = LabelEncoder()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]


# Designate distributions to sample hyperparameters from 
R = np.arange(0.1, 10, 0.2) 
n_features_to_test = np.arange(1, 11)


#RadiusNeighbors
steps = [('scaler', MinMaxScaler()), ('clf', RadiusNeighborsClassifier(outlier_label='most_frequent'))]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'clf__radius':R, 
              'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]


results = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, dim_reduction, name)