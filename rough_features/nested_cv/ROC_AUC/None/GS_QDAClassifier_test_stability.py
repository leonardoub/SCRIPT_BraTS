#Cross Validation on QuadraticDiscriminantAnalysis for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
import load_data
import save_output
import nested_cv

name = 'QDAClassifier'
dim_reduction = 'NONE'


#load data

public_data, public_labels = load_data.function_load_data()


#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]


#QuadraticDiscriminantAnalysis
steps = [('scaler', MinMaxScaler()), ('clf', QuadraticDiscriminantAnalysis())]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test}]


results = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, dim_reduction, name)