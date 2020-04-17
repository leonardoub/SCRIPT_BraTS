#Cross Validation on SVM for classification

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
import load_data
import save_output
import nested_cv

name = 'svm_lin_MMS'
dim_reduction = 'NONE'

#load data

public_data, public_labels = load_data.function_load_data()

#Scalers
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [RobustScaler(), MinMaxScaler()]

#Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, dtype=float))


#SVM
steps = [('scaler', MinMaxScaler()), ('clf', SVC(kernel='linear', random_state=503))]

pipeline = Pipeline(steps)


parameteres = [{'scaler':[MinMaxScaler()], 'red_dim':[None], 
                     'clf__C':list(C_range), 'clf__class_weight':[None, 'balanced']}]

results = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, dim_reduction, name)


