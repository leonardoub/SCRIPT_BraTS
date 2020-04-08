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

name = 'QDAClassifier'
dim_reduction = 'NONE'

#load data

public_data, public_labels = load_data.function_load_data()

encoder = LabelEncoder()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()


for i in range(1, 21):

       #Train test split
       X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, 
       stratify=public_labels, random_state=i*500)

       #Vettorizzare i label
       train_labels_encoded = encoder.fit_transform(y_train)
       test_labels_encoded = encoder.transform(y_test)

       #QuadraticDiscriminantAnalysis
       steps = [('scaler', MinMaxScaler()), ('clf', QuadraticDiscriminantAnalysis())]

       pipeline = Pipeline(steps)

       parameteres = [{'scaler':scalers_to_test}]

       grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5, n_jobs=-1, verbose=1)

       grid.fit(X_train, y_train)

       score_train = grid.score(X_train, y_train)
       score_test = grid.score(X_test, y_test)
       best_p = grid.best_params_

       bp = pd.DataFrame(best_p, index=[i])
       bp['accuracy_train'] = score_train
       bp['accuracy_test'] = score_test
       bp['random_state'] = i*500

       df = df.append(bp, ignore_index=True)

#df.to_csv('/home/users/ubaldi/TESI_PA/result_CV/large_space_NO_fixed_rand_state/QDAClassifier_stability/best_params_QDAClassifier.csv')

#insert sccuracy mean and std

acc_train_mean = df['accuracy_train'].mean()
acc_test_mean = df['accuracy_test'].mean()

acc_train_std = df['accuracy_train'].std()
acc_test_std = df['accuracy_test'].std()


df_train_acc_mean = pd.DataFrame([{'accuracy_train_mean':acc_train_mean}])
df_train_acc_std = pd.DataFrame([{'accuracy_train_std':acc_train_std}])


df_test_acc_mean = pd.DataFrame([{'accuracy_test_mean':acc_test_mean}])
df_test_acc_std = pd.DataFrame([{'accuracy_test_std':acc_test_std}])


df_tot = pd.concat([df, df_train_acc_mean, df_train_acc_std, df_test_acc_mean, df_test_acc_std], axis=1)


#create folder and save

save_output.function_save_output(df_tot, dim_reduction, name)
