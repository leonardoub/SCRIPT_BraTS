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

name = 'svm_sigmoid'
dim_reduction = 'PCA'

#load data

public_data, public_labels = load_data.function_load_data()

encoder = LabelEncoder()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, dtype=float))
gamma_range = np.power(2, np.arange(-10, 11, dtype=float))
n_features_to_test = np.arange(1,11)


for i in range(1, 21):

       #Train test split
       X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, 
       stratify=public_labels, random_state=i*500)

       #Vettorizzare i label
       train_labels_encoded = encoder.fit_transform(y_train)
       test_labels_encoded = encoder.transform(y_test)

       #SVM
       steps = [('scaler', StandardScaler()), ('red_dim', PCA(random_state=i*42)), ('clf', SVC(kernel='sigmoid', random_state=i*503))]

       pipeline = Pipeline(steps)

       n_features_to_test = np.arange(1, 11)

       parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA()], 'red_dim__n_components':n_features_to_test, 'red_dim__whiten':[False, True],
                       'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range), 'clf__class_weight':[None, 'balanced']}]

       grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5, n_jobs=-1, verbose=1)

       grid.fit(X_train, y_train)
       
       score_train = grid.score(X_train, y_train)
       score_test = grid.score(X_test, y_test)
       best_p = grid.best_params_

       bp = pd.DataFrame(best_p, index=[i])
       bp['accuracy_train'] = score_train
       bp['accuracy_test'] = score_test
       bp['random_state'] = i*500
       bp['random_state_pca'] = i*42
       bp['random_state_clf'] = i*503

       df = df.append(bp, ignore_index=True)

#df.to_csv('/home/users/ubaldi/TESI_PA/result_CV/large_space_NO_fixed_rand_state/sigmoid_svm_stability/best_params_svm_sigmoid.csv')


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


