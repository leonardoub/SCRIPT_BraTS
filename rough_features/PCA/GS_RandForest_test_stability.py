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

name = 'RandomForest'
dim_reduction = 'PCA'

#load data

public_data, public_labels = load_data.function_load_data()

encoder = LabelEncoder()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
#n_tree = np.arange(200, 2200, 200)
n_tree = [10, 30, 50, 70, 100, 150, 200, 400, 600, 1000]
n_features_to_test = [0.85, 0.9, 0.95]
depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]

for i in range(1, 6):

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, 
    stratify=public_labels, random_state=i*500)

    #Vettorizzare i label
    train_labels_encoded = encoder.fit_transform(y_train)
    test_labels_encoded = encoder.transform(y_test)

    #RandomForestClassifier
    steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', RandomForestClassifier(random_state=i*503))]

    pipeline = Pipeline(steps)

    parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=i*42)], 'red_dim__n_components':list(n_features_to_test),
                    'red_dim__whiten':[False, True], 
                    'clf__n_estimators':list(n_tree), 'clf__criterion':['gini', 'entropy'], 
                    'clf__max_depth':depth, 'clf__min_samples_split':[2, 5, 10], 
                    'clf__min_samples_leaf':[1, 2, 4], 'clf__class_weight':[None, 'balanced']}]

    grid = GridSearchCV(pipeline, param_grid=parameteres, cv=3, n_jobs=-1, verbose=1)

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

#df.to_csv('/home/users/ubaldi/TESI_PA/result_CV/large_space_NO_fixed_rand_state/RandomForest_stability/best_params_RandomForest.csv')


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


















