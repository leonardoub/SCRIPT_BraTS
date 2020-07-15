#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:45:56 2020

@author: leonardo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
import seaborn as sns

import load_data    
import os

name_clf = 'SVM_rbf'

#load data

data, labels = load_data.function_load_data()    


# Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, 2, dtype=float))
gamma_range = np.power(2, np.arange(-10, 11, 2, dtype=float))



clf = SVC(kernel='rbf', probability=True, random_state=503)

pca = PCA(random_state=42, n_components=0.85)


steps = [('scaler', MinMaxScaler()), ('red_dim', None), ('clf', clf)]

pipeline = Pipeline(steps)



parameteres = [{'clf__C': list(C_range), 'clf__gamma':list(gamma_range)}]

outer_kf = KFold(n_splits=5, shuffle=True, random_state=2)

rf_gridsearch = GridSearchCV(estimator=pipeline, param_grid=parameteres, n_jobs=-1, scoring='roc_auc_ovr_weighted', refit='roc_auc_ovr_weighted', 
                             verbose=1, cv=outer_kf, return_train_score=True)

rf_gridsearch.fit(data, labels)

# and after some hours...
df_gridsearch = pd.DataFrame(rf_gridsearch.cv_results_)


max_scores = df_gridsearch.groupby(['param_clf__C', 
                                    'param_clf__gamma']).max()

#TRAIN
max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
sns_plot = sns.heatmap(max_scores.mean_train_score, annot=True, fmt='.4g')





#create folder and save



outname = f'heatmap_{name_clf}_NONE_TRAIN.png'
outdir = f'/home/users/ubaldi/TESI_PA/fig_heatmap/BRATS/NONE/'


if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    


sns_plot.figure.set_size_inches(9,9)
sns_plot.figure.tight_layout()

sns_plot.figure.savefig(fullname)
plt.close()

#TEST

sns_plot = sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g')


#create folder and save



outname = f'heatmap_{name_clf}_NONE_TEST.png'
outdir = f'/home/users/ubaldi/TESI_PA/fig_heatmap/BRATS/NONE/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

sns_plot.figure.set_size_inches(9,9)
sns_plot.figure.tight_layout()

sns_plot.figure.savefig(fullname)
plt.close()