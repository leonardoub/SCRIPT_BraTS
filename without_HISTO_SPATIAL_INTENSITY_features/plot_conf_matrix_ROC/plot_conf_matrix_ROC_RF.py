#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:05:57 2020

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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score


import load_data
import os

name_clf = 'RandomForestClassifier'


#load data

data, labels = load_data.function_load_data()


#Vettorizzare i label
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)


clf=RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=1, min_samples_split=5, random_state=503)



train_data, test_data, train_labels, test_labels =  train_test_split(data, labels_encoded, test_size=0.20, random_state=8, stratify=labels)


steps = [('scaler', None), ('red_dim', None), ('clf', clf)]

pipeline = Pipeline(steps)

pipeline.fit(train_data, train_labels)



pred_train = pipeline.predict(train_data)
pred_test = pipeline.predict(test_data)

#per prova

pred_proba_train = pipeline.predict_proba(train_data)[:, 1]

pred_proba_test = pipeline.predict_proba(test_data)[:, 1]



   

bp = pd.DataFrame(index=[0])
bp['score'] = pipeline.score

#ROC AUC WITHOUT predict proba: WRONG WAY
bp['train_roc_auc_scores'] = roc_auc_score(train_labels, pred_train)
bp['test_roc_auc_scores'] = roc_auc_score(test_labels, pred_test)

#ROC AUC WITH PREDICT PROBA
bp['train_roc_auc_scores_predict_proba'] = roc_auc_score(train_labels, pred_proba_train)
bp['test_roc_auc_scores_predict_proba'] = roc_auc_score(test_labels, pred_proba_test)
   
bp['train_accuracy_scores'] = accuracy_score(train_labels, pred_train)
bp['test_accuracy_scores'] = accuracy_score(test_labels, pred_test)

bp['train_balanced_accuracy_scores'] = balanced_accuracy_score(train_labels, pred_train)
bp['test_balanced_accuracy_scores'] = balanced_accuracy_score(test_labels, pred_test)


#CONFUSION MATRIX   

from sklearn.metrics import confusion_matrix

#TRAIN
confusion_matrix(train_labels, pred_train)

#TEST
confusion_matrix(test_labels, pred_test)
 

#ROC CURVE TRAIN
from sklearn.metrics import roc_curve, auc   

## Compute ROC curve and ROC area for each class
#fpr_tr = dict()
#tpr_tr = dict()
#roc_auc_tr = dict()
#
#fpr_tr, tpr_tr, _ = roc_curve(train_labels, pred_proba_train)
#roc_auc = auc(fpr_tr, tpr_tr)
#
##PLOT ROC CURVE TRAIN
#
#
#plt.figure()
#lw = 2
#plt.plot(fpr_tr, tpr_tr, color='darkorange',
#         lw=lw, label=f'ROC curve (area = {roc_auc})')
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()





#ROC CURVE TEST
from sklearn.metrics import roc_curve, auc   

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr, tpr, _ = roc_curve(test_labels, pred_proba_test)
roc_auc = auc(fpr, tpr)

#PLOT ROC CURVE TEST


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label=f'ROC curve (area = {roc_auc})')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
