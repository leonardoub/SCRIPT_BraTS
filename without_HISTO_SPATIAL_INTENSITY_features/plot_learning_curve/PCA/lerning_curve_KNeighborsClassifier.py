#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:15:43 2020

@author: leonardo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:10:43 2020

@author: leonardo
"""

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
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier




import plot_learning_curve
import load_data
import os

name_clf = 'KNeighborsClassifier'


#load data

data, labels = load_data.function_load_data()


#Vettorizzare i label
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)


#clf 

pca = PCA(random_state=42)

clf=KNeighborsClassifier()

steps = [('scaler', RobustScaler()), ('red_dim', pca), ('clf', clf)]

pipeline = Pipeline(steps)

title = "Learning Curves (KNeighborsClassifier)"

plot_learning_curve.function_plot_learning_curve(pipeline, title, data, labels_encoded, ylim=(0.0, 1.01),
                    cv=outer_kf, n_jobs=1)

#create folder and save

outname = f'{title}.png'

outdir = f'/home/leonardo/Scrivania/Presentazione/img_learning_curve/BRATS/PCA/{name_clf}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()

#clf optimized

pca = PCA(n_components=0.95, random_state=42)

clf_opt=KNeighborsClassifier(n_neighbors=7, weights='distance')

steps_opt = [('scaler', RobustScaler()), ('red_dim', pca), ('clf', clf_opt)]

pipeline_opt = Pipeline(steps_opt)

title = "Learning Curves (KNeighborsClassifier Optimized)"

plot_learning_curve.function_plot_learning_curve(pipeline_opt, title, data, labels_encoded, ylim=(0.0, 1.01),
                    cv=outer_kf, n_jobs=1)

#create folder and save

outname = f'{title}.png'

outdir = f'/home/leonardo/Scrivania/Presentazione/img_learning_curve/BRATS/PCA/{name_clf}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()
