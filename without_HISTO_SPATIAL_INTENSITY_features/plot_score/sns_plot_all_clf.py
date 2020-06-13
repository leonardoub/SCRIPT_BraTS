#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:55:27 2020

@author: leonardo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import os



clf = ['Adaboost', 'Adaboost','Adaboost','Adaboost',
       'KNeighbors','KNeighbors','KNeighbors','KNeighbors',
       'Random Forest', 'Random Forest', 'Random Forest', 'Random Forest',
       'SVM linear', 'SVM linear', 'SVM linear', 'SVM linear',
       'SVM RBF','SVM RBF','SVM RBF','SVM RBF',
       'SVM sigmoid','SVM sigmoid','SVM sigmoid','SVM sigmoid'     
       ]


value = [0.92, 0.93, 0.87, 0.92, 
         0.90, 0.90, 0.85, 0.89,
         0.94, 0.94, 0.88, 0.94,
         0.76, 0.89, 0.91, 0.91,
         0.76, 0.87, 0.87, 0.89,
         0.93, 0.60, 0.91, 0.85         
         ]

std = [0.04, 0.04, 0.04, 0.06, 
       0.03, 0.05, 0.09, 0.07,
       0.04, 0.03, 0.04, 0.05,
       0.33, 0.07, 0.06, 0.04,
       0.36, 0.05, 0.08, 0.03,
       0.06, 0.43, 0.07, 0.07
       ] 

dim_red = ['ANOVA', 'MI', 'PCA', 'NONE']*6



ax = sns.pointplot(x=clf, y=value, hue=dim_red, dodge=0.3, join=False, ci=None, scale=0.5)

# Find the x,y coordinates for each point
x_coords = []
y_coords = []
for point_pair in ax.collections:
    for x, y in point_pair.get_offsets():
        x_coords.append(x)
        y_coords.append(y)

x_coords_sort, y_coords_sort = zip(*sorted(zip(x_coords,y_coords), reverse=False))

# Calculate the type of error to plot as the error bars
# Make sure the order is the same as the points were looped over
##errors = tips.groupby(['smoker', 'sex']).std()['tip']
##colors = ['steelblue']*2 + ['coral']*2 + ['green']*2 + ['red']*2

colors = ['steelblue', 'coral', 'green', 'red']*6

ax.errorbar(x_coords_sort, y_coords_sort, yerr=std, ecolor=colors, fmt=' ', 
            zorder=-1)


ax.set_xticklabels(ax.get_xticklabels(), rotation=65)

ax.set

ax.set_xlabel('Classifiers')
ax.set_ylabel('ROC AUC score')
ax.set_title('ROC AUC score')

#create folder and save


outname = f'sns_plot_all_clf.png'

outdir = '/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/BRATS/without_HISTO_SPATIAL_INTENSITY/RSoKF_8/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    



ax.figure.set_size_inches(7,5)
ax.figure.tight_layout()

ax.figure.savefig(fullname)

