#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:20:38 2020

@author: leonardo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import os



dataset_path = '/home/leonardo/Scrivania/BRATS_data/data_without_NAN_without_HISTO_without_SPATIAL_without_INTENSITY_with_histologies.csv'



df_data = pd.read_csv(dataset_path)

data_1 = df_data.drop(['ID', 'Date'], axis=1)

data = df_data.drop(['Histology', 'ID', 'Date'], axis=1)



fig, ax1 = plt.subplots()


#Using Pearson Correlation
cor = data.corr()
sns.heatmap(cor, cmap=plt.cm.Reds, ax=ax1)
ax1.set_title('Pearson correlation matrix BraTS dataset')


fig.set_figwidth(10)
fig.set_figheight(8)

plt.subplots_adjust(left=0.125, bottom=0.35, right=0.9, top=0.95, wspace=0, hspace=0)



#create folder and save


outname = f'Pearson_corr_matrix_BRATS'

outdir = '/home/leonardo/Scrivania/scrittura_TESI/img/original/Pearson_corr_matrix/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()
