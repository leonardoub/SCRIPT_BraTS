import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import score_cv


name = 'KNeighbors'
dim_reduction = 'NONE'

#load data
import load_data 
import save_output

public_data, public_labels = load_data.function_load_data()


scaler_ = MinMaxScaler()
n_comp_pca = None
#whiten_ = False
algorithm_ = 'auto'
n_neighbors_ = 10
weights_ = 'distance'
random_state_clf = 'not_present'
random_state_PCA = 42
random_state_outer_kf = 2

dict_best_params = {'SCALER':[scaler_], 'PCA__n_components':[n_comp_pca], 
                    'CLF__algorithm':[algorithm_], 'CLF__n_neighbors':[n_neighbors_], 'CLF__weights':[weights_],
                    'CLF__random_state':[random_state_clf], 'random_state_outer_kf':[random_state_outer_kf]}



df_best_params = pd.DataFrame.from_dict(dict_best_params)

#implmentation of steps
scaler = scaler_
#pca = PCA(n_components=n_comp_pca, whiten=whiten_, random_state=random_state_PCA)
clf = KNeighborsClassifier(n_neighbors=n_neighbors_, weights=weights_, algorithm=algorithm_)


steps = [('scaler', scaler), ('clf', clf)]    
pipeline = Pipeline(steps)


df_score_value, df_mean_std = score_cv.function_score_cv(public_data, public_labels, pipeline)
df_tot=pd.concat([df_best_params, df_score_value, df_mean_std], axis=1, ignore_index=False)


save_output.function_save_output(df_tot, dim_reduction, name)



