#Cross Validation on RandomForestClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import score_cv


name = 'GradientBoosting'
dim_reduction = 'NONE'

#load data
import load_data 
import save_output

public_data, public_labels = load_data.function_load_data()

def create_score_csv_default_HP(scaler_):


    n_comp_pca = None
    #whiten_ = False
    max_depth_ = 'default'
    min_samples_split_ = 'default'
    min_samples_leaf_ = 'default'
    lr = 'default'
    n_estimators_ = 'default'
    random_state_clf = 503
    #random_state_PCA = 42
    random_state_outer_kf = 2

    dict_best_params = {'SCALER':[scaler_], 'PCA__n_components':[n_comp_pca],  
                        'CLF__max_depth_':[max_depth_], 'CLF__lr':[lr], 'CLF__n_estimators':[n_estimators_],
                        'CLF__min_samples_split_':[min_samples_split_], 'CLF__min_samples_leaf_':[min_samples_leaf_],
                        'CLF__random_state':[random_state_clf], 'random_state_outer_kf':[random_state_outer_kf]}


    df_best_params = pd.DataFrame.from_dict(dict_best_params)

    #implmentation of steps
    scaler=scaler_
    #pca = PCA(n_components=n_comp_pca, whiten=whiten_, random_state=random_state_PCA)
    clf = GradientBoostingClassifier(random_state=random_state_clf)

    steps = [('scaler', scaler), ('clf', clf)]    
    pipeline = Pipeline(steps)

    df_score_value, df_mean_std = score_cv.function_score_cv(public_data, public_labels, pipeline)
    df_tot = pd.concat([df_best_params, df_score_value, df_mean_std], axis=1, ignore_index=False)


    return df_tot


df_MMS = create_score_csv_default_HP(MinMaxScaler())
save_output.function_save_output(df_MMS, 'MMS', name)

df_STDS = create_score_csv_default_HP(StandardScaler())
save_output.function_save_output(df_STDS, 'STDS', name)

df_RBT = create_score_csv_default_HP(RobustScaler())
save_output.function_save_output(df_RBT, 'RBT', name)

df_NONE = create_score_csv_default_HP(None)
save_output.function_save_output(df_NONE, 'NONE', name)

