import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

name = 'RandomForest'
dim_reduction = 'NONE_DEFAULT_HP'


#load data
import load_data 
import save_output

public_data, public_labels = load_data.function_load_data()

#Vettorizzare i label
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(public_labels)

########################################
#PIPELINE DEAFULT HP
########################################
#scaler_ = None
#n_comp_pca = None
#whiten_ = False
#n_estimators_ = 250
#criterion_ = 'gini'
#bootstrap_ = True
#max_depth_ = 10
#min_samples_split_ = 5
#min_samples_leaf_ = 2
#class_weight_ = None
random_state_clf = 503
random_state_outer_kf = 2

#implmentation of steps
clf = RandomForestClassifier(random_state=random_state_clf)


steps = [('clf', clf)]    

pipeline = Pipeline(steps)



D={'CLF_default_HP_FOLD_1_FEATURES':[], 'FOLD_1_value':[],
   'CLF_default_HP_FOLD_2_FEATURES':[], 'FOLD_2_value':[],
   'CLF_default_HP_FOLD_3_FEATURES':[], 'FOLD_3_value':[],
   'CLF_default_HP_FOLD_4_FEATURES':[], 'FOLD_4_value':[],
   'CLF_default_HP_FOLD_5_FEATURES':[], 'FOLD_5_value':[]}


# Choose cross-validation techniques for outer loops,
outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

i=0


for train_index, test_index in outer_kf.split(public_data, labels_encoded):


    i+=1


    pipeline.fit(public_data.iloc[train_index, :], labels_encoded[train_index])

    rank_features = pipeline.named_steps["clf"].feature_importances_


    important_features = sorted(zip(rank_features, public_data.columns), reverse=True)[:20]
    #ATTENZIONE HA SENSO SOLO SE NON SI FA PCA
    
    D[f'CLF_FOLD_{i}_FEATURES'] = [item[1] for item in important_features]
    D[f'FOLD_{i}_value'] = [item[0] for item in important_features]


df_best_features = pd.DataFrame.from_dict(D)



save_output.function_save_output(df_best_features, dim_reduction, name)





#dict_best_params = {'SCALER':[scaler_], 'PCA__n_components':[n_comp_pca], 
#                    'CLF__n_estimators':[n_estimators_], 'CLF__criterion':[criterion_], 'CLF__bootstrap':[bootstrap_],
#                    'CLF__max_depth':[max_depth_], 'CLF__min_samples_split':[min_samples_split_], 
#                    'CLF__min_samples_leaf':[min_samples_leaf_], 'CLF__class_weight':[class_weight_],
#                    'CLF__random_state':[random_state_clf], 'random_state_outer_kf':[random_state_outer_kf]}

#df_best_params = pd.DataFrame.from_dict(dict_best_params)