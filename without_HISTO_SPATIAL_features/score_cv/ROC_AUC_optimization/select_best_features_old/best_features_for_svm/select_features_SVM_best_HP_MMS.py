import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


name_clf = 'SVM_linear'
name_scaler = 'MMS'
name_dim_red = 'NONE'
best_or_def_HP= 'BEST_HP'

#load data
import load_data 
import save_output

public_data, public_labels = load_data.function_load_data()

#Vettorizzare i label
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(public_labels)

########################################
#PIPELINE 1
########################################
scaler_ = MinMaxScaler()
C_ = 0.0078125
class_weight_ = 'balanced'
random_state_clf = 503
random_state_outer_kf = 2

#implmentation of steps
clf_1 = SVC(kernel='linear', C=C_, class_weight= class_weight_, probability=True, random_state=random_state_clf)



steps_1 = [('scaler', scaler_), ('clf', clf_1)]    

pipeline_1 = Pipeline(steps_1)



########################################
#PIPELINE 2
########################################
scaler_ = MinMaxScaler()
C_ = 0.25
class_weight_ = 'balanced'
random_state_clf = 503
random_state_outer_kf = 2

#implmentation of steps
clf_2 = SVC(kernel='linear', C=C_, class_weight= class_weight_, probability=True, random_state=random_state_clf)


steps_2 = [('scaler', scaler_), ('clf', clf_2)]    

pipeline_2 = Pipeline(steps_2)



########################################
#PIPELINE 3
########################################
scaler_ = MinMaxScaler()
C_ = 1.25
class_weight_ = 'balanced'
random_state_clf = 503
random_state_outer_kf = 2

#implmentation of steps
clf_3 = SVC(kernel='linear', C=C_, class_weight= class_weight_, probability=True, random_state=random_state_clf)


steps_3 = [('scaler', scaler_), ('clf', clf_3)]    

pipeline_3 = Pipeline(steps_3)



########################################
#PIPELINE 4
########################################
scaler_ = MinMaxScaler()
C_ = 2.25
class_weight_ = None
random_state_clf = 503
random_state_outer_kf = 2

#implmentation of steps
clf_4 = SVC(kernel='linear', C=C_, class_weight= class_weight_, probability=True, random_state=random_state_clf)


steps_4 = [('scaler', scaler_), ('clf', clf_4)]    

pipeline_4 = Pipeline(steps_4)

########################################
#PIPELINE 5
########################################
scaler_ = MinMaxScaler()
C_ = 3.25
class_weight_ = None
random_state_clf = 503
random_state_outer_kf = 2

#implmentation of steps
clf_5 = SVC(kernel='linear', C=C_, class_weight= class_weight_, probability=True, random_state=random_state_clf)


steps_5 = [('scaler', scaler_), ('clf', clf_5)]    

pipeline_5 = Pipeline(steps_5)



pipeline_list = [pipeline_1, pipeline_2, pipeline_3, pipeline_4, pipeline_5]

D={'CLF_best_HP_FOLD_1_FEATURES':[], 'FOLD_1_value':[],
   'CLF_best_HP_FOLD_2_FEATURES':[], 'FOLD_2_value':[],
   'CLF_best_HP_FOLD_3_FEATURES':[], 'FOLD_3_value':[],
   'CLF_best_HP_FOLD_4_FEATURES':[], 'FOLD_4_value':[],
   'CLF_best_HP_FOLD_5_FEATURES':[], 'FOLD_5_value':[]}


# Choose cross-validation techniques for outer loops,
outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

i=0


for train_index, test_index in outer_kf.split(public_data, labels_encoded):


    pipeline = pipeline_list[i]

    i+=1


    pipeline.fit(public_data.iloc[train_index, :], labels_encoded[train_index])

    rank_features = pipeline.named_steps["clf"].coef_

    rank_features_T = rank_features.T
    rank_features_T_bis = rank_features_T.reshape(575,)

    rank_features_T_ter = np.absolute(rank_features_T_bis)

    important_features = sorted(zip(rank_features_T_ter, public_data.columns), reverse=True)[:20]
    #ATTENZIONE HA SENSO SOLO SE NON SI FA PCA
    
    D[f'CLF_best_HP_FOLD_{i}_FEATURES'] = [item[1] for item in important_features]
    D[f'FOLD_{i}_value'] = [item[0] for item in important_features]


df_best_features = pd.DataFrame.from_dict(D)



save_output.function_save_output(df_best_features, name_clf, name_scaler, name_dim_red, best_or_def_HP)





#dict_best_params = {'SCALER':[scaler_], 'PCA__n_components':[n_comp_pca], 
#                    'CLF__n_estimators':[n_estimators_], 'CLF__criterion':[criterion_], 'CLF__bootstrap':[bootstrap_],
#                    'CLF__max_depth':[max_depth_], 'CLF__min_samples_split':[min_samples_split_], 
#                    'CLF__min_samples_leaf':[min_samples_leaf_], 'CLF__class_weight':[class_weight_],
#                    'CLF__random_state':[random_state_clf], 'random_state_outer_kf':[random_state_outer_kf]}

#df_best_params = pd.DataFrame.from_dict(dict_best_params)
