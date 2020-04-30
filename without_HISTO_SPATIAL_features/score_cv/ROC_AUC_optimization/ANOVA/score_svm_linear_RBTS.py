import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import score_cv
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif

name = 'svm_linear_RBTS'
dim_reduction = 'ANOVA'


#load data
import load_data 
import save_output

public_data, public_labels = load_data.function_load_data()


scaler_ = RobustScaler()
ANOVA_percentile_ = 10
C_ = 0.0009765625
class_weight_ = None
random_state_clf = 503
random_state_PCA = 42
random_state_outer_kf = 2


dict_best_params = {'SCALER':[scaler_], 'ANOVA__percentile':[ANOVA_percentile_],
                    'CLF__C':[C_], 'CLF__class_weight':[class_weight_],
                    'CLF__random_state':[random_state_clf], 'random_state_outer_kf':[random_state_outer_kf]}


df_best_params = pd.DataFrame.from_dict(dict_best_params)

#implmentation of steps
scaler=scaler_
svm = SVC(kernel='linear', C=C_, class_weight= class_weight_, probability=True, random_state=random_state_clf)
 

steps = [('scaler', scaler), ('red_dim', SelectPercentile(f_classif, percentile=10)), ('clf', svm)]    
pipeline = Pipeline(steps)


df_score_value, df_mean_std = score_cv.function_score_cv(public_data, public_labels, pipeline, name)
df_tot=pd.concat([df_best_params, df_score_value, df_mean_std], axis=1, ignore_index=False)

save_output.function_save_output(df_tot, dim_reduction, name)
