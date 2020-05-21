import os
import pandas as pd
import numpy as np

def function_save_features_selected_svm(best_est_dictionary, features, name_clf, scaler, dimensionality_reduction, best_or_default_HP, rs_outer_kf):

    i=0

    D={'CLF_best_HP_FOLD_1_FEATURES':[], 'FOLD_1_value':[],
       'CLF_best_HP_FOLD_2_FEATURES':[], 'FOLD_2_value':[],
       'CLF_best_HP_FOLD_3_FEATURES':[], 'FOLD_3_value':[],
       'CLF_best_HP_FOLD_4_FEATURES':[], 'FOLD_4_value':[],
       'CLF_best_HP_FOLD_5_FEATURES':[], 'FOLD_5_value':[]}


    for key, value in best_est_dictionary.items():
        
        i+=1

        rank_features = value.named_steps["clf"].coef_
        rank_features_T = rank_features.T
        rank_features_T_bis = rank_features_T.reshape(575,)
        rank_features_T_ter = np.absolute(rank_features_T_bis)
        
        
        important_features = sorted(zip(rank_features_T_ter, features), reverse=True)[:20]
        #ATTENZIONE HA SENSO SOLO SE NON SI FA PCA
        
        D[f'CLF_best_HP_FOLD_{i}_FEATURES'] = [item[1] for item in important_features]
        D[f'FOLD_{i}_value'] = [item[0] for item in important_features]

        #save best features

    df_best_features = pd.DataFrame.from_dict(D)

    outname = f'features_importance_for_{name_clf}_scaler_{scaler}_dim_red_{dimensionality_reduction}_{best_or_default_HP}_RSoKF_{rs_outer_kf}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/important_features/data_without_HISTO_SPATIAL_INTENSITY/score_ROC_AUC_optimization /best_feature_importances_for_{name_clf}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    df_best_features.to_csv(fullname)
