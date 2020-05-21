import os
import pandas as pd

def function_save_features_selected_ANOVA(dimensionality_reduction, name_clf, features, best_est_dictionary):

    i=0

    for key, value in best_est_dictionary.items():
        
        i+=1
        
        p_values = value.named_steps['red_dim'].pvalues_
        mask_percentile = value.named_steps['red_dim'].get_support()
        
        dict_features_pvalue = {'features':features[mask_percentile], 'p_value':p_values[mask_percentile]}
        
        df_features_pvalue = pd.DataFrame.from_dict(dict_features_pvalue)
        df_features_pvalue_sorted = df_features_pvalue.sort_values(by=['p_value'], ascending=True)

        #save best features

        outname = f'{name_clf}_{key}_features_selected_{dimensionality_reduction}.csv'

        outdir = f'/home/users/ubaldi/TESI_BRATS/result_CV/nested_CV/data_without_HISTO_SPATIAL/ROC_AUC_optimization/{name_clf}_stability/{name_clf}_features_selected_ANOVA'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname = os.path.join(outdir, outname)    

        df_features_pvalue_sorted.to_csv(fullname)




         i=0

    D={'CLF_best_HP_FOLD_1_FEATURES':[], 'FOLD_1_value':[],
       'CLF_best_HP_FOLD_2_FEATURES':[], 'FOLD_2_value':[],
       'CLF_best_HP_FOLD_3_FEATURES':[], 'FOLD_3_value':[],
       'CLF_best_HP_FOLD_4_FEATURES':[], 'FOLD_4_value':[],
       'CLF_best_HP_FOLD_5_FEATURES':[], 'FOLD_5_value':[]}


    for key, value in best_est_dictionary.items():
        
        i+=1

        rank_features = value.named_steps["clf"].feature_importances_
        
        
        important_features = sorted(zip(rank_features, features), reverse=True)[:20]
        #ATTENZIONE HA SENSO SOLO SE NON SI FA PCA
        
        D[f'FOLD_{i}_ANOVA_FEATURES'] = [item[1] for item in important_features]
        D[f'FOLD_{i}_p_value'] = [item[0] for item in important_features]

        #save best features

    df_best_features = pd.DataFrame.from_dict(D)

    outname = f'features_importance_for_{name_clf}_scaler_{scaler}_dim_red_{dimensionality_reduction}_{best_or_default_HP}_RSoKF_{rs_outer_kf}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/important_features/data_without_HISTO_SPATIAL/score_ROC_AUC_optimization /best_feature_importances_for_{name_clf}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)