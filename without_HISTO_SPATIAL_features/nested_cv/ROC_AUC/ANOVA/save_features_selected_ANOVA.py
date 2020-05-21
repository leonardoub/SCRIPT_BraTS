import os
import pandas as pd

def function_save_features_selected_ANOVA(dimensionality_reduction, name_clf, features, best_est_dictionary, rs_outer_kf):

    i=0


    D={'FOLD_1_ANOVA_FEATURES':[], 'FOLD_1_p_value':[],
       'FOLD_2_ANOVA_FEATURES':[], 'FOLD_2_p_value':[],
       'FOLD_3_ANOVA_FEATURES':[], 'FOLD_3_p_value':[],
       'FOLD_4_ANOVA_FEATURES':[], 'FOLD_4_p_value':[],
       'FOLD_5_ANOVA_FEATURES':[], 'FOLD_5_p_value':[]}


    for key, value in best_est_dictionary.items():
        
        i+=1
        
        p_values = value.named_steps['red_dim'].pvalues_
        mask_percentile = value.named_steps['red_dim'].get_support()
        
        
        selected_features = features[mask_percentile]
        selected_p_value = p_values[mask_percentile]
        
        important_features = sorted(zip(selected_p_value, selected_features), reverse=False)


        D[f'FOLD_{i}_ANOVA_FEATURES'] = [item[1] for item in important_features]
        D[f'FOLD_{i}_p_value'] = [item[0] for item in important_features]

        #save best features

    df_best_features = pd.DataFrame.from_dict(D)

    outname = f'{name_clf}_features_selected_ANOVA_RSoKF_{rs_outer_kf}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/important_features/data_without_HISTO_SPATIAL/score_ROC_AUC_optimization /ANOVA_best_features/{name_clf}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    df_best_features.to_csv(fullname)



