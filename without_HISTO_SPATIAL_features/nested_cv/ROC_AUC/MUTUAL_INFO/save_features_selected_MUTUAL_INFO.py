import os
import pandas as pd

def function_save_features_selected_MUTUAL_INFO(dimensionality_reduction, name_clf, features, best_est_dictionary, rs_outer_kf):

    i=0


    D={'FOLD_1_MUTUAL_INFO':[], 'FOLD_1_p_value':[],
       'FOLD_2_MUTUAL_INFO':[], 'FOLD_2_p_value':[],
       'FOLD_3_MUTUAL_INFO':[], 'FOLD_3_p_value':[],
       'FOLD_4_MUTUAL_INFO':[], 'FOLD_4_p_value':[],
       'FOLD_5_MUTUAL_INFO':[], 'FOLD_5_p_value':[]}


    for key, value in best_est_dictionary.items():
        
        i+=1
        
        p_values = value.named_steps['red_dim'].pvalues_
        mask_percentile = value.named_steps['red_dim'].get_support()
        
        
        selected_features = features[mask_percentile]
        selected_p_value = p_values[mask_percentile]
        
        important_features = sorted(zip(selected_p_value, selected_features), reverse=True)


        D[f'FOLD_{i}_MUTUAL_INFO_FEATURES'] = [item[1] for item in important_features]
        D[f'FOLD_{i}_value'] = [item[0] for item in important_features]

        #save best features

    df_best_features = pd.DataFrame.from_dict(D)

    outname = f'{name_clf}_features_selected_MUTUAL_INFO_RSoKF_{rs_outer_kf}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/important_features/data_without_HISTO_SPATIAL/score_ROC_AUC_optimization /MUTUAL_INFO_best_features/{name_clf}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    df_best_features.to_csv(fullname)



