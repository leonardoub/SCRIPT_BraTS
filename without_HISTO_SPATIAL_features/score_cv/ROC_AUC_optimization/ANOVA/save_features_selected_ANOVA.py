import os
import pandas as pd

def function_save_features_selected_ANOVA(dimensionality_reduction, name_clf, features, best_est_dictionary):



    for key, value in best_est_dictionary.items():
        
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