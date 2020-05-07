import os
import pandas as pd

def function_save_output(final_data, name_clf, scaler, dimensionality_reduction, best_or_default_HP):
    outname = f'features_importance_for_{name_clf}_scaler_{scaler}_dim_red_{dimensionality_reduction}_{best_or_default_HP}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/important_features/data_without_HISTO_SPATIAL/score_ROC_AUC_optimization /best_feature_importances_for_{name_clf}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname)





