import os
import pandas as pd

def function_save_output(final_data, dimensionality_reduction, name_clf):
    outname = f'features_importance_for_{name_clf}_best_HP_dim_red_{dimensionality_reduction}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/result_score/data_without_HISTO_SPATIAL/score_ROC_AUC_optimization/{name_clf}_stability/best_feature_importances_for_{name_clf}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname)





