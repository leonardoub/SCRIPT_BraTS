import os
import pandas as pd

def function_save_output(final_data, dimensionality_reduction, name_clf):
    outname = f'score_{dimensionality_reduction}_{name_clf}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/result_score/data_without_HISTO_SPATIAL/ROC_AUC_optimization/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname)





