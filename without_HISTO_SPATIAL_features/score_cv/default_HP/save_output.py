import os
import pandas as pd

def function_save_output(final_data, scaler, name_clf):
    outname = f'score_default_HP_{name_clf}_{scaler}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/result_score/data_without_HISTO_SPATIAL/score_default_HP/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname)





