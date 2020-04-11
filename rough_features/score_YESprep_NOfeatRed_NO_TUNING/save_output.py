import os
import pandas as pd

def function_save_output(final_data, scaler, name_clf, field_names):
    outname = f'score_{name_clf}_{scaler}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/result_score/rough_data/score_YESprep_NOfeatRed_NO_TUNING/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname, index=False, header=field_names)





