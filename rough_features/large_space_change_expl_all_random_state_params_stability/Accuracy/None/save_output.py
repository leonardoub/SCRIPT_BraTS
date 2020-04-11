import os
import pandas as pd

def function_save_output(final_data, dimensionality_reduction, name_clf):
    outname = f'best_params_{dimensionality_reduction}_{name_clf}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/result_CV/large_space_change_expl_all_rand_state/rough_data/{name_clf}_stability'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)




