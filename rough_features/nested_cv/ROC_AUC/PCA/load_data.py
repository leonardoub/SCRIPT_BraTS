import pandas as pd

def function_load_data():

    df_train = pd.read_csv('/home/users/ubaldi/TESI_BRATS/BRATS_data/data_without_NAN_with_histologies.csv')
    pu_data = df_train.drop(['Histology', 'ID', 'Date'], axis=1)  
    pu_labels = df_train.Histology

    return pu_data, pu_labels


