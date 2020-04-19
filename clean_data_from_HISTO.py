import pandas as pd


path_data_without_NAN = '/home/leonardo/Scrivania/BRATS_data/data_without_NAN_with_histologies.csv'


#read csv
data_without_NAN = pd.read_csv(path_data_without_NAN) 




#TOGLIERE LE FEATURES 'HISTO'

features_list = list(data_without_NAN.columns)
index_1 = features_list.index('HISTO_ET_T1Gd_Bin1')
index_2 = features_list.index('HISTO_NET_FLAIR_Bin10')


data_without_NAN_and_HISTO = data_without_NAN.drop(data_without_NAN.iloc[:, index_1:index_2+1], inplace = False, axis = 1) 
data_without_NAN_and_HISTO.to_csv('/home/leonardo/Scrivania/BRATS_data/data_without_NAN_without_HISTO_with_histologies.csv', index=False)
