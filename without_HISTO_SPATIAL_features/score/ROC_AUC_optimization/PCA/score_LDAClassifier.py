import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

name = 'LDAClassifier'
folder = 'score_YESprep_NOfeatRed'

#load data
import load_data 
import save_output

public_data, public_labels = load_data.function_load_data()


#vettorizzare i label
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

def create_csv_score_YES_NO(scaler_):

    #tot_random_state = []
    tot_train_score = []
    tot_test_score = []
    tot_train_auc = []
    tot_test_auc = []

    solver_ = 'lsqr'
    shrinkage_ = 'auto'

    for i in range(1,31):

        #train test split 
        X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, stratify=public_labels)

        #tot_random_state.append(500*i)

        #vettorizzare i label
        train_labels_encoded = encoder.fit_transform(y_train)
        test_labels_encoded = encoder.transform(y_test)


        scaler = scaler_
        clf = LinearDiscriminantAnalysis()

        steps = [('scaler', scaler), ('red_dim', None), ('clf', clf)]    

        pipeline = Pipeline(steps)

        summary = pipeline.named_steps

        pipeline.fit(X_train, train_labels_encoded)

        score_train = pipeline.score(X_train, train_labels_encoded)
        tot_train_score.append(score_train)

        score_test = pipeline.score(X_test, test_labels_encoded)
        tot_test_score.append(score_test)

        y_scores_train = pipeline.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(train_labels_encoded, y_scores_train)
        tot_train_auc.append(auc_train)
        
                
        y_scores_test = pipeline.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(test_labels_encoded, y_scores_test)
        tot_test_auc.append(auc_test)

        y_pred = pipeline.predict(X_test)

        report = classification_report(test_labels_encoded, y_pred, output_dict=True)
        df_r = pd.DataFrame(report)
        df_r = df_r.transpose()
        #df_r.to_csv(f'/home/users/ubaldi/TESI_PA/result_CV/report_{name}/report_{i}')

        #outname = f'report_{i}.csv'

        #outdir = f'/home/users/ubaldi/TESI_PA/result_score/Public/{folder}/report_{name}_{str(abbr_scaler)}_YES_NO'
        #if not os.path.exists(outdir):
        #    os.makedirs(outdir)

        #fullname_r = os.path.join(outdir, outname)    

        #df_r.to_csv(fullname_r)



    #mean value and std

    mean_train_score = np.mean(tot_train_score)
    mean_test_score = np.mean(tot_test_score)
    mean_train_auc = np.mean(tot_train_auc)
    mean_test_auc = np.mean(tot_test_auc)


    std_train_score = np.std(tot_train_score)
    std_test_score = np.std(tot_test_score)
    std_train_auc = np.std(tot_train_auc)
    std_test_auc = np.std(tot_test_auc)


    # pandas can convert a list of lists to a dataframe.
    # each list is a row thus after constructing the dataframe
    # transpose is applied to get to the user's desired output. 
    df = pd.DataFrame([tot_train_score, [mean_train_score], [std_train_score], 
                    tot_test_score, [mean_test_score], [std_test_score], 
                    tot_train_auc, [mean_train_auc], [std_train_auc],
                    tot_test_auc, [mean_test_auc], [std_test_auc],
                    [scaler], ['default'], ['default']])
    df = df.transpose() 

    fieldnames = ['train_accuracy', 'train_accuracy_MEAN', 'train_accuracy_STD',
                'test_accuracy', 'test_accuracy_MEAN', 'test_accuracy_STD',
                'train_roc_auc_score', 'train_roc_auc_score_MEAN', 'train_roc_auc_score_STD',
                'test_roc_auc_score', 'test_roc_auc_score_MEAN', 'test_roc_auc_score_STD',
                'SCALER', 'CLF__solver', 'CLF__shrinkage']


    ## write the data to the specified output path: "output"/+file_name
    ## without adding the index of the dataframe to the output 
    ## and without adding a header to the output. 
    ## => these parameters are added to be fit the desired output. 
    #df.to_csv(f'/home/users/ubaldi/TESI_PA/result_score/Public/score_{name}.csv', index=False, header=fieldnames)




    return df, fieldnames



df_MMS, fieldnames_MMS = create_csv_score_YES_NO(MinMaxScaler())
save_output.function_save_output(df_MMS, 'MMS', name, fieldnames_MMS)

df_STDS, fieldnames_STDS = create_csv_score_YES_NO(StandardScaler())
save_output.function_save_output(df_STDS, 'STDS', name, fieldnames_STDS)

df_RBT, fieldnames_RBT = create_csv_score_YES_NO(RobustScaler())
save_output.function_save_output(df_RBT, 'RBT', name, fieldnames_RBT)