from sklearn import preprocessing
import numpy
def data_preprocess(train_data, normType=2):
    if(normType==1):
        scaler=preprocessing.StandardScaler().fit(train_data)
        train_data=scaler.transform(train_data)
    if(normType==2):
        scaler=preprocessing.MinMaxScaler().fit(train_data)
        train_data=scaler.transform(train_data)
    if(normType==3):
        scaler=preprocessing.Normalizer(norm='l2').fit(train_data)
        train_data=scaler.transform(train_data)

    return train_data