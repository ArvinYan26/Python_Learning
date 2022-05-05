# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 03:21:43 2020

@author: esteb
"""
from sklearn import preprocessing

def preprocess(train_data,test_data, normType=1):
    if(normType==1):
        scaler=preprocessing.StandardScaler().fit(train_data)  #scaler：定标器
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    if(normType==2):
        scaler=preprocessing.MinMaxScaler().fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    if(normType==3):
        scaler=preprocessing.Normalizer().fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    return train_data, test_data