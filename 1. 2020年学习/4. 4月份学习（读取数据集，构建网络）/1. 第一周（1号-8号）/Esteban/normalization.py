# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 03:21:43 2020

@author: esteb
"""
from sklearn import preprocessing

def preprocess(train_data,test_data, normType=1):
    if(normType==1):
        scaler=preprocessing.StandardScaler().fit(train_data) #归一化处理fit（）默认采纳数为均值为0方差为1
        train_data=scaler.transform(train_data)  #将训练集每一列的均值归一化为0，方差归一化为1
        test_data=scaler.transform(test_data)    #将测试集每一列的均值归一化为0，方差归一化为1
    if(normType==2):
        scaler=preprocessing.MinMaxScaler().fit(train_data)  #简单的归一化处理
        train_data=scaler.transform(train_data) #将训练集中的数据归一化为0-1之间的数
        test_data=scaler.transform(test_data)   #将测试集中的数据归一化为0-1之间的数
    if(normType==3):
        scaler=preprocessing.Normalizer().fit(train_data) #训练一个正则器，可应用于测试数据
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    return train_data, test_data

