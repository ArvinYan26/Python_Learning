import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from  sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors



def get_data():
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    return iris_data, iris_target

def data_preprocess():
    """
    切分数据集为训练集，验证集和测试集
    """
    iris_data, iris_target = get_data()
    X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
    X_net, X_items, Y_net, Y_items = train_test_split(X_train, Y_train, test_size=0.2)
    class0_train_data = []
    class1_train_data = []
    class2_train_data = []
    for i in X_net:
        if i.target == 0:
            class0_train_data.append(i)
        if i.target == 1:
            class1_train_data.append(i)
        if i.target == 2:
            class2_train_data.append(i)
    print(class0_train_data)

    print(X_net, X_items, Y_net, Y_items)
    scaler = preprocessing.MinMaxScaler().fit(X_net)
    X_net = scaler.transform(X_net)
    print(X_net)
    return X_train, X_predict, Y_train, Y_predict

"""
def data_preprocess1():
    iris_data, iris_target = get_data()
    X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
    X_net, X_items, Y_net, Y_items = train_test_split(X_train, Y_train, test_size=0.2)
    class0_train_data = []
    class1_train_data = []
    class2_train_data = []
    label =
    for i in X_net:
        if label == 0:
            class0_train_data.append(i)
        if i.target == 1:
            class1_train_data.append(i)
        if i.target == 2:
            class2_train_data.append(i)
    print(class0_train_data)
"""

data_preprocess()
