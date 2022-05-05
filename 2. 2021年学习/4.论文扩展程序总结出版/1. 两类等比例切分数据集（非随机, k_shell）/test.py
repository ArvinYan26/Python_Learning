import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors



def get_data():
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    return iris, iris_data, iris_target

def get_breast_canser():
    data = load_breast_cancer()
    return data

# def get_digits():
#     data = load_digits()
#     return data

def data_preprocess():
    iris_data, iris_target = get_data()
    print(iris_data.shape, iris_target.shape)
    X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    return X_train, X_predict, Y_train, Y_predict

if __name__ == '__main__':
    iris, data, target = get_data()
    # # print(iris[:10], data[:10], target[:10])
    print(iris, data[:10], target[:10])
    #
    # b_data = load_breast_cancer()
    # print(b_data)

    # digits = load_digits()
    # print("digits:", digits, type(digits))

    # digits_data = digits.data
    # print(digits_data[0], digits_data[0].shape)