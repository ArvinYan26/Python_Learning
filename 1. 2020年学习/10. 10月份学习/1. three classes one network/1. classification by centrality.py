import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_blobs, load_digits
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
import pandas as pd
from networkx.algorithms.distance_measures import center as nxCenter
from collections import Counter
from sklearn.model_selection import train_test_split
from GetCOVID_19Data import get_data
#from GetCOVID_19Data1 import get_data
from sklearn import preprocessing
#from BuildNetwork import BuildNetwork
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns


class ClassificationByCentrality(object):

    def __init__(self):
        self.data_len = []
        self.per_class_data_len = None
        self.train_len = None
        #self.t = t
        self.train_x = None
        self.data_idxs_list = []
        self.train_y = None
        self.neigh_models = []  #
        self.G_list = []
        self.mean_dis_list = []
        self.nodes_list = []
        self.edges_list = []
        self.len_list = []  #存储每个组件大小
        self.net_measures = []  # {1:{'averge_degree':[]}}

    def fit(self, x_train, y_train):
        """

        :param x_train: train data
        :param y_train: train target
        :return:
        """
        self.x_train = []
        #all_data = []
        self.y_train = []
        self.train_len =len(x_train)
        labels = [i for i in Counter(y_train)]
        self.labels = labels.sort()
        self.num_class = len(labels)

        for each_class in labels:  #即使是从1开始，也可以直接用
            idxs = np.argwhere(y_train == each_class).reshape(-1)
            #print("idxs:", idxs)
            #for i in list(idxs):
                #self.y_train.append(i)
            #self.y_train = np.hstack((self.y_train, idxs))  #将每一类数据的label添加到新的列表里面
            self.data_idxs_list.append(idxs)
            each_data = x_train[idxs]
            each_target = y_train[idxs]
            #重组target
            for i in each_target:
                self.y_train.append(i)
            #print(len(each_target))
            data_len = len(each_data)
            #重组数据
            for i in range(data_len):
                self.x_train.append(each_data[i])
            self.data_len.append(data_len)  #存储每一类训练集数据长度
            dis_matrix = euclidean_distances(each_data, each_data)
            mean_dis = np.sum(dis_matrix) / (data_len ** 2 - data_len)
            self.mean_dis_list.append(mean_dis)
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        print("训练集：", self.x_train.shape, self.y_train.shape)
        print("平均距离：", self.mean_dis_list)
        print("each_len:", self.data_len)

if __name__ == '__main__':
    def data_preprocessing(data):
        """
        :param data:row data
        :return: Normalized data
        """
        min_max_scaler = preprocessing.MinMaxScaler().fit(data)
        new_data = min_max_scaler.transform(data)

        return new_data

    #data = load_iris()
    #x = data.data
    #y = data.target

    x, y = get_data()
    print(x.shape, y.shape)

    x = data_preprocessing(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    cbc = ClassificationByCentrality()
    cbc.fit(x_train, y_train)
