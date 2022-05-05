from CorePeriphery import CorePeriphery
#from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
from sklearn import preprocessing
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.datasets import make_circles

class DataReorganization(object):
    def __init__(self):

        self.data_idxs_list = []
        self.num_classes = None
        #self.each_data_len = []
        self.mean_dis_list = []


    def data_preprcess(self, x_train):
        min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = min_max_scaler.transform(x_train)
        #x_test = min_max_scaler.transform(x_test)
        return x_train

    def data_reorganization(self, x, y):
        self.train_x = x
        print("x:", len(x))
        self.train_y = y
        self.train_len = len(x)

        labels = [i for i in Counter(y)]
        labels.sort()
        self.labels = labels
        # print("self.labels:", self.labels)
        self.num_classes = len(labels)
        self.data = []
        self.each_data_len = []

        "1. build network"
        for ith_class in labels:
            # label是按照顺序排的，0， 1， 2， 、, # 所以说从图上通过颜色可以看出来是哪一类
            idxs = np.argwhere(y == ith_class).reshape(-1)
            self.data_idxs_list.append(idxs)
            "adjacency matrix"
            dataset = x[idxs]
            print("dataset:", len(dataset))
            self.data.append(dataset)
            data_len = len(dataset) #每类数据长度
            self.each_data_len.append(data_len)
        print("self.data:", len(self.data[0]))
        return self.data

    def draw_confusion_matrix(self, y_true, y_pred):
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
        #print("confusion_matrix:")  # 打印出来看看
        #print(C2)
        sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

        #ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('Predict', fontsize=15)  # x轴
        ax.set_ylabel('True', fontsize=15)  # y轴
        plt.show()

    def plot_scatter(selfx, x, y):
        """
        画出数据散点图
        :return:
        """
        fig = plt.figure(1, figsize=(12, 12))
        plt.title('make_circles function example')
        plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, linewidths=2)
        # print(len(x_train), len(y_train), len(x_test), len(y_test))
        plt.show()

if __name__ == '__main__':



    start = time.time()
    dr = DataReorganization()  # 调用模块
    color_map = {0: "r", 1: "b"}
    #for i in [0.1, 0.15, 0.2, 0.25, 0.3]:

    x, y = make_circles(n_samples=300, factor=0.01, noise=0.25)
    dr.plot_scatter(x, y) #画出原始数据散点图
    #归一化
    x = dr.data_preprcess(x)
    #print("y1:", y)
    dr.plot_scatter(x, y)  #画出归一化后的散点图
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    #print("x_train1:", len(x_train))
    #数据排序
    #x_train = dr.data_reorganization(x_train, y_train)
    #print("x_train2:", len(x_train))
    # 数据归一化
    cp = CorePeriphery(num_class=2, in_rate=0.98)  #调用模块, 类别数
    #G_list, adj = cp.fit(x_train, y_train)
    cp.fit(x_train, y_train)
    acc, y_predict = cp.check(x_test, y_test)
    print("acc:", acc)
    dr.draw_confusion_matrix(y_test, y_predict)


