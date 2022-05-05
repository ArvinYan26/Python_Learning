from CorePeriphery import CorePeriphery
from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
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

class DataReorganization(object):
    def __init__(self):

        self.data_idxs_list = []
        self.num_classes = None
        #self.each_data_len = []
        self.mean_dis_list = []


    def data_preprcess(self, x_train, x_test):
        min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = min_max_scaler.transform(x_train)
        x_test = min_max_scaler.transform(x_test)
        return x_train, x_test

    def data_reorganization(self, x, y):
        self.train_x = x
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
            self.data.append(dataset)
            data_len = len(dataset) #每类数据长度
            self.each_data_len.append(data_len)
            adj_matrix = euclidean_distances(dataset, dataset)
            # 要先求两两平均距离，后面会改动数据。
            mean_dis = np.sum(adj_matrix) / (data_len ** 2 - data_len)
            # print("mean_dis:", mean_dis)
            # mean_dis = mean_dis*self.t #变化阈值
            self.mean_dis_list.append(mean_dis)  # 平均差别
        if not self.mean_dis_list == []:
            self.mean_dis_list = sorted(list(set(self.mean_dis_list)))

        #print("self.mean_dis_list:", self.mean_dis_list, self.each_data_len)
        #print(len(self.data))
        #三类
        #self.new_data = np.vstack((self.data[0], self.data[1], self.data[2]))

        each_data = []
        #正常和新冠
        nc = []
        nc_len = []
        nc_dis = []

        nc_len.extend([self.each_data_len[0], self.each_data_len[2]])
        #self.mean_dis_list.remove(self.mean_dis_list[1])
        nc_dis.extend([self.mean_dis_list[0], self.mean_dis_list[2]])
        nc_data = np.vstack((self.data[0], self.data[2]))
        nc.extend([nc_data, nc_dis, nc_len])
        each_data.append(nc)

        #正常和肺炎
        nv = []
        nv_len = []
        nv_dis = []
        nv_len.extend([self.each_data_len[0], self.each_data_len[1]])
        #self.mean_dis_list.remove(self.mean_dis_list[1])
        nv_dis.extend([self.mean_dis_list[0], self.mean_dis_list[1]])
        nv_data = np.vstack((self.data[0], self.data[1]))
        nv.extend([nv_data, nv_dis, nv_len])
        each_data.append(nv)

        vc = []
        vc_len = []
        vc_dis = []
        vc_len.extend([self.each_data_len[1], self.each_data_len[2]])
        #self.mean_dis_list.remove(self.mean_dis_list[1])
        vc_dis.extend([self.mean_dis_list[1], self.mean_dis_list[2]])
        vc_data = np.vstack((self.data[1], self.data[2]))
        vc.extend([vc_data, vc_dis, vc_len])
        each_data.append(vc)

        return each_data

    def draw_g(self, G, len, i):

        plt.figure("Graph", figsize=(12, 12))
        if i == 0:
            color_map = {0: 'r', 1: 'b', 2: 'b',}
            plt.title("Normal and Covid-19")
            color_list = []
            for idx, thisG in enumerate(len):
                color_list += [color_map[idx]] * (thisG)
        if i == 1:
            color_map = {0: 'r', 1: 'g', 2: 'b', }
            plt.title("Normal and Viral Pneumonia")
            color_list = []
            for idx, thisG in enumerate(len):
                color_list += [color_map[idx]] * (thisG)
        if i == 2:
            color_map = {0: 'g', 1: 'b', 2: 'b', }
            plt.title("Viral Pneumonia and Covid-19")
            color_list = []
            for idx, thisG in enumerate(len):
                color_list += [color_map[idx]] * (thisG)
        #plt.figure("Graph", figsize=(12, 12))
        #Normal and Covid-19, Normal and Viral Pneumonia,
        #plt.title("Normal and Viral Pneumonia")
        #color_list = []
        #for idx, thisG in enumerate(len):
            #color_list += [color_map[idx]] * (thisG)
        pos = nx.spring_layout(G)   #细长
        nx.draw_networkx(G, pos, with_labels=False, node_size=80,
                         node_color=color_list, width=0.1, alpha=1)  #
        plt.show()

    def show_core_periphery(self, G):
        """
        画出黑白边缘结构图
        :return:
        """
        A = np.array(nx.adjacency_matrix(G).todense())
        # print("A:", A)
        plt.figure(1, (12, 12))
        # plt.title("Adj_Matrix")
        plt.imshow(A)
        plt.imshow(A, "gray")
        plt.show()

    def draw_line(self, t, Rho, l, G_list):
        plt.figure(figsize=(12, 12))
        plt.title("Core-periphery Measure")
        plt.plot(t, Rho)
        max_y = np.median(Rho) #找到最大值, 中位数
        max_index = Rho.index(max_y)
        max_x = round(t[max_index], 3) #找到最大值对应的x坐标
        max_G = G_list[max_index] #找到最大值时的图G
        max_G_len = l[max_index]   #训练数据集长度

        #horizontal, values = t[0:max_x+1], [max_y for i in range(max_index+1)]
        plt.plot([max_x, max_x], [0, max_y], 'r--', label='Highest Value')
        print("="*50)
        print([min(t), max_x], [max_y, max_y])
        plt.plot([min(t), max_x], [max_y, max_y], 'r--')
        plt.text(max_x, 0, str(max_x), fontsize='x-large')
        plt.text(min(t), max_y, str(max_y), fontsize='x-large')
        plt.legend(loc='best')


        plt.xlabel("Threshold")
        plt.ylabel("Measure Value")
        plt.grid(True, linestyle="--", color="g", linewidth="0.5")
        plt.show()

        return max_G, max_G_len

    def draw_confusion_matrix(self, y_true, y_pred):
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        #print("confusion_matrix:")  # 打印出来看看
        #print(C2)
        sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('Predict')  # x轴
        ax.set_ylabel('True')  # y轴
        plt.show()

if __name__ == '__main__':

    start = time.time()
    dr = DataReorganization()  #调用模块

    x_train, y_train, x_test, y_test = get_data(percent=0.9)
    print(len(x_train), len(y_train), len(x_test), len(y_test))

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_test = dr.data_preprcess(x_train, x_test)  # 数据归一化

    #返回三组重组的新数据集及相关信息，每组数据中：1.重组的新数据，2.每类平均距离，3.每类数据长度
    #each_data = dr.data_reorganization(x_train, y_train) #返回1.重组的新数据，2.每类平均距离，3.每类数据长度
    #print("each_data:", each_data)
    #print("y_train:", y_train)


    cp = CorePeriphery(num_class=3, in_rate=1.01)  #调用模块, 类别数
    G_list, adj = cp.fit(x_train, y_train)

    for i in range(3):
        dr.draw_g(G_list[i], each_data[i][2], i)
        dr.show_core_periphery(G_list[i])
    acc, y_predict = cp.check(x_test, y_test)
    dr.draw_confusion_matrix(y_test, y_predict)
    print("acc:", acc)


    """
    #t = np.arange(0.82, 1.6, 0.02)
    #print("t:", t)
    Rho = []
    G_list = []
    data_len = []
    data = None
    for i in t:
        cp.get_params(i, 0.88)rich_club_coefficient is not implemented for graphs with self loops.
        rho, g  = cp.fit(nc)
        Rho.append(rho)
        G_list.append(g)
        data_len.append(l)
        #if i == len(t)-1:s
            #data = new_data
    """

    """
    #画节点cloness折线图
    if not len(closess[0]) == len(closess[1]):
        if len(closess[0]) > len(closess[1]):
            closess[0].pop()
            if len(closess[0]) == len(closess[1]):
                break
    """
    #acc, con_m = cp.check(x_test, y_test)
    """
    print("acc and con_m:", acc, con_m)
    
    a.append(acc)
    con_matrix.append(con_m)
    
    #打印最终结果
    print("final:", a, con_matrix)
    mean_acc = np.mean(a)
    max = np.max(a)
    min = np.min(a)
    var = np.var(a)
    print("%f +- %f", (mean_acc, var))
    print(min, max)
    """
    end = time.time()
    print("time:", end - start)


