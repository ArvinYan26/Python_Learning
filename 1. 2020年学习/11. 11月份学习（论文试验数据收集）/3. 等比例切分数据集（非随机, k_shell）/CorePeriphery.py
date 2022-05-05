import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_blobs, load_digits
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
import pandas as pd
from networkx.algorithms.distance_measures import center
from collections import Counter
from sklearn.model_selection import train_test_split
#from GetCOVID_19Data import get_data
from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
from sklearn import preprocessing
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
#import cpalgorithm as cp
import igraph as ig
import matplotlib as mpl


class CorePeriphery(object):

    def __init__(self, num_class, in_rate):
        '''
        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
        self.num_class = num_class
        self.init_rate = in_rate
        self.adj = []  #存储形成的三个网络的邻接矩阵
        self.rcc = []   #存储是哪个网络中每类数据的rich_clup_coefficient
        #self.c_r = c_rate

        self.per_class_data_len = None
        self.train_len = None
        self.train_x = None
        self.data_idxs_list = []
        self.train_y = None

        self.neigh_models = []  #
        self.e_radius = []

        self.G_list = []
        self.mean_dis_list = []
        self.nodes_list = []
        self.edges_list = []
        self.len_list = []  #存储每个组件大小
        self.net_measures = []  # {1:{'averge_degree':[]}}

    def fit(self, data):
        """
        Args:
            data: data[i][0]数据， data[i][1]每类数据平均距离， data[i][2]每类数据长度
        Returns: predict_label

        """
        self.data = data   #三个网络的数据信息
        #print("self.data:", self.data)
        for ith_class in range(self.num_class):

            adj_matrix = euclidean_distances(self.data[ith_class][0], self.data[ith_class][0])
            adj_matrix[adj_matrix == 0] = 10000

            #print("adj_matrix_or:", adj_matrix)
            #每一个节点最小距离找到构建连边，防止单节点出现

            for idx, item in enumerate(adj_matrix):
                min_idx = np.argmin(item)
                # 因为是对称矩阵
                adj_matrix[idx, min_idx] = 1
                adj_matrix[min_idx, idx] = 1

            #小于阈值的设置为1即连边
            adj_matrix[adj_matrix < np.min(self.data[ith_class][1]) * self.init_rate] = 1
            #print("adj_matrix_1:", adj_matrix)
            # 将没有连边的部分都设置为0
            adj_matrix[adj_matrix != 1] = 0
            #self.G_list.append(nx.from_numpy_matrix(adj_matrix))

            #print("adj_matrix:", adj_matrix)
            self.G_list.append(nx.from_numpy_matrix(adj_matrix))  #将邻接矩阵转化为图

            sub_conponents = sorted(nx.connected_components(self.G_list[ith_class]), key=len, reverse=True)

            # print('社区数目',len(sub_conponents)
            center_node = center(self.G_list[ith_class].subgraph(0))[0]

            # print('---Component----')

            if len(sub_conponents) > 1: #如果组件不是一个而是多个，就执行
                for i in sub_conponents:  # 合并节点就是每个子图中中心节点连接即可

                    if i == sub_conponents[0]: #从第二个组件开始找中心节点，防止出现自循环
                        continue
                    sub_G = self.G_list[ith_class].subgraph(i)

                    sub_center_node = center(sub_G)[0]

                    #if not sub_center_node == center_node:
                    edge = (sub_center_node, center_node)

                    self.G_list[ith_class].add_edges_from([edge])

            #k_core = nx.k_core(self.G)
            """
            #计算k_core
            k_shell = nx.k_shell(self.G)
            print("k_core,k_shell:", k_shell)
            print(k_shell.nodes())
            degree = nx.degree(k_shell)
            print("degree:", degree)
            """

            #将图转化为邻接矩阵
            A = np.array(nx.adjacency_matrix(self.G_list[ith_class]).todense())
            #print("A:", A)
            self.adj.append(A)
            #画出0，1棋盘格
            #self.draw_adj_matrix(adj_matrix, self.each_self.data_len[0])

        #print("self.G_list:", self.G_list)
        for i in range(self.num_class):

            #print(self.G_list[i].edges(), self.data[i][2], "*"*100)

            #rcc = nx.rich_club_coefficient(self.G_list[i], self.data[i][1])
            #print("rcc:", rcc)

            rcc = self.calculate_net_measures(self.G_list[i], self.data[i][2], idx=[])
            self.rcc.append(rcc)   #

        #print("self.rcc:", self.rcc)

        return self.G_list, self.adj

    def generate_delta(self, l1, l2, A):

        """
        :param l1: core nodes length (核心节点个数)
        :param l2: periphery nodes length （边缘节点个数）
        :param A: adjacency_matrix
        :return:
        """
        #print("l1, l2", l1, l2)
        delta1 = np.ones(int(l1))
        delta2 = np.zeros(int(l2))

        delta = np.hstack((delta1, delta2))
        #print(delta.shape)
        Delta = delta.reshape(delta.shape[0], 1)*delta
        #print("Delta:", Delta)
        #print(Delta.shape, A.shape)
        Rho = np.sum(Delta*A)/2   #归一化，因为只需要邻接矩阵一半的值

        return Rho

    def draw_adj_matrix(self, adj_matrix, c_n):
        m = np.zeros_like(adj_matrix) - 2
        size = adj_matrix.shape[0]
        m[:c_n, :c_n] = int(0)
        m[:c_n, c_n:] = int(1)
        m[c_n:, :c_n] = int(1)

        for i in range(size):
            m[i, i] = -1
        fig, ax = plt.subplots(figsize=(12, 12))

        colors = ['white', '#000000', '#6495ED', '#FF6A6A']
        # ax.matshow(m, cmap=plt.cm.Blues)
        cmap = mpl.colors.ListedColormap(colors)
        ax.matshow(m, cmap=cmap)

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                v = adj_matrix[j, i]
                ax.text(i, j, int(v), va='center', ha='center')

        plt.show()

    def predict(self, x: np.ndarray, y):

        """
        真正的算法中predict函数中参数不包括 测试标签 y
        Args:
            x: test_data
        Returns:

        """
        y_pred = []
        print("test_x_len:", len(x))
        count = 0
        #x = self.data_preprcess(x)
        for idx, item in enumerate(x):  # 遍历测试数据

            l = y[idx]
            print("label:", l)

            #idx += len(self.G)  # 新节点编号
            #print("new_idx:", idx)
            item = item.reshape(1, -1)
            count += 1
            new_mesures = []

            for i in range(self.num_class):

                idx = len(self.G_list[i]) #新节点label
                #print("idx:", idx)
                dis_matrix = euclidean_distances(item, self.data[i][0])
                min_idx = int(np.argmin(dis_matrix[0]))

                #找到所有小于阈值的节点编号
                edge_idxs = list(np.argwhere(dis_matrix[0] < np.min(self.data[i][1]) * self.init_rate))

                """
                if i == 0:
                    self.min_idx, self.edge_idxs = min_idx, edge_idxs
                if i == 1:
                    self.min_idx = len(self.data[0]) + min_idx
                    self.edge_idxs = [len(self.data[0])+j for j in edge_idxs]
                if i == 2:
                    self.min_idx = (len(self.data[0])+len(self.data[1])) + min_idx
                    self.edge_idxs = [(len(self.data[0])+len(self.data[1])) + j for j in edge_idxs]
                """

                #print(self.min_idx, self.edge_idxs)
                # 添加节点， 添加连边
                test_node = (idx, {'value': None, 'class': 'test', 'type': 'test'})
                self.G_list[i].add_nodes_from([test_node])

                edges = [(idx, min_idx)] #防止出现单节点

                #将小于阈值的节点与新节点连接。
                for edge_idx in edge_idxs:
                    edges.append((idx, int(edge_idx)))

                self.G_list[i].add_edges_from(edges)

                #method one: rich_club_cofficient
                #new_node_m = self.calculate_net_measures(self.G_list[i], self.data[i][2], idx)

                #method two: k_shell
                new_node_m = self.k_shell(self.G_list[i], idx)

                new_mesures.append(new_node_m[0])

                #将新节点移除
                self.G_list[i].remove_node(idx)

            print("new_mesures:", new_mesures)

            if new_mesures[0] == new_mesures[1] == 0:
                label = 0
                y_pred.append(label)
                print("y_pred:", label)
            if new_mesures[1] == 1 and new_mesures[2] == 0:
                label = 1
                y_pred.append(label)
                print("y_pred:", label)
            if new_mesures[0] == 1 and new_mesures[2] == 1:
                label = 2
                y_pred.append(label)
                print("y_pred:", label)

        return np.array(y_pred)

    def check(self, x, y):
        y_hat = self.predict(x, y)  #predict函数中不能有y,此处只是为了验证而已
        print("origanl_y:", y)
        print("predict:", y_hat)
        acc = np.sum(y_hat == y) / len(y)

        return acc, y_hat

    def calculate_net_measures(self, G, data_len, idx):

        rcc = nx.rich_club_coefficient(G, normalized=False, Q=100)
        av_rcc = []

        if idx == []:
            #for each_len in self.each_data_len:
            sum_c0 = [rcc.get(i, 0) for i in range(data_len[0])]
            #print("byebye")
            #print("sum_c0:", sum_c0)
            sum_c1 = [rcc.get(i, 0) for i in range(data_len[0], data_len[0] + data_len[1])]
            #print("sum_c1:", sum_c1)
            #print("over")
            ev_rcc0 = [sum(sum_c0) / data_len[0]]
            ev_rcc1 = [sum(sum_c1) / data_len[1]]
            av_rcc.extend([ev_rcc0, ev_rcc1])

            #fina_rcc.append(sum_c0)
            #fina_rcc.append(sum_c1)

        else:

            print("hello_new_node", "idx:", idx, )
            if idx in rcc.keys():
                av_rcc = [rcc[idx]]
            else:
                av_rcc = [0]
            #av_rcc = [rcc[idx]]

        return av_rcc

    def k_shell(self, G, idx):
        """
        :param G: graph
        :return: 0, 1, 2
        """
        measures = []
        k_shell = nx.k_shell(G)
        #print("k_shell:", k_shell)

        if idx in k_shell.nodes():
            measures.append(0)
        else:
            measures.append(1)

        return measures




