import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from test2 import split_data  # test文件中的spli_data函数

import time


class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        # self.g = []
        self.train_data, self.train_target, self.X_test, self.Y_test = self.get_data()
        self.data_len = len(self.train_data)  # 此程序是24
        self.num_class = num_class
        self.per_class_data_len = int(self.data_len / self.num_class)
        self.plot_data(self.train_data)  # 直接执行此函数
        self.nbrs = []  # 用来存储是哪个类别网络的nbrs
        self.radius = []  # 用来存储是哪个类别的
        self.weight_alpha = [0.1, 0.4, 0.4]  # measures权重
        self.color_map = {0: 'red', 1: 'green', 2: 'yellow'}
        self.net0_measure = []  # 存储每一类别插入Y_items中的数据时的指标
        self.impact0 = []  # storage the impact of the first class
        self.net1_measure = []
        self.impact1 = []
        self.net2_measure = []
        self.G = nx.Graph()

        self.predict_label = []

        # 初始化运行程序，一开始就运行
        self.build_init_network(label=True)
        self.need_classification = []  # 计算模糊分类节点次数
        self.single_node_insert()
        self.accuracy()

    def get_data(self):
        """获取数据集"""

        iris = load_iris()
        iris_data = iris.data  # [:, 2:]
        iris_target = iris.target
        train_data, X_test, train_target, Y_test = train_test_split(iris_data, iris_target, test_size=0.2)

        """
        print("总的数据集:")
        print(iris_data, iris_target)
        #print(iris_data[85])

        print("X_traing:")
        print(train_data, train_target)
        #print(train_data[85])
        #print(iris_data.index(train_data[85]))
        print(len(train_data))
        #print(train_data[85], train_data[84])


        print(X_net, Y_net)
        print("X_items：")
        print(np.array(X_items), np.array(Y_items))


        print("X_test:")

        print(np.array(X_test), np.array(Y_test))
        print(len(X_test))
        """
        print(len(train_data))
        print(len(X_test))

        return train_data, train_target, X_test, Y_test

    def plot_data(self, data):
        """画出数据"""
        node_style = ["ro", "go", "yo"]
        for i in range(self.num_class):
            plt.plot(data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), 0],
                     data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), 1],
                     node_style[i],
                     label=node_style[i])
            plt.legend()
        plt.show()

    def data_preprocess(self, data):
        """特征工程（归一化）"""
        # 归一化
        scaler = preprocessing.MinMaxScaler().fit(data)
        data = scaler.transform(data)

        return data

    def KNN(self, nbrs, train_data):
        """
        KNN获取节点邻居和邻居索引
        :param nbrs:
        :param train_data:
        """
        distances, indices = nbrs.kneighbors(train_data)
        return distances, indices

    def get_radius(self, distances):

        return np.median(distances)  # 中位数

    def epsilon_radius(self, nbrs, train_data, radius):
        """

        :param nbrs:
        :param train_data:
        :param radius:
        :return:
        """
        nbrs.set_params(radius=radius)
        nbrs_distances, nbrs_indices = nbrs.radius_neighbors(train_data)

        return nbrs_distances, nbrs_indices

    def calculate_measure(self, G):
        """
        :param net: 构建的网络g
        :param nodes: 每一类的网络节点
        :return:
        """
        measures = []

        # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        # 1.  平均度
        ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
        ave_deg = round(ave_deg, 3)
        # print("平均度为：%f" % ave_deg)
        measures.append(ave_deg)

        """
        # 2.  平均最短路径长度(需要图是连通的)
        ave_shorest = nx.average_shortest_path_length(G)
        ave_shorest = round(ave_shorest, 3)
        #print("平均最短路径：", ave_shorest)
        #measures.append(ave_shorest)
        """

        # 3.  平均聚类系数
        ave_cluster = nx.average_clustering(G)
        ave_cluster = round(ave_cluster, 3)
        # print("平均聚类系数：%f" % ave_cluster)
        measures.append(ave_cluster)

        # 4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G)
        assortativity = round(assortativity, 3)
        # print("同配性：%f" % assortativity)
        measures.append(assortativity)

        """
        # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(G)
        tran = round(tran, 3)
        #print("三角形分数：%f" % tran)
        measures.append(tran)
        """
        return measures

    def build_init_network(self, label):
        # current_data = self.train_data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]
        # print(current_data)
        for index, instance in enumerate(self.train_data):
            self.G.add_node(str(index), value=instance, typeNode="init_net", label=self.train_target[index])
        # print(self.nodes_list)

        # 切片范围必须是整型
        temp_nbrs = NearestNeighbors(self.k, metric='euclidean')
        temp_nbrs.fit(self.train_data)
        self.nbrs.append(temp_nbrs)  # 将每一类的nbrs都添加进列表， 这个
        knn_distances, knn_indices = self.KNN(temp_nbrs, self.train_data)
        # print(" ")
        temp_radius = self.get_radius(knn_distances)

        self.radius.append(temp_radius)  # 将每一类的radius都添加进radius
        # print("temp_radius", self.radius)
        radius_distances, radius_indices = self.epsilon_radius(temp_nbrs, self.train_data, temp_radius)
        # print(knn_indices, np.array(radius_indices))
        # print(" ")
        # print(np.array(radius_indices))
        # 添加连边
        for idx, one_data in enumerate(self.train_data):  # 这个语句仅仅是获取索引indx，然后给他连边
            # print(knn_indices[idx], radius_indices[idx])
            if (len(radius_indices[idx])) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                print("radius technique:")
                print(idx, radius_indices[idx], knn_indices[idx])

                for indiceNode, indicesNode in enumerate(radius_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            self.G.add_edge(str(indice), str(indiceNode), weight=radius_distances[indiceNode][tmpi])

            else:
                print("KNN technique:")
                print(idx, knn_indices[idx], radius_indices[idx])
                for indiceNode, indicesNode in enumerate(knn_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            self.G.add_edge(str(indice), str(indiceNode), weight=knn_distances[indiceNode][tmpi])

            # do the next steps when there are thingle node (small components)

            num = nx.number_connected_components(self.G)
            if num > self.num_class:
                Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
                list0 = [list(n) for n in Gcc]  # components list
                print("components_list:", list0)
                for i in range(len(list0)):
                    Gi = nx.Graph(list0[i])

            # 单节点情况，和同类最近编号节点连边三个
            count = 0
            new_id = str(idx)  # 因为neighbors迭代器用的节点编号的字符串，所以需要转化为字符串
            # print(self.G.nodes())
            adj = [n for n in self.G.neighbors(new_id)]  # find the neighbors of the new node
            if len(adj) == 0:
                count += 1
                print("sing_node_num:", count)
                print("idx:", idx)
                count = 0
                for i in range(len(self.G.nodes())):
                    if self.G._node[str(idx)]["label"] == self.G._node[str(i)]["label"]:
                        self.G.add_edges_from([(str(idx), str(i))])
                        count += 3
                    if count == self.num_class:
                        break
            print("")

        print("nodes:", self.G.nodes())
        self.draw_graph(self.G)
        self.get_subgraph()
        self.get_subgraph()

        print("self.G0 info:")
        self.draw_graph(self.G0)
        for n in self.G0.nodes():
            if n in self.G0.nodes():
                label = self.G._node[n]["label"]
        print("self.G0_label:", label)

        print("self.G1 info:")
        self.draw_graph(self.G1)
        for n in self.G1.nodes():
            if n in self.G1.nodes():
                label = self.G1._node[n]["label"]
        print("self.G1_label:", label)

        print("self.G2 info:")
        self.draw_graph(self.G2)
        for n in self.G2.nodes():
            if n in self.G2.nodes():
                label = self.G2._node[n]["label"]
        print("label:", label)

    def draw_graph(self, G):
        plt.figure("Graph", figsize=(12, 9))
        pos = nx.spring_layout(G)
        color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]

        for index, (node, typeNode) in enumerate(G.nodes(data='typeNode')):
            if (typeNode == 'test'):
                color_list[index] = 'purple'
                # color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]
        nx.draw_networkx(G, pos, node_color=color_list, with_labels=True, node_size=300)  # 节点默认大小为300
        plt.show()
        print("node_num：", len(G.nodes()))
        print("edges_num：", len(G.edges()))
        print(" ")

    def get_subgraph(self):
        """得到子图，并画出来"""
        print("num_components:", nx.number_connected_components(self.G))
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)

        # self.G0定义为0类数据， self.G1定义为1类数据， self.G2定义为2类数据
        G = self.G.subgraph(Gcc[0])
        for n in G.nodes():
            if G._node[n]["label"] == 0:
                self.G0 = G
            if G._node[n]["label"] == 1:
                self.G1 = G
            if G._node[n]["label"] == 2:
                self.G2 = G

        G = self.G.subgraph(Gcc[1])
        for n in G.nodes():
            if G._node[n]["label"] == 0:
                self.G0 = G
            if G._node[n]["label"] == 1:
                self.G1 = G
            if G._node[n]["label"] == 2:
                self.G2 = G
        # self.calculate_measure(self.G1)
        # plt.subplot(132)
        # self.draw_graph(self.G1)

        G = self.G.subgraph(Gcc[2])
        for n in G.nodes():
            if G._node[n]["label"] == 0:
                self.G0 = G
            if G._node[n]["label"] == 1:
                self.G1 = G
            if G._node[n]["label"] == 2:
                self.G2 = G

        # self.draw_graph(self.G2)

if __name__ == '__main__':
    DataClassification(3, 3)