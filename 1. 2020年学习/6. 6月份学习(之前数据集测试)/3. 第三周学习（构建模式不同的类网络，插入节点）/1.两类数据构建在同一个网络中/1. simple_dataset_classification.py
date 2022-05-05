import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from test import split_data   #test文件中的spli_data函数
from make_simple_dataset import generate_dataset

import time

class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        #self.g = []
        self.train_data, self.train_target, self.X_test, self.Y_test = self.get_data()
        self.data_len = len(self.train_data)  # 此程序是24
        self.num_class = num_class
        self.per_class_data_len = int(self.data_len / self.num_class)
        self.plot_data(self.train_data)  #直接执行此函数
        self.nbrs = []  #用来存储是哪个类别网络的nbrs
        self.radius = []  #用来存储是哪个类别的
        self.weight_alpha = [0.1, 0.4, 0.4]  #measures权重
        self.color_map = {0: 'red', 1: 'green', 3: 'black'}
        self.net0_measure = []   #存储每一类别插入Y_items中的数据时的指标
        self.impact0 = []   #storage the impact of the first class
        self.net1_measure = []
        self.impact1 = []
        self.G = nx.Graph()

        self.predict_label = []

        #初始化运行程序，一开始就运行
        self.build_init_network(label=True)
        self.single_node_insert()
        self.accuracy()

    def get_data(self):
        """获取数据集"""
        """
        iris = load_iris()
        iris_data = iris.data  #[:, 2:]
        iris_target = iris.target
        """
        data, label = generate_dataset()

        #存储切分后的数据，训练集,和测试集
        X_train1 = []
        Y_train1 = []
        train_data = []
        train_target = []
        X_train2 = []
        Y_train2 = []
        X_train3 = []
        Ytrain3 = []

        #第一次划分，train_data, train_target （0.8比例，多数），  X_train1, Y_train1 （0.2，少数）
        train_data, train_target, X_test, Y_test  = split_data(data, label, X_train1, Y_train1, train_data, train_target)

        #第二次划分，X_net 0.8, X_items 0.2
        #X_net, Y_net, X_items, Y_items = split_data(train_data, train_target, X_train2, Y_train2, X_train3, Ytrain3)
        #print("训练集：")
        #print(np.array(X_train1), np.array(Y_train1))



        print("总的数据集:")
        print(data, label)
        print("X_traing")
        print(train_data, train_target)
        print("X_net:")

        """
        print(X_net, Y_net)
        print("X_items：")
        print(np.array(X_items), np.array(Y_items))
        """

        print("X_test:")
        print(np.array(X_test), np.array(Y_test))


        return train_data, train_target, X_test, Y_test


    def plot_data(self, data):
        """画出数据"""
        node_style = ["ro", "go"]
        for i in range(self.num_class):
            plt.plot(data[self.per_class_data_len*i:self.per_class_data_len*(i+1), 0],
                     data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), 1],
                     node_style[i],
                     label=node_style[i])
            plt.legend()
        plt.show()



    def data_preprocess(self, data):
        """特征工程（归一化）"""
        #归一化
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

        return np.median(distances) #中位数

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
        #print("平均度为：%f" % ave_deg)
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
        #print("平均聚类系数：%f" % ave_cluster)
        measures.append(ave_cluster)

        # 4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G)
        assortativity = round(assortativity, 3)
        #print("同配性：%f" % assortativity)
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
        edges_list = []
        nodes_list = []
        # print("类别：", i)
        #current_data = self.train_data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]
        # print(current_data)
        for index, instance in enumerate(self.train_data):
            self.G.add_node(str(index), value=instance, typeNode="init_net", label=self.train_target[index])
        # print(self.nodes_list)
        self.draw_graph(self.G)
        # 切片范围必须是整型
        temp_nbrs = NearestNeighbors(self.k, metric='euclidean')
        temp_nbrs.fit(self.train_data)
        self.nbrs.append(temp_nbrs)  # 将每一类的nbrs都添加进列表， 这个
        knn_distances, knn_indices = self.KNN(temp_nbrs, self.train_data)
        # print(knn_distances, knn_indices)
        temp_radius = self.get_radius(knn_distances)

        self.radius.append(temp_radius)  # 将每一类的radius都添加进radius
        print("temp_radius", self.radius)
        radius_distances, radius_indices = self.epsilon_radius(temp_nbrs, self.train_data, temp_radius)
        # print(radius_distances, radius_indices)
        # 添加连边
        for idx, one_data in enumerate(self.train_data):  # 这个语句仅仅是获取索引indx，然后给他连边
            # print(radius_indices[idx])
            if (len(radius_indices[idx])) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                # print(radius_indices[idx])
                for indiceNode, indicesNode in enumerate(radius_indices):
                    for tmpi, indice in enumerate(indicesNode):
                            if (str(indice) == str(indiceNode)):
                                continue
                            if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)]["label"] or not labels):
                                self.G.add_edge(str(indice), str(indiceNode), weight=radius_distances[indiceNode][tmpi])
            else:
                for indiceNode, indicesNode in enumerate(knn_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)]["label"] or not labels):
                            self.G.add_edge(str(indice), str(indiceNode), weight=knn_distances[indiceNode][tmpi])

        self.G.add_weighted_edges_from(edges_list)
        print("nodes:", self.G.nodes())
        self.draw_graph(self.G)


    def draw_graph(self, G):
        plt.figure("Graph", figsize=(9, 9))
        pos = nx.spring_layout(G)
        color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]

        for index, (node, typeNode) in enumerate(G.nodes(data='typeNode')):
            if (typeNode == 'test'):
                color_list[index] = 'b'
                #color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]
        nx.draw_networkx(G, pos, node_color=color_list, with_labels=True, node_size=300)  # 节点默认大小为300
        plt.show()
        print("node_num：", len(G.nodes()))
        print("edges_num：", len(G.edges()))
        print(" ")


    def get_subgraph(self):
        """得到子图，并画出来"""
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        self.G0 = self.G.subgraph(Gcc[0])
        #self.calculate_measure(self.G0)
        #plt.subplot(131)
        #self.draw_graph(self.G0)

        self.G1 = self.G.subgraph(Gcc[1])
        #self.calculate_measure(self.G1)
        #plt.subplot(132)
        #self.draw_graph(self.G1)




    def single_node_insert(self):
        """

        :param X_items: one node inserted
        :param nodeindex: the index of the inserted node
        :param Y_items: label of the inserted node
        :return:
        """
        # print(self.X_test, self.Y_test)
        # g = self.g
        # print(len(g.nodes()))
        # insert_node_id = len(list(self.g.nodes()))
        # print(insert_node_id)
        # 添加节点
        for index, instance in enumerate(self.X_test):
            #label = 4
            label = self.Y_test[index]
            print("label:", label)
            if (not len(self.Y_test) == 0):
                label = self.Y_test[index]
                # print(index, instance)
            #计算插入节点之前的各个类别网络的measures
            self.get_subgraph()
            measures0 = self.calculate_measure(self.G0)
            self.net0_measure.append(measures0)
            measures1 = self.calculate_measure(self.G1)
            self.net1_measure.append(measures1)

            #插入新的节点构建连边
            insert_node_id = len(list(self.G.nodes()))
            print("insert_node_id:", insert_node_id)
            self.G.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
            #print(len(self.g.nodes()))

            #print(self.nbrs, self.radius)
            radius_distances, radius_indices = self.epsilon_radius(self.nbrs[0], [instance],
                                                                   self.radius[0])
            # print("radius:", radius_indices)
            distances, indices = self.KNN(self.nbrs[0], [instance])
            # print(distances, indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉

            # 添加到训练网络中
            for idx, one_data in enumerate(self.train_data):  # 这个语句仅仅是获取索引indx，然后给他连边
                #print(radius_indices[idx])
                if (len(radius_indices)) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                    # print(radius_indices[idx])
                    for indiceNode, indicesNode in enumerate(radius_indices):
                        for tmpi, indice in enumerate(indicesNode):
                            if (str(indice) == str(indiceNode)):
                                continue
                            self.G.add_edge(str(indice), str(insert_node_id), weight=radius_distances[indiceNode][tmpi])
                else:
                    for indiceNode, indicesNode in enumerate(indices):
                        for tmpi, indice in enumerate(indicesNode):
                            if (str(indice) == str(indiceNode)):
                                continue
                            self.G.add_edge(str(indice), str(insert_node_id), weight=distances[indiceNode][tmpi])
            self.classification(insert_node_id, int(self.Y_test[index]))

            #self.get_subgraph()
            #print("G0_nodes:", self.G0.nodes())



        self.draw_graph(self.G)
        self.get_subgraph()

    def classification(self, insert_node_id, new_node_label):
        new_id = str(insert_node_id)  # 因为neighbors迭代器用的节点编号的字符串，所以需要转化为字符串
        print(self.G.nodes())
        adj = [n for n in self.G.neighbors(new_id)]  # find the neighbors of the new node
        print("adj:", adj)
        # check which class the link of the new node belongs to
        count0 = 0
        count1 = 0
        for n in adj:
            if n in self.G0.nodes():
                count0 += 1
            elif n in self.G1.nodes():
                count1 += 1
        dist_list = []
        if count0 == 3 or count1 == 3:
            self.predict_label.append(new_node_label)
        else:
            self.get_subgraph()
            #如果分类不确定就画出子图分析连边计算measures进行分类
            self.draw_graph(self.G0)
            self.draw_graph(self.G1)
            measures0 = self.calculate_measure(self.G0)
            V1, V2 = np.array(self.net0_measure[len(self.net0_measure) - 1]), np.array(measures0)
            euclidean_dist0 = np.linalg.norm(V2 - V1)
            dist_list.append(euclidean_dist0)
            measures1 = self.calculate_measure(self.G1)
            N1, N2 = np.array(self.net1_measure[len(self.net1_measure) - 1]), np.array(measures1)
            euclidean_dist1 = np.linalg.norm(N2 - N1)
            dist_list.append(euclidean_dist1)
            label = int(dist_list.index(min(dist_list)))
            self.predict_label.append(label)

    def accuracy(self):
        label = list(map(int, self.Y_test))  # 廖雪峰，高阶函数内容
        print("original_label:", label)
        print("predict_label :", self.predict_label)

        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == self.predict_label[i]:
                count += 1
        print(count)
        accuracy = round(count / len(self.Y_test), 3)
        print("accuracy:", accuracy)


if __name__ == '__main__':
    DataClassification(3, 2)