import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
#from test2 import split_data  # test文件中的spli_data函数

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
        #self.plot_data(self.train_data)  # 直接执行此函数
        self.nbrs = []  # 用来存储是哪个类别网络的nbrs
        self.radius = []  # 用来存储是哪个类别的
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
        # self.accuracy()

    def get_data(self):
        """获取数据集"""

        breast_cancer = load_breast_cancer()
        #print(breast_cancer)
        cancer_data = breast_cancer.data  # [:, 2:]
        cancer_target = breast_cancer.target
        train_data, X_test, train_target, Y_test = train_test_split(cancer_data, cancer_target, test_size=0.2)
        train_data,  X_test = self.data_preprocess(train_data, X_test)
        """
        print("总的数据集:")
        print(breast_cancer, breast_cancer)

        print("X_traing:")
        print(train_data, train_target)
        print(len(train_data))

        print("X_test:")
        print(np.array(X_test), np.array(Y_test))
        print(len(X_test))

        print(len(train_data))
        print(len(X_test))
        """

        return train_data, train_target, X_test, Y_test

    """
    def plot_data(self, data):
        
        node_style = ["ro", "go", "yo"]
        for i in range(self.num_class):
            plt.plot(data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), 0],
                     data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), 1],
                     node_style[i],
                     label=node_style[i])
            plt.legend()
        plt.show()
    """
    def data_preprocess(self, data_train, data_test):
        """特征工程（归一化）"""
        # 归一化
        scaler = preprocessing.MinMaxScaler().fit(data_train)
        train_data = scaler.transform(data_train)
        test_data = scaler.transform(data_test)


        return train_data, test_data

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
        """
        measures = []

        # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        # 1.  平均度
        ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
        ave_deg = round(ave_deg, 3)
        # print("平均度为：%f" % ave_deg)
        measures.append(ave_deg)

        
        # 2.  平均最短路径长度(需要图是连通的)
        ave_shorest = nx.average_shortest_path_length(G)
        ave_shorest = round(ave_shorest, 3)
        #print("平均最短路径：", ave_shorest)
        #measures.append(ave_shorest)
        

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

        
        # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(G)
        tran = round(tran, 3)
        #print("三角形分数：%f" % tran)
        measures.append(tran)
        """
        measures = []
        efficiency = nx.global_efficiency(G)
        measures.append(efficiency)

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
                for indiceNode, indicesNode in enumerate(radius_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            self.G.add_edge(str(indice), str(indiceNode), weight=radius_distances[indiceNode][tmpi])

            else:
                for indiceNode, indicesNode in enumerate(knn_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            self.G.add_edge(str(indice), str(indiceNode), weight=knn_distances[indiceNode][tmpi])

            # 单节点情况，和同类最近编号节点连边三个
        for idx, one_data in enumerate(self.train_data):
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
                        count += 1
                    if count == self.k:
                        break

        print("nodes:", self.G.nodes())
        self.get_subgraph()
        self.draw_graph(self.G)

    def draw_graph(self, G):
        plt.figure("Graph", figsize=(12, 12))
        pos = nx.spring_layout(G)
        color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]

        for index, (node, typeNode) in enumerate(G.nodes(data='typeNode')):
            if (typeNode == 'test'):
                color_list[index] = 'purple'
                # color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]
        nx.draw_networkx(G, pos, node_color=color_list, with_labels=True,
                         node_size=80, font_size=6)  # 节点默认大小为300,节点标签默认是12
        plt.savefig('./breast_cancer.jpg')
        plt.show()
        print("node_num：", len(G.nodes()))
        print("edges_num：", len(G.edges()))
        print(" ")

    def get_subgraph(self):
        """得到子图，并画出来"""
        num = nx.number_connected_components(self.G)
        print("num_components:", num)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        list = []
        for n in range(num):
            G = self.G.subgraph(Gcc[n])
            count0 = 0
            count1 = 0

            for m in G.nodes():
                if G._node[m]["label"] == 0:
                    count0 += 1
                    if count0 > 5:
                        self.G0 = G
                if G._node[m]["label"] == 1:
                    count1 += 1
                    if count1 > 5:
                        self.G1 = G
        # 如果组件数大于类别数执行下面步骤,正常情况下，分类阶段不会用到，因为很明显的分3类
        if num > self.num_class:
            for m in range(num):
                G = self.G.subgraph(Gcc[m])
                for n in G.nodes():
                    if G._node[n]["label"] == 0:
                        count = 0
                        for a in self.G0.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:  #
                                break
                    if G._node[n]["label"] == 1:
                        count = 0
                        for a in self.G1.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break

    def single_node_insert(self):
        """

        :param X_items: one node inserted
        :param nodeindex: the index of the inserted node
        :param Y_items: label of the inserted node
        :return:
        """

        # 添加节点
        for index, instance in enumerate(self.X_test):
            label = self.Y_test[index]
            if (not len(self.Y_test) == 0):
                label = self.Y_test[index]
                print("label:", label)
            # 计算插入节点之前的各个类别网络的measures
            self.get_subgraph()
            measures0 = self.calculate_measure(self.G0)
            self.net0_measure.append(measures0)
            measures1 = self.calculate_measure(self.G1)
            self.net1_measure.append(measures1)

            # 插入新的节点构建连边
            insert_node_id = len(list(self.G.nodes()))
            print("insert_node_id:", insert_node_id)
            self.G.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
            radius_distances, radius_indices = self.epsilon_radius(self.nbrs[0], [instance],
                                                                   self.radius[0])
            distances, indices = self.KNN(self.nbrs[0], [instance])

            # 添加到训练网络中
            for idx, one_data in enumerate(self.train_data):  # 这个语句仅仅是获取索引indx，然后给他连边
                if (len(radius_indices)) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
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
            print("=" * 100)

        print("finally_G_info:")
        self.draw_graph(self.G)
        #self.get_subgraph()
        print("finally_subgraph_info:")
        #self.draw_graph(self.G0)
        #self.draw_graph(self.G1)


    def classification(self, insert_node_id, label):
        # 因为在构建离岸边的时候用的就是str()形式，所以需要此处需要转化为字符串

        adj = [n for n in self.G.neighbors(str(insert_node_id))]  # find the neighbors of the new node
        print("adj:", adj)
        # check which class the link of the new node belongs to
        count0 = 0
        count1 = 0
        for n in adj:
            if n in self.G0.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count0 += 1
            elif n in self.G1.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count1 += 1

        print("edges_num:", count0, count1)
        if count0 == len(adj):
            print("classification_result:", 0)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(0)
        elif count1 == len(adj):
            print("classification_result:", 1)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(1)

        else:
            print("模糊分类情况：")
            #self.draw_graph(self.G)

            dist_list = []

            if count0 >= 0 and count0 < len(adj):
                # delate the edges and node
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                # 找到类1中和adj中相同的节点，然后将节点添加到类1中
                node_list = self.G0.nodes()  # 这时候还是插入节点之前的G0
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类0中
                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.get_subgraph()  # get the new sungraph to calclulate the measures

                measures0 = self.calculate_measure(self.G0)  # new subgraph self.G0 measures
                V1, V2 = np.array(self.net0_measure[len(self.net0_measure) - 1]), np.array(measures0)
                print("v1, v2:", V1, V2)
                euclidean_dist0 = np.linalg.norm(V2 - V1)
                dist_list.append(euclidean_dist0)


            if count1 >= 0 and count1 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                # 找到类1中和adj中相同的节点，
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.get_subgraph()

                measures1 = self.calculate_measure(self.G1)
                N1, N2 = np.array(self.net1_measure[len(self.net1_measure) - 1]), np.array(measures1)
                print("N1, N2:", N1, N2)
                euclidean_dist1 = np.linalg.norm(N2 - N1)
                dist_list.append(euclidean_dist1)

            # 确定模糊分类节点的分类标签后，需要将节点移除，然后重新插入到确定的分类标签处
            if str(insert_node_id) in self.G.nodes():
                self.G.remove_node(str(insert_node_id))
            # print(np.array(self.net0_measure), self.net1_measure, self.net2_measure,)
            print("dist_list:", dist_list)
            # get the classfication ruselt
            list = []
            for x in dist_list:
                if not x == 0:
                    list.append(x)
            min_value = min(list)
            label = int(dist_list.index(min_value))
            print("classification_result:", label)
            self.predict_label.append(label)

            if label == 0:
                node_list = self.G0.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges

            if label == 1:
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)
            self.need_classification.append(insert_node_id)

    def accuracy(self):
        label = list(map(int, self.Y_test))  # 廖雪峰，高阶函数内容
        print("original_label:", label)
        print("predict_label :", self.predict_label)

        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == self.predict_label[i]:
                count += 1
        print("正确个数：", count)
        accuracy = round(count / len(self.Y_test), 3)
        print("accuracy:", accuracy)
        print("need_classification:", self.need_classification)

        return accuracy


if __name__ == '__main__':
    DC = DataClassification(2, 2)
    DC.accuracy()

