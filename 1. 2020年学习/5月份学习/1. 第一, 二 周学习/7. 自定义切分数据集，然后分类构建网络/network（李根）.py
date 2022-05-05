# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import Normalizer
import networkx as nx
import math


class NetworkDataClassification():
    def __init__(self, per_class_data_len, num_classes, k): #5， 3， 4
        '''

        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
        self.per_class_data_len = per_class_data_len  #5
        self.data_len = per_class_data_len * num_classes  # 5*3=15个数据
        self.num_classes = num_classes # 3
        self.k = k  #4

        self.knn_areas = []  #
        self.e_radius = []

        self.G = None
        self.nodes_list = []
        self.edges_list = []
        self.color_map = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', -1: 'black'}  # 节点类别颜色地图
        self.train_x, self.train_y = self.generate_labeled_data(per_class_data_len, num_classes)
        self.plot_data(self.train_x, num_classes)
        self.net_measures = {}  # {1:{'averge_degree':[]}}   #直接调用函数了，所以后边不需要再调用函数了
        self.init_network()

    def generate_labeled_data(self, per_class_data_len, num_classes):
        '''
            generate 2d nodes.
            tips: keep data_len % num_classes = 0

        :param per_class_data_len:
        :param num_classes:
        :return:
        '''
        data_x = []
        for i in range(num_classes):
            tmp_data_x = np.random.rand(per_class_data_len, 2) + 1 * i
            # 每一类的数据值都加上1*i是为了区分开来大小，类别。rand:产生0-1之间的随机数不包括0和1
            #print(tmp_data_x.shape)
            data_x.append(tmp_data_x)
        print(data_x)
        data_x = np.array(data_x).reshape(self.data_len, 2)
        print(data_x)
        data_y = np.zeros(self.data_len) #矩阵长度就是维度，生成一个一维15列的全零矩阵
        print(type(data_y))   #1*15的全零矩阵
        data_y = np.split(data_y, num_classes) #将标签分为三个一维的向量，每一个向量长度是5
        print(type(data_y))
        for label, i in enumerate(data_y):
            i[:] = label  #保证i矩阵中，每个里面的元素等于label
            #print(label, i)
        data_y = np.array(data_y)
        print(data_y)
        print('data_x shape:', data_x.shape)
        print('data_y shape:', data_y.shape)
        return data_x, data_y

    def plot_data(self, data, num_classes):
        #data=self.train_x=data_x
        node_style = ['ro', 'go', 'bo', 'yo']
        per_class_data_len = int(data.shape[0] / num_classes)
        #image.shape[0]:表示图像高度，image.shape[1]:表示图像宽度， image.shape[2]:图像通道数
        for i in range(self.num_classes):
            print('i', i)
            print(node_style[i])
            plt.plot(data[per_class_data_len * i:per_class_data_len * (i + 1), 0],
                     data[per_class_data_len * i:per_class_data_len * (i + 1), 1],
                     node_style[i],
                     label=node_style[i])
            plt.legend()  # for add new nodes（给图想加上图例）
        plt.show()

        return 1  #如果结束时返回0那么代表函数为正常结束，返回1表示有异常

    def kNN(self, knn_net, nodes):
        '''
            获取邻域节点
        :param data:  data set
        :param k:     parameter k of KNN
        :return:
        '''

        distances, indices = knn_net.kneighbors(nodes)
        return distances, indices

    def get_radius(self, distances):
        return float(np.median(distances))  #中位数

    def epsilon_radius(self, net, nodes, radius):
        '''
            获取邻域节点
        :param net:
        :param nodes:[[node1],[node2]]
        :param radius:
        :return:
        '''

        net.set_params(radius=radius)

        neigh_distances, neigh_idx = net.radius_neighbors(nodes)
        return neigh_distances, neigh_idx

    def calculate_net_measures(self, net, nodes):
        degree_assortativity = nx.degree_assortativity_coefficient(G=net, nodes=nodes)
        average_clustering_coefficient = nx.average_clustering(G=net, nodes=nodes)

        """有问题"""
        average_degree = average_clustering_coefficient = np.mean(
            [i[1] for i in nx.degree(net, nbunch=nodes)])  # nx.degree 返回的是每个节点的度, 所以要获取再求平均
        return degree_assortativity, average_clustering_coefficient, average_degree

    def generate_net_measures(self):

        for per_class in range(self.num_classes):
            nodes_list = range(per_class * self.per_class_data_len, (per_class + 1) * self.per_class_data_len)
            self.aa = 1
            self.net_measures[per_class] = self.calculate_net_measures(self.G, nodes=nodes_list)
            print(self.net_measures)

    def net_degree_assortativity(self, net, nodes):
        return nx.degree_assortativity_coefficient(G=net, nodes=nodes)

    def average_clustering_coefficient(self, net, nodes):
        return nx.average_clustering(G=net, nodes=nodes)

    def average_degree(self, net, nodes):
        return nx.average_degree_connectivity(net, nodes=nodes)

    def init_network(self):
        '''

        :return:

            1. create network
            2. add nodes
            3. calculate edges by kNN and epsilon radius
                - knn epsilon
                - radius
                - knn
            4. add edges
            5. plot image

        API reference :
            sklearn.neighbors.NearestNeighbors
                - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.radius_neighbors
        '''
        # Step 1
        self.G = nx.Graph()
        nodes_list = []
        edges_list = []
        label_num = 0
        #将每一类数据添加为节点
        for index, node in enumerate(self.train_x):
            node_info = (index, {'value': list(node), 'class_num': label_num, 'type': 'train'})
            if (index + 1) % self.per_class_data_len == 0: #%取余数
                label_num += 1
            self.nodes_list.append(node_info)
        self.G.add_nodes_from(self.nodes_list)
        # Step 2
        base_index = 0

        #计算每一个节点的邻居
        for i in range(self.num_classes):  # 循环每一类数据
            base_index = self.per_class_data_len * i  #5*i

            current_data = self.train_x[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]  #切片读取每一类数据，

            current_knn_area = NearestNeighbors(n_neighbors=self.k, metric='minkowski')
            current_knn_area.fit(current_data)
            self.knn_areas.append(current_knn_area)

            distances, indices = current_knn_area.kneighbors(current_data)
            current_radius = self.get_radius(distances)
            self.e_radius.append(current_radius)

            e_radius_distances, e_radius_indices = self.epsilon_radius(current_knn_area, current_data, current_radius)

            knn_distances, knn_indices = self.kNN(current_knn_area, current_data)

            for index, one_data in enumerate(current_data):  # 循环计算每个数据的边
                if (len(e_radius_indices[index]) > (self.k - 1)):  # 如果半径邻域大于KNN
                    for idx, neigh_idx in enumerate(e_radius_indices[index]):
                        if index == neigh_idx:
                            continue
                        # edge = (index + base_index, neigh_idx + base_index, {'weight': e_radius_distances[index][idx]})
                        edge = (index + base_index, neigh_idx + base_index, e_radius_distances[index][idx])
                        self.edges_list.append(edge)
                else:
                    print(knn_indices)
                    print(index, knn_indices)
                    for idx, neigh_idx in enumerate(knn_indices[index]):
                        print(idx, neigh_idx)
                        if index == neigh_idx:
                            continue
                        # edge = (index + base_index, neigh_idx + base_index, {'weight': knn_indices[index][idx]})
                        edge = (index + base_index, neigh_idx + base_index, knn_indices[index][idx])
                        self.edges_list.append(edge)

        # self.G.add_edges_from(self.edges_list)
        self.G.add_weighted_edges_from(self.edges_list)
        self.generate_net_measures()

        # calculate net measures

        print(self.G.nodes())
        color_list = [self.color_map.get(self.G.nodes[node]['class_num'], 0) for node in self.G.nodes()]
        nx.draw(self.G, pos=self.train_x, with_labels=True, node_color=color_list)
        plt.show()
        print('nodes in network:', self.G.nodes)
        print('egdes in network:', self.G.edges)
        print('egdes num in network:', self.G.number_of_edges())
        print(self.G.nodes[1])

    def add_node_to_net(self, class_num, node_value, node_name='new_node'):
        base_index = self.per_class_data_len * class_num   #class_num : 0-2,三个类别的类名
        e_radius_distances, e_radius_indices = self.epsilon_radius(self.knn_areas[class_num], [node_value], self.e_radius[class_num])
        knn_distances, knn_indices = self.kNN(self.knn_areas[class_num], [node_value])
        print('123',e_radius_indices)
        print(knn_indices)

        """什么意思"""
        if 0 in knn_distances:
            return class_num

        #新节点连边
        edge_list = []
        self.G.add_node(node_name, class_num=-1, value=node_value)  #-1什么意思
        if (len(e_radius_indices) > (self.k - 1)):

            for index, neigh_idx in enumerate(e_radius_indices): #
                edge = (base_index + neigh_idx, node_name, e_radius_distances[0][index])
                edge_list.append(edge)
        else:
            for index, neigh_idx in enumerate(knn_indices[0]):  #
                print(index, neigh_idx)
                edge = (base_index + neigh_idx, node_name, knn_distances[0][index])
                #base_index + neigh_idx：添加连边；  knn_distances[0][index]:节点间距离，权重
                edge_list.append(edge)
        self.G.add_weighted_edges_from(edge_list)
        print('添加边数',len(edge_list))
        print(edge_list)

        current_class_nodes = list(range(base_index, base_index + self.per_class_data_len)).append('new_node')
        net_measures = self.calculate_net_measures(net=self.G, nodes=current_class_nodes)


        v1, v2 = np.array(self.net_measures[class_num]), np.array(net_measures)
        distance = np.linalg.norm(v1 - v2) #求两个向量的欧氏距离（#每一个类前后measures欧差值）

        return distance, net_measures


    def draw_new_node(self,node_value,node_name='new_node'):

        #添加节点信息到字典中
        pos={}
        for i,v in enumerate(self.train_x):
            pos[i] = v
        pos[node_name] = np.squeeze(node_value)  #删除node_value这一维度
        print(pos)

        """0啥意思"""
        color_list = [self.color_map.get(self.G.nodes[node]['class_num'], 0) for node in self.G.nodes()]
        # print(pos[1].shape)
        # print(pos['new_node'].shape)
        # print(color_list)
        nx.draw(self.G, pos=pos, with_labels=True, node_color=color_list)

        plt.show()
        print(self.G.edges)

    def predict(self, node_value, node_name='new_node'):
        variance_list = []
        for idx in range(0, self.num_classes): #循环遍历每一个类别网络
            #添加到不同类别里面比较插入前插入后的measures
            distance, measure = self.add_node_to_net(idx, node_value, node_name)
            variance_list.append(distance)  #每一个类前后measures欧差值
            self.G.remove_node(node_name)

        result = variance_list.index(min(variance_list)) #找到最小欧差值的索引（这里指类别）
        self.add_node_to_net(result, node_value, node_name)

        print(variance_list)
        print('predict', result)
        # print(self.train_x.shape)

        self.draw_new_node(node_value)




node=[1.5, 1.5]
a = NetworkDataClassification(5, 3, 4)
a.predict(node)
