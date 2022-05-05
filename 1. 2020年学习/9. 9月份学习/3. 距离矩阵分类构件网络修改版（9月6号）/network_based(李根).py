import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import Normalizer
import networkx as nx
import math


class NetworkBaseModel():

    def __init__(self, num_classes, k):
        '''

        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
        self.per_class_data_len = None
        self.train_len = None
        self.num_classes = num_classes
        self.k = k

        self.train_x = None
        self.train_y = None

        self.neigh_models = []  #
        self.e_radius = []

        self.G = nx.Graph()
        self.nodes_list = []
        self.edges_list = []
        self.color_map = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', -1: 'black'}  # 节点类别颜色地图
        self.name_map = {0: 'covid', 1: 'normal', 2: 'other'}

        self.net_measures = []  # {1:{'averge_degree':[]}}

    def get_radius(self, distances):
        return float(np.median(distances))

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: array (n, m)
            y: array (n,)
        Returns:

        """
        self.train_x = x
        self.train_y = y
        self.train_len = len(x)
        self.per_class_data_len = int(self.train_len / self.num_classes)

        "1. build network"
        for ith_class in range(self.num_classes):
            nodes = []
            edges = []
            dataset = x[ith_class * self.per_class_data_len:(ith_class + 1) * self.per_class_data_len]
            neigh_model = NearestNeighbors(n_neighbors=self.k, metric='minkowski')
            neigh_model.fit(dataset)
            knn_neigh_dist, knn_neigh_idx = neigh_model.kneighbors(X=dataset)

            self.e_radius.append(self.get_radius(knn_neigh_dist))

            radius_neigh_dist, radius_neigh_idx = neigh_model.radius_neighbors(X=dataset,
                                                                               radius=self.e_radius[ith_class])
            self.neigh_models.append(neigh_model)

            # add nodes
            for idx, item in enumerate(dataset):
                center_node = (
                    self.name_map[ith_class] + '-center',
                    {'value': None, 'class': self.name_map[ith_class], 'type': 'center'})
                nodes.append(center_node)
                node = (self.name_map[ith_class] + '-%s' % idx,
                        {'value': item, 'class': self.name_map[ith_class], 'type': 'train'})
                nodes.append(node)

            self.G.add_nodes_from(nodes)
            # add edges
            for idx, (i, j) in enumerate(zip(knn_neigh_idx, radius_neigh_idx)):
                edge2center = (self.name_map[ith_class] + '-%s' % idx, self.name_map[ith_class] + '-center')
                edges.append(edge2center)
                if len(i) > len(j):
                    for item_idx, item in enumerate(i):
                        # edge = (name_map[ith_class] + '-%s' % idx, name_map[ith_class] + '-%s' % item,
                        #         knn_neigh_dist[idx][item_idx])
                        edge = (self.name_map[ith_class] + '-%s' % idx, self.name_map[ith_class] + '-%s' % item,)
                        edges.append(edge)
                else:
                    for item_idx, item in enumerate(j):
                        # edge = (name_map[ith_class] + '-%s' % idx, name_map[ith_class] + '-%s' % item,
                        #         radius_neigh_dist[idx][item_idx])
                        edge = (self.name_map[ith_class] + '-%s' % idx, self.name_map[ith_class] + '-%s' % item,)
                        edges.append(edge)
            # self.G.add_weighted_edges_from(edges)
            self.G.add_edges_from(edges)

        # 加载各个组件的网络刻度

        seg_point = self.per_class_data_len + 1
        for ith_class in range(self.num_classes):
            nodes = list(self.G.nodes)[seg_point * ith_class:(seg_point * (ith_class + 1))]
            sub_G = self.G.subgraph(nodes=nodes)
            self.net_measures.append(self.calculate_net_measures(sub_G))

        print('训练完成,三个组件的网络刻度', self.net_measures)

        # plt.figure()
        #
        # # pos = nx.spring_layout(self.G)
        #
        # nx.draw(self.G, with_labels=False, node_size=10, width=1)
        # plt.show()

    def predict(self, x: np.ndarray):
        """
        
        Args:
            x: numpy (n,m)

        Returns:

        """

        neigh_model = KNeighborsClassifier(n_neighbors=self.k)
        neigh_model.fit(self.train_x, self.train_y)
        y_pred_prob = neigh_model.predict_proba(
            x)  # [[0.66666667 0.         0.33333333],[0.66666667 0.         0.33333333]]
        y_pred = []
        # print(y_pred_prob)
        for idx, item in enumerate(y_pred_prob):

            max_prob = np.max(item)
            max_prob_idx = np.where(item == max_prob)[0]
            if len(max_prob_idx) > 1:
                print('knn失效,开始计算网络刻度')
                print('idx', idx, 'knn结果', item)
                print('对应的组件', max_prob_idx)
                max_prob_idx = list(max_prob_idx)
                res_measures = []
                for i_th_class in max_prob_idx:

                    neigh_model = self.neigh_models[i_th_class]

                    knn_neigh_dist, knn_neigh_idx = neigh_model.kneighbors(X=x[idx].reshape(1, -1))
                    # print('--', knn_neigh_idx)
                    radius_neigh_dist, radius_neigh_idx = neigh_model.radius_neighbors(X=x[idx].reshape(1, -1),
                                                                                       radius=self.e_radius[i_th_class])

                    node = [(self.name_map[i_th_class] + '-pred',
                             {'value': x[idx], 'class': self.name_map[i_th_class], 'type': 'test'})]

                    self.G.add_nodes_from(node)
                    edges = []
                    for (i, j) in zip(knn_neigh_idx, radius_neigh_idx):

                        if len(i) >= len(j):
                            "knn"
                            for neigh_idx in i:

                                edge = (
                                    self.name_map[i_th_class] + '-pred', self.name_map[i_th_class] + '-%s' % neigh_idx)

                                edges.append(edge)
                        else:
                            "radius"
                            for radius_idx in j:
                                edge = (
                                    self.name_map[i_th_class] + '-pred', self.name_map[i_th_class] + '-%s' % radius_idx)
                                edges.append(edge)

                    self.G.add_edges_from(edges)

                    seg_point = self.per_class_data_len + 1  # include center point

                    nodes = list(self.G.nodes)[seg_point * i_th_class:seg_point * (i_th_class + 1)]+[self.name_map[i_th_class] + '-pred']  # 包括了当前要预测的节点+1

                    sub_G = self.G.subgraph(nodes=nodes)
                    # print(list(sub_G.nodes))
                    sub_G_measures = self.calculate_net_measures(sub_G)
                    print('新的刻度', sub_G_measures)
                    res_measures.append(sub_G_measures)

                    self.G.remove_node(self.name_map[i_th_class] + '-pred')

                pred = self.get_y_by_measure(res_measures, max_prob_idx)
                print(pred)
            else:
                pred = max_prob_idx[0]
            y_pred.append(pred)
        return np.array(y_pred)

    def get_y_by_measure(self, corre_measures, corre_cls_idxs):

        """

        Args:
            corre_measures:  两个组件的网络的网络刻度。 [ [1,2,3], [1,2,3] ]
            max_prob_idx:   对应的组件类别                       [0，1]

        Returns:
            0 or 1
        """
        measures_change = []
        print('dd',corre_measures)
        for idx, class_idx in enumerate(corre_cls_idxs):
            past_measures = np.array(self.net_measures[class_idx])

            new_measures = np.array(corre_measures[idx])
            print('组件%s'%idx,'past', past_measures, 'new',new_measures )
            dist = np.linalg.norm(past_measures - new_measures)

            measures_change.append(float(dist))

        cls_idx = measures_change.index(min(measures_change))
        pred = corre_cls_idxs[cls_idx]
        print('三个组件中的网络刻度变化',measures_change)
        return pred



    def plot_Graph(self, data, num_classes):

        node_style = ['ro', 'go', 'bo', 'yo']
        per_class_data_len = int(data.shape[0] / num_classes)
        for i in range(self.num_classes):
            print('i', i)
            print(node_style[i])
            plt.plot(data[per_class_data_len * i:per_class_data_len * (i + 1), 0],
                     data[per_class_data_len * i:per_class_data_len * (i + 1), 1],
                     node_style[i],
                     label=node_style[i])
            plt.legend()  # for add new nodes
        plt.show()

        return 1

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

    def calculate_net_measures(self, net):
        # print(net.nodes)
        degree_assortativity = nx.degree_assortativity_coefficient(G=net)
        average_clustering_coefficient = nx.average_clustering(G=net)
        average_degree = np.mean([i[1] for i in nx.degree(net)])  # nx.degree 返回的是每个节点的度, 所以要获取再求平均
        dimameter = nx.algorithms.distance_measures.diameter(net)
        return [degree_assortativity, average_clustering_coefficient, average_degree,dimameter]

    def generate_net_measures(self):
        seg_point = self.per_class_data_len + 1
        for per_class in range(self.num_classes):
            nodes = list(self.G.nodes)[seg_point * per_class:seg_point * (per_class + 1)]
            sub_G = self.G.subgraph(nodes=nodes)
            self.net_measures[per_class] = self.calculate_net_measures(sub_G)

    def net_degree_assortativity(self, net):
        return nx.degree_assortativity_coefficient(G=net)

    def average_clustering_coefficient(self, net, ):
        return nx.average_clustering(G=net)

    def average_degree(self, net, nodes):
        return nx.average_degree_connectivity(net, nodes=nodes)


