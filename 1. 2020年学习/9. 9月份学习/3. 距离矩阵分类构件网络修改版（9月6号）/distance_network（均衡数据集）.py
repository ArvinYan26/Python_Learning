import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances,paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
from networkx.algorithms.distance_measures import center as nxCenter

class NetworkBaseModel():

    def __init__(self, num_classes):
        '''

        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
        self.per_class_data_len = None
        self.train_len = None
        self.num_classes = num_classes


        self.train_x = None
        self.train_y = None

        self.neigh_models = []  #
        self.e_radius = []

        self.G_list = []
        self.mean_dis_list = []
        self.nodes_list = []
        self.edges_list = []

        self.net_measures = []  # {1:{'averge_degree':[]}}

    def draw_graph(self):
        nx.draw_networkx(self.G)
        plt.show()

    def fit(self, x: np.ndarray):
        """

        Args:
            x: array (n, m) 输入数据
        Returns:

        """
        self.train_x = x

        self.train_len = len(x)
        self.per_class_data_len = int(self.train_len / self.num_classes)

        "1. build network"
        for ith_class in range(self.num_classes):

            "adjacency matrix"
            dataset = x[ith_class * self.per_class_data_len:(ith_class + 1) * self.per_class_data_len]



            adj_matrix = euclidean_distances(dataset, dataset)

            # 要先求两两平均距离，后面会改动数据。

            mean_dis = np.sum(adj_matrix) / (self.per_class_data_len ** 2 - self.per_class_data_len)
            self.mean_dis_list.append(mean_dis)

            # 把0距离先变成最大999

            adj_matrix[adj_matrix == 0] = 999

            # 先把每一行最小的值设置为1， 即保证每个节点和距离最小的节点有连边

            for idx, item in enumerate(adj_matrix):
                min_idx = np.argmin(item)
                # 因为是对称矩阵
                adj_matrix[idx, min_idx] = 1
                adj_matrix[min_idx, idx] = 1
            # 低于平均距离的建立连边


            adj_matrix[adj_matrix < mean_dis] = 1

            # 将没有连边的部分都设置为0
            adj_matrix[adj_matrix!=1] = 0

            "-- 至此，邻接矩阵生成完毕 --"

            # 构建网络（邻接矩阵构建网络）
            self.G_list.append(nx.from_numpy_matrix(adj_matrix))
            #
            # print(self.G_list[ith_class].nodes)
            # print(self.G_list[ith_class].edges)

            "!- merge diff subgraph in a component -!"


            sub_conponents = nx.connected_components(self.G_list[ith_class])

            # print('社区数目',len(sub_conponents))
            center_node = nxCenter(self.G_list[ith_class])[0]


            print('---Component----')

            for i in sub_conponents:

                sub_G = self.G_list[ith_class].subgraph(i)

                sub_center_node = nxCenter(sub_G)[0]
                edge = (sub_center_node, center_node)

                self.G_list[ith_class].add_edges_from([edge])

        #print(self.G_list)

        # 加载各个组件的网络刻度

        self.G = nx.Graph()
        for i in self.G_list:
            self.G.add_nodes_from(i)



        for ith_class in range(self.num_classes):
            # res = self.calculate_net_measures(self.G_list[ith_class]).append(self.mean_dis_list[ith_class])
            res = self.calculate_net_measures(self.G_list[ith_class])

            res +=[self.mean_dis_list[ith_class]]

            self.net_measures.append(res)


        """
        plt.figure()
        #
        pos = nx.spring_layout(self.G)
        
        nx.draw(self.G, with_labels=False, node_size=10, width=1)
        plt.show()
        """

    def predict(self, x: np.ndarray):
        """

        Args:
            x: numpy (n,m)

        Returns:

        """

        y_pred = []

        for item in x:  # 遍历测试数据
            # 分别插入到三个网络当中

            item = item.reshape(1,-1)

            new_measures = []


            for ith_class in range(self.num_classes):

                dis_matrix = euclidean_distances(item, self.train_x[self.per_class_data_len*ith_class:self.per_class_data_len*(ith_class+1)])

                aver_dis = np.mean(dis_matrix[0])  # 新节点的平均距离

                min_idx = int(np.argmin(dis_matrix[0]) )         # 最小距离

                edge_idxs = list(np.argwhere(dis_matrix[0]<self.mean_dis_list[ith_class]))


                # 添加节点， 添加连边
                test_node = ('test_node', {'value': None, 'class': 'test', 'type': 'test'})

                self.G_list[ith_class].add_nodes_from([test_node])

                edges = [('test_node', min_idx)]
                for edge_idx in edge_idxs:

                    edges.append(('test_node', int(edge_idx)))

                self.G_list[ith_class].add_edges_from(edges)

                res = self.calculate_net_measures(self.G_list[ith_class])

                res.append(aver_dis + self.mean_dis_list[ith_class])

                new_measures.append(res)

                self.G_list[ith_class].remove_node('test_node')

            diff = paired_euclidean_distances(self.net_measures, new_measures)

            y_pred.append(int(np.argmin(diff)))

        return np.array(y_pred)







    def calculate_net_measures(self, net):


        degree_assortativity = nx.degree_assortativity_coefficient(G=net)

        average_clustering_coefficient = nx.average_clustering(G=net)
        average_degree = np.mean([i[1] for i in nx.degree(net)])  # nx.degree 返回的是每个节点的度, 所以要获取再求平均
        dimameter = nx.algorithms.distance_measures.diameter(net)
        return [degree_assortativity, average_clustering_coefficient, average_degree, dimameter]

    def generate_net_measures(self):
        seg_point = self.per_class_data_len + 1
        for per_class in range(self.num_classes):
            nodes = list(self.G.nodes)[seg_point * per_class:seg_point * (per_class + 1)]
            sub_G = self.G.subgraph(nodes=nodes)
            self.net_measures[per_class] = self.calculate_net_measures(sub_G)



# 使用方法

model = NetworkBaseModel(num_classes=3)

# 加载数据

train_x = np.random.randn(150,3)  # 这里假设有三类数据, 前50是第一类 中间50第二类, 最后50第三类
test_x = np.random.randn(15,3)     # 假设要预测15条数据

model.fit(train_x)


y = model.predict(test_x)

print(y)  # 输出[0 0 1 2 0 0 0 2 1 2 0 0 1 0 0]  0表示第一类 1表示第二类 2表示第三类
