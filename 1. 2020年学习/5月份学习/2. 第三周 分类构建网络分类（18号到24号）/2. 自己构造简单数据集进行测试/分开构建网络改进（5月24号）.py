import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from test import split_data  # test文件中的spli_data函数
from make_simple_dataset import generate_dataset
import time


class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        # self.g = []
        self.X_net, self.Y_net, self.X_items, self.Y_items = self.get_iris()
        self.data_len = len(self.X_net)  # 此程序是24
        self.num_class = num_class
        self.per_class_data_len = int(self.data_len / self.num_class)
        self.plot_data(self.X_net)  # 直接执行此函数
        self.nbrs = []  # 用来存储是哪个类别网络的nbrs
        self.radius = []  # 用来存储是哪个类别的
        #self.nodes_list = []  #只要是初始化的变量，那么只要改变了，后续的改变就会继续添加，所以才会边数越来越多
        #self.edges_list = []
        self.color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'black'}
        self.net_measure = {}
        self.G0 = None
        self.G1 = None
        self.G2 = None


        self.build_X_train_network()

    def get_iris(self):
        """获取数据集"""
        """
        iris = load_iris()
        iris_data = iris.data  #[:, 2:]
        iris_target = iris.target
        """
        data, label = generate_dataset()

        # 存储切分后的数据，训练集和测试集
        X_train1 = []
        Y_train1 = []
        X_train2 = []
        Y_train2 = []
        train_data = []
        train_target = []

        # 第一次划分，train_data, train_target （0.8比例，多数），  X_train1, Y_train1 （0.2，少数）
        train_data, train_target, X_train1, Y_train1 = split_data(data, label, X_train1, Y_train1, train_data,
                                                                  train_target)
        # print("训练集：")
        # print(np.array(X_train1), np.array(Y_train1))
        """
        print("总的数据集:")
        print(data, label)
        print("训练集")
        print(train_data, train_target)
        print("维度：", data.ndim)
        print("测试集：")
        print(np.array(X_train1), np.array(Y_train1))
        """
        return train_data, train_target, X_train1, Y_train1

    def plot_data(self, data):
        """画出数据"""
        node_style = ["ro", "go", "bo"]
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
        """
        des = nx.density(g)
        print("密度：%f" % des)

        # 度分布直方图
        # distribution = nx.degree_histogram(g)
        # print(distribution)
        measures.append(des)
        # 节点度
        deg = nx.degree(g)
        #print(deg)
        """
        # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        # 1.  平均度
        ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
        ave_deg = round(ave_deg, 3)
        # print("平均度为：%f" % ave_deg)
        # measures.append(ave_deg)

        # 2.  平均最短路径长度(需要图是连通的)
        ave_shorest = nx.average_shortest_path_length(G)
        ave_shorest = round(ave_shorest, 3)
        # print("平均最短路径：", ave_shorest)
        # measures.append(ave_shorest)

        # 3.  平均聚类系数
        ave_cluster = nx.average_clustering(G)
        ave_cluster = round(ave_cluster, 3)
        # print("平均聚类系数：%f" % ave_cluster)
        # measures.append(ave_cluster)

        # 4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G)
        assortativity = round(assortativity, 3)
        # print("同配性：%f" % assortativity)
        # measures.append(assortativity)

        # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(G)
        tran = round(tran, 3)
        # print("三角形分数：%f" % tran)
        # measures.append(tran)

        return ave_deg, ave_shorest, ave_cluster, assortativity, tran

    def build_edges(self, G, current_data, i):
        """
        # print("类别：", i)
        current_data = self.X_net[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]
        print(current_data)
        for index, instance in enumerate(current_data):
            node_info = (index, {"value": list(instance), "class_num": i, "type": "train"})
            self.nodes_list.append(node_info)
        # print(self.nodes_list)
        """
        edges_list = []
        # 切片范围必须是整型
        temp_nbrs = NearestNeighbors(self.k, metric='euclidean')
        temp_nbrs.fit(current_data)
        self.nbrs.append(temp_nbrs)  # 将每一类的nbrs都添加进列表， 这个
        knn_distances, knn_indices = self.KNN(temp_nbrs, current_data)
        print(knn_distances, knn_indices)
        temp_radius = self.get_radius(knn_distances)

        self.radius.append(temp_radius)  # 将每一类的radius都添加进radius
        print("temp_radius", self.radius)
        radius_distances, radius_indices = self.epsilon_radius(temp_nbrs, current_data, temp_radius)
        print(np.array(radius_distances), np.array(radius_indices))
        # 添加连边
        for idx, one_data in enumerate(current_data):  # 这个语句仅仅是获取索引indx，然后给他连边
            print("idx:", idx)
            print("radius_indices[idx]", len(radius_indices[idx]))
            if (len(radius_indices[idx])) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                # print(radius_indices[idx])
                for index, nbrs_indices in enumerate(radius_indices[idx]):
                    print(index, nbrs_indices)
                    # for indices, eve_index in enumerate(nbrs_indices):
                    # print(indices, eve_index)
                    if idx == nbrs_indices:  # 如果是本身，就跳过，重新下一个循环
                        continue
                    edge = (idx, nbrs_indices) #, radius_distances[idx][index]
                    edges_list.append(edge)
            else:
                print("knn连边")
                print(idx, knn_indices[idx])
                for index, nbrs_indices in enumerate(knn_indices[idx]):
                    # print("信息")
                    print(index, nbrs_indices)
                    # for indices, eve_index in enumerate(nbrs_indices):
                    # print(indices, eve_index)
                    if idx == nbrs_indices:  # 如果是本身，就跳过，重新下一个循环
                        continue
                    edge = (idx, nbrs_indices) # knn_distances[idx][index]
                    edges_list.append(edge)
        print(edges_list)
        G.add_edges_from(edges_list)
        print("边数：", len(G.edges()))
        # self.get_net_measures()  #
        # print(G.nodes())
        # color_list = [self.color_map[self.G.nodes[node]['class_num']] for node in self.G.nodes()]
        # plt.subplot(211)
        # print(self.X_net.shape)
        nx.draw_networkx(G, node_color=self.color_map[i], with_labels=True, node_size=300)  # 节点默认大小为300
        plt.title("X_net")
        plt.show()
        # print(self.G.nodes())
        # print("X_net节点数：", len(G.nodes()))
        # print("X_net边数：", len(G.edges()))

    def build_X_train_network(self):
        """
        分开构建网络
        :return:
        API reference
            klearn.neighbors.NearestNeighbors
                - https://scikit-learn.org/stable/modules/generated
        """

        for i in range(self.num_class):  # 按类别循环遍历每一类别的每一个数据
            nodes_list = []
            # print("类别：", i)
            current_data = self.X_net[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]
            print(current_data)
            for index, instance in enumerate(current_data):
                node_info = (index, {"value": list(instance), "class_num": i, "type": "train"})
                nodes_list.append(node_info)
            # print(self.nodes_list)
            if i == 0:
                self.G0 = nx.Graph()
                self.build_edges(self.G0, current_data, i)
                self.net_measure[i] = self.calculate_measure(self.G0)
                print(self.net_measure)
            if i == 1:
                self.G1 = nx.Graph()
                self.build_edges(self.G1, current_data, i)
                self.net_measure[i] = self.calculate_measure(self.G1)
                print(self.net_measure)
            if i == 2:
                self.G2 = nx.Graph()
                self.build_edges(self.G2, current_data, i)
                self.net_measure[i] = self.calculate_measure(self.G2)
                print(self.net_measure)


if __name__ == '__main__':
    DataClassification(3, 3)
