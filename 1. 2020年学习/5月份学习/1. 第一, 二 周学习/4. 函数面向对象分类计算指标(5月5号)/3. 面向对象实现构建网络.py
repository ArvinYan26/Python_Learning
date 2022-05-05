import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

class DataClassification(object):
    """iris数据分类"""
    def __init__(self, k):
        """初始化"""
        self.X_net, self.X_test, self.Y_net, self.Y_test = self.data_preprocess()
        self.k = k
        self.knn_distances, self.knn_indices = self.KNN()
        self.e_radius_distances,  self.e_radius_indices, self.radius = self.e_radius()
        #self.g = None
        #self.label = True

    def __str__(self):
        return self.X_net, self.Y_net

    def get_iris_data(self):
        """获取数据集"""
        iris = load_iris()
        iris_data = iris.data
        iris_target = iris.target
        return iris_data, iris_target

    def data_preprocess(self):
        """数据集划分和特征工程（归一化）"""
        iris_data, iris_target = self.get_iris_data()
        X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)

        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_predict = scaler.transform(X_predict)
        #print(X_train)
        return X_train, X_predict, Y_train, Y_predict

    def KNN(self):
        """Knn计算距离和索引"""
        nbrs = NearestNeighbors(self.k+1, metric='euclidean')
        nbrs.fit(self.X_net)
        distances, indices = nbrs.kneighbors(self.X_net)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        return distances, indices

    def e_radius(self):
        """电子半径方法，计算距离和索引"""
        radius = np.median(self.knn_distances)
        nbrs = NearestNeighbors(self.k, metric='euclidean')
        nbrs.fit(self.X_net)
        nbrs.set_params(radius=radius)
        distances, indices = nbrs.radius_neighbors(self.X_net)
        return distances, indices, radius


    def build_network(self, label):
        """构建网络"""
        #测试数据
        #print(self.X_net, self.Y_net)
        #print(self.knn_distances, self.knn_indices)
        #print(self.e_radius_distances, self.e_radius_indices)

        self.g = nx.Graph()
        #添加节点

        for index, instance in enumerate(self.X_net):
            self.g.add_node(str(index), values=instance, type="net", label=self.Y_net[index])
            #print(index, instance)
        #print(self.knn_distances)
        #选择方法
        if self.radius > self.k/3 :
            for index, nbrs_index in enumerate(self.e_radius_indices):
                for tempi, eve_index in enumerate(nbrs_index):
                    #if (not str(index)) == str(eve_index):
                    if self.g.nodes()[str(eve_index)]["label"] == self.g.nodes()[str(index)]["label"]:
                        self.g.add_edge(str(index), str(eve_index), weight=self.e_radius_distances[index][tempi])
        else:
            for index, nbrs_index in enumerate(self.knn_indices):
                for tempi, eve_index in enumerate(nbrs_index):
                    #if (not str(index)) == str(eve_index):
                    if self.g.nodes()[str(eve_index)]["label"] == self.g.nodes()[str(index)]["label"]:
                        self.g.add_edge(str(index), str(eve_index), weight=self.knn_distances[index][tempi])

        #plt.subplot(211)
        #self.calculate_measure(self.g)
        color_map = {0: 'r', 1: 'y', 2: 'purple'}
        color_list = [color_map[self.g.nodes[node]['label']] for node in self.g.nodes()]
        nx.draw_networkx(self.g, node_color=color_list, with_labels=False )
        print(len(self.g.nodes()))
        print(len(self.g.edges()))

        plt.show()
        self.get_subgraph()




    def calculate_measure(self, g):
        """

        :param g: X_net构建的网络图
        :return:
        """
        des = nx.density(g)
        print("密度：%f" % des)

        # 节点度
        deg = nx.degree(g)
        print(deg)

        # 平均度
        ave_deg = g.number_of_edges() * 2 / g.number_of_nodes()
        print("平均度为：%f" % ave_deg)

        # 度分布直方图
        #distribution = nx.degree_histogram(g)
        #print(distribution)
        # 同配性

        # Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(g)
        print("同配性：%f" % assortativity)

        # 平均聚类系数
        ave_cluster = nx.average_clustering(g)
        print("平均聚类系数：%f" % ave_cluster)

        # 传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(g)
        print("三角形分数：%f" % tran)

        # 平均最短路径长度(需要图是连通的)
        ave_shorest = nx.average_shortest_path_length(g)
        print("平均最短路径：", ave_shorest)


    def get_subgraph(self):
        """得到子图，并画出来"""
        Gcc = sorted(nx.connected_components(self.g), key=len, reverse=True)
        self.g0 = self.g.subgraph(Gcc[0])
        self.calculate_measure(self.g0)
        plt.subplot(131)
        nx.draw_networkx(self.g0, with_labels=False)
        print("节点数：", len(self.g0.nodes()))
        print("连边：", len(self.g0.edges()))
        print(" ")

        self.g1 = self.g.subgraph(Gcc[1])
        self.calculate_measure(self.g1)
        plt.subplot(132)
        nx.draw_networkx(self.g1, with_labels=False)
        print(len(self.g1.nodes()))
        print(len(self.g1.edges()))
        print(" ")

        self.g2 = self.g.subgraph(Gcc[2])
        self.calculate_measure(self.g2)
        plt.subplot(133)
        nx.draw_networkx(self.g2, with_labels=False)
        print(len(self.g2.nodes()))
        print(len(self.g2.edges()))

        plt.show()


def main():
    dc = DataClassification(5)

    #dc.data_preprocess()
    dc.build_network(label=True)

if __name__ == '__main__':
    main()





