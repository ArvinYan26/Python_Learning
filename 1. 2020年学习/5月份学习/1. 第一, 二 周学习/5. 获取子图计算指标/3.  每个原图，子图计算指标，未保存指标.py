import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import time

class DataClassification(object):
    """iris数据集分类"""
    def __init__(self, k, calss):
        self.k = k
        self.class_num = calss
        self.iris_data, self.iris_target = self.get_iris()
        self.X_net, self.Y_net, self.X_items, self.Y_items, self.X_predict, self.Y_predict = self.data_preprocess()
        self.nbrs, self.radius = self.build_network(label=True)
        #self.color_map = {0: 'r', 1: 'y', 2: 'purple'}
        #self.radius = self.get_radius()
        #self.g = None  如果此处初始化None,那么其他任何函数最初调用它的时候都为空，而不是拿到了已经改变的self.g

    def get_iris(self):
        """获取数据集"""
        iris = load_iris()
        iris_data = iris.data
        iris_target = iris.target
        return iris_data, iris_target

    def data_preprocess(self):
        """特征工程（归一化）"""
        #切分数据集
        X_train, X_predict, Y_train, Y_predict = train_test_split(self.iris_data, self.iris_target, test_size=0.2)
        X_net, X_items, Y_net, Y_items = train_test_split(X_train, Y_train, test_size=0.2)
        #归一化
        scaler = preprocessing.MinMaxScaler().fit(X_net)
        X_net = scaler.transform(X_net)

        return X_net, Y_net, X_items, Y_items,  X_predict, Y_predict

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

    def build_network(self, label):
        """
        :return:
            1. great network
            2. add nodes
            3. calculate distances and indices
            4. add edges
            5. plot graph
        API reference
            klearn.neighbors.NearestNeighbors
                - https://scikit-learn.org/stable/modules/generated
        """
        self.g = nx.Graph()
        #print(len(self.X_net))
        #添加节点
        for index, instance in enumerate(self.X_net):
            self.g.add_node(str(index), values=instance, typeNode='net', label=self.Y_net[index])
        nbrs = NearestNeighbors(self.k, metric='euclidean')
        nbrs.fit(self.X_net)
        distances, indices = self.KNN(nbrs, self.X_net)
        #print(distances, indices)
        radius = self.get_radius(distances)
        radius_distances, radius_indices = self.epsilon_radius(nbrs, self.X_net, radius)
        #print(radius_distances, radius_indices)
        #添加连边
        if radius/self.class_num > self.k:
            for index, nbrs_indices in enumerate(radius_indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if index == eve_index:  #如果是本身，就跳过，重新下一个循环
                        continue
                    if self.g.nodes()[str(eve_index)]['label'] == self.g.nodes()[str(index)]['label']:
                        self.g.add_edge(str(eve_index), str(index), weight=radius_distances[index][indices])
        else:
            for index, nbrs_indices in enumerate(indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if index == eve_index: #如果是本身，就跳过，重新下一个循环
                        continue
                    if self.g.nodes()[str(eve_index)]['label'] == self.g.nodes()[str(index)]['label']:
                        self.g.add_edge(str(eve_index), str(index), weight=distances[index][indices])

        color_map = {0: 'r', 1: 'y', 2: 'purple'}
        color_list = [color_map[self.g.nodes[node]['label']] for node in self.g.nodes()]
        nx.draw_networkx(self.g, node_color=color_list, with_labels=False)
        plt.title("X_net")
        plt.show()
        #print(self.g.nodes())
        print("节点数：", len(self.g.nodes()))
        print("边数：", len(self.g.edges()))
        self.get_subgraph(self.g)
        #number = nx.number_connected_components(self.g)
        #print(number)
        return nbrs, radius


    def single_node_insert(self, X_data, Y_label):
        """
        steps:
            1. 添加节点（for循环）
            2. 依次添加节点到三个子网络中，计算每个插入新节点后的子网络的5个指标，存储起来（列表）
            3. 计算每一个子网络插入前和插入后的网络指标相似度，存储起来
            4. 比较这三个相似度差值大小，将节点插入到差值最小的那个字网络中去
            5. 计算这个插入到子网络中去的节点在这个网络中最近的邻居，进行连边。

        :param X_items: one node inserted
        :param nodeindex: the index of the inserted node
        :param Y_items: label of the inserted node
        :return:
        """
        #print(self.X_test, self.Y_test)
        #g = self.g
        #print(len(g.nodes()))
        #insert_node_id = len(list(self.g.nodes()))
        #print(insert_node_id)

        #添加节点
        for index, instance in enumerate(X_data):
            if (not len(Y_label) == 0):
                label = Y_label[index]
                #print(index, instance)
            insert_node_id = len(list(self.g.nodes()))
            #print(insert_node_id)
            self.g.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
            #print(len(self.g.nodes()))

            radius_distances, radius_indices = self.nbrs.radius_neighbors(X_data)
            distances, indices = self.nbrs.kneighbors(X_data)
            #print(distances, indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉

            #添加到训练网络中
            if self.radius/self.class_num > self.k:
                for index, nbrs_indices in enumerate(radius_indices):
                    for indices, eve_index, in enumerate(nbrs_indices):
                        if index == eve_index:
                            continue
                        self.g.add_edge(str(eve_index), str(insert_node_id), weight=radius_distances[index][indices])
                        print(len(self.g.edges()))
            else:
                #这里一定注意，是和那些邻居的节点索引链接，所以循环的是索引,否则会多出来很多边。
                for index, nbrs_indices in enumerate(indices):
                    for indices, eve_index in enumerate(nbrs_indices):
                        if index == eve_index:
                            continue
                        self.g.add_edge(str(eve_index), str(insert_node_id), weight=distances[index][indices])
            #print(len(self.g.nodes))
            #self.calculate_measure()
            #print(" ")
        color_map = {0: 'r', 1: 'y', 2: 'purple'}
        color_list = [color_map[self.g.nodes[node]['label']] for node in self.g.nodes()]
        nx.draw_networkx(self.g, node_color=color_list, with_labels=False)
        plt.title("node_insert")
        plt.show()
        print("节点数：", len(self.g.nodes()))
        print("边数：", len(self.g.edges()))
        self.get_subgraph(self.g)
        #number = nx.number_connected_components(self.g)
        #print(number)

    def get_subgraph(self, g):
        """得到子图，并画出来"""
        Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
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
        print("节点数：", len(self.g1.nodes()))
        print("边数：", len(self.g1.edges()))
        print(" ")

        self.g2 = self.g.subgraph(Gcc[2])
        self.calculate_measure(self.g2)
        plt.subplot(133)
        nx.draw_networkx(self.g2, with_labels=False)
        print("节点数：", len(self.g2.nodes()))
        print("边数：", len(self.g2.edges()))

        plt.show()

    def calculate_measure(self, g):
        """
        :param g: X_net构建的网络图
        :return:
        """

        des = nx.density(g)
        print("密度：%f" % des)

        # 节点度
        deg = nx.degree(g)
        #print(deg)

        # 平均度
        ave_deg = g.number_of_edges() * 2 / g.number_of_nodes()
        print("平均度为：%f" % ave_deg)

        # 度分布直方图
        #distribution = nx.degree_histogram(g)
        #print(distribution)

        # 同配性 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(g)
        print("同配性：%f" % assortativity)

        # 平均聚类系数
        ave_cluster = nx.average_clustering(g)
        print("平均聚类系数：%f" % ave_cluster)

        # 传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(g)
        print("三角形分数：%f" % tran)

        # 平均最短路径长度(需要图是连通的)
        #ave_shorest = nx.average_shortest_path_length(g)
        #print("平均最短路径：", ave_shorest)



def main():
    dc = DataClassification(6, 3)
    X_net, Y_net, X_items, Y_items, X_predict, Y_predict = dc.data_preprocess()
    #dc.build_network(label=True)
    #time.sleep(1)
    dc.single_node_insert(X_items, Y_items)
    #time.sleep(1)
    dc.single_node_insert(X_predict, Y_predict)


if __name__ == '__main__':
    main()