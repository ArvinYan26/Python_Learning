import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

class DataClassification(object):
    """iris数据集分类"""
    def __init__(self, k, calss):
        self.k = k
        self.class_num = calss
        self.iris_data, self.iris_target = self.get_iris()
        self.X_net, self.Y_net, self.X_test, self.Y_test = self.data_preprocess()
        self.nbrs, self.radius = self.build_network()
        #self.radius = self.get_radius()
        self.e_radius = []
        #self.g = None  如果此处初始化None,那么其他任何函数最初调用它的时候都为空，而不是拿到了已经改变的self.g
        self.color_map = {0: 'red', 1: 'green', 2: 'yellow'}

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
        #归一化
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        return X_train, Y_train,  X_predict, Y_predict

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

    def build_network(self):
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
        #添加节点
        for index, instance in enumerate(self.X_net):
            self.g.add_node(str(index), values=instance, typeNode='net', labels=self.Y_net[index])
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
                    if self.g.nodes()[str(eve_index)]['labels'] == self.g.nodes()[str(index)]['labels']:
                        self.g.add_edge(str(eve_index), str(index), weight=radius_distances[index][indices])
        else:
            for index, nbrs_indices in enumerate(indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if index == eve_index: #如果是本身，就跳过，重新下一个循环
                        continue
                    if self.g.nodes()[str(eve_index)]['labels'] == self.g.nodes()[str(index)]['labels']:
                        self.g.add_edge(str(eve_index), str(index), weight=distances[index][indices])
        color_list = [self.color_map.get(self.g.nodes[node][3], 0) for node in self.g.nodes()]
        nx.draw(self.g, node_color=color_list)
        plt.show()
        #print(self.g.nodes())
        print(len(self.g.nodes()))
        print(len(self.g.edges()))
        return nbrs, radius

    def calculate_measure(self):
        """

        :param g: X_net构建的网络图
        :return:
        """
        des = nx.density(self.g)
        print("密度：%f" % des)
        # 节点度
        deg = nx.degree(self.g)
        print(deg)
        # 平均度
        ave_deg = self.g.number_of_edges() * 2 / self.g.number_of_nodes()
        print("平均度为：%f" % ave_deg)
        # 度分布直方图
        distribution = nx.degree_histogram(self.g)
        print(distribution)
        # 同配性
        # Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(self.g)
        print("同配性：%f" % assortativity)
        # 平均聚类系数
        ave_cluster = nx.average_clustering(self.g)
        print("平均聚类系数：%f" % ave_cluster)
        # 传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(self.g)
        print("三角形分数：%f" % tran)
        # 平均最短路径长度(需要图是连通的)
        # ave_shorest = nx.average_shortest_path_length(g)
        # print(ave_cluster)

    def single_node_insert(self):
        """

        :param instance: one node inserted
        :param nodeindex: the index of the inserted node
        :param label: label of the inserted node
        :return:
        """
        #print(self.X_test, self.Y_test)
        #g = self.g
        #print(len(g.nodes()))
        insert_node_id = len(list(self.g.nodes()))
        #print(insert_node_id)
        #添加节点
        for index, instance in enumerate(self.X_test):
            if (not len(self.Y_test) == 0):
                label = self.Y_test[index]
                #print(index, instance)
            insert_node_id = len(list(self.g.nodes()))
            #print(insert_node_id)
            self.g.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
            #print(len(self.g.nodes()))

            radius_distances, radius_indices = self.nbrs.radius_neighbors(self.X_test)
            distances, indices = self.nbrs.kneighbors(self.X_test)
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
            #self.calculate_measure()
            #print(" ")

        nx.draw(self.g)
        plt.show()
        print(len(self.g.nodes()))
        print(len(self.g.edges()))

    """
    def many_node_insert(self):
        for index, instance in enumerate(self.X_test):
            if (not len(self.Y_test) == 0):
                label = self.Y_test[index]
            print(index, instance)
            self.single_node_insert(instance, index, label)

        nx.draw(self.g)
        plt.show()
        print(len(self.g.nodes()))
        print(len(self.g.edges()))
    """

def main():
    dc = DataClassification(6, 3)
    #dc.build_network()
    dc.calculate_measure()
    dc.single_node_insert()


if __name__ == '__main__':
    main()