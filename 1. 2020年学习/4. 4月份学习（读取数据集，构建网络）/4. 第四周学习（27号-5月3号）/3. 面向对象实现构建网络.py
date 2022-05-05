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
        self.g = None
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
            print(index, instance)
        print(self.knn_distances)
        #选择方法
        #if self.radius > self.k/3 :
        for index, nbrs_index in enumerate(self.e_radius_indices):
            for tempi, eve_index in enumerate(nbrs_index):
                #if (not str(index)) == str(eve_index):
                if self.g.nodes()[str(eve_index)]["label"] == self.g.nodes()[str(index)]["label"]:
                    self.g.add_edge(str(index), str(eve_index), weight=self.e_radius_distances[index][tempi])
        #else:
        for index, nbrs_index in enumerate(self.knn_indices):
            for tempi, eve_index in enumerate(nbrs_index):
                #if (not str(index)) == str(eve_index):
                if self.g.nodes()[str(eve_index)]["label"] == self.g.nodes()[str(index)]["label"]:
                    self.g.add_edge(str(index), str(eve_index), weight=self.knn_distances[index][tempi])

        nx.draw(self.g)
        plt.show()
        print(len(self.g.nodes()))
        print(len(self.g.edges()))



def main():
    dc = DataClassification(5)

    #dc.data_preprocess()
    dc.build_network(label=True)


if __name__ == '__main__':
    main()





