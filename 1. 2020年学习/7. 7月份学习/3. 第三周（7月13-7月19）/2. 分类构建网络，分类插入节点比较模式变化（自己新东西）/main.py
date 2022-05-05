from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from normalization import data_preprocess
import networkx as nx
import numpy as np
import BuildNetwork as BN
from DrawGraph import draw_graph
from ReorganizeData import reorganize_data

class DataClassification(object):

    def __init__(self, k, num):
        self.k = k
        self.class_num = num
        self.G = None
        self.nbrs = {}
        self.radius = {}

    def fit(self, X_train, Y_train):
        """

        :param X_train: train data set
        :param Y_train:  train data traget
        :return:
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.G = nx.Graph()

        for index, instance in enumerate(self.X_train):
            #print(self.Y_train[index])
            self.G.add_node(index, value=instance, TypeNode="train", label=self.Y_train[index])
        #draw_graph(self.G)



    def build_init_network(self):
        self.X_train, self.Y_train, class0, class1 , class2 = reorganize_data(self.X_train, self.Y_train)
        #print(new_data, len(new_data))
        print(self.X_train, self.Y_train)
        print(len(self.X_train), len(self.Y_train))

        self.G = nx.Graph()
        for index, instance in enumerate(self.X_train):
            #print(self.Y_train[index])
            self.G.add_node(index, value=instance, TypeNode="train", label=self.Y_train[index])
        draw_graph(self.G)
        for i in range(self.class_num):

            if i == 0:
                base_index = 0
                current_data =self.X_train[0:len(class0), :]
                print(current_data, len(current_data))
                self.G, temp_nbrs, temp_radius = \
                    BN.build_init_network(current_data, base_index, self.G, self.k, label=True)
                self.nbrs["0"] = temp_nbrs
                self.radius["0"] = temp_radius

            if i == 1:
                base_index = len(class0)
                current_data = self.X_train[len(class0):(len(class0)+len(class1)), :]
                print(current_data, len(current_data))
                self.G, temp_nbrs, temp_radius = \
                    BN.build_init_network(current_data, base_index, self.G, self.k, label=True)
                self.nbrs["1"] = temp_nbrs
                self.radius["1"] = temp_radius

            if i == 2:
                base_index = len(class0)+len(class1)
                current_data = self.X_train[(len(class0)+len(class1)):len(self.X_train), :]
                print(current_data, len(current_data))
                self.G, temp_nbrs, temp_radius = \
                    BN.build_init_network(current_data, base_index, self.G, self.k, label=True)
                self.nbrs["2"] = temp_nbrs
                self.radius["2"] = temp_radius

        print(self.nbrs, self.radius)
        print(self.G.nodes(), self.G.edges())
        draw_graph(self.G)

if __name__ == '__main__':
    data = load_iris()
    #data = load_wine()
    #data = load_breast_cancer()
    X = data.data
    y = data.target
    #X, y = make_moons(n_samples=200, noise=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = data_preprocess(X_train, X_test)
    DC = DataClassification(5, 3)
    DC.fit(X_train, Y_train)
    DC.build_init_network()

