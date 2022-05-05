import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from  sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

def get_data():
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    return iris_data, iris_target

def data_preprocess():
    iris_data, iris_target = get_data()
    X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    return X_train, X_predict, Y_train, Y_predict

def build_network( X_net, Y_net, k, labels):
    #print(X_net, Y_net)
    g = nx.Graph()
    for index, instance in enumerate(X_net):
        g.add_node(str(index), values=instance, typeNode='net', labels=Y_net[index])
    #最近邻方法
    nbrs = NearestNeighbors(k, metric='euclidean')
    nbrs.fit(X_net)
    distances, indices = nbrs.kneighbors(X_net)

    #电子半径方法
    radius = np.median(distances)
    nbrs.set_params(radius=radius)
    e_distances, e_indices = nbrs.radius_neighbors(X_net)

    if radius/3 > k:
        for index, nbrs_indices in enumerate(e_indices):
            for indices, eve_index in enumerate(nbrs_indices):
                if eve_index == index:
                    continue
                if g.nodes()[str(index)]['labels'] == g.nodes()[str(eve_index)]['labels']:
                    g.add_edge(str(eve_index), str(index), weight=e_distances[index][indices])
    else:
        for index, nbrs_indices in enumerate(indices):
            for indices, eve_index in enumerate(nbrs_indices):
                if index == eve_index:  # 如果是本身，就跳过，重新下一个循环
                    continue
                if g.nodes()[str(eve_index)]['labels'] == g.nodes()[str(index)]['labels']:
                    g.add_edge(str(eve_index), str(index), weight=distances[index][indices])

    nx.draw(g)
    plt.show()
    print(len(g.nodes()))
    print(len(g.edges()))
    return g, nbrs, k

def node_insert(g, nbrs, k,  X_predict, Y_predict):

    print(X_predict, Y_predict, g)
    #insert = len(list(g.nodes()))
    for nodeindex, instance in enumerate(X_predict):
        if (not len(Y_predict) == 0):
            label = Y_predict[nodeindex]
        insert_node_id = len(list(g.nodes()))
        #single_node_insert(g, nbrs, k, instance, len(list(g.nodes())), label)
        g.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
        #print(g.nodes()) #120

        #knn方法
        distances, indices = nbrs.kneighbors([instance])

        #电子半径方法
        radius = np.median(distances)
        nbrs.set_params(radius=radius)
        e_distances, e_indices = nbrs.radius_neighbors([instance])

        if radius / 3 > k:  #每一次插入一个数据，计算radius。值具有偶然性，不准确，所以不能在用这个方法
            for index, nbrs_indices in enumerate(e_indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if eve_index == index:
                        continue
                    #if g.nodes()[str(index)]['labels'] == g.nodes()[str(eve_index)]['labels']:
                    g.add_edge(str(eve_index), str(insert_node_id), weight=e_distances[index][indices])

        else:
            for index, nbrs_indices in enumerate(indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if str(index) == str(eve_index):  # 如果是本身，就跳过，重新下一个循环
                        continue
                    #if g.nodes()[str(eve_index)]['labels'] == g.nodes()[str(index)]['labels']:
                    g.add_edge(str(eve_index), str(insert_node_id), weight=distances[index][indices])

    nx.draw(g)
    plt.show()
    #a = len(g.nodes())
    #print(a)
    print(len(g.nodes()))
    #b = len(g.edges())
    #print(b)
    print(len(g.edges()))

"""

def many_node_insert(g, nbrs, k, X_predict, Y_predict):
    for nodeindex, instance in enumerate(X_predict):
        if (not len(Y_predict) == 0):
            label = Y_predict[nodeindex]
        single_node_insert(g, nbrs, k, instance, len(list(g.nodes())), label)

    nx.draw(g)
    plt.show()
    print(len(g.nodes()))
    print(len(g.edges()))
"""

def main():
    X_train, X_predict, Y_train, Y_predict = data_preprocess()
    g, nbrs, k = build_network(X_train, Y_train, 6, labels=True)
    #single_node_insert(g, nbrs, k, X_predict[0], 120, Y_predict[0])
    node_insert(g, nbrs, k, X_predict, Y_predict)




if __name__ == '__main__':
    main()
