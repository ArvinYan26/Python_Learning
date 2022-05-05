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

def single_node_insert(g, nbrs, k,  instance, nodeindex, label):
    """

    :param g:  原来的图，已经添加过节点
    :param nbrs:  训练好的NearestNeighbors
    :param k:  knn中k的值6， 其实这6个邻居会包括他自己，实际上有效邻居只有5个
    :param instance: 添加的每一个数据，一维数组
    :param nodeindex: 因为是在原来的120个节点的图中添加，所以nodeindex范围是从121开始到150结束
    :param label:  #每一个类别，0， 1， 2
    :return:
    """
    g.add_node(str(nodeindex), values=instance, typeNode='test', label=label)
    #knn方法
    distances, indices = nbrs.kneighbors([instance])

    #电子半径方法
    radius = np.median(distances)
    nbrs.set_params(radius=radius)
    e_distances, e_indices = nbrs.radius_neighbors([instance])
    if radius / 3 > k:  #每一次插入一个数据，计算radius。值具有偶然性，不准确，所以不能在用这个方法，因为radius用于密集区域
        count = 0
        for index, nbrs_indices in enumerate(e_indices):
            for indices, eve_index in enumerate(nbrs_indices):
                if eve_index == index:
                    continue
                #if g.nodes()[str(index)]['labels'] == g.nodes()[str(eve_index)]['labels']:
                g.add_edge(str(eve_index), str(nodeindex), weight=e_distances[index][indices])
                count += 1
        print(count)  #测试此方法是否被调用
    else:
        for index, nbrs_indices in enumerate(indices):
            for indices, eve_index in enumerate(nbrs_indices):
                if str(index) == str(eve_index):  # 如果是本身，就跳过，重新下一个循环
                    continue
                #if g.nodes()[str(eve_index)]['labels'] == g.nodes()[str(index)]['labels']:
                g.add_edge(str(eve_index), str(nodeindex), weight=distances[index][indices])

    #print(g.nodes()) 从121数字，每一次增加1 一直到149，共150个节点
    """
    nx.draw(g)
    plt.show()
    print(len(g.nodes()))
    print(len(g.edges()))
    """

def many_node_insert(g, nbrs, k, X_predict, Y_predict):
    for nodeindex, instance in enumerate(X_predict):
        if (not len(Y_predict) == 0):
            label = Y_predict[nodeindex]
        single_node_insert(g, nbrs, k, instance, len(list(g.nodes())), label)

    nx.draw(g)
    plt.show()
    #print(g.nodes())
    print(len(g.nodes()))
    print(len(g.edges()))

def main():
    X_train, X_predict, Y_train, Y_predict = data_preprocess()
    g, nbrs, k = build_network(X_train, Y_train, 6, labels=True)
    #single_node_insert(g, nbrs, k, X_predict[0], 120, Y_predict[0])
    many_node_insert(g, nbrs, k, X_predict, Y_predict)



if __name__ == '__main__':
    main()
