import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from  sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

dataset = load_iris()
#print(dataset)

#切分数据集
X_train, X_predict, Y_train, Y_predict = train_test_split(dataset['data'], dataset['target'], test_size=0.2)
#print(X_train, X_predict, Y_train, Y_predict)
#用MinMaxScaler()方法归一化数据
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_rain = scaler.transform(X_train)
X_predict = scaler.transform(X_predict)
#print(len(train_data))
#print(X_train)
#print(train_data)

def NetworkBulidKNN(X_net, Y_net, knn, labels):
    g = nx.Graph()
    """
    lnNet = len(X_net) #网络大小
    g.graph["lenNet"] = lnNet   #网络大小
    g.graph["classNames"] = list(set(Y_net)) #网络图中的类别名字
    """
    #添加节点
    for index, instance in enumerate(X_net):
        g.add_node(str(index), values=instance, typeNode='net', label=Y_net[index])
        #str（index）：节点， instance：训练数据内容， typeNode：画图用的， label：类别
        #print(instance)
    #print(X_net)
    values = X_net
    #print(values, len(values))
    #定格式
    #if (isinstance(values[0], (int, float, str))): #定格式
        #values = [e[0] for e in values]

    #最近邻方法
    nbrs = NearestNeighbors(knn+1, metric='euclidean')
    #print(nbrs)
    nbrs.fit(values)
    #print(values)
    distances, indices = nbrs.kneighbors(values)
    #print(distances, indices)
    #print(indices)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    #print(distances, indices)

    #电子半径方法
    eRadius = np.quantile(distances, 0.5)
    #print(eRadius)
    nbrs.set_params(radius=eRadius)

    #建立连边
    #KNN中的连边
    for indiceNode, indicesNode in enumerate(indices): #indices:是每个点的五个邻居的索引,5个索引。
        #print(indiceNode, indicesNode) #0 [15 119 111 35 108], 1 [36 10 38 47 3]......
        for tmpi, indice in enumerate(indicesNode): #indicesNode:
            #print(tmpi, indice) #0 15, 1 119, 2 111, 3 35, 4 108
            if g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"]:
                g.add_edge(str(indice), str(indiceNode), weight=distances[indiceNode][tmpi])
    #print(len(list(g.edges())))

    #eRadius中的连边
    #print(instance)
    distances, indices = nbrs.radius_neighbors(X_net)
    print(indices, distances)
    for indiceNode, indicesNode in enumerate(indices):
        #print(indiceNode, indicesNode)
        for tmpi, indice in enumerate(indicesNode):
            #print(tmpi, indice)
            if (not str(indice)) == str(indiceNode): #保证不是节点自己连自己，因为搬经计算可能会出现自己
                if g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"]:
                    g.add_edge(str(indice), str(indiceNode), weight=distances[indiceNode][tmpi])

    #print(len(list(g.edges())))
    #g.graph["index"] = lnNet
    return g, nbrs  #返回给调用函数，以便其他函数调用

def NodeInseration(g, nbrs, instance, nodeIndex, label):
    g.add_node(str(nodeIndex), values=instance, typeNode='test', label=label)
    #if (isinstance(instance, (int, float, str))):
        #instance = [instance]
    #print(instance)
    distances, indices = nbrs.kneighbors([instance])  #instance不是矩阵，必须转化为矩阵，所以加[]
    #print(distances, indices)
    for indiceNode, indicesNode in enumerate(indices):
        #print(indiceNode, indicesNode)
        for tmpi, indice in enumerate(indicesNode):
            #print(tmpi, indice)
            g.add_edge(str(indice), str(nodeIndex), weight=distances[indiceNode][tmpi])

    distances, indices = nbrs.radius_neighbors([instance])
    for indiceNode, indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if (not str(indice)) == str(indiceNode):
                g.add_edge(str(indice), str(nodeIndex), weight=distances[indiceNode][tmpi])

def manyInserts(g,nbrs,X_predict,Y_predict):
    for nodeIndex, instance in enumerate(X_predict):
        if(not len(Y_predict) == 0):
            label = Y_predict[nodeIndex]
        NodeInseration(g, nbrs, instance, len(list(g.nodes())), label)


if __name__ == "__main__":
    g, nbrs = NetworkBulidKNN(X_train, Y_train, 6, labels=True)
    nx.draw(g)
    plt.show()
    print(len(list(g.nodes())))
    print(len(list(g.edges())))
    #list = list(range(130-150))
    #print(list)

    """
    Inseration = NodeInseration(g, nbrs, X_predict[29], 136, Y_predict[29])
    nx.draw(g)
    plt.show()
    print(len(list(g.nodes())))
    print(len(list(g.edges())))
    """
    manyInserts(g, nbrs, X_predict, Y_predict)
    nx.draw(g)
    plt.show()
    print(len(list(g.nodes())))
    print(len(list(g.edges())))


