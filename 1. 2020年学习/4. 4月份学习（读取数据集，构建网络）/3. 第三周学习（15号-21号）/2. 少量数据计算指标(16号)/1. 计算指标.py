import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances

data =[ [5.1,3.5,1.4,0.2],
        [4.9,3.0,1.4,0.2],
        [4.7,3.2,1.3,0.2],
        [4.6,3.1,1.5,0.2],
        [5.0,3.6,1.4,0.2],
        [5.4,3.9,1.7,0.4],
        [4.6,3.4,1.4,0.3],
        [5.0,3.4,1.5,0.2],
        [4.4,2.9,1.4,0.2],
        [4.9,3.1,1.5,0.1],
        [7.0,3.2,4.7,1.4],
        [6.4,3.2,4.5,1.5],
        [6.9,3.1,4.9,1.5],
        [5.5,2.3,4.0,1.3],
        [6.5,2.8,4.6,1.5],
        [5.7,2.8,4.5,1.3]]

#print(len(data))  #数据或者矩阵的长度指的就是矩阵的维度（多少维）

#数据归一化
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
Nor_Matrix = min_max_scaler.transform(data)
#print(Nor_Matrix)

#计算相似度，这里采用欧几里德距离
Matrix = euclidean_distances(Nor_Matrix)
#print(Matrix)

def BuildNetwork():
    g = nx.Graph()

    for i in range(len(Matrix)):
        for j in range(i+1, len(Matrix)):
            if Matrix[i][j] > 0 and Matrix[i][j] < 1:
                Matrix[i][j] = Matrix[j][i] = 1
            elif Matrix[i][j] > 1:
                Matrix[i][j] = Matrix[j][i] = 0
    #print(Matrix)
    NodeDegree = []
    for i in range(len(Matrix)):
        CountDegree = 0
        # 此处是遍历所有节点的所有边，所以范围是len(Matrix),而不是(i+1,len(Matrix))
        for j in range(len(Matrix)):
            if Matrix[i][j] == 1:
                g.add_edge(i, j)
                CountDegree += 1
        NodeDegree.append(CountDegree)
    #print(NodeDegree)
    nx.draw(g)
    plt.show()
    #计算度
    deg = nx.degree(g)
    #print(deg)
    #计算平均度
    Average_degree = float(sum(NodeDegree) / len(data))
    #print(Average_degree)
    return g

def CalculateDegreeDistribution():
    """计算度分布"""
    #初始化矩阵度矩阵
    degree = np.zeros((len(data)), dtype=int)
    sum_degree = 0
    degree_distribution = np.zeros((len(data)), dtype=float)

    #计算度
    for i in range(len(data)):
        for j in range(len(data)):
            degree[i] += Matrix[i][j]  #此时的矩阵是0,1邻接连边矩阵，所以直接加就是边数
    print(degree)

    #计算平均度
    #法1
    for i in range(len(data)):
        sum_degree += degree[i]
    #average_degree = float(sum_degree/len(data))
    #法2
    average_degree = float(sum(degree) / len(data))
    print(average_degree)

    #统计度数为degree[i]的节点个数
    for i in range(len(data)):
        degree_distribution[degree[i]] += 1
    print(degree_distribution)

    #计算度分布
    for i in range(len(data)):
        degree_distribution[i] = degree_distribution[i] / len(data)
    print(degree_distribution)

def OtherMeasures(G):
    """计算其他网络指标"""
    #计算聚类系数
    clustering = nx.clustering(G)
    #平均聚类系数
    average_clustering = nx.average_clustering(G)
    print(clustering)
    print(average_clustering)

    #直径
    #diameter = nx.diameter(G)
    #print(diameter)

    #同配性（复杂网络的用色弹性）
    assortativity = nx.degree_assortativity_coefficient(G)
    print(assortativity)

    #平均最短路径长度
    aver_shortest_path = nx.average_shortest_path_length(G)
    print(aver_shortest_path)

if __name__ == "__main__":
    G = BuildNetwork()
    CalculateDegreeDistribution()
    OtherMeasures(G)
    #print(len(list(G.nodes())))
    #print(len(list(G.edges())))
