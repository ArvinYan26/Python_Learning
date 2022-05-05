import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

Network_size = 50
pin = 0.5

#生成16*16的全零矩阵
Amtrix = np.zeros((Network_size, Network_size))
print(Amtrix)

def GenerateNetwork():
    """生成随机网络，连边"""
    #下面两个方法任选其一。
    """
    for i in range(Network_size):
        for j in range(Network_size):
            if i == j:
                continue
            probability = np.random.random()
            if probability <= pin:
                Amtrix[i][j] = Amtrix[j][i] =1
    """
    for i in range(Network_size):
        for j in range(i+1, Network_size):
            probability = np.random.random()
            if probability <= pin:
                Amtrix[i][j] = Amtrix[j][i] =1

def GenerateGraph():
    """创建图，并且计算度和平均度，度等级图"""
    G = nx.Graph()
    NodeDegree = []
    for i in range(len(Amtrix)):
        CountNodeDegree = 0
        for j in range(len(Amtrix)):
            if Amtrix[i][j] == 1:
                G.add_edge(i, j)
                CountNodeDegree += 1
        NodeDegree.append(CountNodeDegree)
    AverageDegree = sum(NodeDegree) / len(Amtrix)
    print(AverageDegree)
    deg = nx.degree(G) #deg = G.degree() 一样
    print(deg)
    #节点图
    nx.draw_spring(G)
    plt.show()

    #画度等级折线图
    degree_sequence = sorted([d for n, d in deg], reverse=True)
    #degree_sequence = nx.degree_histogram(G)
    print((degree_sequence))
    #x = range(len(degree_sequence))
    #y = [(z/float(sum(degree_sequence)) for z in degree_sequence)]
    plt.loglog(degree_sequence, 'b-', marker='o')
    plt.title("Degree Rank Plot")
    plt.ylabel("degree")
    plt.xlabel("rank")

    nx.draw(G)
    plt.show()

GenerateNetwork()
GenerateGraph()

#计算程序运行时间
end_time = time.time()
print("用时：%d"%(end_time-start_time))