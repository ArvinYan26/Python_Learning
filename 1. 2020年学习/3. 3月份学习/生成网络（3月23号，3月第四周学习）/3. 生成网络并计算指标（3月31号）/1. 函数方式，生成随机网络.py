import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()  #程序开始时间
Network_size = int(input("请输入网络节点数："))
pin = float(input("请输入连边概率："))

#生成16*16的全零矩阵
Amtrix = np.zeros((Network_size, Network_size))
print(Amtrix)

def GenerateNetwork():
    """生成邻接矩阵"""
    for i in range(Network_size):
        for j in range(Network_size):
            if i == j:
                continue
            probability = np.random.random()
            if probability <= pin:
                Amtrix[i][j] = Amtrix[j][i] =1
    print(Amtrix)
    
def GenerateGraph():
    """构建网络图"""
    G = nx.Graph()
    for i in range(len(Amtrix)):
        for j in range(len(Amtrix)):
            if Amtrix[i][j] == 1:
                G.add_edge(i, j)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.show()

GenerateNetwork()
GenerateGraph()

#计算程序运行时间
end_time = time.time()
print("用时：%d"%(end_time-start_time))

