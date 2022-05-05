import matplotlib.pyplot as plt
import networkx as nx
#import igraph
import scipy as sp
import numpy as np
import time

TotalNones = 512 #节点总数
M = 8   #社区数量
pin = 0.9  #社区内部连边概率
pout = 0.001 #社区外部连边概率

#随机生成0-1之间的一个实数，概率，所以一次只能生产方法一个，想要循环生成，必须放在循环内部
#probability = np.random.random()


#生成全零矩阵
#amtrix = [[0 for i in range(TotalNones)] for i in range(TotalNones)]
#print(amtrix)
Amtrix = np.zeros((TotalNones, TotalNones))
print(Amtrix)
InNodes = int(TotalNones / M)  # 每个社区节点数是48/4=12

InBegin = 0
InEnd = InNodes  #12
#社区内部生成邻接矩阵
#外层循环控制遍历每一个社区
for m in range(M):
    #内层循环控制遍历每一个社区内部的节点连边情况
    for i in range(InBegin, InEnd): #0-12
        for j in range(InBegin, InEnd): #0-12
            if i == j:
                continue   #如果i==j跳过此步，继续下一个循环
            #probability = np.random.random()
            probability = np.random.random()
            if probability <= pin:
                Amtrix[i][j] = Amtrix[j][i] = 1
    #内层循环结束时，遍历完第一个社区的节点连边，接下来遍历下面的每一个社区的节点连边
    InBegin += InNodes
    InEnd += InNodes
print(Amtrix)

#社区外部，遍历每一个社区之间的连边
OutBgin = 0
OutEnd = InNodes   #12
for m1 in range(M-1):  #0-3, (0,1,2,)
    OutBgin1 = OutEnd
    OutEnd1 = OutEnd + InNodes
    for m2 in range(M-m1-1):
        for i in range(OutBgin, OutEnd):
            for j in range(OutBgin1, OutEnd1):
                #random：一次只能生成一个，所以想要遍历，只能放在循环内部，循环一次执行一次
                probability = np.random.random()
                if probability <= pout:
                    Amtrix[i][j] = Amtrix[j][i] = 1
        OutBgin1 += InNodes
        OutEnd1 += InNodes
    OutBgin += InNodes
    OutEnd += InNodes
print(Amtrix)

#画出社区网络图，并计算度和平均度
G = nx.Graph()
NodeDegree = []
for i in range(len(Amtrix)):
    CountNode = 0
    for j in range(len(Amtrix)):
        if Amtrix[i][j] == 1:
            CountNode += 1
            G.add_edge(i, j)
    NodeDegree.append(CountNode)

AverageDegree = sum(NodeDegree) / len(Amtrix)
print(AverageDegree)
deg = nx.degree(G)
print(deg)
#pos = nx.spring_layout(G)

plt.figure(figsize=(12, 12))
nx.draw_networkx(G,  node_size=100, with_labels=False, node_color="blue", width=0.8, alpha=0.7)
plt.show()
