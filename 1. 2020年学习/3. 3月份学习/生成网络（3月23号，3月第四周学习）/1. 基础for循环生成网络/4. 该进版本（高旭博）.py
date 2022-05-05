import numpy as np
import igraph
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt

NETWORK_SIZE = 128
# PROBABILITY_OF_EAGE=0.8  #Limited to global
pin = 0.7
# pout=0.01
pout = 0.01

M = 8  # Community Number
#Amatrix = [[0 for i in range(NETWORK_SIZE)] for i in range(NETWORK_SIZE)]
Amatrix = np.zeros((NETWORK_SIZE, NETWORK_SIZE) )
print(Amatrix)

intvl = int(NETWORK_SIZE / M) #16
bgIntvl = 0
endIntvl = intvl

#最外层循环控制便利的社区0-7，共8个社区
for m in range(M):
	#内层循环控制索引为0的第一个社区的16个节点
    for i in range(bgIntvl, endIntvl):  #范围：（0，16），节点编号是0-15
        for j in range(bgIntvl, endIntvl): #范围：（0，16）
            if (i == j):
                continue   #i==j 时不执行，直接跳过，回到初始继续下一个操作
            probability = np.random.random()  #随机生成一个实数，在0-1之间
            if (probability <= pin):  #pin = 0.7
                Amatrix[i][j] = Amatrix[j][i] = 1  #对角元素相等为1，生成邻接矩阵
    # Update Interval （控制社区索引，循环遍历完第一个社区的所有16个节点，第二个社区节点从第17各节点开始编号）
    bgIntvl += intvl   #bgIntvl=16，此时m=2,代表第二个时区的16各节点的编号是从16还是记录，
    endIntvl += intvl  #endTntvl=32, 此时m=2,代表是第二个时区的16各节点的编号范围是16-31
#以上循环结束时，会遍历所有元素，然后生成邻接矩阵
print(Amatrix)


# INTERcommunity #社区内部
bg1 = 0
end1 = intvl  # end1 = 16
#最外层循环控制17-128行，没16行为一个循环，总共为7个循环 （即M-1=7）
for m1 in range(M - 1):  # M -1 = 7,m1范围是 0-7
    # Destiny Range Initial Conditions #概率初始范围
    bg2 = end1     # bg = 16, 32, 48, 64, 80, 96, 112
    end2 = end1 + intvl   #end2 = 32, 48, 64, 80, 96, 112
	#内层循环控制0-16行，列数为17-128，没16列作为第二个内层循环
    for m2 in range(M - m1 - 1):  # m2 范围是 0-7, 0-6, 0-5
        for i in range(bg1, end1):  #范围0-16, 16-32, 32-48，
            for j in range(bg2, end2):  #范围是：16-32, 32-48, 48-64
                probability = np.random.random()
                if (probability <= pout):
                    Amatrix[i][j] = Amatrix[j][i] = 1
        # Destiny Range Update
        bg2 = end2 # bg2 = 32， 48, 64
        end2 = end2 + intvl  #end2 = 32+16=48, 64, 80
    bg1 = end1 #bg1=16, 32, 48, 64, 80, 96, 112,
    end1 = end1 + intvl  #end1 = 32, 48, 64, 80, 96, 112,
print(Amatrix)

#构建网络图
G = nx.Graph()
NodeDegree = []
for i in range(len(Amatrix)):
    CountNode = 0
    for j in range(len(Amatrix)):
        if Amatrix[i][j] == 1:
            CountNode += 1
            G.add_edge(i, j)
    NodeDegree.append(CountNode)
AverageDegree = sum(NodeDegree) / len(Amatrix)
print(AverageDegree)

nx.draw(G)
plt.show()
print(G)
