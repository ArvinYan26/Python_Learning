# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:35:03 2019
@author: ZHJ
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time

print('请输入ER网络的顶点个数：')
NETWORK_SIZE = int(input())
print('请输入连边概率：')
PROBABILITY_OF_EAGE = float(input())
adjacentMatrix = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=int)  # 初始化邻接矩阵
random.seed(time.time())  # 'random.random()#生成[0,1)之间的随机数


# 生成ER网络
def generateRandomNetwork():
    count = 0
    probability = 0.0
    for i in range(NETWORK_SIZE):
        for j in range(i + 1, NETWORK_SIZE):
            probability = random.random()
            if probability < PROBABILITY_OF_EAGE:
                count = count + 1
                adjacentMatrix[i][j] = adjacentMatrix[j][i] = 1
    print('您所构造的ER网络边数为：' + str(count))


# 用于绘制ER图
def showGraph():
    G = nx.Graph()
    for i in range(len(adjacentMatrix)):
        for j in range(len(adjacentMatrix)):
            if adjacentMatrix[i][j] == 1:  # 如果不加这句将生成完全图，ER网络的邻接矩阵将不其作用
                G.add_edge(i, j)
    nx.draw(G)
    plt.show()
"""
# 将ER网络写入文件中
def writeRandomNetworkToFile():
    ARRS = []
    f = open('randomNetwork01.txt', 'w+')
    for i in range(NETWORK_SIZE):
        t = adjacentMatrix[i]
        ARRS.append(t)
        for j in range(NETWORK_SIZE):
            s = str(t[j])
            f.write(s)
            f.write(' ')
        f.write('\n')
    f.close()
"""

# 计算度分布并将其存入文件中
def calculateDegreeDistribution():
    averageDegree = 0.0
    identify = 0.0
    statistic = np.zeros((NETWORK_SIZE), dtype=float)  # statistic将用于存放度分布的数组，数组下标为度的大小，对应数组内容为该度的概率
    degree = np.zeros((NETWORK_SIZE), dtype=int)  # degree用于存放每个节点的度
    for i in range(NETWORK_SIZE):
        for j in range(NETWORK_SIZE):
            degree[i] = degree[i] + adjacentMatrix[i][j]
    print(degree)
    for i in range(NETWORK_SIZE):
        averageDegree += degree[i]
    print('平均度为' + str(averageDegree / NETWORK_SIZE))  # 计算平均度
    for i in range(NETWORK_SIZE):
        statistic[degree[i]] = statistic[degree[i]] + 1
    for i in range(NETWORK_SIZE):
        statistic[i] = statistic[i] / NETWORK_SIZE
        identify = identify + statistic[i]
    identify = int(identify)
    print('如果output为1则该算法正确\toutput=' + str(identify))  # 用于测试算法是否正确
    f = open('degree01.txt', 'w+')  # 将度分布写入文件名为degree01文件中，若磁盘中无此文件将自动新建
    for i in range(NETWORK_SIZE):
        f.write(str(i))
        f.write(' ')
        s = str(statistic[i])  # 注意写入操作要求是字符串格式，因此用str进行格式转换
        f.write(str(s))  # 写入的每一行由两部分组成，一个元素为度的下标，第二个元素为度的概率
        f.write('\n')  # 每个节点的度及概率写入完成将进行换行，输入下一个节点的度及度分布
    f.close()


# 主程序开始
start = time.perf_counter()  # 用以程序计时开始位置
generateRandomNetwork()  # 生成ER随机网络
#writeRandomNetworkToFile()  # 将随机网络写入randomNetwork01.txt文件中
calculateDegreeDistribution()  # 计算此ER随机网络的度分布并将结果写入文件degreee01.txt文件中
finish = time.perf_counter()  # 程序计时结束
duration = finish - start
print('生成这个ER网络需要的时间为：' + str(duration) + 's')
print('您所构造的ER网络如下：')
showGraph()
