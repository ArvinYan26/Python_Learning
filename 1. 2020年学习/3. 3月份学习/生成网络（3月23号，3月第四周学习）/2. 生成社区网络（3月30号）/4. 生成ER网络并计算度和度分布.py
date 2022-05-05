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
AdjacencyMatrix = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=int)  # 初始化邻接矩阵
random.seed(time.time())  # 'random.random()#生成[0,1)之间的随机数


# 生成ER网络
def generateRandomNetwork():
    count = 0
    for i in range(NETWORK_SIZE):
        for j in range(i + 1, NETWORK_SIZE):
            probability = random.random()
            if probability < PROBABILITY_OF_EAGE:
                count = count + 1
                AdjacencyMatrix[i][j] = AdjacencyMatrix[j][i] = 1
    #print('您所构造的ER网络边数为：' + str(count)) #str()用于将值转化为适于人阅读的字符串的形式,
    print("您所构造的ER网络边数为：%d" % count)

# 用于绘制ER图
def showGraph():
    G = nx.Graph()
    for i in range(len(AdjacencyMatrix)):
        for j in range(len(AdjacencyMatrix)):
            if AdjacencyMatrix[i][j] == 1:  # 如果不加这句将生成完全图，ER网络的邻接矩阵将不其作用
                G.add_edge(i, j)
    nx.draw(G)
    plt.show()

# 将ER网络写入文件中
def writeRandomNetworkToFile():
    ARRS = []
    f = open('randomNetwork01.txt', 'w+') #创建一个可写文档
    for i in range(NETWORK_SIZE):
        t = AdjacencyMatrix[i]   #循环遍历邻接矩阵的每一行
        ARRS.append(t)
        for j in range(NETWORK_SIZE):
            s = str(t[j])  #循环遍历每一行的每一个元素
            f.write(s)     #将所有元素添加进文档中
            f.write(' ')   #用空格隔开
        f.write('\n')      #循环完一行元素，换行
    f.close()              #写完所有元素后关闭文件，节省内存


# 计算度分布并将其存入文件中
def calculateDegreeDistribution():
    AverageDegree = 0.0
    identify = 0.0
    statistic = np.zeros((NETWORK_SIZE), dtype=float)  # 一维数组，statistic将用于存放度分布的数组，数组下标为度的大小，对应数组内容为该度的概率
    degree = np.zeros((NETWORK_SIZE), dtype=int)  # degree用于存放每个节点的度，下标为节点索引
    for i in range(NETWORK_SIZE):
        for j in range(NETWORK_SIZE):
            degree[i] = degree[i] + AdjacencyMatrix[i][j]
    #degree表示的是
    AverageDegree = sum(degree) / len(AdjacencyMatrix)
    #for i in range(NETWORK_SIZE):
        #AverageDegree += degree[i]

    print("平均度为：%d" % AverageDegree)
    #print('平均度为:' + str(AverageDegree))  # 计算平均度

    #验证所有节点的度分布的和是否是1，如果是1说明算法正确，
    #度分布Pk 就是节点度数为k的节点占总结点个数的比例
    for i in range(NETWORK_SIZE):
        statistic[degree[i]] += 1 #统计相同度数的个数，存放在statistic，此时statistic只是统计相同度数的一维数组
    print(statistic)
    for i in range(NETWORK_SIZE):
        statistic[i] = statistic[i] / NETWORK_SIZE #此时最后的: statistic= 度数为i的节点数 / 总的节点个数
        # 验证所有节点的度分布的和是否是1，如果是1说明算法正确，
        identify += statistic[i]
    print(statistic)
    identify = int(identify)
    print('如果output为1则该算法正确\toutput=' + str(identify))  # 用于测试算法是否正确

    #将度分布写入文件中
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
writeRandomNetworkToFile()  # 将随机网络写入randomNetwork01.txt文件中
calculateDegreeDistribution()  # 计算此ER随机网络的度分布并将结果写入文件degreee01.txt文件中
finish = time.perf_counter()  # 程序计时结束
duration = finish - start
print('生成这个ER网络需要的时间为：' + str(duration) + 's')
print('您所构造的ER网络如下：')
showGraph()
