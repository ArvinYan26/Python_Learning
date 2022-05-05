import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy import random

N = int(input("请输入节点数："))
probability_of_edge = float(input("请输入连边概率："))
G = nx.Graph()
Matrix = np.array(random.randint((2), size=(N, N)))  #生成一个元素大小为0,1的N*N的随机矩阵
print(Matrix)

#下面两种方法任选其一，注意方式即可
for i in range(N):
    for j in range(i+1, N):
        probability = np.random.random()
        if probability <= probability_of_edge:
                Matrix[i][j] = Matrix[j][i] = 1#添加连边
"""
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        probability = np.random.random()
        if probability <= probability_of_edge:
                Matrix[i][j] = Matrix[j][i] = 1#添加连边
"""

print(Matrix)



