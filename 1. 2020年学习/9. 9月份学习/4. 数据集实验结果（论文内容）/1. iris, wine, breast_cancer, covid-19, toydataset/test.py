import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

l = np.arange(1, 2, (2-1)/5)
s = np.arange(1, 2, 0.2)
print(l, s)




data =[[5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [5.5, 2.3, 4.0, 1.3],
        [7.3, 2.9, 6.3, 1.8],
        [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5]
        ]
#数据归一化
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
Nor_Matrix = min_max_scaler.transform(data)
#print(Nor_Matrix)

#计算相似度，这里采用欧几里德距离
#这个函数用于求一个矩阵中所有向量之间的欧氏距离，包括向量本身与本身的距离（0）
Matrix = euclidean_distances(Nor_Matrix, Nor_Matrix)
Matrix = np.round(Matrix, decimals=2)
print("eud_Matrix:")
print(Matrix)

s = sorted(set(Matrix.reshape(1, -1)[0]))
print("s:", len(s), s)
print(min(s), max(s), np.mean(s))

c = np.arange(0.25, 2, 0.25)
d = np.round([i*np.mean(s) for i in c], decimals=2)
print("d:", d)
print("c:", c)

#print("step:", (max(s)-min(s))/6)
#l = np.arange(min(s), max(s), (max(s)-min(s))/6)
#print("l:", l)