#欧几里德距离
#几个数据集之间的相似度一般是基于每对对象间的距离计算。最常用的当然是欧几里德距离，其公式为:
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

data1 = [[5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [7.3, 2.9, 6.3, 1.8],
        [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5]]
x = np.array([5.1, 3.5, 1.4, 0.2])
y = np.array([4.9, 3.0, 1.4, 0.2])
#print(data1[0])  #[5.1, 3.5, 1.4, 0.2] 是一个列表，不能直接进行加减
#print(len(data1))

#method一：
dist_list = []
for i in range(len(data1)):
    for j in range(len(data1)):
        dist = np.sqrt(np.sum(np.square(np.array(data1[i])-np.array(data1[j]))))
        dist_list.append(dist)
print(dist_list)


#method二：计算两个列表或者是向量之间的欧氏距离
x = np.array([5.1, 3.5, 1.4, 0.2])
y = np.array([4.9, 3.0, 1.4, 0.2])
dist = np.linalg.norm(y - x)
print(dist)
#0.5385164807134502

list = []
list1 = [5.1, 3.5, 1.4, 0.2]
list2 = [4.9, 3.0, 1.4, 0.2]
dist = np.linalg.norm(y - x)
print(dist)
list.append(dist)
print(list)
#0.5385164807134502


#dist_list1 = np.array(dist_list)
#dist1 = np.reshape((9,9), dist_list1)

#a = 100
#b = np.sqrt(a)
#print(b)


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
#Matrix1 = euclidean_distances(Nor_Matrix)

Matrix = euclidean_distances(Nor_Matrix, Nor_Matrix)
#print(Matrix1.shape, Matrix.shape)

Matrix = np.round(Matrix, decimals=2)
print("eud_Matrix:")
print(Matrix)

s = sorted(set(Matrix.reshape(1, -1)[0]))
print("s:", len(s), s)
print(min(s), max(s), np.mean(s))
print("step:", (max(s)-min(s))/6)
l = np.arange(min(s), max(s), (max(s)-min(s))/6)
print("l:", l)
#threshold =



g = nx.Graph()

for i in range(len(Matrix)):
    for j in range(i+1, len(Matrix)):
        if Matrix[i][j] > 0 and Matrix[i][j] < 1:
            Matrix[i][j] = Matrix[j][i] = 1
        elif Matrix[i][j] > 1:
            Matrix[i][j] = Matrix[j][i] = 0
print(Matrix)

for i in range(len(Matrix)):
    for j in range(i+1, len(Matrix)):
        if Matrix[i][j] == 1:
            g.add_edge(i, j)

nx.draw(g)
plt.show()

"""
eud_Matrix: 10 * 10  距离矩阵

[[0.   0.39 0.27 0.35 0.52 0.32 1.16 1.48 1.45 1.53]
 [0.39 0.   0.17 0.13 0.19 0.09 0.9  1.45 1.32 1.63]
 [0.27 0.17 0.   0.09 0.25 0.12 1.03 1.52 1.42 1.64]
 [0.35 0.13 0.09 0.   0.17 0.11 0.97 1.5  1.39 1.65]
 [0.52 0.19 0.25 0.17 0.   0.24 0.92 1.55 1.39 1.74]
 [0.32 0.09 0.12 0.11 0.24 0.   0.96 1.46 1.36 1.62]
 [1.16 0.9  1.03 0.97 0.92 0.96 0.   0.92 0.61 1.33]
 [1.48 1.45 1.52 1.5  1.55 1.46 0.92 0.   0.38 0.61]
 [1.45 1.32 1.42 1.39 1.39 1.36 0.61 0.38 0.   0.91]
 [1.53 1.63 1.64 1.65 1.74 1.62 1.33 0.61 0.91 0.  ]
 
"""