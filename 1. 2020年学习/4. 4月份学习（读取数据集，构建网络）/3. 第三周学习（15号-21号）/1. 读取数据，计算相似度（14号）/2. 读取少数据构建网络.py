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
#数据归一化
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
Nor_Matrix = min_max_scaler.transform(data)
print(Nor_Matrix)

#计算相似度，这里采用欧几里德距离
Matrix = euclidean_distances(Nor_Matrix)
print(Matrix)

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
