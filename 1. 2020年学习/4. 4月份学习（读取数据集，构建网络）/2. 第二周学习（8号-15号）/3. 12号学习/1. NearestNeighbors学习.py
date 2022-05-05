import numpy as np
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(2, 0.4)
neigh.fit(samples)   #用samples作为训练数据，拟合模型
neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False) #找到[0, 0, 1.3]这个点的最近的两个邻居，不返回距离
nbrs = neigh.radius_neighbors([0, 0, 1.3], 0.4, return_distance=False) #找到以某个点为圆心半径为0.4以内的所有该点的邻居，不返距离
np.asarray(nbrs[0][0])
