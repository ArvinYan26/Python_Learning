import numpy as np
from sklearn import preprocessing
"""
l = np.zeros(5)
print(l.shape, l)

a = np.array([[1, 2],
             [2, 2]])
b = np.array([[2, 1],
            [1, 1]])
print(a*b)
c = np.sum(a*b)
print("c:", c)

x = np.array([[1, 3],
             [2, 4]])
d = x.T
print(d)

f = np.array([2, 3, 4])
print(f.reshape(f.shape[0], 1))
"""

data = np.array([1, 2, 3]).reshape(1, -1)

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
new_data = min_max_scaler.transform(data)
print(new_data)

