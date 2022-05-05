import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

z = np.dot(x, y)  #求x*y这里并非是向量乘积
print(z)

print(np.sum(x, axis=0)) #列求和
print(np.sum(x, axis=1)) #行求和

#求转至
print(x.T)
"""
[[1 3]
 [2 4]]
"""

#把一个数组的每一行加上相同的元素
#方法一
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
for i in range(len(x)):
    y[i, :] = x[i, :] + v
print(y)

"""
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
"""
#方法二
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(x)
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))
print(vv)
y = x + vv
print(y)
