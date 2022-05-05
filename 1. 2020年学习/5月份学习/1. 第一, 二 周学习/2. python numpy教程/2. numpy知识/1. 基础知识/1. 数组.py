import numpy as np
"""
a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
"""

#Numpy还提供了许多创建数组的函数：
a = np.zeros((2, 2))
print(a)
########################生成6*6的全零矩阵
#法一：
x = 6
b = [[0 for i in range(6)] for i in range(6)]
#[0 for i in range(6)] :每一维度6个，在重复生成6个维度，也就是最后生成6*6的全零矩阵
b = np.array(b)  #将列表生成式生成的矩阵转换成np矩阵
print("创建的矩阵是：")
print(b)
#法二
c = np.zeros((6, 6))
print(c)

d = np.ones((3, 3))  #生成全1矩阵
print(d)

e = np.full((3, 3), 7) #3*3的全7矩阵
print(e)
"""
[[7 7 7]
 [7 7 7]
 [7 7 7]]
"""

f = np.eye(2)  #生成2*2的单位矩阵
print(f)
"""
[[1. 0.]
 [0. 1.]]
"""

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
"""
[[0.80289339 0.92833686]
 [0.79590999 0.50150431]]
"""


import numpy as np

"""
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
c = np.arange (5)
d = np.linspace(0, 2*np.pi, 5)
f = [a, b]


print(a, b, c, d)
print(f)

#多维数组表示矩阵
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])
print(a[2, 4])

#切片多维数组 MD slicing
a1 = a[0, 1:4 ]
print(a1)  #[12 13 14]
print(a[0, 1:4]) # >>>[12 13 14]
print(a[1:4, 0]) # >>>[16 21 26]
print(a[::2, ::2]) # >>>[[11 13 15]
                  #     [21 23 25]
                  #     [31 33 35]]
print(a[:, 1]) # >>>[12 17 22 27 32]

print(type(a)) # >>><class 'numpy.ndarray'>
print(a.dtype) # >>>int64
print(a.size) # >>>25
print(a.shape) # >>>(5, 5)
print(a.itemsize) # >>>8
print(a.ndim) # >>>2
print(a.nbytes) # >>>200


#基础操作符
# Basic Operators
a = np.arange(25)
a = a.reshape((5, 5))

b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78])
b = b.reshape((5,5))
"""
# dot, sum, min, max, cumsum
#arange #生成10个一维数组，范围是0-10，不包括10
a = np.arange(10)
print(a)

print(a.sum()) # >>>45
print(a.min()) # >>>0
print(a.max()) # >>>9
#print(a.cumsum()) # >>>[ 0  1  3  6 10 15 21 28 36 45]

#索引进阶
# Fancy indexing
a = np.arange(0, 100, 10)
indices = [1, 5, -1]
b = a[indices]
print(a) # >>>[ 0 10 20 30 40 50 60 70 80 90]
print(b) # >>>[10 50 90]

# Incomplete Indexing
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(b) # >>>[ 0 10 20 30 40]
print(c) # >>>[50 60 70 80 90]

# Where
a = np.arange(0, 100, 10)
b = np.where(a < 50)
c = np.where(a >= 50)[0]
print(b) # >>>(array([0, 1, 2, 3, 4]),)
print(c) # >>>[5 6 7 8 9]