import numpy as np


#取自己想要的数据
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
a = arr[arr%2==0]
print(a)
"""
[0 2 4 6 8]
"""

#**问题：**将arr中的所有奇数替换为-1。
arr[arr%2==1] = -1
print(arr)
"""
[ 0 -1  2 -1  4 -1  6 -1  8 -1]
"""

#**问题：**将arr中的所有奇数替换为-1，而不改变arr。
arr = np.arange(10)
out = np.where(arr %2 == 1, -1, arr)
print(arr)
"""
[ 0 -1  2 -1  4 -1  6 -1  8 -1]
[0 1 2 3 4 5 6 7 8 9]
"""

#改变数组的形状,**问题：**将一维数组转换为2行的2维数组
a = np.arange(10)
a = a.reshape(2, -1) #设置-1：表示是numpy自动计算形成的新的数组的列数，
print(a)
#也可以手动设置，自己计算
a = a.reshape(2, 5)
print(a)
print(len(a))
"""
[[0 1 2 3 4]
 [5 6 7 8 9]]
"""

#**问题：**垂直堆叠数组a和数组b
#repeat(1, 10):1这个数字重复10次
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
# Answers
# Method 1:
np.concatenate([a, b], axis=0) #水平叠加，axis=1

# Method 2:
np.vstack([a, b])

"""
x = np.arange(0,10,2)                     # x=([0,2,4,6,8])
y = np.arange(5)                          # y=([0,1,2,3,4])
m = np.vstack([x,y])                      # m=([[0,2,4,6,8],
                                          #     [0,1,2,3,4]])
xy = np.hstack([x,y])                     # xy =([0,2,4,6,8,0,1,2,3,4])
"""

#**问题：**从数组a中删除数组b中的所有项。
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print(len(a))
a = np.setdiff1d(a, b)
print(a)

#**问题：**获取a和b元素匹配的位置。
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
pos = np.where(a == b)
print(pos)

#**问题：**创建一个形状为5x3的二维数组，以包含5到10之间的随机十进制数
arr = np.arange(9).reshape(3, 3)
#method 1
rand_arr = np.random.randint(low=5, high=10, size=(5, 3)) + np.random.random((5,3))
#method 2
rand_arr = np.random.uniform(5, 10, size=(5, 3))
print(rand_arr)


