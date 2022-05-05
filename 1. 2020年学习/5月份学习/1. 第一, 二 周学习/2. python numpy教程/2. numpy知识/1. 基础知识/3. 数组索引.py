import numpy as np

"""
#方法一：
a = np.arange(1, 13)
print(a)
a = a.reshape(3, 4)
print(a)

[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

#方法二：
#a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

b = a[:2, 1:3]
print(b)

[[2 3]
 [6 7]]

print(a[0, 1])
b[0, 0] = 77 #b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])

#你还可以将整数索引与切片索引混合使用。 但是，这样做会产生比原始数组更低级别的数组。 请注意，这与MATLAB处理数组切片的方式完全不同：
a = np.arange(1, 13)
print(a)
a = a.reshape(3, 4)
print(a)
#a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

＃两种访问数组中间行数据的方式。
＃将整数索引与切片混合会产生一个较低等级的数组，
＃仅使用切片时，产生的数组与
＃原始数组：


row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
"""
"""
#整数数组索引: 使用切片索引到numpy数组时，生成的数组视图将始终是原始数组的子数组。 相反，整数数组索引允许你使用另一个数组中的数据构造任意数组。 这是一个例子：
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"
# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"

#整数数组索引的一个有用技巧是从矩阵的每一行中选择或改变一个元素：
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3, ],
          #                [ 4,  5, 16, ],
          #                [17,  8,  9, 2],
          #                [10, 21, 12]])
"""


#数据类型
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)        # Prints "int64"






