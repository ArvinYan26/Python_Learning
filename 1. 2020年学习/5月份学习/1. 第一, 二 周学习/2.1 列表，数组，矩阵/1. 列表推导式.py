"""
列表推导式提供了从序列创建列表的简单途径。通常应用程序将一些操作应用于某个序列的每个元素，用其获得的结果作为生成新列表的元素，
或者根据确定的判定条件创建子序列。每个列表推导式都在 for 之后跟一个表达式，然后有零到多个 for 或 if 子句。返回结果是一个根据表达从其后的
for 和 if 上下文环境中生成出来的列表。如果希望表达式推导出一个元组，就必须使用括号。
这里我们将列表中每个数值乘三，获得一个新的列表：
"""
li = [2, 4, 6]
li_1 = [3*x for x in li]
print("li_1:", li_1)

li_2 = [[x, x**2] for x in li]
print("li_2:", li_2)

li_3 = [[x, x**2, x*2] for x in li]
print("li_3:", li_3)

#这里我们对序列里每一个元素逐个调用某方法：
freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
s = [weapon.strip() for weapon in freshfruit]  #strip:去掉字符串开头和结尾的空格
print(s)

#我们可以用 if 子句作为过滤器：
li_4 = [3*x for x in li if x > 3]
print("li_4:", li_4)

#练习
#如果list中既包含字符串，又包含整数，由于非字符串类型没有lower()方法
#请修改列表生成式，通过添加if语句保证列表生成式能正确地执行：
L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = [s.lower() for s in L1 if isinstance(s, str)]
print(L2)
L2 = [s.lower() if isinstance(s, str) else s for s in L1]
print(L2)

#生成数组
import numpy as np
#生成列表
a = [[0 for i in range(4)] for i in range(4)]
#转化成数组
a = np.array(a)
print(a)

#直接生成数组
a1 = np.zeros((4, 4))
print("a1:", a1)

"""
#列表嵌套解析
"""
Matrix = np.arange(1, 13).reshape(3, 4)
print(Matrix)
print(Matrix[0])
"""
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

"""
Matrix_new = [[row[i] for row in Matrix] for i in range(4)]
print(Matrix_new)
print(Matrix_new[0])


