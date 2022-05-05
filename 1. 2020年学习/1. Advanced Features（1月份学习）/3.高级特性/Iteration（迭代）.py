#如果给定一个list或tuple，我们可以通过for循环来遍历这个list或tuple，这种遍历我们称为迭代（Iteration）。

#在Python中，迭代是通过for ... in来完成的，而很多语言比如C语言，迭代list是通过下标完成的，比如Java代码：

#不管有没有下标，只要是可迭代对象，都可以迭代
d = {'a':1, 'b':2, 'c':3}
for key in d:
    print(key)

#默认情况下，dict迭代的是key。如果要迭代value，可以用for value in d.values()，如果要同时迭代key和value，可以用for k, v in d.items()。
for k, v in d.items():
    print(k, v)
for ch in 'abc':
    print(ch)

#如何判断一个对象是可迭代对象呢？方法是通过collections模块的Iterable类型判断：
from collections import Iterable
isinstance('abc', Iterable)

#最后一个小问题，如果要对list实现类似Java那样的下标循环怎么办？Python内置的enumerate函数可以把一个list变成索引-元素对，这样就可以在for循环中同时迭代索引和元素本身：
for i , value in enumerate(['A', 'B', 'C']):
    print(i, value)

#上面的for循环里，同时引用了两个变量，在Python里是很常见的，比如下面的代码：
for x, y in [(1, 1), (2, 2), (3, 3), (4, 4)]:
    print(x, y)

#作业：请使用迭代查找一个list中最小和最大值，并返回一个tuple：
def findMinAndMax(L):
    if len(L)==0:
        return(None, None)
    elif len(L)==1:
        return(L[0], L[0])
    elif len(L)==2:
        if L[0] > L[1]:
            return(L[1], L[0])
        else:
            return(L[0], L[1])
    else:
        i = 0
        s = L[i]
        b = L[i]
        for i in range(len(L)-2):
            if s > L[i +1]:
                s = L[i+1]
            if b < L[i+1]:
                b = L[i+1]
            return(s, b)
L =(1, 4, 8, 9 ,16, 3)