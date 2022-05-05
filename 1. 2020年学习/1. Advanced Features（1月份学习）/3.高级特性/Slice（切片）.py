#1. 回忆之前的学习
"""
L = []
n = 1
while n <= 99:
    L.append(n)
    n = n + 2
print(L)
"""
#可以用循环取一个列表的前n个元素
"""
L = ['a', 'b', 'c', 'd']
r = []
n = 3
for i in range(n):
    r.append(L[i])
print(r)

#2. 用切片操作
L = ['a', 'b', 'c', 'd']
r = L[0:3]    #如果第一个索引是0，可以省略0即：L[:3]
print(r)

#也可以从1开始取两个元素
s = L[1:3]  #取第一个和第二个元素
print(s)

#类似的，既然Python支持L[-1]取倒数第一个元素，那么它同样支持倒数切片，试试：
z = L[-2:]  #取索引为-1，-2的元素。也就是倒数第一和第二和正数有所不同
print(z)
"""
#3. 创建一个0-99的数列
L = list(range(100))
print(L)

#取前10个数
r = L[:11]
print(r)

#取后10个数
s = L[-10:]
print(s)

#前10个每两个取一个
z = L[:11:2]
print(z)

#所有数每5个取一个
a = L [::5]
print(a)

#赋值一个列表，即取所有数
q = L[:]
print(q)

#4. tuple也是一种list，唯一区别是tuple不可变。因此，tuple也可以用切片操作，只是操作的结果仍是tuple：
T = (1, 2, 3, 4, 5, 6)
t = T[:3]
print(t)

#5. 字符串也一样
L = 'ABCDEFG'
s = L[:3]
print(s)
q = L[::2]
print(q)

#作业：用切片完成一个trim函数（清楚两端的空格）
