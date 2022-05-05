L = list(range(1, 11))
print(L)

#但如果要生成[1x1, 2x2, 3x3, ..., 10x10]怎么做？方法一是循环：

L = []
for x in range(1, 11):
    L.append(x*x)
print(L)

#用列表生成式一句话完成

Z = [x * x for x in range(1, 11) ]
print(Z)

#for循环后面还可以加上if判断，这样我们就可以筛选出仅偶数的平方：
Z = [x * x for x in range(1, 11) if x % 2 ==0 ]
print(Z)

#还可以使用两层循环，可以生成全排列：
H = [m + n for m in 'ABC' for n in 'XYZ']
print(H)

#for循环其实可以同时使用两个甚至多个变量，比如dict的items()可以同时迭代key和value：
d = {'x': 'A', 'y': 'B', 'z': 'C' }
for k, v in d.items():
    print(k, '=', v)

#因此，列表生成式也可以使用两个变量来生成list：
d = {'x': 'A', 'y': 'B', 'z': 'C' }
s = [k + '=' + v for k, v in d.items()]
print(s)
#练习
#如果list中既包含字符串，又包含整数，由于非字符串类型没有lower()方法
#请修改列表生成式，通过添加if语句保证列表生成式能正确地执行：
L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = [s.lower() for s in L1 if isinstance(s, str)]
print(L2)
L2 = [s.lower() if isinstance(s, str) else s for s in L1]
print(L2)