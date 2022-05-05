#传统方法
a = []
i = 10
while i <= 77:
    a.append(i)
    i += 1

print(a)

#range
for i in range(10, 78):
    print(i)

#range
for i in range(10, 24, 2):  #2是步长
    print(i)

#列表生成式 ,i和11是存放的数据，range只是负责次数，每循环一次就放在列表里一次，
#如果是固定值，那么就是固定值存放列表中n次，也就是循环的次数
a = [i for i in range(1, 18)]
print(a)
a = [11 for i in range(1, 18)]
print(a)

#生成10以内的偶数
c = [i for i in range(10) if i%2 == 0]
print(c)

#两个变量的列表生成式
d = [(i, j) for i in range(3) for j in range(2)]
print(d)

#三个变量
e = [(i, j, k) for i in range(3) for j in range(2) for k in range(3)]
print(e)

#列表去重
a = [11, 22, 33, 44, 11, 22, 33, 44]
b = []
for i in a:
    if i not in b:
        b.append(i)
print(b)

 
