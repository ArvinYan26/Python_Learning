#Python的循环有两种，一种是for...in循环，依次把list或tuple中的每个元素迭代出来，看例子：
names = ['a', 'c', 'c']
for name in names:
    print(name)

#所以for x in ...循环就是把每个元素代入变量x，然后执行缩进块的语句。
sum = 0
for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    sum = sum + x
print(sum)

sum = 0
for x in range(101):
    #.list(range(101))
    sum = sum + x
print(sum)

for x in range(5, 10):
    print(x)