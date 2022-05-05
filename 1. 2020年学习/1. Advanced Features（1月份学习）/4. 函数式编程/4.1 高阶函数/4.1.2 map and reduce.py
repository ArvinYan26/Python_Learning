#Python内建了map()和reduce()函数。
#我们先看map。map()函数接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回。
#举例说明，比如我们有一个函数f(x)=x**2，要把这个函数作用在一个list [1, 2, 3, 4, 5, 6, 7, 8, 9]上，就可以用map()实现如下：
def f(x):
    return x * x
r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
list(r)

L = []
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    L.append(f(n))
print(L)

#所以，map()作为高阶函数，事实上它把运算规则抽象了，因此，我们不但可以计算简单的f(x)=x**2，还可以计算任意复杂的函数，比如，把这个list所有数字转为字符串：

s = list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(s)

#reduce函数
#再看reduce的用法。reduce把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算，其效果就是：
#reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
#比方说对一个序列求和，就可以用reduce实现：
from functools import reduce
def add(x, y):
    return x + y
sum = reduce(add, [1, 3, 5, 7, 9])
print(sum)

#当然求和运算可以直接用Python内建函数sum()，没必要动用reduce。
#但是如果要把序列[1, 3, 5, 7, 9]变换成整数13579，reduce就可以派上用场：
def fn(x, y):
    return x*10 + y
I = reduce(fn, [1, 3, 5, 7, 9])
print(I)

#这个例子本身没多大用处，但是，如果考虑到字符串str也是一个序列，对上面的例子稍加改动，配合map()，我们就可以写出把str转换为int的函数：
from functools import reduce
def fn(x, y):
    return x*10 + y

def char2num(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digits[s]

X = reduce(fn, map(char2num, '13579'))
print(X)

#练习
#1. 利用map()函数，把用户输入的不规范的英文名字，变为首字母大写，其他小写的规范名字。输入：['adam', 'LISA', 'barT']，输出：['Adam', 'Lisa', 'Bart']：
#2. Python提供的sum()函数可以接受一个list并求和，请编写一个prod()函数，可以接受一个list并利用reduce()求积：
#3. 利用map和reduce编写一个str2float函数，把字符串'123.456'转换成浮点数123.456：

#1.
def normalize(name):
    def nor(name):
        return name[0].upper()+name[1:].lower  #upper:转换为大写字母
    return list(map(nor, name)
L = ['adam', 'LISA', 'BARt', 'guCCy', '5465']
print(normalize(L))

#2.
def prod(X)
    def Fun(a, b):
        return a * b
    return reduce(Fun, X)

#3.
from functools import reduce

def str2float1(k):

     s = k.split('.')

     sk = s[0] + s[1]

     def char2num(k):

          digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

          return digits[k]

     return reduce(lambda x, y: x * 10 + y, map(char2num, sk))/10**len(s[1])



