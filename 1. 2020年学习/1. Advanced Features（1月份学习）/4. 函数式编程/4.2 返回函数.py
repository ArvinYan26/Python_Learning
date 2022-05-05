#我们来实现一个可变参数的求和。通常情况下，求和的函数是这样定义的：
"""
def calc_sum(*args):   #args:参数，可以使一个list
    ax = 0
    for n in args:
        ax = ax + n
    return ax
#L = list(range(100))
sum = calc_sum(1, 2, 3, 4)
print(sum)
"""
#但是，如果不需要立刻求和，而是在后面的代码中，根据需要再计算怎么办？可以不返回求和的结果，而是返回求和的函数：
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum()
f = lazy_sum(1, 3, 4, 5)

#闭包
def count():
    fs = []
    for i in range(1, 4):
        def f():
            return i * i
        fs.append(f)
    return fs
f1, f2, f3 = count()

#如果一定要引用循环变量怎么办？方法是再创建一个函数，用该函数的参数绑定循环变量当前的值，无论该循环变量后续如何更改，已绑定到函数参数的值不变：
def count():
    def f(j):
        def g():
            return j * j
        return g
    fs = []
    for i in range(1, 4):
        fs.append(f(i)) # f(i)立即被执行，因此i的当前值被传入分f（）
    return fs
f1, f2, f3 = count()

#作业
#利用闭包返回一个计数器函数，每次调用它返回递增整数：
def creatCounter():
    n = 0
    def counter():
        nonlocal n   #nonlocal:非本地
        n = n + 1
        return n
    return counter()
