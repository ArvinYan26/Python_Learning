#所以，如果列表元素可以按照某种算法推算出来，那我们是否可以在循环的过程中不断推算出后续的元素呢？这样就不必创建完整的list，
#从而节省大量的空间。在Python中，这种一边循环一边计算的机制，称为生成器：generator。
#要创建一个generator，有很多种方法。第一种方法很简单，只要把一个列表生成式的[]改成()，就创建了一个generator：
#1. 列表生成式
L = [x * x for x in range(10)]
print(L)

#2. 生成式
g = (x * x for x in range(10))
print(g)

#当然，上面这种不断调用next(g)实在是太变态了，正确的方法是使用for循环，因为generator也是可迭代对象：
g = (x * x for x in range(10))
for n in g:
    print(n)

#3. 比如，著名的斐波拉契数列（Fibonacci），除第一个和第二个数外，任意一个数都可由前两个数相加得到：

#1, 1, 2, 3, 5, 8, 13, 21, 34, ...

#斐波拉契数列用列表生成式写不出来，但是，用函数把它打印出来却很容易：
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a+b
        n = n + 1
    return 'done'
fib(6)  #把6的值赋给函数fib中的max

#赋值语句：
"""
a, b = b, a+b
#相当于：
t = (b, a + b)
a = t[0]
b = t[1]
"""

#生成式，把上述fib函数的print（b）改为yield b即可
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a+b
        n = n + 1
    return 'done'
f = fib(6)  #把6的值赋给函数fib中的max

#用生成式，生成杨辉三角
def triangles():
    L = [1]
    n = 0
    while True:
        yield L
        L = [1] + [L[n] + L[n+1] for n in range(len(L)-1)] + [1]
        n = n + 1
    return done
t = triangles()