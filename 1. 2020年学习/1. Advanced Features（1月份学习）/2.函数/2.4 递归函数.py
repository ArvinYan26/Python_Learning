#计算一个数的阶乘，只需要在n=1时需要特别处理
"""
def fact(n):
    if n == 1:
        return 1
    return n * fact(n - 1)
Result = fact(int(input('请输入n的值：')))
print(Result)

#解决递归调用栈溢出的方法是通过尾递归优化，事实上尾递归和循环的效果是一样的，所以，把循环看成是一种特殊的尾递归函数也是可以的。
#尾递归是指，在函数返回的时候，调用自身本身，并且，return语句不能包含表达式。这样，编译器或者解释器就可以把尾递归做优化，使递归本身无论调用多少次，都只占用一个栈帧，不会出现栈溢出的情况。
def fact(n):
    return fact_iter(n, 1)
def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)
Result = fact(int(input('请输入n的值：')))

#尾递归调用时，如果做了优化，栈不会增长，因此，无论多少次调用也不会导致栈溢出。
#遗憾的是，大多数编程语言没有针对尾递归做优化，Python解释器也没有做优化，所以，即使把上面的fact(n)函数改成尾递归方式，也会导致栈溢出
"""
# 汉诺塔（递归问题）的关键就在于放弃我们让大脑（或用手和笔）去跟踪函数运行每一步的执行的习惯，利用用抽象和自动化的思想解决问题。
# 汉诺塔是一个非常好的帮助我们从传统的数学思维转变到计算思维的小问题，汉诺塔问题让我第一次认识到计算机/算法的神奇和魅力所在！
def hanoi(n, a, b, c):
    if n == 1:
        print(a, '-->', c)
    else:
        hanoi(n - 1, a, c, b)
        print(a, '-->', c)
        hanoi(n - 1, b, a, c)
hanoi(10, 'a', 'b', 'c')
