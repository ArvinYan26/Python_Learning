#1.位置参数
#我们先写一个计算x**2的函数：
#对于power(x)函数，参数x就是一个位置参数。
"""
def power(x):
    return x * x
y = power(float(input("请输入x的值:")))
print(y)

#计算x的n次幂

def power(x, n):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s
x = float(input("请输入x的值:"))
n = float(input("请输入n的值:"))
y = power(x, n)
print(y)

#默认参数
#这个时候,默认参数就排上用场了。由于我们经常计算x2，所以，完全可以把第二个参数n的默认值设定为2：

def power(x, n=2):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s
y = power(float(input("请输入x的值：")))
print(y)
"""

#可变参数
#在Python函数中，还可以定义可变参数。顾名思义，可变参数就是传入的参数个数是可变的，可以是1个、2个到任意个，还可以是0个。
#我们以数学题为例子，给定一组数字a，b，c……，请计算a2 + b2 + c2 + ……。
#要定义出这个函数，我们必须确定输入的参数。由于参数个数不确定，我们首先想到可以把a，b，c……作为一个list或tuple传进来，这样，函数可以定义如下：
def calc(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
l = list(range(1, 5))
y = calc(l)
print(y)

#关键字参数
"""
def person(name, age, **kw):
extra = {'city': 'Beijing', 'job': 'Engineer'}
person('Jack', 24, **extra)
"""
#命名关键字参数
#如果要限制关键字参数的名字，就可以用命名关键字参数，例如，只接收city和job作为关键字参数。这种方式定义的函数如下：
def person(name, age, *, city, job):
    print(name, age, city, job)
#和关键字参数**kw不同，命名关键字参数需要一个特殊分隔符*，*后面的参数被视为命名关键字参数。
#调用方式如下：
def person(name, age, *, city, job):
    print(name, age, city, job)

person('jack', 24, city = 'Beijing', job = 'Engineer')

#由于命名关键字参数city具有默认值，调用时，可不传入city参数：
def person(name, age, *, city='Beijing', job):
    print(name, age, city, job)
person('Jack', 24, job='Engineer')


#参数组合
"""
要注意定义可变参数和关键字参数的语法：
*args是可变参数，args接收的是一个tuple；
**kw是关键字参数，kw接收的是一个dict。
可变参数既可以直接传入：func(1, 2, 3)，又可以先组装list或tuple，再通过*args传入：func(*(1, 2, 3))；
关键字参数既可以直接传入：func(a=1, b=2)，又可以先组装dict，再通过**kw传入：func(**{'a': 1, 'b': 2})。
"""

##########    传播智课的内容   ###################
def test(a, b=22,c=72):
    result = a + b
    print("result=%d"%result)
test(11)  #默认使用b = 22这个值，默认参数
test(33, 44)  #只要b赋值了，那么传进去的b的值就会改变，不在用默认的b= 222这个值
test(55)
test(11, c=44) #可以指定形参赋值，但是形参名字必须一致即c。这个c=44叫命名参数

#不定长参数1:*args，必须放在形参最后位置，切记,以元组存储。*：告诉python解释器，args可以存储多余的不带变量名的实参
def sum_2_nums(a, b,*args): #*具有特殊功能，形参还是args，*args可以接受多个实参
    print("_"*30)
    print(a)
    print(b)
    print(args)
    #result = a + b
    #print("result=%d"%result)

sum_2_nums(11,22,8, 4,3,55,44)
sum_2_nums(11,33,77)  #显示结果是（77，）如果要告诉别人一个元组只有一个元素，元素后边必须加逗号。

#不定长参数2：**kwargs，以字典方式存储，**：告诉python，kwagrs可以存储带变量名的实参
def sum_2_nums(a, b,*args,**kwargs): #*具有特殊功能，形参还是args，*args可以接受多个实参
    print("_"*30)
    print(a)
    print(b)
    print(args)
    print(kwargs)
#多余的实参，如果不带变量名就传给*args这个形参，用元组存储，如果带变量名，全部传给**kwargs这个形参，用字典存储
sum_2_nums(11,22,8, 4,3,task = 55,done=44)

#不定长参数3:*args，必须放在形参最后位置，切记,以元组存储。*：告诉python解释器，args可以存储多余的不带变量名的实参
def sum_2_nums(a, b,*args): #*具有特殊功能，形参还是args，*args可以接受多个实参
    print("_"*30)
    print(a)
    print(b)
    print(args)
    result = a + b
    for num in args:
        result += num

    print("result=%d"%result)

sum_2_nums(11,22,8, 4,3,55,44)
sum_2_nums(11,33,77)  #显示结果是（77，）如果要告诉别人一个元组只有一个元素，元素后边必须加逗号。
