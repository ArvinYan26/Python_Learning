#由于函数也是一个对象，而且函数对象可以被赋值给变量，所以，通过变量也能调用该函数。
def now():
    print('2020-02-27')
f = now()

#现在，假设我们要增强now()函数的功能，比如，在函数调用前后自动打印日志，但又不希望修改now()函数的定义，这种在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）。
#本质上，decorator就是一个返回函数的高阶函数。所以，我们要定义一个能打印日志的decorator，可以定义如下：

def log(func):
    def wrapper(*args, **kw):   #关键字参数  #  闭包：：从此处到12那个的所有代码成为闭包，在函数内部在定义一个函数并且要用到原来函数的变量
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

#闭包  (传智播客教程 )
#case1
def test(number):
    print("---1---")
    def test_in(number2):
        print("---3---")
        print(number+number2)
    print("--3--")
    return test_in
#调用函数
ret = test(100) # 传递参数给number=100   ,ret最终指向的是test_in这个函数体
print("*"*30)
ret(100)     #因为ret函数指向的是test_in这个函数体，所以100是给number2传参数

#case2
def test(a,b):
    def test_in(x):
        print(a*x+b)
    return test_in
line1 = test(1,1)  #line1指向test这个函数，test函数返回的是test_in函数，所以最终line1指向test_in函数体
line1(0)            #此处是调用line1函数，将0传参给x
line2 = test(10,4)
line2(2)

line1(2)

#########################################装饰器-1#####################
#装饰器是程序开发中经常会用到的一个功能，用好了装饰器，开发效率如虎添翼，所以这也是Python面试中必问的问题，但对于好多初次接触这个知识的人来讲，
# 这个功能有点绕，自学时直接绕过去了，然后面试问到了就挂了，因为装饰器是程序开发的基础知识，这个都不会，别跟人家说你会Python, 看了下面的文章，
# 保证你学会装饰器。
def test1():
    peint("------1-------")
def test1():
    print("-----2------")
test1()

#开发代码的时候别轻易删代码，因为会导致其他代码的崩溃，非常麻烦
def w1(func):
    #定义一个闭包
    def inner():
        print("---正在验证权限------")
        func()  #调用func指向的那个函数
    return inner #闭包定义完成
def f1():
    print("----f1------")
def f2():
    print("-----f2------")

#innerFunc = w1(f1) #innnerFunc指向w1函数，并将f1函数传给w1中的func参数
#innerFunc()   #调用函数innnerFunc，返回inner函数，闭包性质。返回的inner函数中调用了func函数，而它只想的是f1函数，此时调用了f1函数，执行f1函数

f1 = w1(f1)
f1()  #虽然调用了f1，但是f1的功能已经变了，因为指向的函数变为了inner（inner中调用的func函数其实是指向的原来的f1），
      #这就给人感觉是在没有修改f1的前提下扩展了f1的功能，这就是装饰器的意义。

#########################################装饰器-2#####################
def w1(func):
    #定义一个闭包
    def inner():
        print("---正在验证权限------")
        if False:  #改为True将会是另外一种情况
            func()
        else:
            print("-----没有权限----")
    return inner #闭包定义完成

#f1 = w1(f1)
@w1  #等价于f1 = w1(f1)
def f1():
    print("----f1------")

@w1  #等价于 等价于f2 = w1(f2)
def f2():
    print("-----f2------")

#f1 = w1(f1) #等价于@1 ：语法糖

f1()
f2()

#########################再议装饰器####################################
#定义函数，完成包裹数据
def makeBold(fn):
    def wrapped():
        print("-----1------")  #为了了解其运行逻辑，所以加此语句
        return "<b>" + fn() + "</b>"
    return wrapped

#定义函数，完成包裹数据
def makeItalic(fn):
    #定义一个闭包
    def wrapped():
        print("-----2------")
        return "<i>" + fn() + "</i>"
    return wrapped

@makeBold
def test1():
    return "hello world-1"

@makeItalic
def test2():
    return "hello world-2"

@makeBold  #等价于test3=makeBold(test3())#虽然先执行这句程序，但是装饰的时候紧跟着的代码不是函数，如果不是，暂停，所以要等下面的装饰函数完成以后再装饰，所以执行到此就直接执行下一句，
           #下一句直接装饰了test3函数，然后返回<i>hello world-3</i>，接下来在执行@makeBold进行装饰，结果返回
           #<b><i>hello world-3</i></b>

@makeItalic   #装饰器装饰的时候不管多少层，都是从下向上装饰，调用的时候是从上向下调用。牢记
def test3():
    print("-----3------")
    return "hello world-3"

test3()
print(test1())
print(test2())
ret = test3()
print(ret)

#########################装饰器装饰的时间####################################
def w1(func):   #func变量名指向f1这个函数
    print("-----正在装饰------")
    def inner():  #其实执行到此处的时候，因为这只是定义一个函数，不执行，所以执行return inner(),
        print("-----正在验证权限-------")
        func()
    return inner()

#只要python解释器执行到此代码，那么就会自动进行装饰，而不是等到调用的时候才装饰
@w1  #f1 = w1(f1)  f1表面上看没有改变，其实已经指向了函数体inner（），不改变f1的情况下，扩展了f1的功能，这就是装饰器
def f1():
    print("-------f1--------")

#在调用f1之前就已经进行装饰了
f1()
##################使用装饰器对有参数的函数进行装饰###################################
#1.对普通有参数函数进行调用
def func(functionName):
    print("-------1-------")
    def func_in():
        print("-----func_in----1---")
        functionName()   #调用functionName只想的那个函数
        print("------func_in-----2-----")
    print("0-----func------2")
    return func_in  #不带括号时表示只是返回这个函数

def test(a, b):
    print("------test-a=%d, b=%d---"%(a, b))

test(11, 22)
