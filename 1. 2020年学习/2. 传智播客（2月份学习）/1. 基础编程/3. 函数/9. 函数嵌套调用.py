def test1():
    pass

def test2():
    print("===21===")
    print("===haha===")
    print("====22===")

def test3():
    print("=======31======")
    test2()
    print("=====32=====")

test3()

############################应  用##################
def print_line():
    print("-"*50)

print_line()

#打印5条线，重复时使用函数使用，研发时一般这样
#快速迭代就是在原有版本上更新，
def print_line():
    print("-" * 50)

def print_5_line():
    i = 0
    while i < 5:
        print_line()
        i += 1
print_5_line()

###
def sum(a, b, c):  #形参
    sum = a + b + c
    return sum

def average(a1, b1, c1): #形参
    result = sum(a1, b1, c1)  #实参，嵌套调用了sum函数，计算来了三个数的和存在result中
    result /= 3 #result = result/3
    print("平均值是：%d" %result)

#获取三个数
num1 = int(input("请输入第1个数的值："))
num2 = int(input("请输入第1个数的值："))
num3 = int(input("请输入第1个数的值："))

average(num1, num2, num3) #实参
