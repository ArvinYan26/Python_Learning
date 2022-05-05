#测量温度,return:把一个值返回到调用的地方去

def get_wendu():
    wendu = 22
    print("当前的室温是：%d" %wendu )
    return wendu  #如果一个函数有return，就会将结果返回到调用处

def get_wendu_huashi(wendu):
    wendu = wendu  + 273.15
    print("当前的温度（华氏）是：%d"%wendu)

result = get_wendu()

get_wendu_huashi(result)

#将函数值返回，想要使用函数的值，就必须要找参数接收
#*args：表示可以传入多个参数.如果返回的参数是多个，要么用多个变量去接收，要么，就是以一个元组的整体形式返回给一个参数
def sum(*args):
    sum = 0
    for n in args:
        sum = sum + n
    return sum, 9

sum = sum(1, 2, 3)
print(sum)