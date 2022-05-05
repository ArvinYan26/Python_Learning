
def get_wendu():
    wendu = 22    #局部变量
    print("当前的室温是：%d" %wendu )
    return wendu  #如果一个函数有return，就会将结果返回到调用处

def get_wendu_huashi(wendu):
    wendu = wendu  + 273.15
    print("当前的温度（华氏）是：%d"%wendu)
#如果一个函数有返回值，但是没有在调用之前用个变量保存的话，那么没有任何意义
result = get_wendu()

get_wendu_huashi(result)

#全局变量

#定义一个全局变量
wendu = 0
def get_wendu():
    #如果wendu这个变量已经在全局变量的位置定义了，此时还想修改它，
    #仅仅是wendu = 一个值是不够的，此时这个wendu还是局部变量，只是名字相同罢了
    #使用global对一个全局部变量的声明，此时的wendu变量就会变成全局变量
    global wendu
    wendu = 33  #此时wendu这个去全局变量被修改成了33

def print_wendu():
    print("当前的温度是：%d"%wendu)
#如果一个函数有返回值，但是没有在调用之前用个变量保存的话，那么没有任何意义
get_wendu()
print_wendu()
