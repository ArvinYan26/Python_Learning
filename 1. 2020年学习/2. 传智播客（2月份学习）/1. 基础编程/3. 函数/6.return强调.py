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