def test(a, b,*args,**kwargs): #*具有特殊功能，形参还是args，*args可以接受多个实参
    print("_"*30)
    print(a)
    print(b)
    print(args)
    print(kwargs)
#多余的实参，如果不带变量名就传给*args这个形参，用元组存储，如果带变量名，全部传给**kwargs这个形参，用字典存储
#test(11,22,8, 4,3,task = 55,done=44)

A = (44, 55, 66)
B = {"name": "laowang", "age": 18}

test(11, 22, 33, A, B)  #如果不加*，A,B偶给args
#把A拆出来给args，把B给kwargs
test(11, 22, 33, *A, **B)
#一个*:把元组拆成一个个的值，把字典拆成一个key，一个值，


