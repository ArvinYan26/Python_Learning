#函数里面的只要运行单return就会介数，下面的return不再运行，break是结束循环 
def test():
    a = 11
    b = 22
    c = 33
    return a
    return b  #第一个return已经结束了运行，下面的return不再返回了
    return c

num = test()
print(num)

#另一个方法，返回时三个值################
def test():
    a = 11
    b = 22
    c = 33
    #用列表封装三个值
    d = [a, b, c]
    return d

num = test()
print(num)

#第二种方法################
def test():
    a = 11
    b = 22
    c = 33
    #用列表封装三个值
    return [a, b, c]

num = test()
print(num)

#第三种方法################
def test():
    a = 11
    b = 22
    c = 33
    #用列表封装三个值
    #return (a, b, c)
    #return {a, b, c}
    return a, b, c #相当于用元组封装，返回

num = test()
print(num)