def func(functionName):

    def func_in(*args, **kwargs):
        print("-----记录日志-----")
        ret = functionName(*args, **kwargs)  #ret是用来保存返回来的hah
        return ret  #把haha返回到ret = test()调用处

    return func_in  # 不带括号时表示只是返回这个函数

@func  #test = func(test) ,对下面这个函数进行封装
def test():
    print("------test----")
    return "haha"

@func     #对没有返回值的函数进行封装
def test2():
    print("------test2----")

@func   #对有参数的函数进行封装
def test3(a):
    print("----test3--a=%d---"%a)

ret = test()
print("test return value is %s" %ret)

a = test2()
print("test2 return value is %s"%a)

test3(11)


