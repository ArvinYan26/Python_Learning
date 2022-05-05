def func(functionName):
    print("-------1-------")

    def func_in():
        print("-----func_in----1---")
        ret = functionName()  #ret是用来保存返回来的hah
        print("------func_in-----2-----")
        return ret  #把haha返回到ret = test()调用处
    print("0-----func------2")
    return func_in  # 不带括号时表示只是返回这个函数

@func  #test = func(test) ,对下面这个函数进行封装
def test():
    print("------test-------")
    return "haha"

ret = test()
print("test return value is %s" %ret)