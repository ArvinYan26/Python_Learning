#装饰器就相当于调用了一个函数
def func_arg(arg):
    def func(functionName):   #3-8行是一个闭包
        def func_in():
            print("---记录日志--arg=%s-" %arg)
            if arg == "heihei":
                functionName()
                functionName()
            else:
                functionName()
        return func_in
    return func

#1. 先执行func_arg("heihei")函数，这个函数return的结果是func这个函数的引用
#2. @func("heihei")就变成了@func
#3. 使用func对test进行装饰

#带有参数的装饰器，能够起到在运行时起到不同的功能
@func_arg("heihei")
def test():
    print("----test----")

@func_arg("haha")
def test2():
    print("-----test-----")

test()
test2()