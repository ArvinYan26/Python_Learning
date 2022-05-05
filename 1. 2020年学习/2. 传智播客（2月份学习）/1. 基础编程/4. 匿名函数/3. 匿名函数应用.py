"""
def test(a,b,func):
    result = func(a,b)
    return result

ret = test(3, 4, lambda x,y:x+y)
print(ret)
"""

#另一种方式
def test(a,b,func):
    result = func(a,b)
    return result
func_new = input("请输入一个匿名函数：")
func_new = eval(func_new)
ret = test(3, 4, func_new)
print(ret)

#c++和c是静态语言，在编译前所有功能都得确定，不能让你那个到运行了才输入功能
#python是静态语言，可以编译的时候再确定功能，更加灵活多变


