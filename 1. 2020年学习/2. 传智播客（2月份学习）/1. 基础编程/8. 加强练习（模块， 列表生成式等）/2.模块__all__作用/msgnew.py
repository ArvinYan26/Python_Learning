#__all__,放入将来允许别人调用的函数名，不让用的不放在列表里即可，这样就防止别人调用你所有的函数

__all__ = ["test1", "Test"]

def test1():
    print("-----test1----")

def test2():
    print("-----test2----")

num = 100

class Test(object):
    pass