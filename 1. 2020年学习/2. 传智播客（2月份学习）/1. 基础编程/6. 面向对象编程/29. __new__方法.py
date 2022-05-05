#__new__方法只负责创建，__init__方法只负责初始化
#C中是一个构造方法就完成了上面的两个方法的功能
class Dog(object):
    def __init__(self):
        pass
    def __del__(self): #所有的引用结束时，此方法执行
        pass
    def __str__(self):
        return "对象的返回信息"

    # cls此时是Dog指向的那个类对象，
    def __new__(cls):
        print("----new----")
        return object.__new__(cls)
xtq = Dog() #此处是先调用new方法然后接收返回值，自己再调用init方法，而不是通过new方法直接调用的init方法
#相当于做了三件事
#1. 调用new方法来创建对象，然后找了一个变量来接收new方法的返回值，这个返回值表示创建出来的对象的引用
#2. 调用init方法
#3. 返回对象的引用
