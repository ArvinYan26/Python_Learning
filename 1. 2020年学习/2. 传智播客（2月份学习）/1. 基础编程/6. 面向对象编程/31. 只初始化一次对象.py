#确保某一个类只有一个实例，而且自行实例化并向整个系统提供这个实例，这个类称为单例类，单例模式是一种对象创建型模式。
#下面这个例子初始化了两次，
class Dog(object):

    __instance = None

    def __new__(cls, name):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    def __init__(self, name):
        self.name = name

a = Dog("旺财")
print(id(a))
print(a.name)

b = Dog("哮天犬")
print(id(b))
print(b.name)