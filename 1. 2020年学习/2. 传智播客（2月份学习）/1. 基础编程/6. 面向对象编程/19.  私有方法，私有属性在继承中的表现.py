class A:
    def __init__(self):
        self.num1 = 100
        self.num2 = 200

    def test1(self):
        print("-------test1------")

    def __test2(self):
        print("------test2------")

    def test3(self):
        self.__test2()
        print(self.num2)

class B(A):
    def test4(self):
        self.__test2()
        print(self.num2)

b = B()
b.test1()
#b.__test2()  #私有方法并不会被继承
print(b.num1)
#print(b.__num2)  #私有方法不能继承。这些类似于父类有秘密，不能让子类知道继承

b.test3() #继承的父类里的方法中调用了父类的私有属性或者私有方法，这种是可以的
b.test4() #但是如果是子类里定义一个新方法，这个方法想要调用父类里的私有属性和私有方法的话是不行的


