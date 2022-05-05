#object:称为新式类，如果写上，就代表是所有类的基类，即boss，python3中独有的，如果不写就默认为经典类

class Base(object):  #python3默认协商这个object，新式类，有一些多出来的功能
    def test(self):
        print("-----Base----")

class A(Base):
    def test1(self):
        print("----a-----")

class B(Base):
    def test2(self):
        print("-----b-----")

class C(A, B):
    def test2(self):
        print("-----c-----")

c = C()
c.test() #如果c类里自己有就调用自己的，如果没有就调用别人的，最后才会调用新式类（object类）
c.test1()
c.test()
c.test2()

print(C.__mro__)

#一般架构师在设计多个类时会注意到尽量避免不同类里面有相同的方法和方法名，真的不能避免了，就用之前讲过的调用被重写的父类方法方法即可
