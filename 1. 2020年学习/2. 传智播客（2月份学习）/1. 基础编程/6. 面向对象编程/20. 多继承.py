#object:称为新式类，如果写上，就代表是所有类的基类，即boss，python3中独有的，如果不写就默认为经典类

class Base(object):  #python3默认协商这个object，新式类，有一些多出来的功能
    def test(self):
        print("-----Base----")

class A(Base):
    def test1(self):
        print("----test1-----")

class B(Base):
    def test2(self):
        print("-----test2-----")

class C(A, B):
    pass

c = C()
c.test1()
c.test()
c.test2()
