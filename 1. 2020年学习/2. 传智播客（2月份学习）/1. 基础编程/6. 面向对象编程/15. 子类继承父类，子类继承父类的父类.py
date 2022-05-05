#子类可以继承父类的方法，但是子类之间不能相互继承和使用方法
#继承父类的父类的功能，扩展为后可以继承祖先的所有功能
class Animal:
    def eat(self):
        print("吃")
    def drink(self):
        print("喝")
    def sleep(self):
        print("睡")
    def run(self):
        print("跑")
    def sajiao(self):
        print("撒娇")

class Dog(Animal):
    def bark(self):
        print("汪汪")
class Xiaotq(Dog):
    def fly(self):
        print("fly")

xiaotq = Xiaotq()
xiaotq.fly()
xiaotq.bark() #继承的父类功能
xiaotq.sajiao() #继承的父类的父类（爷爷）的功能
