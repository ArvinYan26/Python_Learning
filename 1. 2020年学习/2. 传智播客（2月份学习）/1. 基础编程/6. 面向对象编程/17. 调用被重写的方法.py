#子类可以继承父类的方法，但是子类之间不能相互继承和使用方法
#继承父类的父类的功能，扩展为后可以继承祖先的所有功能
#重写：父类里面已经有的方法，子类里面重新定义了一个相同名字的方法，这就是重写
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

    def bark(self):
        print("狂叫")
        # 调用被重写的第一种方法
        Dog.bark(self)
        #第二种
        super().bark()  

xiaotq = Xiaotq()
xiaotq.fly()
xiaotq.bark() #在xiaotq类里找，如果有（就是重写）就直接调用自己的，如果没有就用父类的，
xiaotq.sajiao() #继承的父类的父类（爷爷）的功能
