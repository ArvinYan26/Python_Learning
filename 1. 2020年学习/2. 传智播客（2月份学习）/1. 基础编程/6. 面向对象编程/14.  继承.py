#子类可以继承父类的方法，但是子类之间不能相互继承和使用方法
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
    """
    def eat(self):
        print("吃")
    def drink(self):
        print("喝")
    def sleep(self):
        print("睡")
    def run(self):
        print("跑")
    """
    def bark(self):
        print("汪汪")

class Cat(Animal):
    def catah(self):
        print("抓老鼠")

#a = Animal()
#a.eat()
wangcai = Dog()
wangcai.eat()

tom = Cat()
tom.eat()
tom.sajiao()