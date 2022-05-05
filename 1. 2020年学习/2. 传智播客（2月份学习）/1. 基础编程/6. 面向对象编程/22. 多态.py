class Dog(object):
    def print_self(self):
        print("大家好----")

class Xiaotq(Dog):
    def print_self(self):
        print("hello everybody-----")

def introduce(tem): #不能知道tem到底是指向的谁，只有执行的时候才知道，这就是多态。C和C++里面会很明确的告诉你调用的谁的
    tem.print_self()

dog1 = Dog()
dog2 = Xiaotq()

introduce(dog1)
introduce(dog2)
#Dog and Xiaotq 都有这个print_self()方法，但是调用函数时并不知道调用的是哪个类的，只有真正执行时才知道具体调用的谁的，这就是多态
#python即支持面向过程也支持面向对象编程，面向对象编程的三要素是：封装，继承，多态
#封装：把函数和全局变量封在一起就是封装
#继承：子类继承父类功能
#多态：定义的时候不知道调用谁的功能，只有执行的时候才知道是调用的子类还是父类的功能