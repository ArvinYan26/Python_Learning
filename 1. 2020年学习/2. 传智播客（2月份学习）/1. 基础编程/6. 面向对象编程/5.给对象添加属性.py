class Cat:
    #属性

    #方法()
    def eat(self):
        print("毛正在吃鱼，嘻嘻、、、")
    def drink(self):
        print("猫正在喝水,哈哈、、、")

#创建一个对象
tom = Cat()  #tom之下那个Cat（）的地址即引用

#调用tom指向的对象中的方法
tom.eat()
tom.drink()

#给tom指向的这个对象添加属性
tom.name = "汤姆"
tom.age = 40

#获取属性
print("%s的年龄：%d" % (tom.name, tom.age))
