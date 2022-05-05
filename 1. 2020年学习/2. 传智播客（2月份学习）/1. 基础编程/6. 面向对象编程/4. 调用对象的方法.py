class Cat:
    #属性

    #方法()
    def eat(self):
        print("毛正在吃鱼，嘻嘻、、、")
    def drink(self):
        print("猫正在喝水,哈哈、、、")

#创建一个对象
tom = Cat()  #tom之下那个Cat（）的地址即引用

#调用对象的方法
tom.eat()
tom.drink()