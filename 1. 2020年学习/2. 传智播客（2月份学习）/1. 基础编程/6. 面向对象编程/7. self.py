class Cat:
    #属性

    #方法()
    def eat(self):
        print("毛正在吃鱼，嘻嘻、、、")
    def drink(self):
        print("猫正在喝水,哈哈、、、")
    def introduce(self):  #self是形参，至少要保证接受一个参数，用来传递当前的对象
                          #其他的变量名也行，但是为了都能读懂，一般用self
        print("%s的年龄：%d" % (self.name, self.age))

#创建一个对象
tom = Cat()  #tom之下那个Cat（）的地址即引用

#调用tom指向的对象中的方法
tom.eat()
tom.drink()

#给tom指向的这个对象添加属性
tom.name = "汤姆"
tom.age = 40

#获取属性的一种方式
#print("%s的年龄：%d" % (tom.name, tom.age))

#获取属性的二种方式
tom.introduce()

lanmao = Cat()
lanmao.name = "蓝猫"
lanmao.age = 10
lanmao.introduce()  #相当于lanmao.introduce(lanmao)
