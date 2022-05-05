class Cat:
    """定义了一个Cat类"""

    def __init__(self, new_name, new_age): #self:哪个对象指向self，self指向谁
        """初始化对象（默认的属性）"""
        self.name = new_name
        self.age = new_age

    #python解释器会自动的调用这个方法
    def __str__(self): #此时的self指向的是通过__init__方法添加过属性的新的对象，所以下面的self.name可用
        return "%s的年龄是：%d"%(self.name, self.age)

    def eat(self):
        print("猫正在吃鱼，嘻嘻、、、")
    def drink(self):
        print("猫正在喝水,哈哈、、、")
    def introduce(self):  #self是形参，至少要保证接受一个参数，用来传递当前的对象
                          #其他的变量名也行，但是为了都能读懂，一般用self
        print("%s的年龄：%d" % (self.name, self.age))

#创建一个对象(每一二个)
#不同对象的属性之间不影响，因为每个对象所指向的地址不同，开辟的都是新地址
tom = Cat("汤姆", 40)  #tom之下那个Cat（）的地址即引用

#创建第二个对象
lanmao = Cat("蓝猫", 10)

print(lanmao) #__str__方法里面return的是什么就打印什么
print(tom)
