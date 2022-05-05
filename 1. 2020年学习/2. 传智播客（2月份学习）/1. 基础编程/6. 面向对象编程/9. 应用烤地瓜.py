class SweetPotato:

    def __init__(self):
        """默认属性（初始化），如果后边变化，会再变化，这里只是初始化"""
        self.cookedString = "生的"
        self.cookedLevel = 0  #时间是0表示还没开始烤是生的
        self.condiments = []
    def __str__(self):
        return "地瓜 状态：%s(%d),添加的作料有：%s"%(self.cookedString, self.cookedLevel, str(self.condiments))
            #不管是字典列表或者元组，只要加了str，那么打印出来的就是字符串

    def cook(self, cooked_time):
        #因为这个方法被调用多次，为了能够在一次调用这个发那个发誓能够获取上一次调用这个方法的cooked_time
        #所以需要在此，把cooked_time保存到这个对象的属性中去，因为属性不会随着方法的调用而结束
        self.cookedLevel += cooked_time #要把时间存起来

        if self.cookedLevel >= 0 and self.cookedLevel < 3:
            self.coookedString = "生的"
        elif self.cookedLevel >= 3 and self.cookedLevel < 5:
            self.coookedString = "半生不熟"
        elif self.cookedLevel >=5 and self.cookedLevel < 8:
            self.coookedString = "熟了"
        elif self.cookedLevel >8:
            self.coookedString = "烤糊了"
    def addCondiments(self, item):
        self.condiments.append(item)

#创建了地瓜对象
di_gua = SweetPotato()
print(di_gua)

#开始烤地瓜
di_gua.cook(3)
di_gua.addCondiments("大蒜")
di_gua.addCondiments("番茄酱")
di_gua.addCondiments("辣椒")
print(di_gua)
