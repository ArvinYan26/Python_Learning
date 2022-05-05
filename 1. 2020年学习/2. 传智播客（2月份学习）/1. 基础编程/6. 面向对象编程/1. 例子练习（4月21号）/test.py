class SweetPoato:
    def __init__(self):
        """默认属性初始化"""
        self.cookedstring = "生的"
        self.cookedleavel = 0 #刚开始是0，因为没有开始烤
        self.condiments = [] #定义空列表，后期加进去作料

    def __str__(self): #默认调用的方法
        return "地瓜：状态： %s(%d), 添加的作料有：%s" %(self.cookedstring, self.cookedleavel, str(self.condiments))

    def cook(self, cook_time):
        self.cookedleavel += cook_time

        if self.cookedleavel >= 0 and self.cookedleavel < 3:
            self.cookedstring = "生的"
        elif self.cookedleavel >= 3 and self.cookedleavel < 5:
            self.cookedstring = "半生不熟"
        elif self.cookedleavel >= 5 and self.cookedleavel < 8:
            self.cookedstring = "熟了"
        elif self.cookedleavel >= 8:
            self.cookedstring = "糊了"

    def addcondiments(self, item):
        self.condiments.append(item)



di_gua = SweetPoato()
print(di_gua)
di_gua.cook(6)
di_gua.addcondiments("大蒜")
print(di_gua)

