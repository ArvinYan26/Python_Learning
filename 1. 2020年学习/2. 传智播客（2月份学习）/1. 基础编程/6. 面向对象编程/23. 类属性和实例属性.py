#类在程序里也是对象，叫类对象，类对象里的属性叫类属性
#通过名字创建的对象也是对象，叫实例对象，实例对象里面的的属性叫实例属性
#实例对象之间的属性不能相互调用，而类属性是共享的，所有实例对象都可以调用它
#；类属性只会定义一次，不会因为创立了多个实例对象而改变

class Tool(object):  #类对象
    #类属性
    num = 0
    #方法
    def __init__(self, new_name):
        #实例属性
        self.name = new_name
        #对类属性+=1
        Tool.num += 1

tool1 = Tool("铁锹")  #tool1：实例对象
tool2 = Tool("工兵铲")
tool3 = Tool("水桶")

print(Tool.num)
