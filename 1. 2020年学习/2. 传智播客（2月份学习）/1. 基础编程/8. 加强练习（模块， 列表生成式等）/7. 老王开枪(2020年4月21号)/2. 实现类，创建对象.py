class Person(object):
    """人的类"""
    def __init__(self, name):
        #调用父类的方法一
        super(Person, self).__init__() #self不用传
        # 可能会调用父类的方法二
        #Person.__init__(self) #self需要传

        self.name = name

class Gun(object):
    """枪类"""
    def __init__(self, name):
        super(Gun, self).__init__() #可能会调用父类的方法
        self.name = name #用来记录枪的类型

class Danjia(object):
    """弹夹类"""
    def __init__(self, max_num):
        super(Danjia, self).__init__()   #可能会调用父类的方法
        self.max_num = max_num    #记录弹夹的最大容量

class Zidan(object):
    """子弹类"""
    def __init__(self, sha_shang_li):
        super(Zidan, self).__init__() #可能会调用父类的方法
        self.sha_shang_li = sha_shang_li


def main():
    """用来控制整个程序的流程"""

    #1. 创建老王对象
    laowang = Person("老王")
    #2. 创建一个枪对象
    ak47 = Gun("AK47")
    #3. 创建一个弹夹对象
    dan_jia = Danjia(20)
    #4. 创建一些子弹对象
    zi_dan = Zidan(10)
    #5. 老王把子弹安装到弹夹中
    #6. 老王把弹夹安装到枪中
    #7. 老王拿枪
    #8. 老王开强打敌人
    #9. 创建一个敌人对象

if __name__ == "__main__":
    main()