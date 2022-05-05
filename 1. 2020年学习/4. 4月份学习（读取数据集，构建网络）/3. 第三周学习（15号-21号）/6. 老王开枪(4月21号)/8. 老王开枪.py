class Person(object):
    """人的类"""    #核心思想，创建类1，想要在类2中代用类1的方法，就在类2中定义一个方法，指向类1就行

    def __init__(self, name):
        super(Person, self).__init__()  # 可能会调用父类的方法
        self.name = name
        self.gun = None #用来保存枪对象的引用
        self.hp = 100  #初始化对象血量为满血
    def anzhuang_zidan(self, dan_jia_temp, zi_dan_temp):
        """把子弹安装到弹夹中"""
        # 弹夹.保存子弹（子弹）#调用方法
        dan_jia_temp.baocun_zidan(zi_dan_temp)

    def anzhuang_danjia(self, gun_temp, dan_jia_temp):
        """把弹夹安装到枪中"""
        # 枪.保存弹夹（弹夹）
        gun_temp.baocun_danjia(dan_jia_temp)

    def naqiang(self, gun_temp):
        """老王拿起一把枪"""
        self.gun = gun_temp

    def __str__(self): #老王的拿枪信息，和血量信息
        if self.gun:
            return "%s老王的血量是：%d， 他有枪%s"%(self.name, self.hp, self.gun)
        else:
            if self.hp > 0:
                return "%s敌人的血量是：%d， 他没枪"%(self.name, self.hp)
            else:
                return "%s 已挂。。。" % self.name

    def kou_ban_ji(self, diren):
        """让枪发射子弹打敌人"""
        #枪.开火（敌人）
        self.gun.fire(diren) #枪开火，所以开火属性是枪的去枪中找fire方法

    def diao_xue(self, sha_shang_li):
        """根据杀伤力，掉响应的血量"""
        self.hp -= sha_shang_li

class Gun(object):
    """枪类"""

    def __init__(self, name):
        super(Gun, self).__init__()  # 可能会调用父类的方法
        self.name = name  # 用来记录枪的类型
        self.danjia = None  # 因为最初这个类没有弹夹

    def baocun_danjia(self, dan_jia_temp):
        """用一个属性来保存弹夹的引用"""
        self.danjia = dan_jia_temp

    def __str__(self):
        if self.danjia:
            return "枪的信息为：%s，%s"%(self.name, self.danjia)
        else:
            return "枪的信息为：%s，枪中没有弹夹"%(self.name)

    def fire(self, diren):
        """枪从弹夹中获取一发子弹，来击中敌人"""
        #先从弹夹中取子弹
        zidan_temp = self.danjia.tanchu_zidan() #弹夹弹出子弹，所以方法属于弹夹
        #让这个子弹去打敌人(zi_dan_temp可能是空，所以先判断)
        if zidan_temp:
            #子弹.打中敌人（敌人） #打中敌人是子弹的方法，所以去子弹中找方法
            zidan_temp.dazhong(diren)
        else:
            print("弹夹中没有子弹了。。。。。")

class Danjia(object):
    """弹夹类"""

    def __init__(self, max_num):
        super(Danjia, self).__init__()  # 可能会调用父类的方法
        self.max_num = max_num  # 记录弹夹的最大容量
        self.zidan_list = []  # 用来保存子弹的引用(当前容量)

    def baocun_zidan(self, zi_dan_temp):
        """将子弹保存下来"""
        self.zidan_list.append(zi_dan_temp)

    def __str__(self):
        return "弹夹的信息：%d/%d"%(len(self.zidan_list), self.max_num)

    def tanchu_zidan(self):
        """让弹夹取出来弹夹中最上面的那颗子弹"""
        if self.zidan_list: #如过列表不是空的，就返回True，然后执行if语句，
            return self.zidan_list.pop() #必须有返回，否则子弹就弹不出来
        else:
            return None

class Zidan(object):
    """子弹类"""
    def __init__(self, sha_shang_li):
        super(Zidan, self).__init__()  # 可能会调用父类的方法
        self.sha_shang_li = sha_shang_li

    def dazhong(self, diren):
        """让敌人掉血"""
        #敌人.掉血（一颗子弹的威力） #掉血这个方法，实在敌人那里面，所以回人的类中找方法
        diren.diao_xue(self.sha_shang_li)

def main():
    """用来控制整个程序的流程"""
    # 1. 创建老王对象
    laowang = Person("老王")
    # 2. 创建一个枪对象
    ak47 = Gun("AK47")
    # 3. 创建一个弹夹对象
    dan_jia = Danjia(20)
    # 4. 创建一些子弹对象
    for i in range(15): #面向对象设计的时候想要重复执行，前面直接加for循环记性
        zi_dan = Zidan(10)
        # 5. 老王把子弹安装到弹夹中（老王有这个能力，即方法）
        laowang.anzhuang_zidan(dan_jia, zi_dan)
    # 6. 老王把弹夹安装到枪中
    laowang.anzhuang_danjia(ak47, dan_jia)
    #测试弹夹
    #print(dan_jia)  #用__str__方法
    #测试枪的信息
    #print(ak47)  # 用__str__方法
    # 7. 老王拿枪
    #老王.拿枪（枪）
    laowang.naqiang(ak47)
    #测试老王
    print(laowang) #yong __str__方法
    # 8. 创建一个敌人对象
    gebi_laosong = Person("隔壁老宋")
    print(gebi_laosong)
    # 9. 老王开枪打敌人
    #老王.扣扳机（隔壁老宋）
    for i in range(12):
        laowang.kou_ban_ji(gebi_laosong)
        print(gebi_laosong)

if __name__ == "__main__":
    main()