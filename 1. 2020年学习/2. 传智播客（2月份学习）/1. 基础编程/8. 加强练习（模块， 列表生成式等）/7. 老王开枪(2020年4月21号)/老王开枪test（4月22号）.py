class Person(object):
    """创建人类"""
    def __init__(self, name):
        #调用父类方法,用下面两种方法
        #Person.__init__(self)
        #super(Person, self).__init__()
        self.name = name
        self.gun = None
        self.hp = 100 #初始化血量

    def anzhuang_zidan(self, danjia_temp, zidan_temp):
        """把子弹装到弹夹里面"""
        danjia_temp.baocun_zidan(zidan_temp)

    def anzhuang_danjia(self, gun_temp, danjia_temp):
        """把弹夹安装到枪里面"""
        gun_temp.baocun_danjia(danjia_temp)

    def naqiang(self, gun_temp):
        """老王拿枪"""
        self.gun = gun_temp

    def __str__(self): #
        if self.gun:
            return "%s hp is ：%d,  he has one gun %s" % (self.name, self.hp, self.gun)
        else:
            if self.hp > 0:
                return "%s hp is ：%d, he has no gun" % (self.name, self.hp)
            else:
                return "%s is over......." % self.name

    def kou_ban_ji(self, enemy):
        """老枪扣扳机，是老王的方法，但是开火是枪的方法"""
        self.gun.fire(enemy)
        # 因为self.gun此时指向的是gun_temp,而gun_temp指向的是ak47即Gun,所以从Gun调用fire方法

    def diao_xue(self, sha_shang_li):
        """掉血功能"""
        self.hp -= sha_shang_li

class Gun(object):
    """枪类"""
    def __init__(self, name):
        #Gun.__init__(self)
        self.name = name
        self.danjia = None #初始化枪的属性，没有弹夹

    def baocun_danjia(self, danjia_temp):
        """用一个属性保存弹夹"""
        self.danjia = danjia_temp

    def __str__(self):
        """打印信息"""
        #先判断是否有弹夹
        if self.danjia:
            return "gun's info ：%s， %s" % (self.name, self.danjia)
        else:
            return "gun's info ：%s，ther is no dnajai in gun" % (self.name)

    def fire(self, enemy):
        """枪从弹夹中获取一枚最上边的子弹，来打敌人"""
        #先从弹夹中获取子弹
        zidan_temp = self.danjia.tanchu_zidan() #tanchu_zidan()是弹夹的方法，需要一个变量去接收这枚子弹
        #先判断弹夹是否为空，真，就可以取出来子弹，打敌人，假就是没子弹，
        if zidan_temp:
            zidan_temp.dazhong(enemy) #打中是子弹打中，所以方法属于子弹
        else:
            print("there is no bullet in the danjia......")


class DanJia(object):
    """弹夹类"""
    def __init__(self, max_num): #弹夹需要有容量
        #DanJia.__init__(self)
        self.max_num = max_num #弹夹最大容量
        self.zidan_list = []   #用来保存子弹的引用，添加一颗，列表中就多一颗

    def baocun_zidan(self, zidan_temp):
        self.zidan_list.append(zidan_temp)

    def __str__(self):
        return "dnajai's info ：%d/%d" % (len(self.zidan_list), self.max_num)

    def tanchu_zidan(self):
        """从弹夹中取出最上面的一枚子弹"""
        #同样先判断弹夹是都为空
        if self.zidan_list:
            return self.zidan_list.pop() #必须有返回，否则子弹弹不出去
        else:
            return None


class ZiDan(object):
    """"子弹类"""
    def __init__(self, sha_shang_li):
        self.sha_shang_li = sha_shang_li

    def __str__(self):
        return "子弹掉血：%d" % (self.sha_shang_li)

    def dazhong(self, enemy): #打中敌人，所以形参是enemy，最终是实参老宋
        """让敌人掉血， 所以掉血是人类的方法,掉的血就是sha_shang_li"""
        enemy.diao_xue(self.sha_shang_li)



def main():
    """用来控制整个流程"""

    #1.创建老王对象
    laowang = Person("Soldier")
    #2. 创建枪对象
    ak47 = Gun("AK47")
    #3. 创建弹夹对象
    danjia = DanJia(20) #传递给弹夹容量20
    #4. 创建子弹对象
    zidan = ZiDan(10) #子弹杀伤力为10
    for i in range(15):
        #5. 老王把子弹装到弹夹里
        laowang.anzhuang_zidan(danjia, zidan)
    #6. 老王把弹夹装到枪里
    laowang.anzhuang_danjia(ak47, danjia)
    #测试弹夹和枪
    print(danjia)
    print(ak47)
    print(ZiDan)
    #7. 老王拿枪
    laowang.naqiang(ak47)
    #测试老王
    print(laowang)
    #8. 创建敌人对象
    laosong  = Person("Master")
    #测试敌人老宋
    print(laosong)
    #9. 老王开枪打敌人老宋
    for i in range(12):
        laowang.kou_ban_ji(laosong)
        print(laosong)


if __name__ == "__main__":
    main()