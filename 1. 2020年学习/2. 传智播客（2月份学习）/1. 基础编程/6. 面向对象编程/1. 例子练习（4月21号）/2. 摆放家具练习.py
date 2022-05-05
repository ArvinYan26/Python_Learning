class House:
    def __init__(self, new_area, new_info, new_addr):
        """初始化房子信息，房子面积，大小信息，地址"""
        self.area = new_area
        self.info = new_info
        self.addr = new_addr
        self.left_area = new_area
        self.contain_items = []


    def __str__(self):
        """打印房子当前信息"""
        msg = "房子的面积是：%d, 房子的可用面积：%d, 房子的户型：%s, \n地址是：%s"%(self.area, self.left_area, self.info, self.addr)
        msg += "\n当前房子里的物品有：%s"%(str(self.contain_items))
        return msg

    def add_item(self, item):
        """添加家具"""
        self.contain_items.append(item.get_name())
        self.left_area -= item.get_area()


class Bed:
    def __init__(self, new_name, new_area):
        """初始化床de信息"""
        self.name = new_name
        self.area = new_area

    def __str__(self):
        bed_msg = "床的种类是：%s, 床的面积是：%d"%(self.name, self.area)
        return bed_msg

    def get_area(self):
        return self.area

    def get_name(self):
        return self.name

home = House(129, "三室一厅", "北京市 朝阳区 长安街 666号")
print(home)

bed = Bed("席梦思", 4)
print(bed)

home.add_item(bed)
print(home)

bed1 = Bed("三人床", 6)
home.add_item(bed1)
print(home)