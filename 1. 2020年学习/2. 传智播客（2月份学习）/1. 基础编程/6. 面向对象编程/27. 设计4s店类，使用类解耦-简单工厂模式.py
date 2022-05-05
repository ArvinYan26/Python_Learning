class Carstore(object):

    def __init__(self):
        self.factory = Factory() #添加了factory这个属性，又创建了factory这实例对象指向Factory这个对象

    def order(self, car_type):
        return self.factory.select_car_by_type(car_type)
        #取出来factory这个属性，即是指向了Factory这个对象，去调用它的方法
        #如果下面创建了suonata对象，那么self.factory.select_car_by_type就指向了索纳塔这个对象
class Factory(object):
    def select_car_by_type(self, car_type):
        if car_type == "索纳塔":
            return Suonata()
        elif car_type == "名图":
            return Mingtu()
        elif car_type == "Ix35":
            return Ix35()


class Car(object):
    def move(self):
        print("车在移动、、、、、")

    def music(self):
        print("音乐、、、、、")

    def stop(self):
        print("停止、、、、、")

class Suonata(Car):
    pass

class Mingtu(Car):
    pass

class Ix35(Car):
    pass

car_store = Carstore()
car = car_store.order("索纳塔")
car.move()
car.music()
car.stop()