class CarStore(object):
    def __init__(self):
        """初始化属性"""
        self.factory = Factory()

    def order(self):
        return self.factory.select_car_by_type()

#肯定是4s店预定车，然后工厂根据预定去选择要造的车，4s店不能造车，所以，通过工厂这个类来解耦
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
        print("run")

    def music(self):
        print("music")

    def stop(self):
        print("stop")

class Suonata(Car):
    pass

class Mingtu(Car):
    pass

class Ix35(Car):
    pass

car_store = CarStore()
car = CarStore.order("名图")
car.move()
car.music()
car.stop()
