class Store(object):
    """基类只是设计大的框架，然后让子类去完成，起到控制全局作用"""
    def select_car(self):
        pass
    def order(self, car_type):
        return self.select_car(car_type)

class Carstore(Store):  #继承的基类
    def select_car(self, car_type): #方法和基类相同意味着是重写方法
        return Factory().select_car_by_type(car_type)

class BMWCarstore(Store):
    def select_car(self, car_type):
        return Factory().select_car_by_type(car_type)

class BMWFactory(object):
    def select_car_by_type(self, car_type):
        pass


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

bmw_store = BMWCarstore()
bmw = bmw_store.order("720li")