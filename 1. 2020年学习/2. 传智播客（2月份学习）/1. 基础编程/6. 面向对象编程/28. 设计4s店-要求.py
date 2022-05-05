class Store(object):
    """基类只是搭建框架，然后负责调用子类的方法实现整个代码的运行"""
    def select_car(self):
        """虽然定义了这种方法，但是不去实现，让子类去实现"""
        pass

    def order(self, car_type):
        return self.select_car(car_type)

class BMWCarStore(Store): #继承基类
    def select_car(self, car_type): #方法名字和基类的相同，意味着是重写方法，
        return BMWFactory().select_car_by_type(car_type)

class CarStore(Store):
    def select_car(self, car_type):
        return Factory().select_car_by_type(car_type)

class BMWFactory(object):
    def select_car_by_type(self, car_type):
        pass

        """
        if car_type == "索纳塔":
            return Suonata()
        elif car_type == "名图":
            return Mingtu()
        elif car_type == "Ix35":
            return Ix35()
        """
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

bmw_store = BMWCarStore()
bmw = bmw_store.order("720li")