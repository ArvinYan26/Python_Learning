class Carstore(object):
    def order(self, car_type):
        if car_type == "索纳塔":
            return Suonata()
        elif car_type == "名图":
            return Mingtu()

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


car_store = Carstore()
car = car_store.order("索纳塔")
car.move()
car.music()
car.stop()